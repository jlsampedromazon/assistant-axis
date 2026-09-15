#!/usr/bin/env python3
"""
Score role responses using a judge LLM.

This script scores how well model responses adhere to their assigned roles
using an LLM judge (e.g., GPT-4). Scores are on a 0-3 scale:
    0: Model refused to answer
    1: Model says it can't be the role, but can help with related tasks
    2: Model identifies as AI/LLM but has some role attributes
    3: Model is fully playing the role

Alongside --output_dir/{role}.json (the parsed int scores, unchanged format),
this also writes --output_dir/../judge_raw/{role}.json: the same keys mapped
to {"score": ..., "raw_completion": ...}, so the judge's raw completion text
survives for audit/spot-check even when parse_judge_score() fails or a
completion looks ambiguous.

Usage:
    uv run scripts/3_judge.py \
        --responses_dir outputs/gemma-2-27b/responses \
        --roles_dir data/prompts/roles \
        --output_dir outputs/gemma-2-27b/scores \
        --judge_model gpt-4.1-mini
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import jsonlines
import yaml
from dotenv import load_dotenv
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from assistant_axis.judge import RateLimiter, call_judge_batch, parse_judge_score
import openai

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress verbose HTTP logging from httpx/openai
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)


def load_role_eval_prompt(role_file: str) -> str:
    """Load eval_prompt from role JSON file."""
    with open(role_file, 'r') as f:
        data = json.load(f)
    return data.get("eval_prompt", "")


def load_responses(responses_file: Path) -> List[dict]:
    """Load responses from JSONL file."""
    responses = []
    with jsonlines.open(responses_file, 'r') as reader:
        for entry in reader:
            responses.append(entry)
    return responses


def load_questions_by_index(questions_file: Optional[str]) -> Dict[int, dict]:
    """Load optional question metadata keyed by zero-based question_index."""
    if not questions_file:
        return {}
    path = Path(questions_file)
    if not path.exists():
        raise FileNotFoundError(f"Questions file does not exist: {path}")

    questions_by_index = {}
    # Stage 3 score keys only retain q-index, so metadata is joined by JSONL order.
    with jsonlines.open(path, 'r') as reader:
        for question_index, question in enumerate(reader):
            if not isinstance(question, dict):
                raise ValueError(f"Question row {question_index} must be an object: {path}")
            questions_by_index[question_index] = question
    return questions_by_index


def load_role_question_guidance(judge_guidance_file: Optional[str]) -> Optional[dict]:
    """Load optional role-by-question judge guidance YAML."""
    if not judge_guidance_file:
        return None
    path = Path(judge_guidance_file)
    if not path.exists():
        raise FileNotFoundError(f"Judge guidance file does not exist: {path}")
    with path.open('r') as f:
        guidance = yaml.safe_load(f) or {}
    if not isinstance(guidance, dict):
        raise ValueError(f"Judge guidance file must contain a mapping: {path}")
    return guidance


def shared_trait_rule_from_guidance(judge_guidance: Optional[dict]) -> str:
    """Return shared guidance text when configured."""
    if not judge_guidance:
        return ""
    shared_trait_rule = judge_guidance.get("shared_trait_rule", "")
    if shared_trait_rule is None:
        return ""
    if not isinstance(shared_trait_rule, str):
        raise ValueError("judge guidance shared_trait_rule must be a string when provided.")
    return shared_trait_rule


def role_question_guidance_entry(judge_guidance: dict, role_id: str, question_id: str) -> Any:
    """Find guidance for one role/question pair across supported YAML layouts."""
    if isinstance(judge_guidance.get("roles"), dict):
        role_block = judge_guidance["roles"].get(role_id)
        if isinstance(role_block, dict):
            questions_block = role_block.get("questions", role_block)
            if isinstance(questions_block, dict) and question_id in questions_block:
                return questions_block[question_id]

    if isinstance(judge_guidance.get("role_question_guidance"), dict):
        role_block = judge_guidance["role_question_guidance"].get(role_id)
        if isinstance(role_block, dict) and question_id in role_block:
            return role_block[question_id]

    # The v1 guidance schema stores entries under guidance -> role_id -> question_id.
    if isinstance(judge_guidance.get("guidance"), dict):
        role_block = judge_guidance["guidance"].get(role_id)
        if isinstance(role_block, dict) and question_id in role_block:
            return role_block[question_id]

    if isinstance(judge_guidance.get("guidance"), list):
        for entry in judge_guidance["guidance"]:
            if (
                isinstance(entry, dict)
                and entry.get("role_id") == role_id
                and entry.get("question_id") == question_id
            ):
                return entry

    role_block = judge_guidance.get(role_id)
    if isinstance(role_block, dict) and question_id in role_block:
        return role_block[question_id]

    raise KeyError(f"Missing judge guidance for role_id={role_id!r}, question_id={question_id!r}")


def format_guidance_values(values: Any) -> str:
    """Format scalar or list guidance values for prompt insertion."""
    if values is None:
        return ""
    if isinstance(values, list):
        return "\n".join(f"- {value}" for value in values)
    return f"- {values}"


def format_role_question_guidance(
    judge_guidance: Optional[dict],
    role_id: str,
    question_id: str,
) -> str:
    """Format optional role-question guidance using advisory wording."""
    if not judge_guidance:
        return ""

    entry = role_question_guidance_entry(judge_guidance, role_id, question_id)
    if isinstance(entry, str):
        entry = {"strong_signals": [entry]}
    if not isinstance(entry, dict):
        raise ValueError(f"Judge guidance for {role_id}/{question_id} must be a mapping or string.")

    # Guidance is phrased as examples so the role rubric remains the scoring authority.
    sections = [
        ("Strong signals may include...", entry.get("strong_signals", entry.get("strong"))),
        ("Valid subtle signals may include...", entry.get("valid_subtle_signals", entry.get("subtle"))),
        ("Weaker or generic signs may include...", entry.get("weaker_generic_signs", entry.get("weak"))),
        (
            "Saturation or wrong-role risks may include...",
            entry.get("saturation_wrong_role_risks", entry.get("risks")),
        ),
    ]
    lines = [
        "Role-question advisory guidance:",
        "The following guidance is advisory. Do not require every listed item.",
    ]
    for heading, values in sections:
        formatted_values = format_guidance_values(values)
        if formatted_values:
            lines.append(heading)
            lines.append(formatted_values)
    return "\n".join(lines)


def format_eval_prompt(
    eval_prompt_template: str,
    role_id: str,
    question_index: int,
    question: str,
    answer: str,
    questions_by_index: Dict[int, dict],
    judge_guidance: Optional[dict],
) -> str:
    """Populate legacy and v3 eval prompt placeholders."""
    question_metadata = questions_by_index.get(question_index, {})
    question_id = str(question_metadata.get("id", question_index))
    probe_family = str(question_metadata.get("probe_family", ""))
    return eval_prompt_template.format(
        question=question,
        answer=answer,
        shared_trait_rule=shared_trait_rule_from_guidance(judge_guidance),
        role_id=role_id,
        question_id=question_id,
        probe_family=probe_family,
        role_question_guidance=format_role_question_guidance(judge_guidance, role_id, question_id),
    )


def validate_guidance_for_responses(
    role_id: str,
    responses: List[dict],
    questions_by_index: Dict[int, dict],
    judge_guidance: Optional[dict],
) -> None:
    """Fail before judging if configured role-question guidance is incomplete."""
    if not judge_guidance:
        return
    for response in responses:
        question_index = response["question_index"]
        question_metadata = questions_by_index.get(question_index, {})
        question_id = str(question_metadata.get("id", question_index))
        role_question_guidance_entry(judge_guidance, role_id, question_id)


async def process_role(
    role: str,
    responses: List[dict],
    eval_prompt_template: str,
    client: openai.AsyncOpenAI,
    rate_limiter: RateLimiter,
    judge_model: str,
    max_tokens: int,
    batch_size: int,
    existing_scores: Dict[str, int],
    questions_by_index: Dict[int, dict],
    judge_guidance: Optional[dict],
) -> tuple[dict, dict]:
    """Process a single role and return (scores, raw_completions).

    raw_completions carries the judge model's raw completion text alongside
    the parsed score for every prompt actually sent this call, keyed the same
    way as scores -- including prompts where parsing failed or the API call
    returned nothing, so a mis-parse is auditable rather than silently lost.
    """
    # Build prompts for each response
    prompts = []
    keys = []

    for resp in responses:
        prompt_idx = resp["prompt_index"]
        question_idx = resp["question_index"]
        question = resp["question"]
        label = resp["label"]

        # Get assistant response from conversation
        assistant_response = ""
        for msg in resp["conversation"]:
            if msg["role"] == "assistant":
                assistant_response = msg["content"]
                break

        key = f"{label}_p{prompt_idx}_q{question_idx}"

        # Skip if already scored
        if key in existing_scores:
            continue

        # Extra metadata placeholders are empty unless optional files are supplied.
        judge_prompt = format_eval_prompt(
            eval_prompt_template=eval_prompt_template,
            role_id=role,
            question_index=question_idx,
            question=question,
            answer=assistant_response,
            questions_by_index=questions_by_index,
            judge_guidance=judge_guidance,
        )
        prompts.append(judge_prompt)
        keys.append(key)

    if not prompts:
        return {}, {}

    # Call judge
    logger.info(f"Scoring {len(prompts)} new responses for {role}...")
    responses_text = await call_judge_batch(
        client=client,
        prompts=prompts,
        model=judge_model,
        max_tokens=max_tokens,
        rate_limiter=rate_limiter,
        batch_size=batch_size
    )

    # Parse scores
    scores = {}
    raw_completions = {}
    for key, response_text in zip(keys, responses_text):
        score = parse_judge_score(response_text) if response_text else None
        if score is not None:
            scores[key] = score
        raw_completions[key] = {"score": score, "raw_completion": response_text}

    return scores, raw_completions


async def main_async():
    parser = argparse.ArgumentParser(description="Score role responses with judge LLM")
    parser.add_argument("--responses_dir", type=str, required=True, help="Directory with response JSONL files")
    parser.add_argument("--roles_dir", type=str, default="../data/roles/instructions", help="Directory containing role JSON files")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for score JSON files")
    parser.add_argument("--judge_model", type=str, default="gpt-4.1-mini", help="Judge model to use")
    parser.add_argument("--max_tokens", type=int, default=10, help="Max tokens for judge response")
    parser.add_argument("--batch_size", type=int, default=50, help="Concurrent batch size")
    parser.add_argument("--requests_per_second", type=int, default=100, help="Rate limit")
    parser.add_argument("--roles", nargs="+", help="Specific roles to process")
    parser.add_argument("--dry_run", action="store_true", help="Preview what would be processed without making API calls")
    parser.add_argument("--questions_file", type=str, help="Optional JSONL file with question metadata")
    parser.add_argument("--judge_guidance_file", type=str, help="Optional YAML role-question judge guidance")
    args = parser.parse_args()

    # Check for API key (not needed for dry run)
    if not args.dry_run and not os.getenv("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY not found")
        sys.exit(1)

    # Create output directory
    output_dir = Path(args.output_dir)
    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    responses_dir = Path(args.responses_dir)
    roles_dir = Path(args.roles_dir)
    questions_by_index = load_questions_by_index(args.questions_file)
    judge_guidance = load_role_question_guidance(args.judge_guidance_file)

    # Get response files
    response_files = sorted(responses_dir.glob("*.jsonl"))
    logger.info(f"Found {len(response_files)} response files")

    # Filter roles if specified
    if args.roles:
        response_files = [f for f in response_files if f.stem in args.roles]

    logger.info(f"Processing {len(response_files)} roles")

    # Dry run mode
    if args.dry_run:
        logger.info("Dry run mode - no API calls will be made")
        total_prompts = 0
        sample_shown = False

        for response_file in response_files:
            role = response_file.stem
            output_file = output_dir / f"{role}.json"

            # Load existing scores
            existing_scores = {}
            if output_file.exists():
                try:
                    with open(output_file, 'r') as f:
                        existing_scores = json.load(f)
                except Exception:
                    pass

            # Get role eval prompt
            role_file = roles_dir / f"{role}.json"
            if not role_file.exists():
                logger.info(f"  {role}: no role file, skipping")
                continue

            eval_prompt_template = load_role_eval_prompt(role_file)
            if not eval_prompt_template:
                logger.info(f"  {role}: no eval_prompt, skipping")
                continue

            # Load responses and count prompts to be scored
            responses = load_responses(response_file)
            validate_guidance_for_responses(role, responses, questions_by_index, judge_guidance)
            prompts_for_role = 0
            sample_prompt = None

            for resp in responses:
                prompt_idx = resp["prompt_index"]
                question_idx = resp["question_index"]
                label = resp["label"]
                key = f"{label}_p{prompt_idx}_q{question_idx}"

                if key not in existing_scores:
                    prompts_for_role += 1
                    if sample_prompt is None:
                        # Build sample prompt
                        assistant_response = ""
                        for msg in resp["conversation"]:
                            if msg["role"] == "assistant":
                                assistant_response = msg["content"]
                                break
                        sample_prompt = format_eval_prompt(
                            eval_prompt_template=eval_prompt_template,
                            role_id=role,
                            question_index=question_idx,
                            question=resp["question"],
                            answer=assistant_response,
                            questions_by_index=questions_by_index,
                            judge_guidance=judge_guidance,
                        )

            if prompts_for_role > 0:
                total_prompts += prompts_for_role
                logger.info(f"  {role}: {prompts_for_role} prompts")

                # Show one sample
                if not sample_shown and sample_prompt:
                    logger.info("\n" + "=" * 60)
                    logger.info("SAMPLE JUDGE PROMPT:")
                    logger.info("=" * 60)
                    logger.info(f"Model: {args.judge_model}")
                    logger.info(f"Max tokens: {args.max_tokens}")
                    logger.info("-" * 60)
                    logger.info(sample_prompt)
                    logger.info("=" * 60 + "\n")
                    sample_shown = True

        logger.info(f"\nTotal prompts to send: {total_prompts}")
        return

    # Initialize client and rate limiter
    client = openai.AsyncOpenAI()
    rate_limiter = RateLimiter(args.requests_per_second)

    # Track results
    successful = 0
    skipped = 0
    failed = 0
    errors = []

    # Raw completions are saved alongside scores in a sibling directory,
    # keyed identically to scores/{role}.json, rather than folded into that
    # file's value type -- 4_vectors.py and several analysis scripts read
    # scores/{role}.json as a plain {key: int} mapping, so changing its
    # schema would ripple well past this script.
    raw_output_dir = output_dir.parent / "judge_raw"
    if not args.dry_run:
        raw_output_dir.mkdir(parents=True, exist_ok=True)

    # Process each role
    for response_file in tqdm(response_files, desc="Scoring roles"):
        role = response_file.stem
        output_file = output_dir / f"{role}.json"
        raw_output_file = raw_output_dir / f"{role}.json"

        # Load existing scores
        existing_scores = {}
        if output_file.exists():
            try:
                with open(output_file, 'r') as f:
                    existing_scores = json.load(f)
            except Exception:
                pass

        # Load existing raw completions
        existing_raw_completions = {}
        if raw_output_file.exists():
            try:
                with open(raw_output_file, 'r') as f:
                    existing_raw_completions = json.load(f)
            except Exception:
                pass

        # Get role eval prompt
        role_file = roles_dir / f"{role}.json"
        if not role_file.exists():
            logger.info(f"Skipping {role}: no role file found")
            skipped += 1
            continue

        eval_prompt_template = load_role_eval_prompt(role_file)
        if not eval_prompt_template:
            logger.info(f"Skipping {role}: no eval_prompt in role file")
            skipped += 1
            continue

        # Load responses
        responses = load_responses(response_file)
        validate_guidance_for_responses(role, responses, questions_by_index, judge_guidance)
        if not responses:
            errors.append(f"{role}: no responses found")
            failed += 1
            continue

        # Check if all responses are already scored
        all_scored = True
        for resp in responses:
            key = f"{resp['label']}_p{resp['prompt_index']}_q{resp['question_index']}"
            if key not in existing_scores:
                all_scored = False
                break

        if all_scored:
            logger.info(f"Skipping {role}: all {len(responses)} responses already scored")
            skipped += 1
            continue

        # Score responses
        try:
            new_scores, new_raw_completions = await process_role(
                role=role,
                responses=responses,
                eval_prompt_template=eval_prompt_template,
                client=client,
                rate_limiter=rate_limiter,
                judge_model=args.judge_model,
                max_tokens=args.max_tokens,
                batch_size=args.batch_size,
                existing_scores=existing_scores,
                questions_by_index=questions_by_index,
                judge_guidance=judge_guidance,
            )

            # Merge scores
            all_scores = {**existing_scores, **new_scores}
            all_raw_completions = {**existing_raw_completions, **new_raw_completions}

            # Save scores
            with open(output_file, 'w') as f:
                json.dump(all_scores, f, indent=2)

            # Save raw completions (same keys as scores, sibling file)
            with open(raw_output_file, 'w') as f:
                json.dump(all_raw_completions, f, indent=2)

            logger.info(f"Saved {len(all_scores)} scores for {role} ({len(new_scores)} new)")
            successful += 1

        except Exception as e:
            errors.append(f"{role}: {e}")
            failed += 1

    # Print summary
    logger.info("\n" + "=" * 40)
    logger.info("SUMMARY")
    logger.info("=" * 40)
    logger.info(f"Successful: {successful}")
    logger.info(f"Skipped:    {skipped}")
    logger.info(f"Failed:     {failed}")

    if errors:
        logger.info("\nErrors:")
        for error in errors[:10]:
            logger.info(f"  - {error}")
        if len(errors) > 10:
            logger.info(f"  ... and {len(errors) - 10} more")


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
