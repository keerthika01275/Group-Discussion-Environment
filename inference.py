import json
import os
from typing import List, Optional

from openai import OpenAI

from app.env import GDMultiEvalEnv
from app.models import GDAction, GDObservation

API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
TASK_NAME = os.getenv("GD_TASK", "hard_full_moderation")
DIFFICULTY = os.getenv("GD_DIFFICULTY", "hard")

client = OpenAI(
    api_key=API_KEY or "DUMMY_KEY",
    base_url=API_BASE_URL,
)


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    err = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={err}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


def observation_to_prompt(obs: GDObservation) -> str:
    turns_text = "\n".join(
        f"[{turn.speaker}] {turn.text}" for turn in obs.visible_turns
    )

    stats_dict = {
        speaker: {
            "speaking_time_sec": stats.speaking_time_sec,
            "interruptions": stats.interruptions,
            "filler_count": stats.filler_count,
            "confidence_score": stats.confidence_score,
            "engagement_score": stats.engagement_score,
            "relevance_score": stats.relevance_score,
        }
        for speaker, stats in obs.speaker_stats.items()
    }

    return f"""
You are evaluating a placement-style group discussion.

Topic:
{obs.topic}

Visible Transcript:
{turns_text}

Speaker Stats:
{json.dumps(stats_dict, indent=2)}

Available actions:
{obs.available_actions}

Return exactly one JSON object with this format:
{{
  "action_type": "score_speaker | flag_dominance | flag_low_participation | flag_irrelevant_speaker | request_more_context | select_winner | submit_final_evaluation",
  "speaker": "speaker name or null",
  "score": 0.0,
  "reason": "short reason"
}}

Rules:
- Return valid JSON only.
- Use score only for action_type="score_speaker".
- For other action types, score can be null.
""".strip()


def heuristic_action(obs: GDObservation, step_num: int) -> GDAction:
    stats = obs.speaker_stats

    if step_num == 1:
        dominant = max(stats.items(), key=lambda x: x[1].speaking_time_sec)[0]
        return GDAction(
            action_type="flag_dominance",
            speaker=dominant,
            reason="Highest speaking time"
        )

    if step_num == 2:
        irrelevant = min(stats.items(), key=lambda x: x[1].relevance_score)[0]
        return GDAction(
            action_type="flag_irrelevant_speaker",
            speaker=irrelevant,
            reason="Lowest relevance score"
        )

    if step_num == 3:
        low_participant = min(stats.items(), key=lambda x: x[1].speaking_time_sec)[0]
        return GDAction(
            action_type="flag_low_participation",
            speaker=low_participant,
            reason="Lowest speaking time"
        )

    if step_num == 4:
        winner = max(
            stats.items(),
            key=lambda x: (
                x[1].relevance_score * 0.40
                + x[1].confidence_score * 0.25
                + x[1].engagement_score * 0.20
                - x[1].interruptions * 0.03
                - x[1].filler_count * 0.01
            )
        )[0]
        return GDAction(
            action_type="select_winner",
            speaker=winner,
            reason="Best overall weighted score"
        )

    return GDAction(
        action_type="submit_final_evaluation",
        reason="Heuristic evaluation complete"
    )


def llm_action(obs: GDObservation) -> GDAction:
    prompt = observation_to_prompt(obs)

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": "You are a strict JSON-only decision maker for a GD evaluation environment."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.2,
        max_tokens=200,
    )

    content = response.choices[0].message.content.strip()
    data = json.loads(content)
    return GDAction(**data)


def main() -> None:
    env = GDMultiEvalEnv(difficulty=DIFFICULTY)
    obs = env.reset()

    rewards: List[float] = []
    step_num = 0
    done = False

    log_start(TASK_NAME, "gd_multieval", MODEL_NAME)

    while not done and step_num < 8:
        step_num += 1

        try:
            if API_KEY:
                action = llm_action(obs)
            else:
                action = heuristic_action(obs, step_num)

            obs, reward, done, info = env.step(action)
            rewards.append(reward.value)

            action_str = f"{action.action_type}({repr(action.speaker) if action.speaker else ''})"
            log_step(step_num, action_str, reward.value, done, None)

        except Exception as e:
            log_step(step_num, "error()", 0.0, done, str(e))
            break

    final_score = rewards[-1] if rewards else 0.0
    log_end(done, step_num, final_score, rewards)


if __name__ == "__main__":
    main()