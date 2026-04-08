import json
import os
from typing import Dict, List, Optional, Tuple

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


def speaker_weighted_score(stat) -> float:
    score = (
        stat.relevance_score * 0.40
        + stat.confidence_score * 0.25
        + stat.engagement_score * 0.20
        - stat.interruptions * 0.03
        - stat.filler_count * 0.01
    )
    return round(score, 4)


def build_score_breakdown(obs: GDObservation) -> Dict[str, Dict[str, float]]:
    breakdown = {}
    for speaker, stat in obs.speaker_stats.items():
        breakdown[speaker] = {
            "relevance": round(stat.relevance_score, 3),
            "confidence": round(stat.confidence_score, 3),
            "engagement": round(stat.engagement_score, 3),
            "interruptions_penalty": round(stat.interruptions * 0.03, 3),
            "filler_penalty": round(stat.filler_count * 0.01, 3),
            "speaking_time_sec": round(stat.speaking_time_sec, 2),
            "final_weighted_score": speaker_weighted_score(stat),
        }
    return breakdown


def winner_confidence(obs: GDObservation, winner: str) -> float:
    scores = {
        speaker: speaker_weighted_score(stat)
        for speaker, stat in obs.speaker_stats.items()
    }
    ordered = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    if len(ordered) == 1:
        return 1.0

    top_score = ordered[0][1]
    second_score = ordered[1][1]
    margin = max(0.0, top_score - second_score)

    # Normalize into a simple 0 to 1 confidence
    conf = min(1.0, 0.6 + margin)
    return round(conf, 3)


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

    score_breakdown = build_score_breakdown(obs)

    return f"""
You are evaluating a placement-style group discussion.

Topic:
{obs.topic}

Visible Transcript:
{turns_text}

Speaker Stats:
{json.dumps(stats_dict, indent=2)}

Current Score Breakdown:
{json.dumps(score_breakdown, indent=2)}

Available actions:
{obs.available_actions}

Return exactly one JSON object in this format:
{{
  "action_type": "score_speaker | flag_dominance | flag_low_participation | flag_irrelevant_speaker | request_more_context | select_winner | submit_final_evaluation",
  "speaker": "speaker name or null",
  "score": 0.0,
  "reason": "short reason with evidence"
}}

Rules:
- Return valid JSON only.
- Use score only for action_type="score_speaker".
- For other action types, score can be null.
- Prefer explainable decisions based on transcript and structured stats.
""".strip()


def choose_dominant(obs: GDObservation) -> Tuple[str, str]:
    stats = obs.speaker_stats
    dominant, stat = max(stats.items(), key=lambda x: x[1].speaking_time_sec)
    reason = (
        f"{dominant} has the highest speaking time "
        f"({stat.speaking_time_sec:.1f}s) and appears most dominant."
    )
    return dominant, reason


def choose_irrelevant(obs: GDObservation) -> Tuple[str, str]:
    stats = obs.speaker_stats
    irrelevant, stat = min(stats.items(), key=lambda x: x[1].relevance_score)
    reason = (
        f"{irrelevant} has the lowest relevance score "
        f"({stat.relevance_score:.2f}), indicating weaker topic alignment."
    )
    return irrelevant, reason


def choose_low_participant(obs: GDObservation) -> Tuple[str, str]:
    stats = obs.speaker_stats
    low_participant, stat = min(stats.items(), key=lambda x: x[1].speaking_time_sec)
    reason = (
        f"{low_participant} has the lowest speaking time "
        f"({stat.speaking_time_sec:.1f}s), indicating low participation."
    )
    return low_participant, reason


def choose_winner(obs: GDObservation) -> Tuple[str, str]:
    stats = obs.speaker_stats
    winner = max(stats.items(), key=lambda x: speaker_weighted_score(x[1]))[0]
    stat = stats[winner]
    conf = winner_confidence(obs, winner)
    final_score = speaker_weighted_score(stat)
    reason = (
        f"{winner} has the best overall weighted score ({final_score:.3f}) "
        f"based on relevance, confidence, engagement, and penalties. "
        f"Decision confidence={conf:.2f}."
    )
    return winner, reason


def heuristic_action(obs: GDObservation, step_num: int) -> GDAction:
    visible_turn_count = len(obs.visible_turns)

    if visible_turn_count < 2 and "request_more_context" in obs.available_actions:
        return GDAction(
            action_type="request_more_context",
            reason="Too few visible turns to make a reliable evaluation."
        )

    if step_num == 1 and "flag_dominance" in obs.available_actions:
        speaker, reason = choose_dominant(obs)
        return GDAction(
            action_type="flag_dominance",
            speaker=speaker,
            reason=reason
        )

    if step_num == 2 and "flag_irrelevant_speaker" in obs.available_actions:
        speaker, reason = choose_irrelevant(obs)
        return GDAction(
            action_type="flag_irrelevant_speaker",
            speaker=speaker,
            reason=reason
        )

    if step_num == 3 and "flag_low_participation" in obs.available_actions:
        speaker, reason = choose_low_participant(obs)
        return GDAction(
            action_type="flag_low_participation",
            speaker=speaker,
            reason=reason
        )

    if step_num == 4 and "select_winner" in obs.available_actions:
        speaker, reason = choose_winner(obs)
        return GDAction(
            action_type="select_winner",
            speaker=speaker,
            reason=reason
        )

    return GDAction(
        action_type="submit_final_evaluation",
        reason="Evaluation completed using structured moderation and score analysis."
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
        max_tokens=250,
    )

    content = response.choices[0].message.content.strip()
    data = json.loads(content)
    return GDAction(**data)


def format_action_for_log(action: GDAction, obs: Optional[GDObservation] = None) -> str:
    extra = ""
    if action.action_type == "select_winner" and action.speaker and obs is not None:
        conf = winner_confidence(obs, action.speaker)
        extra = f", confidence={conf:.2f}"

    speaker_part = repr(action.speaker) if action.speaker else ""
    reason_part = action.reason.replace("\n", " ") if action.reason else ""
    return f"{action.action_type}({speaker_part}, reason={reason_part!r}{extra})"


def print_score_breakdown(obs: GDObservation) -> None:
    print("[INFO] Speaker Score Breakdown:", flush=True)
    breakdown = build_score_breakdown(obs)
    for speaker, info in breakdown.items():
        print(f"[INFO] {speaker}: {json.dumps(info)}", flush=True)


def main() -> None:
    env = GDMultiEvalEnv(difficulty=DIFFICULTY)
    obs = env.reset()

    rewards: List[float] = []
    step_num = 0
    done = False

    log_start(TASK_NAME, "gd_multieval", MODEL_NAME)
    print_score_breakdown(obs)

    while not done and step_num < 8:
        step_num += 1

        try:
            prev_obs = obs

            if API_KEY:
                try:
                    action = llm_action(obs)
                except Exception:
                    action = heuristic_action(obs, step_num)
            else:
                action = heuristic_action(obs, step_num)

            obs, reward, done, info = env.step(action)
            rewards.append(reward.value)

            action_str = format_action_for_log(action, prev_obs)
            log_step(step_num, action_str, reward.value, done, None)

        except Exception as e:
            log_step(step_num, "error()", 0.0, done, str(e))
            break

    final_score = rewards[-1] if rewards else 0.0
    log_end(done, step_num, final_score, rewards)


if __name__ == "__main__":
    main()