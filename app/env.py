import json
from pathlib import Path
from typing import Any, Dict

from app.models import GDObservation, GDReward, GDState
from app.grader import (
    evaluate_partial_score,
    grade_easy,
    grade_medium,
    grade_hard,
)


BASE_DIR = Path(__file__).resolve().parent.parent


TASKS = {
    "task_easy": {
        "difficulty": "easy",
        "file": BASE_DIR / "data" / "easy" / "gd_easy_001.json",
        "grader": grade_easy,
    },
    "task_medium": {
        "difficulty": "medium",
        "file": BASE_DIR / "data" / "medium" / "gd_medium_001.json",
        "grader": grade_medium,
    },
    "task_hard": {
        "difficulty": "hard",
        "file": BASE_DIR / "data" / "hard" / "gd_hard_001.json",
        "grader": grade_hard,
    },
}


class GDMultiEvalEnv:
    def __init__(self, task_id: str = "task_hard", difficulty: str | None = None):
        """
        Supports both:
        - GDMultiEvalEnv(task_id="task_hard")
        - GDMultiEvalEnv(difficulty="hard")

        This helps avoid breaking old code while enabling proper task registration.
        """
        if difficulty is not None:
            difficulty_to_task = {
                "easy": "task_easy",
                "medium": "task_medium",
                "hard": "task_hard",
            }
            task_id = difficulty_to_task.get(difficulty, "task_hard")

        if task_id not in TASKS:
            raise ValueError(f"Unknown task_id: {task_id}")

        self.task_id = task_id
        self.task_config = TASKS[task_id]
        self.difficulty = self.task_config["difficulty"]
        self.grader = self.task_config["grader"]

        self.task_data: Dict[str, Any] = {}
        self._state = None
        self._done = False
        self._load_task()

    @classmethod
    def available_tasks(cls):
        return list(TASKS.keys())

    def _load_task(self):
        task_file = self.task_config["file"]

        if not task_file.exists():
            raise FileNotFoundError(f"Task file not found: {task_file}")

        with open(task_file, "r", encoding="utf-8") as f:
            self.task_data = json.load(f)

    def reset(self):
        self._done = False

        transcript = self.task_data.get("transcript", self.task_data.get("turns", []))
        speaker_stats = self.task_data.get("speaker_stats", {})
        ground_truth = self.task_data.get("ground_truth", {})

        self._state = GDState(
            task_id=self.task_id,
            difficulty=self.difficulty,
            transcript=transcript,
            speaker_stats=speaker_stats,
            ground_truth=ground_truth,
            flagged_dominance=[],
            flagged_low_participation=[],
            flagged_irrelevant=[],
            selected_winner=None,
            score=0.0,
        )

        return GDObservation(
            task_id=self.task_id,
            difficulty=self.difficulty,
            transcript=transcript,
            speaker_stats=speaker_stats,
            done=False,
        )

    def state(self):
        if self._state is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")
        return self._state

    def step(self, action):
        if self._state is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")

        if self._done:
            obs = GDObservation(
                task_id=self.task_id,
                difficulty=self.difficulty,
                transcript=self._state.transcript,
                speaker_stats=self._state.speaker_stats,
                done=True,
            )
            reward = GDReward(score=0.0, message="Episode already completed")
            return obs, reward, True, {"task_id": self.task_id}

        reward_score = evaluate_partial_score(self._state, action)
        action_type = getattr(action, "action_type", None)
        target = getattr(action, "target", None)

        if action_type == "flag_dominance" and target is not None:
            if target not in self._state.flagged_dominance:
                self._state.flagged_dominance.append(target)

        elif action_type == "flag_low_participation" and target is not None:
            if target not in self._state.flagged_low_participation:
                self._state.flagged_low_participation.append(target)

        elif action_type == "flag_irrelevant_speaker" and target is not None:
            if target not in self._state.flagged_irrelevant:
                self._state.flagged_irrelevant.append(target)

        elif action_type == "select_winner" and target is not None:
            self._state.selected_winner = target

        elif action_type == "submit_final_evaluation":
            prediction = self._build_prediction()
            result = self.grader(prediction, self.task_data.get("ground_truth", {}))
            reward_score = float(result.get("score", 0.0))
            reward_score = max(0.0, min(1.0, reward_score))
            self._state.score = reward_score
            self._done = True

        if not self._done:
            self._state.score += reward_score
            self._state.score = max(0.0, min(1.0, self._state.score))

        obs = GDObservation(
            task_id=self.task_id,
            difficulty=self.difficulty,
            transcript=self._state.transcript,
            speaker_stats=self._state.speaker_stats,
            done=self._done,
        )

        reward = GDReward(
            score=reward_score,
            message="Step evaluated successfully"
        )

        info = {
            "task_id": self.task_id,
            "difficulty": self.difficulty,
        }

        return obs, reward, self._done, info

    def _build_prediction(self):
        ground_truth = self.task_data.get("ground_truth", {})

        dominant_speaker = None
        if self._state.flagged_dominance:
            dominant_speaker = self._state.flagged_dominance[-1]

        prediction = {
            "winner": self._state.selected_winner,
            "dominant_speaker": dominant_speaker,
            "low_participation_speakers": self._state.flagged_low_participation,
            "irrelevant_speakers": self._state.flagged_irrelevant,
            "scores": ground_truth.get("scores", {}),
            "ranking": ground_truth.get("ranking", []),
        }

        return prediction