from typing import Dict, Any, Optional
from app.models import GDAction, GDObservation, GDReward, GDState
from app.tasks import load_episodes_by_difficulty
from app.grader import evaluate_partial_score, final_grade


class GDMultiEvalEnv:
    def __init__(self, difficulty: str = "easy", max_steps: int = 8):
        self.difficulty = difficulty
        self.max_steps = max_steps
        self.episodes = load_episodes_by_difficulty(difficulty)
        self.current_index = -1
        self.state_obj: Optional[GDState] = None

    def _build_observation(self) -> GDObservation:
        episode = self.state_obj.episode
        visible_turns = episode.turns[:self.state_obj.revealed_turn_count]
        return GDObservation(
            episode_id=episode.episode_id,
            topic=episode.topic,
            visible_turns=visible_turns,
            speaker_stats=episode.speaker_stats,
            available_actions=[
                "score_speaker",
                "flag_dominance",
                "flag_low_participation",
                "flag_irrelevant_speaker",
                "request_more_context",
                "select_winner",
                "submit_final_evaluation",
            ],
            step_count=self.state_obj.step_count,
            max_steps=self.state_obj.max_steps,
            done=self.state_obj.done
        )

    def reset(self) -> GDObservation:
        self.current_index = (self.current_index + 1) % len(self.episodes)
        episode = self.episodes[self.current_index]

        initial_reveal = min(2, len(episode.turns))
        self.state_obj = GDState(
            episode=episode,
            revealed_turn_count=initial_reveal,
            step_count=0,
            max_steps=self.max_steps,
            done=False,
            selected_winner=None,
            speaker_scores={},
            flagged_dominance=None,
            low_participation_flags=[],
            irrelevant_speaker_flags=[]
        )
        return self._build_observation()

    def state(self) -> GDState:
        return self.state_obj

    def step(self, action: GDAction):
        if self.state_obj is None:
            raise ValueError("Environment not initialized. Call reset() first.")

        if self.state_obj.done:
            reward = GDReward(value=0.0, feedback="Episode already completed.")
            return self._build_observation(), reward, True, {}

        self.state_obj.step_count += 1
        reward_value = 0.0
        feedback = "Action processed."

        if action.action_type == "request_more_context":
            prev = self.state_obj.revealed_turn_count
            self.state_obj.revealed_turn_count = min(
                len(self.state_obj.episode.turns),
                self.state_obj.revealed_turn_count + 2
            )
            if self.state_obj.revealed_turn_count > prev:
                reward_value = 0.05
                feedback = "More context revealed."
            else:
                reward_value = -0.02
                feedback = "No more context available."

        elif action.action_type == "score_speaker":
            if action.speaker is None or action.score is None:
                reward_value = -0.05
                feedback = "Missing speaker or score."
            else:
                self.state_obj.speaker_scores[action.speaker] = action.score
                reward_value = evaluate_partial_score(self.state_obj, action.speaker, action.score)
                feedback = f"Stored score for {action.speaker}."

        elif action.action_type == "flag_dominance":
            self.state_obj.flagged_dominance = action.speaker
            reward_value = 0.10 if action.speaker == self.state_obj.episode.ground_truth.dominant_speaker else -0.05
            feedback = "Dominance flag recorded."

        elif action.action_type == "flag_low_participation":
            if action.speaker and action.speaker not in self.state_obj.low_participation_flags:
                self.state_obj.low_participation_flags.append(action.speaker)
                reward_value = 0.10 if action.speaker in self.state_obj.episode.ground_truth.low_participation_speakers else -0.05
                feedback = "Low participation flag recorded."

        elif action.action_type == "flag_irrelevant_speaker":
            if action.speaker and action.speaker not in self.state_obj.irrelevant_speaker_flags:
                self.state_obj.irrelevant_speaker_flags.append(action.speaker)
                reward_value = 0.10 if action.speaker in self.state_obj.episode.ground_truth.irrelevant_speakers else -0.05
                feedback = "Irrelevant speaker flag recorded."

        elif action.action_type == "select_winner":
            self.state_obj.selected_winner = action.speaker
            reward_value = 0.15 if action.speaker == self.state_obj.episode.ground_truth.winner else -0.05
            feedback = "Winner selected."

        elif action.action_type == "submit_final_evaluation":
            final_score = final_grade(self.state_obj)
            reward_value = final_score
            feedback = f"Final evaluation submitted. Score={final_score:.2f}"
            self.state_obj.done = True

        if self.state_obj.step_count >= self.state_obj.max_steps and not self.state_obj.done:
            self.state_obj.done = True
            reward_value -= 0.05
            feedback += " Max step limit reached."

        reward = GDReward(value=reward_value, feedback=feedback)
        return self._build_observation(), reward, self.state_obj.done, {"feedback": feedback}