from typing import Dict
from app.models import GDState


def score_closeness(pred: float, truth: float) -> float:
    diff = abs(pred - truth)
    if diff <= 0.05:
        return 1.0
    if diff <= 0.10:
        return 0.8
    if diff <= 0.20:
        return 0.5
    return 0.0


def evaluate_partial_score(state: GDState, speaker: str, score: float) -> float:
    truth = state.episode.ground_truth.scores.get(speaker)
    if truth is None:
        return -0.05
    return 0.15 * score_closeness(score, truth)


def final_grade(state: GDState) -> float:
    gt = state.episode.ground_truth
    total = 0.0

    if state.selected_winner == gt.winner:
        total += 0.30

    if state.flagged_dominance == gt.dominant_speaker:
        total += 0.20

    if set(state.low_participation_flags) == set(gt.low_participation_speakers):
        total += 0.20

    if set(state.irrelevant_speaker_flags) == set(gt.irrelevant_speakers):
        total += 0.20

    if state.speaker_scores:
        score_part = 0.0
        count = 0
        for speaker, pred in state.speaker_scores.items():
            if speaker in gt.scores:
                score_part += score_closeness(pred, gt.scores[speaker])
                count += 1
        if count > 0:
            total += 0.10 * (score_part / count)

    return min(total, 1.0)