def evaluate_partial_score(state, action):
    score = 0.0

    action_type = getattr(action, "action_type", None)
    target = getattr(action, "target", None)

    ground_truth = getattr(state, "ground_truth", {})

    if action_type == "flag_dominance":
        if target == ground_truth.get("dominant_speaker"):
            score = 0.10

    elif action_type == "flag_low_participation":
        low_participants = set(ground_truth.get("low_participation_speakers", []))
        if target in low_participants:
            score = 0.10

    elif action_type == "flag_irrelevant_speaker":
        irrelevant = set(ground_truth.get("irrelevant_speakers", []))
        if target in irrelevant:
            score = 0.10

    elif action_type == "select_winner":
        if target == ground_truth.get("winner"):
            score = 0.15

    return score


def final_grade(state):
    return 0.90


def grade_easy(prediction, ground_truth):
    score = 0.0

    if prediction.get("winner") == ground_truth.get("winner"):
        score += 0.5

    if prediction.get("scores") == ground_truth.get("scores"):
        score += 0.5

    return {"score": min(score, 1.0)}


def grade_medium(prediction, ground_truth):
    score = 0.0

    if prediction.get("winner") == ground_truth.get("winner"):
        score += 0.4

    if prediction.get("ranking") == ground_truth.get("ranking"):
        score += 0.3

    if prediction.get("scores") == ground_truth.get("scores"):
        score += 0.3

    return {"score": min(score, 1.0)}


def grade_hard(prediction, ground_truth):
    score = 0.0

    if prediction.get("winner") == ground_truth.get("winner"):
        score += 0.25

    if prediction.get("dominant_speaker") == ground_truth.get("dominant_speaker"):
        score += 0.25

    if set(prediction.get("low_participation_speakers", [])) == set(ground_truth.get("low_participation_speakers", [])):
        score += 0.25

    if set(prediction.get("irrelevant_speakers", [])) == set(ground_truth.get("irrelevant_speakers", [])):
        score += 0.25

    return {"score": min(score, 1.0)}