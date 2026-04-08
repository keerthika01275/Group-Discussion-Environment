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