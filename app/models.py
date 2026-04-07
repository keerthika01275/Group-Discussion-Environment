from typing import Dict, List, Literal, Optional
from pydantic import BaseModel, Field


class Turn(BaseModel):
    speaker: str
    text: str
    timestamp: float


class SpeakerStats(BaseModel):
    speaking_time_sec: float
    interruptions: int
    filler_count: int
    confidence_score: float = Field(ge=0.0, le=1.0)
    engagement_score: float = Field(ge=0.0, le=1.0)
    relevance_score: float = Field(ge=0.0, le=1.0)


class GroundTruth(BaseModel):
    winner: str
    dominant_speaker: Optional[str] = None
    low_participation_speakers: List[str]
    irrelevant_speakers: List[str]
    scores: Dict[str, float]


class GDEpisode(BaseModel):
    episode_id: str
    difficulty: Literal["easy", "medium", "hard"]
    topic: str
    turns: List[Turn]
    speaker_stats: Dict[str, SpeakerStats]
    ground_truth: GroundTruth


class GDObservation(BaseModel):
    episode_id: str
    topic: str
    visible_turns: List[Turn]
    speaker_stats: Dict[str, SpeakerStats]
    available_actions: List[str]
    step_count: int
    max_steps: int
    done: bool


class GDAction(BaseModel):
    action_type: Literal[
        "score_speaker",
        "flag_dominance",
        "flag_low_participation",
        "flag_irrelevant_speaker",
        "request_more_context",
        "select_winner",
        "submit_final_evaluation"
    ]
    speaker: Optional[str] = None
    score: Optional[float] = None
    reason: Optional[str] = None


class GDReward(BaseModel):
    value: float
    feedback: str


class GDState(BaseModel):
    episode: GDEpisode
    revealed_turn_count: int
    step_count: int
    max_steps: int
    done: bool
    selected_winner: Optional[str] = None
    speaker_scores: Dict[str, float] = {}
    flagged_dominance: Optional[str] = None
    low_participation_flags: List[str] = []
    irrelevant_speaker_flags: List[str] = []