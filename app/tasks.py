import json
from pathlib import Path
from typing import List
from app.models import GDEpisode


DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_episode(file_path: Path) -> GDEpisode:
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return GDEpisode(**data)


def load_episodes_by_difficulty(difficulty: str) -> List[GDEpisode]:
    folder = DATA_DIR / difficulty
    episodes = []
    for file in sorted(folder.glob("*.json")):
        episodes.append(load_episode(file))
    return episodes