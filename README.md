# GD-MultiEval

## Overview
GD-MultiEval is an OpenEnv-style environment for evaluating placement-style group discussions using transcript turns and structured multimodal signals such as speaking time, interruptions, confidence, engagement, and relevance.

## Motivation
Group discussions are widely used in campus placements, interviews, and training assessments. Human evaluators score candidates based on relevance, clarity, participation, confidence, balance, and communication quality. This environment simulates that process so AI agents can be evaluated on realistic GD assessment tasks.

## Real-World Utility
This environment models a genuine evaluation workflow used in recruitment and education. It is designed for agent benchmarking in participant scoring, moderation, fairness analysis, and winner selection.

## Observation Space
Each observation contains:
- GD topic
- visible transcript turns
- per-speaker structured stats
- available actions
- current step count
- max step limit
- done flag

## Action Space
The agent can take the following actions:
- `score_speaker`
- `flag_dominance`
- `flag_low_participation`
- `flag_irrelevant_speaker`
- `request_more_context`
- `select_winner`
- `submit_final_evaluation`

## Tasks
### Easy — Single Speaker Evaluation
The agent scores one participant in a simple GD setup.

### Medium — Multi-Speaker Ranking
The agent evaluates multiple participants, compares them, and selects the best candidate.

### Hard — Full Moderation
The agent detects dominant behavior, low participation, irrelevant content, and selects the winner fairly.

## Reward Design
Rewards are shaped across the trajectory:
- positive reward for correct intermediate actions
- positive reward for correct winner selection
- positive reward for correct moderation flags
- final reward based on full evaluation quality
- penalties for wrong or unnecessary actions

## Deterministic Grading
Each GD episode has hidden ground-truth labels:
- winner
- dominant speaker
- low participation speakers
- irrelevant speakers
- reference per-speaker scores

This makes grading deterministic and reproducible.

## Project Structure
```text
gd-multieval/
│
├── app/
│   ├── env.py
│   ├── grader.py
│   ├── models.py
│   ├── tasks.py
│   ├── utils.py
│   └── __init__.py
│
├── data/
│   ├── easy/
│   ├── medium/
│   └── hard/
│
├── tests/
│   └── test_env.py
│
├── inference.py
├── openenv.yaml
├── Dockerfile
├── requirements.txt
└── README.md