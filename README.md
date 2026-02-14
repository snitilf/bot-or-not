# bot or not challenge detector
i built this project to detect bot accounts in english and french x/twitter datasets for the mchacks bot or not challenge.

## quick run for evaluators
from the repo root:

```bash
pip install -r requirements.txt
python3 src/run_final.py <english_dataset.json> <french_dataset.json> --team localhost
```

this creates:
- `localhost.detections.en.txt`
- `localhost.detections.fr.txt`

runtime is usually about 15 seconds.

## what i implemented
the detector is a hybrid pipeline with rules + machine learning.

1. feature extraction (`src/features.py`)
- extracts user-level behavioral and text signals (timing, repetition, hashtags, punctuation style, collision patterns, account metadata, topic/activity normalization).

2. english detector (`src/en_detector.py`)
- 4 high-confidence rules:
- out-of-range timestamps
- high hashtag + near 24/7 posting
- same-second batch posting
- extreme question-only behavior
- ensemble model: random forest + gradient boosting + xgboost.
- robust threshold selection with stress testing support (`src/stress_eval_en.py`).

3. french detector (`src/fr_detector.py`)
- 6 high-confidence behavioral rules (different rule set from english).
- ensemble model: random forest + xgboost + gradient boosting.
- extra stability controls:
- rate clamp
- human-likeness veto
- robust threshold selection with fallback guardrail (`src/stress_eval_fr.py`).

4. final runner (`src/run_final.py`)
- runs both language detectors and writes final submission files.

## strict benchmark results (primary)
these are the strict non-leaky results from `src/strict_eval.py` on the original practice datasets:

- ds30 en: 248/264 (93.9%)
- ds32 en: 240/252 (95.2%)
- ds31 fr: 108/108 (100.0%)
- ds33 fr: 107/112 (95.5%)

totals:
- english total: 488/516 (94.6%)
- french total: 215/220 (97.7%)
- overall total: 703/736 (95.5%)

## scoring objective and design choice
challenge score:
- +4 true positive
- -1 false negative
- -2 false positive

because false positives are costly, i tuned the system to be precision-first and conservative on uncertain cases.

## final evaluation datasets (ds34 english / ds35 french)
i ran the final pipeline on the released evaluation datasets using:

`python3 src/run_final.py data/dataset.posts&users.34.json data/dataset.posts&users.35.json --team localhost`

generated outputs:

- localhost.detections.en.txt: 74 detected users out of 438 total users (16.9%)
- localhost.detections.fr.txt: 35 detected users out of 283 total users (12.4%)

## files evaluators should inspect
- `src/features.py`
- `src/en_detector.py`
- `src/fr_detector.py`
- `src/run_final.py`
- `src/strict_eval.py`
- `src/stress_eval_en.py`
- `src/stress_eval_fr.py`
- `src/scorer.py`

## team
team name: localhost

members: just me 
