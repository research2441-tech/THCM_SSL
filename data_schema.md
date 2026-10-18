# Data schema and chronological protocol

One row represents **one independently timestamped post**. UTF-8 CSV fields:

| Field | Required | Example | Role |
|---|---|---|---|
| text | yes | `Great #launch 😀` | Raw sentiment text; emoji/hashtags preserved |
| label | yes | positive | Human/verified corpus polarity annotation |
| timestamp | yes | `2024-06-30T08:00:00Z` | Original publication or collection time, not invented download time |
| platform | recommended | X | Cross-platform drift provenance |
| user_id | recommended | opaque-user-key | Optional user-overlap audit (avoid publishing identifiable handles) |
| post_id | recommended | opaque-post-key | Traceable exact-post deduplication |
| language | recommended | hi-en | Code-mixing stratification |
| corpus | recommended | verified-collection-name | Corpus audit identifier |

No data are shipped. Missing actual original timestamps invalidate genuine chronological holdout claims. `prepare_data.py` sorts by real timestamps, excludes timestamps tied across temporal boundaries, removes exact normalized duplicates and flags near duplicates using 3-gram Jaccard overlap. User overlap is filtered when usable user identifiers are present. This is a strict preliminary audit; for publication also check repost IDs and duplicated texts across the *entire six-corpus collection* before splitting.

Split defaults are 70% training, 15% validation (itself split into tuning and calibration), and 15% future test. These are reference defaults, **not** verified original paper dates or ratios. Labels are mapped using training-only classes; previously unseen classes need an explicitly implemented class-incremental procedure rather than silently mapping them.
