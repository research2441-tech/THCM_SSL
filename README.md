# THCM-SSL — reproducible research reference implementation

**Temporal Hypergraph Co-Occurrence Memory with Selective State-Space Learning for Continual Social Media Sentiment Analysis**. This is an independent **reference reimplementation from the supplied manuscript**, not a claim to be the original code that produced the published/reported metrics. The full experimental raw datasets, original checkpoints, precise chronological cutoffs, and original SSM kernel were not provided. No numeric manuscript claim is prefilled or fabricated.

## Design

`prepare_data.py` → chronological, duplicate-audited partitions → `TemporalHypergraph` with heterogeneous hashtag/emoji/word tokens and bounded contextual hyperedges → weighted incidence message propagation → two learnable hypergraph projections → Gumbel-sigmoid sparsity gate → input-dependent diagonal selective recurrent SSM → masked average pooling → sentiment classifier → held-out calibration → selective prediction. `ContinualMemory` supplies class prototypes and Fisher-diagonal penalties for sequential window training experiments.

**Critical implementation distinction:** This reference uses an explicit PyTorch diagonal selective scan. It is **not the official Mamba selective-scan CUDA implementation**; the reference has a recurrent state of `hidden_dim` per sequence and the configured `ssm_state_dim` is descriptive only. Its runtime is linear in sequence length for fixed dimensions but the Python loop may be slower than optimized Mamba kernels. For an exact original-model reproduction, the paper must specify and the code must match the original SSM implementation. The hypergraph relevance and salience are bounded reference approximations where the manuscript equations lack machine-readable detail. Do not describe this as proof of the reported 18.6M parameters or 291 ms/1,000 posts without measurement.

## Required data

Provide a real CSV with columns `text,label,timestamp,platform,user_id` (only first three required; UTC parseable timestamp mandatory). Labels may be strings or integers. Dataset source, chronological metadata and permissions are the user's responsibility. Dataset files are never invented or included. Six datasets described in the paper must be handled as separate independently documented corpora; the same command can run on each. Cross-corpus post/user comparisons need a unified pre-split audit.

## Install and run

```bash
python -m pip install -r requirements.txt
python run_all.py --data YOUR_VERIFIED_DATA.csv --config config.yaml --out outputs
# Short functional run, not a scientific experiment:
python run_all.py --data YOUR_VERIFIED_DATA.csv --config config.yaml --out outputs --smoke
pytest -q test_pipeline.py
```

`outputs/` is generated locally, not included in this intentionally flat source archive. The pipeline writes split CSVs, a leakage audit, best weights, metadata, actual test predictions, evaluation metrics, coverage-risk CSVs, a sensitivity *structure* table, shift masks, and training figures. Experiments producing absent benchmark metrics are not falsely reported as completed.

## Reproduction boundaries

- Chronological splitting is enforced with non-overlapping timestamps, training-fitted vocabulary and causal hypergraph statistics. Calibration is fitted only on a separate validation-derived calibration partition. Never train on test-set vocabulary or labels.
- `run_all.py` currently executes **one train–validation–calibration–test run**. It does **not** execute a complete five-task continual-learning benchmark; `ContinualMemory.consolidate` and `.penalty` are exposed for implementing that sequential protocol with verified task windows. Do not claim retention results until that protocol is run.
- `baselines.py` includes an executable TF-IDF+linear SVM and neural variant constructors. These architectures need to be trained and evaluated on the same data to support comparative claims; no baseline benchmark results are assumed.
- `reviewer_experiments.py` measures active-edge counts under grid variants and identifies shift masks. Its CSV is **not predictive performance sensitivity**: each variant requires independent model retraining and validation to obtain F1/accuracy conclusions.
- `statistics_tests.py` enables paired instance/cluster bootstrap and seed summaries when genuine aligned predictions from multiple runs are available. A single run does not justify significance claims.
- The author's obsolete FCM/top-50 Fisher/Multi-Stacked BiLSTM prose should not be described as the THCM-SSL model. Such techniques are only optional independent baselines.

## Reproduce figures and tables

All plots must be generated from real experimental logs. `visualize.py` currently plots training convergence and coverage-risk; it does not fabricate paper Fig. 2 edge evolution or Fig. 3 gate heatmaps. To produce those, record successive `TemporalHypergraph.edges`, last_pruned, and gate activation tensors on actual, permitted corpus windows.

## Manuscript consistency

See `REPRODUCIBILITY.md` for the precise evidence required before asserting that the code reproduces the reported macro-F1, ECE, retention and latency. A ZIP of code alone cannot establish the six original dataset provenance records, the claimed significance or Figure 1 icon copyright permission. Contact the editor separately about Figure 1 provenance.
