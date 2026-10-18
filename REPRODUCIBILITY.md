# Reviewer-facing reproducibility and claims audit

**Study:** THCM-SSL / Scientific Reports revision. **Status:** reference implementation requiring empirical reproduction on verified original data.

| Reviewer concern | Evidence/code supplied | What must still be completed |
|---|---|---|
| Method inconsistency FCM/BiLSTM | THCM-SSL exclusively in `thcm_ssl.py`; comparison variants in `baselines.py` | Remove obsolete prose from manuscript and identify the original authors' exact checkpoint/configuration |
| Hyperedge composition and pruning | `TemporalHypergraph`, `.update`, `.stats`, `last_pruned` | Verify normalized PMI, salience and semantic-similarity equations vs source implementation |
| Mixing coefficients / tau sensitivity | `config.yaml`, `reviewer_experiments.py` | Retrain each configuration on genuine datasets; report validation selection and future test performance |
| Trainable gate and sparsity | `SparseFeatureGate` + joint loss in `train.py` | Log activation distributions across epochs and compare fixed gating |
| SSM architecture / complexity | `SelectiveSSMBlock`, model parameter reporting | Match the original SSM kernel and benchmark CPU/GPU timing by sequence length; reference recurrent code is not fused Mamba |
| Fisher and class prototypes | `ContinualMemory` | Run genuine multi-window continual training, update prototypes only with eligible training labels |
| Temporal dataset leakage | `prepare_data.py` and split audit | Supply verified exact dates, source IDs, duplicates across all six corpora and user identifiers where permitted |
| Statistical reliability | `statistics_tests.py` | Collect aligned predictions across genuine five-seed runs, paired bootstrap, per-corpus uncertainty and multiple-testing correction |
| Calibration/selective risk | `TemperatureCalibrator`, `evaluate.py` | Report pre/post-temperature metrics on future-time test without test-fit temperature |
| Drift ablations/baselines | `baselines.py`, `reviewer_experiments.py` | Fit all comparison methods and calculate per-dataset numeric outcomes |
| Latency and memory | `train.py` metadata and model parameter count | Benchmark matched batch sizes/sequences, warmed-up accelerator, forward-only AND end-to-end latency |
| Figure 1 image permissions | Not a code task | Identify original creators or obtain permission per editor's letter |

## Sources and unverified claims

The supplied manuscript claims mean macro-F1 0.852, balanced accuracy 0.845, ECE 0.024 after calibration, task-1 final retention 0.849 and latency 291 ms per 1,000 posts. These are **manuscript claims, not repository-generated results**. No raw examples, checkpoints, logged predictions or exact SSM kernel proving these values were supplied. Until they are reproduced, do not attribute them to this reference repository.

The manuscript identifies SemEval-T17 as the hyperparameter source, and states alpha/beta/gamma 0.4/0.35/0.25, gate coefficient .01, retention .4, prototype eta .1 and a 20th percentile hyperedge threshold. The reference tau=3.0, data fractions, default max length, and diagonal SSM are implementation choices, not asserted original settings. Parameters and code paths must be audited against original experiment files before publication-grade reproducibility is claimed.

The requested editorial code deposit is in a recognized DOI-assigning repository. GitHub alone does not meet that explicit condition. No DOI is supplied because no deposit has been made.
