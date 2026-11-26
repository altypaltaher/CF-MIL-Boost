# CF-MIL-Boost

Code, per-repetition results and figures for the paper

> **Deciding at the Right Granularity: Cross-Fitted Multiple-Instance Boosting for Session-Level Android Malware Detection from Network Traffic**
> Altyeb Taha, King Abdulaziz University (manuscript submitted to *Electronics*, MDPI).

CF-MIL-Boost treats each one-minute window of an application's network traffic as a bag of flows. A flow-level LightGBM model produces instance scores that are cross-fitted by capture. A window-level LightGBM model then combines window statistics of these scores with robust statistics of the raw flow features. Application (capture) decisions average the window probabilities.

![Pipeline](paperfigs/Fig1_pipeline.png)

## Evaluation protocol

All models, including the baselines, are evaluated under the same leakage-free protocol:

- **Grouped splits by capture.** Flows are never split. 20% of captures form the test set, and the rest is divided into fit 70% / validation 15% / aggregation 15%, stratified by category.
- **300-s horizon.** Every capture is restricted to its first 300 s. This removes the capture-length shortcut (elapsed-time flow ROC-AUC 0.877 → 0.494).
- **Time-only control.** In every binary repetition, the window index is also used as the only score (ROC-AUC 0.498 ± 0.002).
- **Identical baselines.** LightGBM, XGBoost, AdaBoost and GBM receive the same windows, captures, raw statistics, class weights and MCC-based model selection as CF-MIL-Boost.

## Main results (MCC, mean over repetitions)

| Task | Level | Best tuned baseline | CF-MIL-Boost |
|---|---|---|---|
| Binary (10 reps) | 1-min window | 0.552 (LightGBM) | **0.669** |
| Binary (10 reps) | Capture | 0.644 (GBM) | **0.787** |
| Five-class (5 reps) | 1-min window | 0.515 (XGBoost) | **0.668** |
| Five-class (5 reps) | Capture | 0.566 (LightGBM/XGBoost) | **0.772** |

CF-MIL-Boost won every repetition against every baseline. The full tables and corrected resampled t-tests (Nadeau & Bengio, 2003) are in `mil/*_summary.txt` and `mil/*_pairwise.csv`.

## Repository layout

| Path | Content |
|---|---|
| `features.txt` | The 80 CICFlowMeter features used as model inputs |
| `build_ctx.py` | Builds the per-flow table (`ctx_flows.parquet`) from the CICAndMal2017 per-capture CSV files, with capture IDs, start times and causal host-context features (used only in the ablation) |
| `mil_run.py` | All experiments: `python mil_run.py <bin\|cat> <rep> <stage>` (stages below) |
| `mil_loop.sh` | Resumable driver over tasks, repetitions and stages |
| `mil_st.py` | Summaries and pairwise statistics (corrected resampled t-test, Wilcoxon) |
| `mil_pred.py` | Re-runs one repetition and saves window-level test predictions (used for Figure 6) |
| `mil_fig.py`, `fig_pipeline.py`, `fig_stats.py`, `fig_roc.py` | Figures of the paper |
| `mil/` | Raw per-repetition results (JSON lines), summaries, pairwise tests, repetition-0 predictions |
| `figs/`, `paperfigs/` | Figures (PNG and PDF, 600 dpi) |

Stages of `mil_run.py`:

| Stage | Content |
|---|---|
| `A` | Flow-level LightGBM, without and with host-context features |
| `B` | Flow-level XGBoost, without and with host-context features |
| `C` | Flow-level AdaBoost and GBM |
| `D1` | Window-level baselines (LightGBM, XGBoost, AdaBoost, GBM) and the time-only control |
| `D2` | Ablations: mean vote and MIL-LR pooling of flow scores |
| `D3` | CF-MIL-Boost (scores + raw statistics) and the scores-only ablation |
| `E` | Random-flow-split leakage check |

## Reproducing the experiments

1. Download the CICAndMal2017 CSV files (CICFlowMeter output per capture) from the Canadian Institute for Cybersecurity: <https://www.unb.ca/cic/datasets/andmal2017.html>. Arrange them as `<root>/<Category>/<Family>/<capture>.csv`, with the categories `Adware`, `Benign`, `Ransomware`, `SMSmalware` and `Scareware`. The dataset is not redistributed here.
2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Build the flow table:
   ```bash
   CFMIL_DATA=/path/to/root python build_ctx.py
   ```
4. Run the experiments. The paper uses 10 binary and 5 five-class repetitions; results are appended to `mil/<task>_results.jsonl`.
   ```bash
   TASKS="bin" REPS="0 1 2 3 4 5 6 7 8 9" bash mil_loop.sh
   TASKS="cat" REPS="0 1 2 3 4" STAGES="A B C D1 D2 D3" bash mil_loop.sh
   ```
5. Compute the statistics and figures:
   ```bash
   python mil_st.py bin && python mil_st.py cat
   python mil_fig.py bin && python mil_fig.py cat
   python mil_pred.py bin 0 D3 && python mil_pred.py bin 0 D1
   python mil_pred.py cat 0 D3 && python mil_pred.py cat 0 D1
   python fig_pipeline.py && python fig_stats.py && python fig_roc.py
   ```

The output directory defaults to the repository folder and can be changed with `CFMIL_WORK`. Random seeds are fixed per repetition (`1000 + rep`).

The experiments were run with Python 3, LightGBM 4.7.0, XGBoost 3.4.1, scikit-learn 1.9.1, pandas 3.0.5, NumPy 2.5.3, SciPy 1.18.1 and matplotlib, on a two-core machine with 7 GB of memory.

## Citation

If you use this code, please cite the paper (full reference to be added on publication):

```
A. Taha, "Deciding at the Right Granularity: Cross-Fitted Multiple-Instance Boosting for
Session-Level Android Malware Detection from Network Traffic," Electronics, 2026 (submitted).
```

## License

MIT License (see `LICENSE`).
