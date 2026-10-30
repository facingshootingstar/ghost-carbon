# Independent single-thread re-runs (evidence for Results / Threats)

These are **independent re-runs** of `src/run_experiment.py` (unmodified).
**The paper's tables and figures use the locked `results/emissions.csv`, not these files.**

Protocol (same as the locked run):
- Digits, 75/25 stratified split, `random_state=42`; RF `n_estimators=100, n_jobs=1`;
  inference = 50 × `predict` over the 450-row test set.
- Shell: `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1` (the script does not set them).
- PROXY meter (65 W CPU × 475 gCO2/kWh); MLP = sklearn `MLPClassifier`.
- Each run in a scratch copy so `results/emissions.csv` was never written.

Summary (family kg = train + infer):

| file | RF/LR | MLP/LR | order | RF infer share | MLP train share |
|---|---|---|---|---|---|
| emissions.csv (locked) | 25.4× | 19.3× | RF > MLP | 62.6% | 96.3% |
| rerun_01.csv | 23.2× | 17.0× | RF > MLP | 64.5% | 96.6% |
| rerun_02.csv | 29.8× | 21.2× | RF > MLP | 64.0% | 96.7% |
| rerun_03.csv | 26.9× | 21.4× | RF > MLP | 63.2% | 96.8% |
| rerun_04.csv | 27.9× | 22.4× | RF > MLP | 62.8% | 96.5% |
| rerun_05.csv | 28.0× | 20.7× | RF > MLP | 62.8% | 96.5% |

**What these files support:** phase shares stay in a narrow band; family ratios move (RF/LR about 23–30×, MLP/LR about 17–22×).
**What they do not show:** an RF/MLP order swap, or the multi-thread ratio collapse (~4×) mentioned in Threats (that case needs a multi-thread or other-environment run, e.g. a Kaggle session with default BLAS).
