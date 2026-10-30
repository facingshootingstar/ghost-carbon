# Independent single-thread re-runs: local + Kaggle (evidence for Results / Threats)

These are **independent re-runs** of the Digits PROXY protocol.
**The paper's tables and figures use the locked `results/emissions.csv`, not these files.**

Protocol (same as the locked run):
- Digits, 75/25 stratified split, `random_state=42`; RF `n_estimators=100, n_jobs=1`;
  inference = 50 × `predict` over the 450-row test set.
- `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1` (shell for `src/run_experiment.py`; notebook sets them before importing sklearn).
- PROXY meter (65 W CPU × 475 gCO2/kWh); MLP = sklearn `MLPClassifier`.
- Local `rerun_0*.csv`: each run in a scratch copy so `results/emissions.csv` was never written.
- `kaggle_01.csv`: public notebook output from Kaggle (`drowningcraft/ghost-carbon-claimed-green-metered-gray`), same protocol, single-thread pin.

Summary (family kg = train + infer):

| file | RF/LR | MLP/LR | order | RF infer share | MLP train share |
|---|---|---|---|---|---|
| emissions.csv (locked) | 25.4× | 19.3× | RF > MLP | 62.6% | 96.3% |
| rerun_01.csv | 23.2× | 17.0× | RF > MLP | 64.5% | 96.6% |
| rerun_02.csv | 29.8× | 21.2× | RF > MLP | 64.0% | 96.7% |
| rerun_03.csv | 26.9× | 21.4× | RF > MLP | 63.2% | 96.8% |
| rerun_04.csv | 27.9× | 22.4× | RF > MLP | 62.8% | 96.5% |
| rerun_05.csv | 28.0× | 20.7× | RF > MLP | 62.8% | 96.5% |
| kaggle_01.csv (Kaggle, OMP=1) | 13.1× | 17.6× | **MLP > RF** | 62.4% | 96.9% |

`rerun_01..05` were run on the local machine. `kaggle_01.csv` is the `results/emissions.csv` written by the public Kaggle notebook run
([ghost-carbon-claimed-green-metered-gray](https://www.kaggle.com/code/drowningcraft/ghost-carbon-claimed-green-metered-gray)),
downloaded unmodified (the notebook omits the `proxy_cpu_w` / `proxy_gco2_per_kwh` columns; values are 65 W and 475 gCO2/kWh).
Single-thread (`OMP/OPENBLAS/MKL_NUM_THREADS=1` set before sklearn import). LR train share on Kaggle: 74.4%.

**What these files support:** phase shares stay in a narrow band on both machines
(RF infer about 62–65%, MLP train about 96–97%); this is the stable claim.
Family ratios move (local RF/LR 23–30×, MLP/LR 17–22×; Kaggle RF/LR ≈13.1×, MLP/LR ≈17.6×).

**Family order is machine-dependent:** all five local re-runs keep RF > MLP > LR, while the Kaggle
single-thread run ranks MLP > RF > LR. Default multi-thread BLAS can collapse the ratios further;
no multi-thread run is deposited here.
