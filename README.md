# Ghost Carbon — Claimed Green, Metered Gray

**VI:** So “xanh / tiết kiệm” trên giấy với CO₂ train+infer đo được (CodeCarbon hoặc proxy). Ba mô hình nhỏ, chạy CPU &lt;10 phút.

## Claim
ML papers and model cards often claim “green”, “efficient”, or “low-carbon” without reporting **measured** training and inference emissions. We meter a few standard small models and contrast narrative claims vs kgCO₂e.

## Research questions
- **RQ1:** How much kgCO₂e do train + infer produce for RF, logistic regression, and a tiny MLP on public tabular/digits data?
- **RQ2:** How do measured (or proxy) emissions compare to typical “green / efficient” narrative claims in small-model papers?
- **RQ3:** Does inference dominate or training dominate for these tiny models under a fixed CPU budget?

## Method (short)
1. Load sklearn Digits.
2. Train RF, LR, tiny Torch MLP; run a fixed inference batch.
3. Meter with CodeCarbon when available; otherwise a **clearly labeled** proxy (energy ≈ power × time × carbon intensity).
4. Log CSV; plot claim-vs-metered bars.

## How to run
```bash
cd ghost-carbon
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python src/run_experiment.py
```
Outputs: `results/emissions.csv`, `results/claim_vs_metered.png`.

Kaggle: open `notebooks/ghost_carbon.ipynb` and Run All.

## Independence
Shares **no** code or data with the other env-papers folders.
