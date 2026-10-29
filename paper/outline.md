# Claimed Green, Metered Gray: Ghost Carbon in “Efficient” Machine Learning

## Abstract bullets
- Small ML models are often marketed as “green” or “efficient” without metered emissions.
- We measure train+infer kgCO₂e (CodeCarbon or labeled proxy) for RF, LR, and a tiny MLP on Digits.
- Narrative “near-zero” claims diverge from even tiny metered footprints; inference vs train share is reported.
- Takeaway: meter before you claim green — ghost carbon is the gap between story and joules.

## Related work & positioning (added 2026-10-29)
**Contribution.** We contribute an *audit* of narrative “negligible/unreported” green claims against metered (or labeled-proxy) train+infer kgCO₂e on small models, reporting the relative ghost gap and train/infer share.

**Related work & positioning.**
Prior work established that ML training has a measurable carbon cost (Strubell et al., 2019) and gave practitioners estimators such as the ML CO₂ Impact calculator (Lacoste et al., 2019) and CodeCarbon (Courty et al.). Tool-focused studies then asked how accurate those estimators are: Bouza et al. (2023) compare seven trackers against wattmeters for deep-learning training, and Fischer (arXiv:2509.22092) ground-truths CodeCarbon and static estimates against external measurements, finding errors of up to 40%. On the reporting side, Chiu's ICML 2026 position paper surveys 1,000 ICML 2025 papers and finds carbon data in only 5.9% of them, arguing that efficiency claims without energy data are incomplete. We share that reporting-gap motivation. Gap: prior work measures joules or tool error; few audit the “negligible/unreported” claim itself, which yields a 100% relative ghost gap on small RF/LR/MLP models, together with the train↔infer share.

- **Not the same as Fischer (2509.22092) / Bouza et al. (2023):** we do not validate CodeCarbon or any tool against a wall meter; our meter is CodeCarbon or a labeled CPU-watt proxy, and the object under test is the claim, not the meter.
- **Not the same as Chiu (ICML 2026 position):** we do not survey reporting rates across a venue; we take one class of claim (“negligible/unreported” on small models) and quantify what it omits, split by train vs. inference.
- **Not the same as Strubell/Lacoste-style footprinting:** we make no large-model or planetary estimate; absolute kg are micro-scale and the finding is the relative gap.

**References.**
1. Strubell, E., Ganesh, A., McCallum, A. (2019). Energy and Policy Considerations for Deep Learning in NLP. ACL. arXiv:1906.02243.
2. Lacoste, A., Luccioni, A., Schmidt, V., Dandres, T. (2019). Quantifying the Carbon Emissions of Machine Learning. arXiv:1910.09700.
3. Courty, B., Schmidt, V., et al. CodeCarbon (software). https://github.com/mlco2/codecarbon
4. Bouza, L., Bugeau, A., Lannelongue, L. (2023). How to estimate carbon footprint when training deep learning models? A guide and review. Environ. Res. Commun. 5, 115014. https://pmc.ncbi.nlm.nih.gov/articles/PMC10661046/
5. Fischer, R. (2025). Ground-Truthing AI Energy Consumption: Validating CodeCarbon Against External Measurements. arXiv:2509.22092. https://arxiv.org/abs/2509.22092
6. Chiu, G.-M. (2026). Position: Carbon Footprint Reporting Should Be Routine in Machine Learning Research. ICML 2026. https://icml.cc/virtual/2026/poster/67123 ; https://github.com/guan404ming/carbon-footprint-reporting

## RQs
RQ1 metered kgCO₂e; RQ2 claim vs meter; RQ3 train vs infer share.

## Method
Digits; RF/LR/MLP; CodeCarbon else proxy (CPU W × s × gCO₂/kWh); claim baseline = “negligible” narrative (0 or paper-typical understatement).

## Expected tables/figures
- Table 1: per-model train/infer energy, kgCO₂e, duration
- Fig 1: claim-vs-metered bar chart
- Table 2: train share vs infer share


## Paper type & scope (q34 upgrade)
Short conference paper: **reproducible audit protocol** (subtitle tag: Protocol & Synthetic Demonstration).

*This paper presents a reproducible audit protocol demonstrated on synthetic/proxy data; a REAL_DATA path is documented for follow-on measurement, not claimed as completed observation.*

**Limitations.** Synthetic/PROXY demonstration only; REAL_DATA path documented (README) but not run; contribution is an audit metric, not a first discovery; no journal-tier or indexing claim.

## Camera-ready figure captions
- Fig. 1 (`results/claim_vs_metered.{pdf,svg,png}`): Reported vs. metered footprint per model family on Digits. Grey baseline: the “negligible/unreported” claim encoded as 0 kg. Stacked bars: PROXY-metered training (blue) and inference (orange, dotted) emissions in mg CO₂e; labels give totals. Ghost gap = 100% of metered for every family.

Figures: 300 dpi PNG + vector PDF/SVG, ≥9 pt fonts, Okabe–Ito / viridis-family palettes; regenerate with `python src/make_figures.py` (reads frozen results/seeds, never rewrites CSV/JSON).
