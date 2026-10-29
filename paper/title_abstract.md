## 1. Ghost Carbon

**Title:** Claimed Green, Metered Gray: Ghost Carbon in “Efficient” Machine Learning

**Subtitle:** When a model is called green before anyone reads the watt-hour meter

**Abstract.**
Small models are often described as climate-friendly by narrative alone—“lightweight,” “efficient,” “negligible impact”—while reporting no metered emissions at all. We train and run inference for random forests, logistic regression, and a small MLP on the Digits workload and record train and infer energy with CodeCarbon when available, otherwise with a clearly labeled CPU-watt proxy (PROXY_labeled; this draft). We treat the “negligible / unreported” claim as a reported value of zero and define *ghost carbon* as metered minus reported. In absolute terms the footprint is tiny (≈7.8e-06 kgCO₂e for all six runs), and we make no planetary claim; the point is relative: 100% of the metered footprint is missing from the claim, and footprints differ ≈25× across model families that would all be called “green.” Train versus inference shares are reported so claims cannot hide behind training-only accounting. Unlike tool-validation studies (CodeCarbon vs. wattmeter) or venue-level surveys of how often carbon is reported, we audit the green claim itself against a meter. **Takeaway:** meter before you market green.

**Keywords:** green AI, carbon accounting, CodeCarbon, ML emissions, sustainability claims

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

**Scope note:** PROXY_labeled proxy; micro-benchmark (lab) numbers, not planetary impact.
