# Dengue Risk Prediction Model — Research

Interpretable dengue risk models built on **9,449,900 cleaned notification records** from Brazil's
national notifiable disease surveillance system (SINAN, 2023–2025), with external validation
against the **IDAMS** model published in *Lancet Global Health* 2023.

> A Chinese-language version of this document is preserved at [`README.zh-CN.md`](README.zh-CN.md).

---

## Contents

| Path | What it holds |
|---|---|
| [`code/`](code/) | The three-stage pipeline: feature engineering → model fitting → external validation |
| [`results/`](results/) | Fitted coefficients and metrics (`模型结果_三模型指标与系数.json`), IDAMS validation output (`模型结果_IDAMS校验.json`) |
| [`figures/`](figures/) | Publication-ready figures (ROC curves, coefficient plots, gradients) |
| [`reports/`](reports/) | Full methods report, project record, and teaching slides (Chinese) |

The fitted coefficients in `results/` are the sole artefact consumed by the deployed service —
see [`../service/`](../service/).

---

## Three findings

**1. Severity is predictable.** Model B2 reaches **AUC 0.810**. The dominant drivers are
leukopenia and comorbidity burden (haematological disease, renal disease, autoimmune disease,
diabetes, hypertension).

**2. "Is this dengue?" is hard to learn — and the obstacle is the data.** Model A reaches only
0.686. SINAN has no genuine non-dengue febrile control group: patients are notified *because*
dengue is suspected, so there are no true negatives. No change of algorithm can recover
information the sampling frame never captured.

**3. Published models require recalibration before cross-population use.** Transferring the IDAMS
calculator directly to Brazilian data dropped AUC from the original 0.75–0.86 to **0.617**, even
though every overlapping coefficient had the correct sign. Its score nevertheless increases
monotonically with final severity (high-risk tier: 9% of ordinary cases → 24% of severe cases),
which supports positioning it as a **triage aid** rather than a diagnostic tool.

---

## The three models

| Model | Target | n | Positives | AUC | Sensitivity | Specificity |
|---|---|---|---|---|---|---|
| **A** | Dengue (10/11/12) vs. inconclusive (8) | 300,000 | 150,000 | 0.686 | 0.718 | 0.547 |
| **B** | Warning signs + severe (11/12) vs. ordinary (10) | 364,246 | 164,246 | 0.722 | 0.605 | 0.725 |
| **B2** | Severe (12) vs. other dengue (10/11) | 212,310 | 12,310 | **0.810** | 0.699 | 0.780 |

`CLASSI_FIN` outcome coding: `8` = inconclusive/discarded · `10` = ordinary dengue ·
`11` = dengue with warning signs · `12` = severe dengue.

> ⚠ Metrics are computed on **balanced (downsampled) test sets** and are therefore optimistic.
> Under true prevalence, positive predictive value will be markedly lower. Re-evaluate on a
> prevalence-preserving test set before deployment.

### Why downsampling

Severe cases are roughly 0.15% of confirmed dengue. Trained directly, a model that always predicts
"not severe" scores 99.8% accuracy while catching nothing. The pipeline keeps every minority-class
record and randomly subsamples the majority class, then additionally applies
`class_weight="balanced"`.

---

## Features (26)

**14 symptoms** (binary): fever, myalgia, headache, rash, vomiting, nausea, back pain,
conjunctivitis, arthritis, arthralgia, petechiae, **leukopenia**, positive tourniquet test,
retro-orbital pain.

**7 comorbidities** (binary): diabetes, haematological disease, liver disease, renal disease,
hypertension, peptic ulcer disease, autoimmune disease.

**5 further terms**: age (years), sex (female = 1), days of illness (notification date −
symptom onset), and two seasonal harmonics `sin(2πw/52)`, `cos(2πw/52)` derived from the
epidemiological week.

Symptom coding follows SINAN: `1 = yes`, `2 = no`, `9 = unknown`; the pipeline maps anything other
than `"1"` to 0, so **"no" and "unknown" are indistinguishable to the model**. The deployed
service reproduces this convention exactly.

---

## External validation against IDAMS

| Check | Result |
|---|---|
| ① Coefficient direction agreement | ✅ **Passed** — all 4 overlapping variables agree with IDAMS (4/4) |
| ② Day-of-illness gradient | ❌ **Not replicated** — IDAMS rises with illness duration (0.75 → 0.86); ours is flat (0.65–0.67) |
| ③ Whole-calculator transfer | ⚠ **Degraded but usable** — AUC 0.617 (direct transfer) < 0.650 (local refit) < 0.668 (full feature set) |

Finding ② is an informative negative result. SINAN lacks the variables carrying IDAMS's strongest
late-stage discrimination (mucosal bleeding, skin flushing), and records symptoms once at
notification rather than through daily follow-up.

---

## Reproducing the analysis

### Dependencies

```bash
pip install pandas numpy scikit-learn pyarrow matplotlib
```

### Data

Download from Brazil's Ministry of Health, DATASUS: <https://datasus.saude.gov.br/>
(datasets `DENGBR23`, `DENGBR24`, `DENGBR25`). Raw data is **not** vendored in this repository.

### Pipeline

```bash
# 1. Feature engineering, one year at a time
python3 code/01_prepare_data.py 2023 /path/to/DENGBR23.csv
python3 code/01_prepare_data.py 2024 /path/to/DENGBR24.csv --chunked   # 6.56M rows, needs chunking
python3 code/01_prepare_data.py 2025 /path/to/DENGBR25.csv

# 2. Fit the three models
python3 code/02_fit_models.py A
python3 code/02_fit_models.py B
python3 code/02_fit_models.py B2

# 3. External validation against IDAMS
python3 code/03_idams_validation.py
```

Intermediate files are written to `/tmp/`; final outputs are `/tmp/full_results.json` and
`/tmp/idams_eval.json`, which correspond to the two files checked into [`results/`](results/).

> **A parsing trap worth flagging:** SINAN dates are ISO `%Y-%m-%d`, not `%d/%m/%Y`. Using the
> wrong format silently fails every date parse and empties the sample. This cost real debugging
> time during the project.

### Data volumes

| Year | Raw notifications | After cleaning | Severe (code 12) |
|---|---|---|---|
| 2023 | 1.646 M | 1.577 M | 1,549 |
| 2024 | 6.565 M | 6.288 M | 8,226 |
| 2025 | 1.645 M | 1.585 M | 2,535 |
| **Total** | **9.856 M** | **9.4499 M** | **12,310** |

The 2024 anomaly is real: Brazil experienced its most severe recorded dengue epidemic that year.

---

## Known limitations

- **No true negatives** — the most fundamental constraint; it structurally caps what Model A can learn.
- **Optimistic metrics** — balanced test sets inflate specificity and positive predictive value.
- **Insufficient variable depth** — platelet count and haematocrit, among the strongest laboratory
  predictors, are absent from the notification form.
- **Passive surveillance bias** — symptoms are self-reported and captured once, with care-seeking
  selection effects.
- **Uncalibrated thresholds** — the IDAMS appendix gives no intercept, so its calculator yields a
  relative score rather than an absolute probability. The same applies to the coefficients exported
  here: `intercept_` was not saved, which is why the deployed service reports a relative index.

---

## References

1. *Early diagnostic indicators of dengue versus other febrile illnesses in Asia and Latin America*
   (IDAMS study). **Lancet Global Health** 2023; appendix 7, Table S11.
2. WHO. *Dengue: Guidelines for Diagnosis, Treatment, Prevention and Control.* 2009.
3. *Machine learning for predicting severe dengue in Puerto Rico.* **Infectious Diseases of
   Poverty** 2025.
4. Brazilian Ministry of Health — DATASUS / SINAN dengue notification data.
