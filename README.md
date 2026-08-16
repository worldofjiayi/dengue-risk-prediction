# Dengue Risk Prediction: From Surveillance Data to a Deployed Self-Assessment Tool

An interpretable dengue risk model built on **9.45 million notification records** from Brazil's
national surveillance system (SINAN, 2023–2025), externally validated against a published
*Lancet Global Health* model, and deployed as a multilingual public self-assessment web service.

**Live service:** http://35.88.114.45

---

## What this repository contains

This is a two-part project: the **research** that produced the model, and the **engineering**
that put it in front of users.

| | Directory | Contents |
|---|---|---|
| **1. Research** | [`model/`](model/) | Feature engineering and training pipeline, fitted coefficients, external validation against IDAMS, figures, and the full methods report |
| **2. Service** | [`service/`](service/) | FastAPI backend that serves the fitted model, hand-written multilingual frontend, test suite, and deployment configuration |

The two halves are deliberately separable: the service ships only a **2.9 KB coefficient file**
([`service/app/model/dengue_models.json`](service/app/model/dengue_models.json)). No raw
surveillance data is vendored, downloaded, or required at inference time.

---

## Key findings

**1. Severity is predictable; case definition is not.**
The severe-dengue model reaches **AUC 0.810**, driven by leukopenia and comorbidities
(haematological, renal, autoimmune disease, diabetes, hypertension). The "is this dengue at all?"
model reaches only 0.686 — and the bottleneck is the data, not the algorithm: SINAN contains no
true negative controls, because non-dengue febrile illnesses are never notified.

**2. A published model lost most of its discrimination when transferred.**
The IDAMS calculator (*Lancet Global Health* 2023) fell from a reported AUC of 0.75–0.86 to
**0.617** on Brazilian data, even though all four overlapping coefficients pointed in the correct
direction. Its score still rose monotonically with final severity (high-risk tier: 9% of ordinary
cases → 24% of severe cases), which suggests it is better positioned as a **triage aid** than as a
diagnostic instrument.

**3. A negative result worth reporting.**
IDAMS's day-of-illness gradient did not replicate. The original model improves as illness
progresses (0.75 → 0.86); ours stays flat (0.65–0.67). SINAN lacks the variables that carry
IDAMS's late-stage discrimination (mucosal bleeding, skin flushing), and records symptoms once at
notification rather than through daily follow-up.

---

## Model performance

| Model | Target | n | AUC | Sensitivity | Specificity |
|---|---|---|---|---|---|
| **A** | Dengue vs. inconclusive | 300,000 | 0.686 | 0.718 | 0.547 |
| **B** | Warning signs + severe vs. ordinary | 364,246 | 0.722 | 0.605 | 0.725 |
| **B2** | Severe vs. other dengue | 212,310 | **0.810** | 0.699 | 0.780 |

> Metrics come from balanced (downsampled) test sets and are therefore **optimistic**. Under the
> true low prevalence of severe dengue (~0.15% of confirmed cases), positive predictive value will
> be substantially lower. Re-evaluation on a prevalence-preserving test set is required before any
> clinical use.

Full methodology: [`model/README.md`](model/README.md) ·
Reports and figures: [`model/reports/`](model/reports/), [`model/figures/`](model/figures/)

---

## The deployed service

```
Web questionnaire  →  FastAPI  →  deterministic feature encoding (26 features)
                                        ↓
                          three logistic regression models (A / B / B2)
                                        ↓
                      LLM-generated advice in the user's language  →  browser
```

- **Six-step questionnaire**: age, sex, days of illness, 14 symptoms and 7 comorbidities, each
  answered *yes / no / don't know*
- **Five languages**: Simplified Chinese, Traditional Chinese, English, Spanish, Portuguese —
  chosen for equatorial dengue-endemic regions
- **Two risk gauges plus a rule-based safety layer** (see below)
- **44 tests**, including a hand-computed check of the coefficient dot product

Engineering details, API contract, and deployment steps: [`service/README.md`](service/README.md)

### Three design decisions worth explaining

**Scores are relative, not probabilities.** The training pipeline exported `coef_` but not
`intercept_`, and used downsampling with `class_weight="balanced"`. There is therefore no
calibrated absolute risk to report. The service anchors each score against a *reference person*
— same epidemiological week, no symptoms, no comorbidities, 30 years old — and reports the
position between that reference and the model's ceiling. Every user-facing string states this
explicitly; nowhere does the interface claim a percentage probability.

**"Don't know" is encoded as 0, faithfully.** SINAN codes symptoms as `1 = yes`, `2 = no`,
`9 = unknown`, and the original feature engineering treats anything other than `"1"` as 0. The
questionnaire therefore offers a genuine "don't know" option and encodes it the same way the
model was trained. This matters: leukopenia and the tourniquet test are the two strongest
severity predictors, and no member of the public knows their own values.

**WHO warning signs bypass the model entirely.** Both severity models lean heavily on leukopenia
— it is the single strongest predictor in Model B (β = 1.600) and second only to haematological
disease in Model B2 (β = 1.400 vs. 1.529). A patient with persistent vomiting and petechiae who
has not had blood drawn can therefore score "low" on all three models — a false reassurance in
exactly the situation where WHO says to seek care. The service therefore applies an independent rule: if such signs are
reported, a prominent alert is shown and the advice generator is instructed to recommend prompt
medical assessment regardless of score.

---

## Quick start

```bash
git clone <this repository>
cd jiayi/service

python3 -m venv .venv
./.venv/bin/pip install -r requirements.txt

cp .env.example .env          # defaults to MOCK_MODE=true — no API key needed
./.venv/bin/uvicorn app.main:app --port 8000
```

Open <http://localhost:8000>. In mock mode the **risk scores are computed by the real model**;
only the natural-language advice is canned. Run the tests with `pytest tests`.

Reproducing the research pipeline requires downloading SINAN data from
[DATASUS](https://datasus.saude.gov.br/) — see [`model/README.md`](model/README.md).

---

## Limitations

- **No true negatives.** SINAN notifies suspected dengue only, so Model A learns "dengue vs.
  inconclusive", not "dengue vs. other febrile illness". This is a structural ceiling.
- **Optimistic metrics.** Balanced test sets inflate specificity and PPV relative to deployment.
- **Missing laboratory depth.** Platelet count and haematocrit — among the strongest known
  predictors — are absent from the notification form.
- **Passive surveillance bias.** Symptoms are self-reported and recorded once, with care-seeking
  selection effects.
- **Uncalibrated thresholds.** The low/medium/high cut-offs are engineering defaults. The service
  logs de-identified feature vectors and scores to support local recalibration.
- **Hemisphere transfer.** Seasonal terms were fitted on southern-hemisphere data. The service
  cancels the seasonal term out of its relative score, so this does not bias individual results,
  but it does mean seasonality is not currently modelled for users.

---

## Disclaimer

This project produces **risk indicators, not medical diagnoses**. Output must not be used to
diagnose, treat, or rule out disease, and cannot substitute for assessment by a qualified
clinician. Anyone who is unwell, or whose symptoms worsen, should seek medical care.

---

## References

1. *Early diagnostic indicators of dengue versus other febrile illnesses in Asia and Latin America*
   (IDAMS study). **Lancet Global Health** 2023; appendix 7, Table S11.
2. WHO. *Dengue: Guidelines for Diagnosis, Treatment, Prevention and Control.* 2009.
3. *Machine learning for predicting severe dengue in Puerto Rico.* **Infectious Diseases of
   Poverty** 2025.
4. Brazilian Ministry of Health — DATASUS / SINAN dengue notification data (DENGBR23/24/25).
