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
(haematological, renal and autoimmune disease, plus diabetes and hypertension), consistent with
prior severity-prediction work [3]. The "is this dengue at all?" model reaches only AUC 0.686 —
and the bottleneck is the data, not the algorithm: SINAN contains no
true negative controls, because non-dengue febrile illnesses are never notified.

**2. A published model lost most of its discrimination when transferred.**
The IDAMS calculator (*Lancet Global Health* 2023) fell from a reported AUC of 0.75–0.86 to
**0.617** on Brazilian data, even though all four overlapping coefficients pointed in the correct
direction. Its score still rose monotonically with final severity (high-risk tier: 9% of ordinary
cases → 24% of severe cases), which suggests it is better positioned as a **triage aid** than as a
diagnostic instrument.

**3. A negative result worth reporting.**
IDAMS's day-of-illness gradient did not replicate. The original model improves as illness
progresses (0.75 → 0.86); the transferred calculator is flat on our data (0.60–0.64 by day), as
is a per-day refit of our own model (0.65–0.67). SINAN lacks the variables that carry IDAMS's
late-stage discrimination (cough, rhinorrhoea, skin flushing, serial temperature), and records
symptoms once at notification rather than through daily follow-up.

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
Chinese methods report and slides: [`model/reports/`](model/reports/) ·
Figures: [`model/figures/`](model/figures/)

**Reports.** The full English technical report lives at
[`docs/technical-report.pdf`](docs/technical-report.pdf)
([HTML source](docs/technical-report.html)). A formal project delivery report, covering the same
ground plus deployment and billing operations, is available in English
([`docs/project-report.pdf`](docs/project-report.pdf)) and Chinese
([`docs/项目报告.pdf`](docs/项目报告.pdf)). Research
artifacts under `model/` intentionally keep their original Chinese filenames; the directory
tables in [`model/README.md`](model/README.md) give an English description of each file.

---

## The deployed service

```
Web questionnaire  →  FastAPI  →  deterministic feature encoding (26 features)
                                        ↓
                          three logistic regression models (A / B / B2)
                                        ↓
                      LLM-generated advice in the user's language  →  browser
```

- **Seven-step questionnaire**: age, sex and days of illness; 14 symptoms; 3 exposure questions;
  7 comorbidities; and an optional free-text note — every symptom, comorbidity and exposure
  question answered *yes / no / don't know*
- **Five languages**: Simplified Chinese, Traditional Chinese, English, Spanish, Portuguese,
  covering the Americas, Chinese-speaking users, and other dengue-endemic regions
- **Two rule layers outside the model**: a WHO warning-sign alert and an epidemiological exposure
  tier (nearby confirmed case, local fever cluster, travel to an outbreak area) — see below
- **Beyond assessment**: a citation-verified follow-up chat, a pre-travel destination lookup
  (WHO outbreak news plus web search), and an adaptive questionnaire planner with provable early
  stopping — details in [`service/README.md`](service/README.md)
- **407 tests** — including a hand-computed check of the coefficient dot product — plus a
  26-scenario eval harness

Engineering details, API contract, and deployment steps: [`service/README.md`](service/README.md)

### Three design decisions worth explaining

**Scores are relative, not probabilities.** The training pipeline exported `coef_` but not
`intercept_`, and used downsampling with `class_weight="balanced"`. There is therefore no
calibrated absolute risk to report. The service anchors each score against a *reference person*
— same epidemiological week, no symptoms, no comorbidities, a 30-year-old male on day 0 of
illness — and reports the
position between that reference and the model's ceiling. Every user-facing string states this
explicitly; nowhere does the interface claim a percentage probability.

**"Don't know" is encoded as 0, faithfully.** SINAN codes symptoms as `1 = yes`, `2 = no`,
`9 = unknown`, and the original feature engineering treats anything other than `"1"` as 0. The
questionnaire therefore offers a genuine "don't know" option and encodes it the same way the
model was trained. This matters: leukopenia, the single strongest severity predictor, is a
laboratory finding, and the tourniquet test is a clinical manoeuvre; no member of the public
knows either without having been examined.

**WHO warning signs bypass the model entirely.** Both severity models lean heavily on leukopenia
— it is the single strongest predictor in Model B (β = 1.600) and second only to haematological
disease in Model B2 (β = 1.400 vs. 1.529). A patient with persistent vomiting and petechiae who
has not had blood drawn can therefore score "low" on all three models — a false reassurance in
exactly the situation where WHO says to seek care. So the service applies an independent rule:
if such signs are reported, a prominent alert is shown and the advice generator is instructed to
recommend prompt medical assessment regardless of score.

---

## Quick start

Requires Python 3.10+.

```bash
git clone https://github.com/worldofjiayi/dengue-risk-prediction.git
cd dengue-risk-prediction/service

python3 -m venv .venv
./.venv/bin/pip install -r requirements.txt

cp .env.example .env          # defaults to MOCK_MODE=true — no API key needed
./.venv/bin/uvicorn app.main:app --port 8000
```

Open <http://localhost:8000>. In mock mode the **risk scores are computed by the real model**;
only the natural-language advice is canned. Run the tests with `./.venv/bin/pytest tests`.
On Windows, use `.venv\Scripts\` in place of `.venv/bin/`.

Reproducing the research pipeline requires downloading SINAN data from
[DATASUS](https://datasus.saude.gov.br/) — see [`model/README.md`](model/README.md).

---

## Limitations

- **No true negatives.** Only suspected dengue is notified to SINAN, so Model A learns "dengue
  vs. inconclusive", not "dengue vs. other febrile illness". This is a structural ceiling.
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

## Data availability

All training data are public, de-identified notification records from Brazil's
[DATASUS](https://datasus.saude.gov.br/) (DENGBR23/24/25). No record-level data is redistributed
in this repository; only fitted coefficients ship with the service. The deployed service logs
de-identified feature vectors and scores only — free-text input is never stored (see
[`service/README.md`](service/README.md)).

---

## License and citation

Code and configuration are released under the [MIT License](LICENSE). Reports, figures, and
documentation may be reused with attribution. Raw SINAN surveillance data is not distributed in
this repository and remains subject to DATASUS terms; the fitted coefficient files carry no
record-level information. To cite this repository, see [`CITATION.cff`](CITATION.cff).

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
