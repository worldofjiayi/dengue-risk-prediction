# Evaluation harness

Scenario-based regression gate for the dengue risk assessment service, and the
seed of its **failure-case library**. Per the project's methodology, this
harness is *the only thing that can tell you whether to (re)train the model*:
scenarios pin down the currently verified behavior of the API — model tiers,
rule-channel outputs (WHO warning signs, epidemiological exposure), advice
contracts, and language coverage. When a model retrain or a rule change shifts
any of that, the harness turns red **before** users see it, and every failing
scenario is dumped with its full request/response for offline triage. Those
dumps, together with the `data/assessments.jsonl` feedback loop (see
`scripts/eval_stats.py`), are the evidence base for the "should we retrain /
recalibrate" decision called out in the recalibration caveat at the end of the
README's Scoring section.

Everything runs in-process against the FastAPI app with `MOCK_MODE=true`
forced — no server, no network, no DeepSeek calls — so results are
deterministic and safe to run anywhere. Scores are season-anchored
(the seasonal terms cancel in the 0–100 normalization), so pinned score ranges
are stable across calendar dates.

## Running

From the project root (Windows):

```
.venv\Scripts\python.exe service\scripts\eval_run.py                 # human-readable
.venv\Scripts\python.exe service\scripts\eval_run.py --json          # machine-readable
.venv\Scripts\python.exe service\scripts\eval_run.py --only id1,id2  # subset
.venv\Scripts\python.exe service\scripts\eval_run.py --scenarios path\to\other.json
```

Exit code 0 only if every scenario passes; 1 if any check fails; 2 for usage
or file errors. Each failing scenario writes
`service/eval/failures/<id>.json` containing the request, the response, and
the failed checks (the directory is cleared at the start of every run, so it
always reflects the latest run only).

## Adding a scenario

Append an object to `scenarios.json`:

```json
{
  "id": "kebab-case-unique-id",
  "description": "What behavior this pins down and why it matters",
  "endpoint": "assess",            // or "chat" / "destination"
  "request": { ...full request body for that endpoint... },
  "checks": [ {"type": "status", "expect": 200}, ... ]
}
```

Guidelines:

- Write the **full** request body (all 14 symptoms, 7 comorbidities,
  3 exposure answers) so a scenario never depends on defaulting behavior
  unless that defaulting *is* the behavior under test.
- Encode **current verified behavior**, not desired behavior: run the request
  once, read the actual response, then pin it. If the app and a scenario
  disagree, the scenario is wrong until proven otherwise.
- Pin scores with a small window (e.g. `±0.2`) around the observed value to
  absorb float rounding, never wide "anything goes" ranges.

## Check types

| type | parameters | passes when |
| --- | --- | --- |
| `status` | `expect` | HTTP status code equals `expect`. |
| `level` | `model`, `expect_in` | `body[model]["level"]` is one of `expect_in`. `model` is `dengue`/`worsening`/`severe`. |
| `score_between` | `model`, `lo`, `hi` | `body[model]["score"]` is within `[lo, hi]`. |
| `warning_signs` | `expect` | `body["warning_signs"]` equals `expect` as a set (exact match). |
| `exposure_level` | `expect` | `body["exposure_context"]["level"]` equals `expect`. |
| `field_equals` | `path`, `expect` | Value at dotted `path` (list segments by integer index, e.g. `advice.medical.0`) equals `expect`. Generic escape hatch. |
| `advice_order` | — | The advice object's keys serialize exactly as `medical, monitoring, protection` (medical first — a frontend contract). |
| `medical_urgency` | — | At least one `advice.medical` item contains a seek-care keyword for the request's language (就医/就诊/急诊, 就醫/就診/急診, "seek medical/care", "atención médica"/"acuda", "procure atendimento", …). Lexicons live in `eval_run.py`. |
| `no_probability_language` | — | No advice/summary string pairs a numeric percentage with probability wording (概率/機率/probabilit…/chance). The scores are relative indicators, never infection probabilities. |
| `explanations_present` | — | Each of the three models has 1–5 explanation entries, sorted by absolute contribution descending. |
| `reply_nonempty` | — | Chat only: `reply` is a non-empty string. |
| `reply_mentions_tier` | — | Chat only: the reply references the overall tier computed from the request's context (highest of the three levels), using the per-language tier labels in `eval_run.py`. |
| `scores_match_scenario` | `ref` | The runner executes the scenario with id `ref` and all three `z` values match exactly. Used to prove that exposure answers and unknown-vs-no answers never move the model scores. |
| `sources_urls_allowed` | `min_sources`, `max_sources` (both optional) | Chat / destination: every URL appearing in the generated text (`reply`, `recent_findings`, `advice`) is present in `sources`, and the source count falls within the given bounds. This is the harness-side guard on the no-fabricated-citation invariant — the app enforces it too, in `app/verifier.py`, and the two checks are written independently. |
| `sources_origins` | `expect_origins` (optional) | Every source carries an `origin` of `who` or `search`, and each origin listed in `expect_origins` appears at least once. Provenance is the point: a WHO Disease Outbreak News item and a page a search engine returned are not equally hard, and the reader has to be able to tell them apart. |
| `search_count` | `expect` or `max` | Chat only: `body["search_count"]` equals `expect` (or is at most `max`). `expect: 0` is how "a question with no place in it must not buy a single web search" is pinned. |
| `no_model_scores` | — | Destination only: the response carries none of `dengue`, `worsening`, `severe`, `epi_week`, `advice_source`. A destination lookup produces travel context, never a score — inventing a "destination risk score" from a coarse country table is exactly what this service does not do. |
| `advice_source` | `expect` | `body["advice_source"]` equals `expect` (`llm` or `template`). Distinguishes advice the model produced and that passed verification from the template fallback used when generation fails or keeps violating the rules. |

The lexicons and tier labels are deliberately **duplicated** in `eval_run.py`
rather than imported from `app/` — the harness pins expected behavior, so if
the app's wording drifts, the harness should fail loudly instead of silently
following along.

## Tests

`service/tests/test_eval_run.py` is the meta-test: it runs the shipped
scenario file (must pass), verifies that a failing scenario produces exit
code 1 plus a failure dump with actual values, and covers `--only`, `--json`,
and `scores_match_scenario` mismatch detection.
