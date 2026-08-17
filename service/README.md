# Dengue Risk Self-Assessment — Web Service

FastAPI service that puts the dengue models from [`../model/`](../model/) in front of the public:
a six-step questionnaire, three relative-risk scores, and LLM-generated guidance in five languages.

**Live:** http://35.88.114.45

> ⚠️ Scores are **relative risk indicators, not infection probabilities**, and thresholds have not
> been calibrated for any local population. See [Scoring](#scoring) below.

---

## Architecture

```mermaid
flowchart LR
    A[Questionnaire<br>static/index.html] -->|POST /api/assess| B[FastAPI<br>app.main:app]
    B --> C[Deterministic encoding<br>26 features]
    B -.optional.-> D[LLM call #1<br>extract symptoms from free text]
    D -.-> C
    B --> R[Rule checks<br>WHO warning signs<br>exposure context]
    C --> E[Three logistic models<br>A dengue / B worsening / B2 severe]
    E --> X[Contribution breakdown<br>top-5 per model]
    E --> F[LLM call #2<br>advice in user's language]
    R --> F
    F --> V{{"Output verifier<br>app.verifier"}}
    V -->|clean| B
    V -->|"violations: re-ask once"| F
    V -.->|twice failed, or upstream error| T["Template advice<br>advice_source = template"]
    T --> B
    X --> B
    B -->|AssessmentResult| G["Dual gauges + banners + advice"]
    G -->|POST /api/chat| H{"Place named in the question?<br>app.intel.find_location"}
    H -->|no| H1["OpenAI /chat/completions<br>function tool, no search"]
    H -->|yes| H2["Anthropic /v1/messages<br>server-side web_search"]
    H1 <-->|tool_calls| I["lookup_dengue_context<br>app.intel"]
    H2 --> I
    G -->|POST /api/destination| D2["Pre-travel lookup<br>app.destination<br>cached 6h per place × language"]
    D2 --> I
    D2 --> H2
    I --> J[("dengue_endemicity.json<br>WHO + CDC, 2026")]
    I --> K["WHO Disease Outbreak News<br>live, cached 12h"]
    H2 --> S[("Search results<br>origin: search")]
    H1 --> W{{"Output verifier<br>allowed URLs = WHO ∪ search results"}}
    H2 --> W
    W -->|clean| G
    W -.->|twice failed| Y["Localised fallback reply<br>/ empty recent_findings"]
```

Feature encoding is **deterministic and authoritative**. The first LLM call has one narrow job:
read the free-text notes field and flag symptoms the user described but did not tick. It may only
promote `unknown → yes`; it can never override an explicit answer, and any failure is logged and
ignored rather than failing the request.

Two things run **beside** the model rather than inside it — the WHO warning-sign check and the
epidemiological exposure tier. Both are plain rules over the questionnaire, and both are described
below.

Three more layers wrap the LLM calls themselves: a rule-based
[output verifier](#output-verification) that every generated string must pass before it reaches a
user, an [epidemic-intelligence tool](#epidemic-intelligence) the chat model can call on its
own initiative, and [web search](#web-search) — reached over a *different* API protocol, and only
when the user actually named a place.

- Endpoints: `POST /api/assess`, `POST /api/chat`, `POST /api/destination`, `POST /api/plan`
- Entry point: `app.main:app` — port `80` in production, `8000` for local development
- Static frontend served directly by FastAPI (`GET /` returns `static/index.html`)
- Health: `GET /api/health` → `{"status":"ok","mock_mode":bool,"models":["A","B","B2"]}`
- Validation errors return 422; server errors return 500 with `{"detail": "..."}` (localised to the
  request's `language`). `/api/chat` returns 502 when the upstream model is unreachable;
  **`/api/assess` no longer does** — see [Output verification](#output-verification)

---

## The model

Coefficients live in [`app/model/dengue_models.json`](app/model/dengue_models.json) (2.9 KB) — a
copy of the research output in [`../model/results/`](../model/results/). No training data is
required at inference time. A test asserts the two copies never drift apart.

| Model | Field in response | Target | AUC |
|---|---|---|---|
| A | `dengue` | Dengue vs. inconclusive | 0.686 |
| B | `worsening` | Warning signs + severe vs. ordinary | 0.722 |
| B2 | `severe` | Severe vs. other dengue | **0.810** |

### Features (26)

Order is defined by `FEATS` in [`app/schemas.py`](app/schemas.py) and matches the training script
exactly.

- **14 symptoms** (binary): `FEBRE`, `MIALGIA`, `CEFALEIA`, `EXANTEMA`, `VOMITO`, `NAUSEA`,
  `DOR_COSTAS`, `CONJUNTVIT`, `ARTRITE`, `ARTRALGIA`, `PETEQUIA_N`, `LEUCOPENIA`, `LACO`,
  `DOR_RETRO`
- **7 comorbidities** (binary): `DIABETES`, `HEMATOLOG`, `HEPATOPAT`, `RENAL`, `HIPERTENSA`,
  `ACIDO_PEPT`, `AUTO_IMUNE`
- `age` (0–110), `sex_f` (female = 1), `day_ill` (0–14), `wk_sin` / `wk_cos` (seasonal harmonics,
  computed server-side from the current ISO week)

The three exposure questions are **not** part of this vector — see
[Epidemiological exposure](#epidemiological-exposure--deliberately-outside-the-model).

### Tri-state answers

Every symptom and comorbidity is answered **yes / no / don't know**, encoded `yes → 1`,
`no → 0`, `unknown → 0`.

This is not a shortcut. SINAN codes `1 = yes`, `2 = no`, `9 = unknown`, and the training pipeline's
`(df[c] == "1")` collapses "no" and "unknown" to the same value. Reproducing that convention keeps
inference faithful to training — and it matters in practice, because leukopenia and the tourniquet
test are among the strongest predictors and no member of the public knows their own values.

### Scoring

The training pipeline exported `coef_` but **not `intercept_`**, and used downsampling with
`class_weight="balanced"`. Two obvious approaches fail:

- `sigmoid(z)` — with no intercept, z is persistently positive and nearly every symptomatic user
  saturates near 100.
- Normalising against the theoretical range — `z_min` corresponds to holding *every*
  negative-coefficient symptom, an unreachable state; a genuinely asymptomatic person then lands
  mid-range (measured: a healthy 25-year-old scored "medium").

The service instead anchors on a **reference person**:

```
score = 100 × (z − z_ref) / (z_ceil − z_ref)          clipped to [0, 100]

z_ref  = same epidemiological week, no symptoms, no comorbidities, 30-year-old male, day 0
z_ceil = same week, every risk-increasing feature at its bound
```

The score therefore answers: *relative to a symptom-free person right now, where does this
respondent sit on this model?* The seasonal term appears identically in all three quantities and
cancels — deliberately, since `wk_sin`/`wk_cos` capture a population-level seasonal baseline
rather than individual risk. It still contributes to `z`, which is returned and logged.

Observed spread:

| Case | dengue | worsening | severe |
|---|---|---|---|
| Healthy 25M, no symptoms | 0 low | 0 low | 0 low |
| Mild: 30F, fever + headache, 2 days | 30.4 low | 0 low | 0 low |
| Typical: 35F, 4 symptoms, 3 days | 45.1 medium | 1.1 low | 0 low |
| High risk: 72M, leukopenia + 4 comorbidities | 53.1 medium | 69.3 **high** | 70.1 **high** |

The low/medium/high cut-offs (35 / 65) are engineering defaults. Recalibration on a
prevalence-preserving sample is required before clinical use; the eval log exists to support that.

### Adaptive questioning — provable early stopping

Twenty-one tri-state questions is a lot to ask someone with a fever. `POST /api/plan`
([`app/planner.py`](app/planner.py)) lets the frontend ask only the questions that can still
change the outcome — and prove when it is safe to stop.

**The bounds maths.** All three models are linear in their features, and every coefficient is
known. For a partially-answered questionnaire, each model's final score is therefore *hard-bounded*:

- an **answered** binary question contributes exactly `yes → coef`, `no`/`unknown` → 0;
- an **unasked** binary question can only end up contributing 0 or its coefficient, so over the
  unasked set:

  ```
  z_min = z_answered + Σ min(0, coef_f)
  z_max = z_answered + Σ max(0, coef_f)
  ```

- `age` / `sex` / `day_ill` are mandatory in the questionnaire's first step, and the seasonal
  terms are server-computed constants for the day — neither adds uncertainty.

`z_min` / `z_max` pass through the *same* reference-anchored normalisation as the real score
(the shared `score_from_z` helper, clipped to [0, 100]), yielding `[score_min, score_max]` per
model. `score_now` is the score with unasked questions treated as 0 — exactly what `/api/assess`
would return if the user stopped right now, byte-for-byte.

**The stop rule.** A model is `decided` when `[score_min, score_max]` lies inside a single risk
band (same `level` at both ends, honouring the 35 / 65 cut-offs). When all three models are
decided, **no combination of remaining answers can change any tier** — `can_stop` is true and
that is a proof, not a heuristic. In practice a healthy respondent who answers "no" to the
half-dozen highest-impact questions is fully decided with 14 questions still unasked.

The request distinguishes *answered "don't know"* (a deterministic 0 — the interval tightens
exactly as for "no") from *not asked yet* (genuinely uncertain): **a present key means answered,
a missing key means unasked**. This is why the endpoint has its own `PlanRequest` model instead
of reusing `FormInput`, whose validators fill missing keys with `unknown` and would erase the
distinction.

**Contract.**

```json
POST /api/plan
{
  "age": 35, "sex": "F", "day_ill": 3,
  "symptoms":      { "FEBRE": "yes", "VOMITO": "no" },
  "comorbidities": { "DIABETES": "unknown" },
  "language": "zh-CN"
}
```

```json
{
  "bounds": {
    "dengue":    { "score_now": 45.1, "score_min": 30.2, "score_max": 88.7,
                   "level_now": "medium", "decided": false },
    "worsening": { "...": "..." },
    "severe":    { "...": "..." }
  },
  "can_stop": false,
  "next": [
    { "kind": "symptom",     "code": "LEUCOPENIA", "why_model": "severe" },
    { "kind": "comorbidity", "code": "HEMATOLOG",  "why_model": "severe" }
  ],
  "answered": 3,
  "remaining": 18
}
```

Unknown codes and out-of-range values return 422, as in `/api/assess`.

**Question ranking.** `next` holds up to 5 unasked questions, ordered by information value:

```
impact(f) = Σ over undecided models m of |coef_f^m| / (z_ceil^m − z_ref^m)
```

Dividing by each model's own normalisation span makes coefficients comparable across models
before summing. `why_model` names the undecided model where the feature's normalised |coef| is
largest — the frontend uses it to say "this question mainly informs the severe-disease estimate".
Ordering is fully deterministic: impact descending, ties broken by `FEATS` order. Once every
model is decided, `next` is `[]` no matter how many questions remain.

**Deterministic by design.** The planner calls no LLM. The fitted coefficients *are* the
information value of each question — which answer could move which score by how much is a closed-
form fact, and the stop rule is a proof over score bounds. An LLM choosing the next question
could only add noise (and non-reproducibility) to a decision the model already answers exactly.

The two WHO warning-sign symptoms (`VOMITO`, `PETEQUIA_N`) and the three exposure questions are
treated as mandatory by the frontend regardless of planning — the planner does not special-case
them; they simply tend to arrive already answered.

### WHO warning signs — independent of the model

Both severity models lean heavily on leukopenia — the single strongest predictor in Model B
(β = 1.600), and second only to haematological disease in Model B2 (β = 1.400 vs. 1.529). A
patient reporting persistent vomiting and petechiae who has not had blood drawn can therefore
score "low" on all three models — a false reassurance precisely where WHO advises seeking care.

The backend therefore applies a rule check on `VOMITO` and `PETEQUIA_N` and returns
`warning_signs`. The frontend shows a prominent alert, and the advice prompt is instructed to
recommend prompt medical assessment regardless of score.

### Epidemiological exposure — deliberately outside the model

Three questions ask about exposure rather than symptoms:

| Code | Question |
|---|---|
| `FEVER_CLUSTER` | An unusual increase in fever cases among people nearby |
| `CONFIRMED_CASE` | A confirmed dengue case nearby (household, workplace, neighbourhood) |
| `OUTBREAK_TRAVEL` | Recent travel to, or residence in, a dengue outbreak area |

Clinically these are among the strongest clues a physician has. Statistically, this service cannot
weight them: **SINAN notification records contain no such variables**, so the fitted models have no
coefficients for them. Adding them to the feature vector would mean inventing numbers, and it would
break the byte-for-byte correspondence with the training script that the rest of this document
depends on.

So they take the same route as the WHO warning signs — a rule, reported alongside the scores:

```
high   = CONFIRMED_CASE == yes  or  OUTBREAK_TRAVEL == yes
medium = FEVER_CLUSTER == yes   (and not high)
low    = otherwise
```

Only an explicit `yes` counts; `unknown` never raises the tier, for the same reason `unknown`
encodes as 0 in the model. The result is returned as `exposure_context`, with `factors` listing the
codes answered `yes` so the frontend can render translated labels. The advice prompt receives the
tier and is told to treat a `high` exposure as raising the case for seeing a clinician even when
the scores are low. A test asserts the 26-feature vector is identical with every exposure answer
set to `yes` versus `no`.

### Score explanations

Because all three models are logistic regressions without interaction terms, `z` decomposes exactly:
each feature contributes `coef[name] × value`, and the parts sum to `z`. No approximation, no
surrogate model. `explanations` returns the top 5 contributors per model:

```json
{ "feature": "FEBRE_x", "code": "FEBRE", "contribution": 0.904, "direction": "up" }
```

- Zero contributions are omitted — a symptom the user does not have moved nothing.
- Sorted by `|contribution|` descending, capped at 5, rounded to 4 decimals.
- `direction` is `"up"` when the term raised `z`, `"down"` when it lowered it.
- `code` strips the `_x` suffix so the frontend can reuse the questionnaire's translated labels;
  the five non-binary features (`age`, `sex_f`, `day_ill`, `wk_sin`, `wk_cos`) keep their own names.

Note that `wk_sin`/`wk_cos` can appear as contributors: they move `z`, even though they cancel out
of the 0–100 score (see [Scoring](#scoring)).

---

## Output verification

Everything above this line is deterministic and checkable. The generated prose is not — so it gets
checked on the way out. [`app/verifier.py`](app/verifier.py) is a **pure rule engine**: no model, no
network, no I/O. It does not judge whether advice is *good*; it judges whether it crossed a line
this service is not allowed to cross, and each of those lines is expressible as a rule.

| Code | Rule | Deliberately *not* a violation |
|---|---|---|
| `dosage` | A number adjacent to a dose unit (`mg`, `ml`, `mcg`, `g`, `片`, `粒`, `毫克`, `毫升`, `comprimido`, `tableta`, `tablet`) or a dosing interval (`每 N 小时`, `cada N horas`, `a cada N horas`, `every N hours`) | "avoid aspirin or ibuprofen" — a safety warning with no number is not a prescription. Unit-tested in all five languages |
| `probability` | A `%` figure, probability wording, **and** a second-person reference, all in the same sentence | "90% of dengue cases are mild" — a population statistic, tested in both directions per language |
| `urgency_missing` | `overall_tier == "high"` or any warning sign present, yet no `medical` item matches the language's seek-care lexicon | Low tier with no warning signs — advice may legitimately give a threshold instead |
| `language_mismatch` | zh-CN/zh-TW below 30% CJK characters; en/es/pt above 5% CJK **or** carrying fewer than two of the language's function words | Loanwords and place names — the thresholds are deliberately loose |
| `structure` | Any of `medical` / `monitoring` / `protection` outside 1–5 non-blank items, or an item over 400 characters | — |
| `fabricated_url` | *(chat / destination)* A link that does not prefix-match one returned by this turn's **tools or web search** | A reply with no links at all |
| `empty` | *(chat only)* A blank reply | — |

Each violation carries a developer-facing `message` written in the second person, because that
message is fed straight back to the model.

**Retry, then fall back.** On the advice path: generate → verify → if anything fires, re-ask once
with the violation messages appended ("Your previous answer violated: … Regenerate the same JSON,
fixing only these issues") → verify again → if it still fails, serve the built-in template for that
language and tier. `/api/chat` gets the same single retry, then falls back to a localised
"I can't produce a reliable answer right now — please consult a local health service."

There is exactly **one** copy of the fallback text: `fallback_advice(language, tier)` in
`deepseek_client.py`, the same function that powers mock mode. Demo prose and production fallback
cannot drift apart because they are the same strings. Mock mode runs the verifier too — it is free —
and a test asserts **every template × 5 languages × 3 tiers × with/without warning signs yields zero
violations**. That test is what makes the fallback path trustworthy; a dirty template would mean the
safety net is itself unsafe.

The urgency lexicon exists twice on purpose: once in `verifier.py`, once in `scripts/eval_run.py`.
Neither imports the other. If someone rewords the advice templates, one side goes red.

### `advice_source`, and the deliberate 502 → 200 change

`AssessmentResult` gained `advice_source: "llm" | "template"`. Mock mode and every fallback report
`"template"`; only verified live output reports `"llm"`.

**An advice-stage `DeepSeekError` no longer fails the assessment.** It used to return 502. It now
returns **200 with template advice and `advice_source: "template"`.**

The reasoning: the scores, the WHO warning-sign check, the exposure tier and the contribution
breakdown are all computed locally and are the part of this service with actual evidence behind
them. Throwing all of that away because one paragraph of natural language was unavailable is a bad
trade — especially for a user who has just answered twenty-one questions about their symptoms. The
response stays honest about what happened via `advice_source`.

`/api/chat` keeps its 502: there the reply *is* the entire output, and there is nothing to fall
back to but a localised apology.

---

## Epidemic intelligence

The chat model can call one tool, on its own initiative:
`lookup_dengue_context(location)` — [`app/intel.py`](app/intel.py).

```json
{
  "location": "Singapore",
  "matched": true,
  "endemicity": "high",
  "season_note": "Year-round; warmer months of June to October usually see the highest counts.",
  "who_notices": [
    { "title": "Dengue - Global situation", "date": "2024-05-30",
      "url": "https://www.who.int/emergencies/disease-outbreak-news/item/2024-DON518" }
  ],
  "lookup_failed": false
}
```

**Provenance.** Two sources, both named in the payload's own data file:

- [`app/data/dengue_endemicity.json`](app/data/dengue_endemicity.json) — 81 countries and
  territories tiered `high` / `moderate` / `low` / `none`, each with a short English season note,
  compiled from the **WHO dengue and severe dengue fact sheet** and the **CDC dengue risk map**
  (2026, recorded in the file's `_sources` block). Sub-national reality that a country tier would
  misrepresent lives in the note: China is `moderate` "confined to the south — Guangdong, Yunnan,
  Fujian, Guangxi and Hainan", the United States is `moderate` "south Florida, the Texas Gulf coast
  and Hawaii", Australia is `moderate` "far north Queensland only". A separate `Northern China`
  entry is `none`. An alias map (~360 keys) resolves English, zh-CN, zh-TW, Spanish and Portuguese
  spellings plus common variants — `USA`, `美国`, `Estados Unidos`, `u.s.` all reach `United States`.
  **This table never touches a score.** It is travel context, reported next to the model output the
  same way the exposure tier is.
- **WHO Disease Outbreak News**, live, via the public OData endpoint
  `GET https://www.who.int/api/news/diseaseoutbreaknews?$filter=contains(Title,'Dengue')&$orderby=PublicationDateAndTime desc`
  (httpx, 8 s timeout, no key). Items whose `Title` contains the canonical country name are kept,
  newest first, capped at three. If none match, the newest **global** notices are returned
  unchanged — their titles literally read "Global situation", so they describe their own scope and
  cannot be mistaken for a country-specific alert. Each URL is built as
  `https://www.who.int/emergencies/disease-outbreak-news/item/{UrlName}`; nothing is hand-assembled.

**The no-fabricated-URL invariant.** This is the load-bearing part. `ChatResponse.sources` holds
exactly what this turn retrieved — WHO tool results plus, on the search path,
[what the web search returned](#web-search) — and those URLs are the *only* ones the reply may
contain: `verify_chat_reply(reply, language, allowed_urls=<this turn's source URLs>)`. An empty
`allowed_urls` means no tool returned anything, so **any** link is a violation. Fail twice and the
turn is replaced with the localised fallback and empty `sources`. The tool description tells the
model the same thing in words — cite only what came back, never reconstruct a who.int link — but
the verifier is what enforces it, and the eval harness re-checks it end to end with its own,
separately written URL regex.

**Caching.** The WHO list is cached in-process for 12 hours (module-level timestamped cache,
injectable so tests drive it directly). DONs are published rarely; hitting who.int on every chat
turn would be both slow and rude. A *failed* fetch is never cached — the next turn tries again.

**Honest failure.** Network error, HTTP error, or a malformed payload all produce
`lookup_failed: true` with `who_notices: []`. There is no "best guess" branch anywhere in this
module. The local endemicity table still answers, because it is on disk — so a user asking about
Brazil during a WHO outage still learns Brazil is highly endemic, and simply gets no citation.

**Mock mode** makes no network call and serves a canned list of *real* DON entries through the
*same* selection logic, so the payload shape is byte-identical to production. Singapore, Brazil,
Thailand and their zh aliases are the demo path; an unrecognised place returns `matched: false`.
The mock is the environment, not a different contract.

### Function calling

`DeepSeekClient.chat_with_tools(system, messages, tools, tool_executor, …)` runs the OpenAI
`tools` / `tool_calls` loop and stays purely transport: it never imports the intel module's logic.
The pipeline injects `tool_executor(name, args) -> dict`, which is where argument cleaning, the
actual lookup, and result collection live. On `tool_calls` the client executes each call in a worker
thread (an 8-second HTTP call must not block the event loop), appends `role: "tool"` messages, and
continues. After `max_rounds` (default 2) it sends one final turn **without** tools and with an
explicit "answer now, using only the tool results above" instruction, rather than looping forever.
Malformed `arguments` JSON becomes `{}` and a crashing tool becomes `{"error": …,
"lookup_failed": true}` — the model is told what went wrong instead of the request dying.

In mock mode, if the question or recent history names a known location, the client simulates one
tool round by **actually invoking the injected executor** and citing a URL that genuinely came back
from it. That matters: the mock exercises the no-fabricated-URL invariant rather than side-stepping
it. Only the user's own text is scanned for place names — not the rendered prompt, which contains
language names like "葡萄牙语" that would otherwise be read as Portugal.

---

## Web search

The endemicity table is a travel almanac and the WHO Disease Outbreak News feed is a slow
publication channel. Neither can answer *"what has been happening there for the last three
months?"* — so `/api/chat` and `/api/destination` can run a real web search, with real citations.

### Two protocols in one client

DeepSeek's web search is a **server-side tool** and it exists only on the **Anthropic-compatible**
endpoint. The OpenAI-compatible endpoint this service already used does not offer it; sending the
tool there just gets an ordinary function-call request back. So `deepseek_client.py` now speaks two
protocols:

| | Endpoint | Used by | Search |
|---|---|---|---|
| `chat_json` | `POST /chat/completions` (OpenAI) | notes extraction, advice generation | never |
| `chat_with_tools` | `POST /chat/completions` (OpenAI) | `/api/chat` with **no place** named | never |
| `chat_anthropic_search` | `POST /anthropic/v1/messages` (Anthropic) | `/api/chat` **with** a place, `/api/destination` | yes |

The Anthropic path differs in every layer: `x-api-key` + `anthropic-version: 2023-06-01` instead of
a bearer token, `system` as a top-level parameter instead of a message, and a reply that arrives as
a list of content blocks — `thinking`, `text`, `server_tool_use`, `web_search_tool_result`. The
parser concatenates the `text` blocks, **ignores `thinking` entirely**, and harvests every `url`
(with its sibling `title` and `page_age`) out of the `web_search_tool_result` blocks, de-duplicated,
order preserved. `usage.server_tool_use.web_search_requests` reports what was actually spent. On
`stop_reason: "max_tokens"` the answer is still returned but a warning is logged — a truncated
answer beats a 502, as long as the log says which one you got.

Advice generation stays on the OpenAI path with no search: it summarises a score the service
already computed, and there is nothing on the web that would improve it.

### Search is bought only when a place is named

Measured against the live API, one ordinary question triggered **4 searches and ~13.9k input
tokens** (a second probe: 2 searches, 16.5k input tokens). Search is the only per-call, per-use
cost in this service and the *number* of uses is the model's decision, not ours. So the gate is
not a prompt instruction, it is control flow:

`/api/chat` runs `intel.find_location(question + recent history)` first — a local alias table,
zero cost, matching English, zh-CN, zh-TW, Spanish and Portuguese spellings. **No hit, no search
tool, not even attached.** "What does the worsening score mean?" costs exactly what it cost before.
A hit switches the turn to the Anthropic path, where the intel lookup is *also* performed and
pasted into the prompt as known facts — the free layer grounds the paid one, and its WHO URLs join
the citation whitelist.

Three settings, all in `.env`:

| Setting | Default | Effect |
|---|---|---|
| `SEARCH_ENABLED` | `true` | Master switch. Off → chat falls back to the function-tool path; `/api/destination` still answers from the WHO layer with `search_status: "disabled"` |
| `SEARCH_MAX_USES` | `2` | Passed to the tool as `max_uses`. `0` means the tool is not attached at all — that is how a verification retry re-writes prose without buying a second search |
| `SEARCH_CACHE_TTL_SECONDS` | `21600` (6 h) | Destination lookups are cached by `(canonical location, language)`. A second request for the same country makes **no external call at all** — not even the WHO one, since the whole response is cached. Only successful lookups are cached; a failure must not be pinned for six hours |

Every request that *could* have searched writes one line to the evaluation log with its
`search_count` — including the ones that ended up spending nothing. Counting only the requests that
cost money makes it impossible to compute what fraction of traffic is free, which is the number
that says whether the gate is working. `scripts/eval_stats.py` reports the total, the mean, the
maximum and the zero-search share, broken down by `chat` / `destination`.

### Provenance, and the extended no-fabricated-URL invariant

Every citation now carries an `origin`:

```json
{ "title": "Dengue - Global situation", "date": "2024-05-30",
  "url": "https://www.who.int/emergencies/disease-outbreak-news/item/2024-DON518",
  "origin": "who" }
```

`who` means the WHO Disease Outbreak News API returned it; `search` means the web-search tool did.
They are not equally hard, and a reader has to be able to tell them apart, so the label travels
with the link rather than being guessed from the domain. `date` is nullable because search results
usually arrive without a `page_age` — the field is left `null` rather than filled with today.

The invariant from [Epidemic intelligence](#epidemic-intelligence) is unchanged in spirit and wider
in scope: **allowed URLs = the union of this turn's WHO tool results and this turn's search
results.** Anything else in the generated text is a `fabricated_url` violation → one re-ask with
the violation message → still failing means the localised fallback reply (chat) or empty
`recent_findings` with `search_status: "degraded"` (destination). Unverified text is never shipped.

Two practical consequences of live testing:

- A search round returns about **10 results**, so a two-round request arrives with 18–20 URLs. The
  service caps what it exposes at 8 — but the selection puts every URL the reply actually cites
  first, because `sources` is simultaneously the citation list *and* the verifier's allow-list.
  Truncating a genuinely cited link would make the service reject its own correct answer.
- That raw result list contains whatever the search engine ranked highly — national news sites and
  aggregators alongside the health authority. Asking for official sources in the prompt shapes the
  *findings* (a live run attributed every figure to Singapore's NEA) but not the *result list*.
  This is exactly why the `origin` label exists rather than a claim that every source is official:
  the reader can see that a link is a search hit and weigh it accordingly.

**Mock mode never searches.** It returns a canned reply plus three real, stable public-health pages
(a WHO DON item, the PAHO dengue topic page, the ECDC dengue overview) with one deliberately
date-less entry, driven by the detected location — same code path, same shapes, same verification,
no network and no spend. A canned demo that cited invented URLs would make the "no fabricated
sources" invariant a lie in exactly the environment where people look at it most.

---

## API

### `POST /api/assess`

```json
{
  "age": 34,
  "sex": "F",
  "day_ill": 3,
  "symptoms":      { "FEBRE": "yes", "CEFALEIA": "yes", "LEUCOPENIA": "unknown" },
  "comorbidities": { "DIABETES": "no" },
  "exposure":      { "CONFIRMED_CASE": "yes", "FEVER_CLUSTER": "no" },
  "language": "zh-CN",
  "notes": ""
}
```

Omitted symptom / comorbidity / exposure keys default to `"unknown"`; unrecognised keys return 422.
The whole `exposure` object may be omitted, which yields `{"level": "low", "factors": []}`.

```json
{
  "dengue":    { "score": 42.2, "level": "medium", "z": 1.9308 },
  "worsening": { "score": 20.3, "level": "low",    "z": 1.9528 },
  "severe":    { "score": 20.0, "level": "low",    "z": 3.2024 },
  "epi_week": 33,
  "warning_signs": ["VOMITO", "PETEQUIA_N"],
  "exposure_context": { "level": "high", "factors": ["CONFIRMED_CASE"] },
  "summary": "...",
  "advice": {
    "medical":    ["..."],
    "monitoring": ["..."],
    "protection": ["..."]
  },
  "explanations": {
    "dengue":    [{ "feature": "FEBRE_x", "code": "FEBRE", "contribution": 0.904, "direction": "up" }],
    "worsening": [{ "feature": "LEUCOPENIA_x", "code": "LEUCOPENIA", "contribution": 1.6, "direction": "up" }],
    "severe":    [{ "feature": "CEFALEIA_x", "code": "CEFALEIA", "contribution": -0.725, "direction": "down" }]
  },
  "disclaimer": "...",
  "model_note": "Scores are relative risk indicators, not infection probabilities. ...",
  "advice_source": "llm"
}
```

`advice` is ordered **`medical` → `monitoring` → `protection`**, and that order is the display
order: the first thing a reader wants is whether to see a doctor, then what to watch at home, and
only then long-term mosquito protection. The ordering lives in the `Advice` model's field
declaration (Pydantic serialises in declaration order), and the advice prompt states both the order
and the reason so a live LLM produces the same shape.

The advice content also varies by **overall tier** — the highest of the three model levels
(`high` > `medium` > `low`). At `low` the medical advice gives a threshold for when to see someone;
at `high` its first line says to seek care promptly. In mock mode this is real: `medical` and
`summary` have three written variants per language, while `protection` and `monitoring` stay
constant.

`advice_source` is `"llm"` only when a live model produced the text **and** it passed the output
verifier. Mock mode, verifier fallback and upstream failure all report `"template"`. An advice
failure returns 200, not 502 — see [Output verification](#output-verification).

### `POST /api/chat`

A stateless follow-up conversation about the user's own result. The server keeps nothing — the
frontend replays the context and recent history on every turn.

```json
{
  "language": "zh-CN",
  "question": "Should I go to hospital now?",
  "context": {
    "dengue":    { "score": 42.2, "level": "medium" },
    "worsening": { "score": 20.3, "level": "low" },
    "severe":    { "score": 20.0, "level": "low" },
    "warning_signs": ["VOMITO"],
    "exposure_level": "high",
    "symptoms":      { "FEBRE": "yes" },
    "comorbidities": { "DIABETES": "no" },
    "age": 34, "sex": "F", "day_ill": 3
  },
  "history": [
    { "role": "user", "content": "..." },
    { "role": "assistant", "content": "..." }
  ]
}
```

```json
{
  "reply": "...",
  "sources": [
    { "title": "Dengue - Global situation",
      "date": "2024-05-30",
      "url": "https://www.who.int/emergencies/disease-outbreak-news/item/2024-DON518",
      "origin": "who" },
    { "title": "Dengue Cases - National Environment Agency",
      "date": null,
      "url": "https://www.nea.gov.sg/dengue-zika/dengue/dengue-cases",
      "origin": "search" }
  ],
  "search_count": 2
}
```

- `question` is 1–500 characters; blank or longer returns 422.
- `history` is **truncated to the last 6 messages** rather than rejected — being chatty should not
  produce an error. Roles are `user` / `assistant` only.
- Every `context` field is optional; unrecognised symptom keys are dropped instead of rejected,
  since the context is a snapshot replayed by the client.
- The system prompt requires the assistant to answer in `language`, forbids diagnosis,
  prescriptions and drug dosages, forbids stating any probability of infection, requires
  recommending a clinician when the user reports worsening or warning signs, redirects off-topic
  questions, and marks the user's text as data so embedded instructions are not followed.
- `sources` is what this turn actually retrieved and is therefore citable — `[]` when nothing ran
  or nothing was found, in which case the reply must contain no links at all. It is both the
  citation list and the verifier's allow-list. Each entry carries `origin` (`who` | `search`) and
  a nullable `date`. See [Epidemic intelligence](#epidemic-intelligence) and
  [Web search](#web-search).
- `search_count` is how many web searches this turn bought. **It is `0` whenever the question named
  no place** (the search tool is not even attached) or `SEARCH_ENABLED=false`. Both fields are
  additive — a client that ignores them behaves exactly as before.
- Replies are plain prose. `DeepSeekClient.chat_with_tools` omits `response_format`, unlike
  `chat_json` which the two assessment calls still use.
- In mock mode a question naming a known place runs both layers and cites what came back;
  otherwise the reply is canned per language and quotes the user's own risk tier. Either way, no
  network call and no spend.
- Errors: `DeepSeekError` → 502, unexpected → 500, both with a `detail` localised to `language`.
  A reply that fails verification twice returns **200** with the localised fallback text and empty
  `sources` — the model failed, not the request.

### `POST /api/destination`

A pre-travel lookup: what is the dengue situation in a place, over the **last three months**.

```json
{ "location": "Singapore", "language": "en" }
```

`location` is 1–120 characters in any language; blank returns 422. It is resolved through the same
alias table as chat, so `新加坡`, `Singapur` and `Singapore` are one cache entry.

```json
{
  "location": "Singapore",
  "matched": true,
  "endemicity": "high",
  "season_note": "Year-round; warmer months of June to October usually see the highest counts.",
  "who_notices": [
    { "title": "Dengue - Global situation", "date": "2024-05-30",
      "url": "https://www.who.int/emergencies/disease-outbreak-news/item/2024-DON518" }
  ],
  "recent_findings": ["Weekly case counts have been falling since June 2026."],
  "sources": [
    { "title": "Dengue - Global situation", "date": "2024-05-30",
      "url": "https://www.who.int/emergencies/disease-outbreak-news/item/2024-DON518",
      "origin": "who" },
    { "title": "Dengue Cases - National Environment Agency", "date": null,
      "url": "https://www.nea.gov.sg/dengue-zika/dengue/dengue-cases", "origin": "search" }
  ],
  "advice": { "protection": ["..."], "medical": ["..."], "monitoring": ["..."] },
  "search_status": "ok",
  "disclaimer": "...",
  "model_note": "..."
}
```

**There are no scores here, and there will not be.** A location has never fed the model and never
changes the exposure tier; a "destination risk score" derived from a coarse country table would be
a number invented to look quantitative. An eval check (`no_model_scores`) fails the build if
`dengue`, `worsening`, `severe`, `epi_week` or `advice_source` ever appear in this response.

Three layers, degrading independently:

1. `endemicity`, `season_note`, `who_notices` — the local table and the WHO feed. Free, stable,
   and the reason this endpoint still answers when everything else is down.
2. `recent_findings` — 2–4 short factual bullets from the web search, with dates, preferring
   government and public-health sources. **Non-empty if and only if `search_status == "ok"`.**
3. `advice` — fixed per-language travel guidance, keyed only by whether the place is endemic.
   It never comes from a model, so it is always available and always compliant. Its key order is
   deliberately `protection → medical → monitoring`, unlike `/api/assess`: nobody here is ill yet,
   so "how not to get bitten" comes before "when to see a doctor".

`search_status`:

- `ok` — the search ran and returned sources; `recent_findings` passed verification
- `degraded` — search was attempted and failed, returned nothing, was skipped because the input
  does not read like a place name, or produced text that failed verification twice. Layers 1 and 3
  are returned as usual and `recent_findings` is `[]`
- `disabled` — `SEARCH_ENABLED=false`

An upstream failure returns **200**, not 502, for the same reason `/api/assess` does: the layers
that are still valid are worth more than a clean error.

### `POST /api/plan`

Adaptive-questioning planner: hard score bounds for a partially-answered questionnaire, a
provable stop signal, and the next questions worth asking. Fully deterministic, no LLM call.
Request/response contract and the underlying maths are documented in
[Adaptive questioning](#adaptive-questioning--provable-early-stopping).

### Languages

`language` accepts `zh-CN` (default), `zh-TW`, `en`, `es`, `pt`, and controls `summary`, `advice`,
`disclaimer`, `model_note`, chat replies, and error details. Static UI strings — including the
labels behind `explanations[*].code` and `exposure_context.factors` — are localised in the frontend
([`static/app.js`](static/app.js)); the language selector persists to `localStorage` and falls back
to `navigator.language`.

---

## Running locally

Requires Python 3.10+.

```bash
python3 -m venv .venv
./.venv/bin/pip install -r requirements.txt

cp .env.example .env        # defaults to MOCK_MODE=true
./.venv/bin/uvicorn app.main:app --reload --port 8000
```

Open <http://localhost:8000>.

**In mock mode the risk scores are real** — computed by the actual model from your answers, and so
are the exposure tier and the contribution breakdown. Only the natural-language advice, chat
replies and search findings are canned (localised per language, and varied by risk tier), and
`advice_source` / `search_status` say so. The output verifier and the epidemic-intelligence tool
both run for real, the latter against its canned WHO list rather than the network. **No API key
required, and no web search is ever bought.**

```bash
pytest tests                    # 386 tests (+1 live test, skipped unless RUN_LIVE_TESTS=1)
python scripts/eval_run.py      # 26 scenarios
```

The one live test calls DeepSeek for real and therefore costs money; it is skipped by default so
CI never spends anything:

```bash
RUN_LIVE_TESTS=1 pytest tests/test_search.py -k live   # one real search, needs DEEPSEEK_API_KEY
```

---

## Using a real LLM

```ini
DEEPSEEK_API_KEY=sk-...
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-v4-flash        # or deepseek-v4-pro
MOCK_MODE=false

# Web search — per-use billing, read the Web search section first
SEARCH_ENABLED=true
SEARCH_MAX_USES=2
SEARCH_CACHE_TTL_SECONDS=21600
```

The client speaks the OpenAI-compatible `/chat/completions` format, so any compatible provider
works by changing `DEEPSEEK_BASE_URL` and `DEEPSEEK_MODEL`. The two assessment calls request
`response_format: json_object` and feed parse failures back to the model for up to two retries;
a place-free `/api/chat` turn uses the same endpoint without `response_format`, since a chat reply
should be prose, and adds `tools` / `tool_choice: auto` for the epidemic-intelligence function. A
provider that ignores `tools` degrades gracefully: no tool call, no sources, and the verifier then
forbids links outright.

Search is the one thing that is **not** portable: it uses the Anthropic-compatible
`<BASE_URL>/anthropic/v1/messages` endpoint and the `web_search_20250305` server-side tool. A
provider without that endpoint should run with `SEARCH_ENABLED=false` — chat then keeps the
function-tool path and `/api/destination` reports `search_status: "disabled"` while still serving
the endemicity table and WHO notices. Restart the service after editing `.env`
(`sudo systemctl restart jiayi`).

Going live also switches on the retry-then-fallback path: watch the logs for
`未通过输出校验` (a violation was caught) and `建议退回模板文案` (both attempts failed). A steady
stream of either means the prompt and the verifier disagree about something, and the prompt is
usually the one to fix. For search, watch `检索完成` (how many searches a turn actually bought)
and `检索回复被 max_tokens 截断` (the output budget is too small for the results that came back).

---

## Updating model coefficients

1. Re-run the training pipeline in [`../model/`](../model/)
2. Copy the output over both `../model/results/模型结果_三模型指标与系数.json` and
   `app/model/dengue_models.json`
3. Restart

Coefficients are looked up **by feature name**, not position, so ordering within the file does not
matter — but key names must match `FEATS` in `app/schemas.py`. Missing keys are treated as 0 with a
warning.

Two tests guard this path: `test_z_matches_hand_computed_value` recomputes a known z by hand, and
`test_bundled_coefficients_match_research_output` fails if the two copies drift apart.

If a future training run exports `intercept_`, the reference-anchored score can be replaced with a
calibrated probability — update `MODEL_NOTES` in `app/schemas.py` and the frontend wording at the
same time.

---

## Evaluation logging

Each assessment appends one **de-identified** JSON line to `data/assessments.jsonl`, and each
request that *could* have searched appends one line of its own:

```json
{"timestamp": "2026-08-16T02:10:10+00:00", "language": "zh-CN", "mock_mode": true,
 "epi_week": 33,
 "features": {"FEBRE_x": 1, "LEUCOPENIA_x": 0, "age": 34.0},
 "scores": {"dengue": {"score": 45.1, "level": "medium", "z": 2.219}},
 "exposure": {"FEVER_CLUSTER": "no", "CONFIRMED_CASE": "yes", "OUTBREAK_TRAVEL": "unknown"},
 "exposure_level": "high",
 "has_notes": true}
{"timestamp": "2026-08-16T02:11:04+00:00", "kind": "destination", "language": "en",
 "mock_mode": false, "location": "Singapore", "matched": true,
 "search_count": 2, "search_status": "ok"}
```

- The two record kinds are told apart by **fields, not order**: an assessment record has `scores`,
  a search record has `search_count`. Anything with neither is counted as a bad line.
- **Neither the free-text notes nor the chat question is ever written to disk** — only a
  `has_notes` boolean, and for search records the resolved place name, which is a country label
  with no identifying content.
- Search records are written for every chat turn and destination lookup that *could* have
  searched, including the ones where `search_count` is 0. Logging only the paid ones would make it
  impossible to measure what share of traffic the place-name gate keeps free.
- Records contain numeric features, scores, and language.
- `exposure` and `exposure_level` are recorded in their own block, never inside `features` —
  they are categorical answers with no identifying content, and they are exactly the covariates
  worth testing during recalibration ("does knowing about a nearby confirmed case add
  discrimination?"). Keeping them out of `features` preserves the 26-column contract.
- Path is set by `EVAL_LOG_PATH` (relative paths resolve against the service root); **empty
  disables logging**. Write failures are logged and never affect the response.
- `mock_mode` marks demo traffic so it can be filtered during analysis.

```bash
python scripts/eval_stats.py            # per-model distributions, level shares,
                                        # exposure-tier distribution, languages,
                                        # and search spend (total / mean / max / zero share)
python scripts/eval_stats.py --json     # machine-readable
```

Records written before the exposure questions existed are simply left out of the exposure
distribution rather than counted as a phantom tier.

This log is the raw material for the threshold recalibration described above.

---

## Deployment

Current production: AWS Lightsail, Oregon (us-west-2), 2 vCPU / 2 GB / Ubuntu 24.04, running as
`ubuntu` on port 80.

Package locally, excluding secrets and local state:

```bash
cd ..                       # repository root
tar czf /tmp/jiayi.tar.gz --exclude=.venv --exclude=__pycache__ --exclude='*.pyc' \
  --exclude=.pytest_cache --exclude=.git --exclude=data --exclude='.env' \
  --exclude='*.pem' --exclude=account.txt model service
scp -i ~/.ssh/your-key.pem /tmp/jiayi.tar.gz ubuntu@<PUBLIC_IP>:/tmp/
```

On the server:

```bash
sudo apt update && sudo apt install -y python3-venv python3-pip

sudo mkdir -p /opt/jiayi && sudo chown -R $USER:$USER /opt/jiayi
tar xzf /tmp/jiayi.tar.gz -C /opt/jiayi
cd /opt/jiayi/service

python3 -m venv .venv
./.venv/bin/pip install -r requirements.txt

cp .env.example .env && chmod 600 .env
vim .env                    # start with MOCK_MODE=true to validate the deployment
mkdir -p data

sudo cp deploy/jiayi.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now jiayi

systemctl is-active jiayi
curl http://127.0.0.1/api/health
journalctl -u jiayi -f
```

Finally, open port 80 in the cloud firewall (Lightsail: *Networking → IPv4 Firewall → Add rule →
HTTP*), then browse to `http://<PUBLIC_IP>`.

The service binds port 80 as the unprivileged `ubuntu` user via systemd's
`AmbientCapabilities=CAP_NET_BIND_SERVICE` — no root, no nginx. To use an unprivileged port
instead, drop the two capability lines from the unit and change `--port`.

### Redeploying

```bash
cd /opt/jiayi && tar xzf /tmp/jiayi.tar.gz
cd service && ./.venv/bin/pip install -r requirements.txt   # if dependencies changed
sudo systemctl restart jiayi
```

### Optional: nginx + HTTPS

Only needed for a domain and TLS. Move uvicorn back to 8000 (edit `--port` and remove the
capability lines in `deploy/jiayi.service`), then let nginx own 80/443:

```nginx
server {
    listen 80;
    server_name example.com;
    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

```bash
sudo apt install -y certbot python3-certbot-nginx
sudo certbot --nginx -d example.com
```

---

## Layout

```
service/
├── app/
│   ├── main.py             FastAPI entry point, routes (/api/assess, /api/chat, /api/destination, /api/plan), static mounting
│   ├── schemas.py          FormInput / MLFeatures / AssessmentResult / Chat* / Destination* / Plan*, FEATS, localised strings, source merging
│   ├── ml_model.py         coefficient loading, feature encoding, scoring, contribution breakdown
│   ├── planner.py          adaptive questioning: score bounds, provable stop rule, next-question ranking
│   ├── pipeline.py         orchestration: encode → rules → score → advise → verify → assemble; chat (both paths)
│   ├── destination.py      pre-travel lookup: WHO layer + web search + travel advice, cached per place × language
│   ├── prompt_builder.py   the LLM prompts (features, advice, chat, search) + the tool schema
│   ├── deepseek_client.py  two-protocol client: OpenAI JSON / tool-calling loop + Anthropic web search; shared fallback text
│   ├── verifier.py         rule engine over generated text: dosage / probability / urgency / language / structure / URLs
│   ├── intel.py            lookup_dengue_context: endemicity table + live WHO outbreak news, 12h cache
│   ├── eval_log.py         de-identified evaluation logging (assessments + search spend)
│   ├── config.py           .env settings
│   ├── data/
│   │   └── dengue_endemicity.json  81 countries/territories + alias map (WHO + CDC, 2026)
│   └── model/
│       └── dengue_models.json    fitted coefficients (mirror of ../model/results/)
├── static/                 hand-written frontend, zero external dependencies
├── tests/                  386 pytest tests (+1 live search test, skipped by default)
├── eval/scenarios.json     26 declarative regression scenarios
├── scripts/eval_run.py     scenario runner (regression gate + failure library)
├── scripts/eval_stats.py   evaluation log statistics, including search spend
├── deploy/                 systemd unit + manual launch script
├── .env.example
└── requirements.txt
```

---

## Disclaimer

Output is algorithmically generated and is **for reference only — not a medical diagnosis or
treatment recommendation**. It cannot replace assessment by a qualified clinician. Anyone who is
unwell, or whose symptoms worsen, should seek medical care promptly.
