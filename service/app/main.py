"""FastAPI entry point: API routes + static page hosting."""

import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles

from app.config import get_settings
from app.deepseek_client import DeepSeekError
from app.destination import run_destination
from app.pipeline import run_assessment, run_chat
from app.planner import plan
from app.schemas import (
    SERVER_ERRORS,
    UPSTREAM_ERRORS,
    AssessmentResult,
    ChatRequest,
    ChatResponse,
    DestinationRequest,
    DestinationResponse,
    FormInput,
    PlanRequest,
    PlanResponse,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Dengue risk assessment service",
    description=(
        "A risk self-assessment back end built on a questionnaire + DeepSeek + dengue "
        "logistic regression models (Brazilian SINAN 2023–2025). Scores are relative "
        "risk indicators, not infection probabilities."
    ),
)


# Static assets are served under stable URLs, and browsers apply heuristic caching
# when no Cache-Control header is present -- after a deploy, users kept seeing the
# previous app.js until their cache expired on its own (observed in production).
# no-cache does not mean "don't cache": it means revalidate before every use, so an
# unchanged file still answers 304 via its ETag and an updated one is picked up at
# once. API responses are dynamic and get the same header harmlessly.
@app.middleware("http")
async def _revalidate_everything(request, call_next):
    response = await call_next(request)
    response.headers.setdefault("Cache-Control", "no-cache")
    return response


# Note: the API routes must be registered first and the static directory mounted on "/" last,
# otherwise it would shadow the API.


@app.post("/api/assess", response_model=AssessmentResult)
async def assess(form: FormInput) -> AssessmentResult:
    """Take the questionnaire, return the risk assessment result.

    Note: **an upstream LLM failure no longer fails this endpoint**. The scores, the warning
    signs, the exposure context and the contribution breakdown are all computed locally; if
    the advice text fails, the pipeline falls back to the template and marks advice_source
    as "template" (see the pipeline module docstring). The 502 branch below is therefore
    purely defensive now -- actually reaching it means a new, uncaught failure point has
    appeared in the pipeline.
    """
    try:
        return await run_assessment(form)
    except DeepSeekError as exc:
        logger.error("Upstream DeepSeek service error (not absorbed by the assessment flow): %s", exc)
        raise HTTPException(
            status_code=502,
            detail=UPSTREAM_ERRORS.get(form.language, UPSTREAM_ERRORS["zh-CN"]),
        ) from exc
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Unknown error in the assessment flow")
        raise HTTPException(
            status_code=500,
            detail=SERVER_ERRORS.get(form.language, SERVER_ERRORS["zh-CN"]),
        ) from exc


@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    """Follow-up questions about the user's own assessment result.

    Stateless: the context and history are sent back by the front end. Error messages are
    localised to the request language -- a Chinese error popping up in the chat window is
    more confusing to a Spanish-speaking user than no reply at all.
    """
    try:
        return await run_chat(req)
    except DeepSeekError as exc:
        logger.error("Upstream error in the follow-up chat: %s", exc)
        raise HTTPException(
            status_code=502, detail=UPSTREAM_ERRORS.get(req.language, UPSTREAM_ERRORS["zh-CN"])
        ) from exc
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Unknown error in the follow-up chat")
        raise HTTPException(
            status_code=500, detail=SERVER_ERRORS.get(req.language, SERVER_ERRORS["zh-CN"])
        ) from exc


@app.post("/api/destination", response_model=DestinationResponse)
async def destination(req: DestinationRequest) -> DestinationResponse:
    """Pre-travel lookup: dengue in a place over the last three months.

    (Regional table + WHO notices + web search.)

    **This endpoint returns no score at all**: the location is pre-travel background and
    never takes part in scoring.

    An upstream search failure does not fail it -- the regional table and the WHO notices
    are local/public data and are returned as usual, with search_status merely degraded
    (see the app.destination module docstring). The 502 branch here is therefore purely
    defensive.
    """
    try:
        return await run_destination(req)
    except DeepSeekError as exc:
        logger.error("Upstream error in the destination lookup (not absorbed): %s", exc)
        raise HTTPException(
            status_code=502,
            detail=UPSTREAM_ERRORS.get(req.language, UPSTREAM_ERRORS["zh-CN"]),
        ) from exc
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Unknown error in the destination lookup")
        raise HTTPException(
            status_code=500,
            detail=SERVER_ERRORS.get(req.language, SERVER_ERRORS["zh-CN"]),
        ) from exc


@app.post("/api/plan", response_model=PlanResponse)
async def plan_questions(req: PlanRequest) -> PlanResponse:
    """Adaptive questioning plan: hard score bounds, whether we can provably stop, what to ask next.

    A fully deterministic computation (model coefficients only), with no LLM call and no
    side effects.
    """
    try:
        return plan(req)
    except Exception as exc:
        logger.exception("Unknown error in the questioning plan")
        raise HTTPException(
            status_code=500,
            detail=SERVER_ERRORS.get(req.language, SERVER_ERRORS["zh-CN"]),
        ) from exc


@app.get("/api/health")
async def health() -> dict:
    """Health check: also confirms that all three models are loaded."""
    from app.ml_model import get_model

    return {
        "status": "ok",
        "mock_mode": get_settings().mock_mode,
        "models": list(get_model().info()),
    }


# ---- Static pages: the static/ directory is maintained by the front-end agent ----
# check_dir=False: do not crash at import time when the directory does not exist yet
# (it may be created after start-up)
_static_dir = Path(__file__).resolve().parent.parent / "static"
try:
    if not _static_dir.is_dir():
        logger.warning("Static directory does not exist yet: %s, serving the API only", _static_dir)
    app.mount(
        "/",
        StaticFiles(directory=str(_static_dir), html=True, check_dir=False),
        name="static",
    )
except Exception:
    logger.exception("Failed to mount the static directory (%s), serving the API only", _static_dir)


if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run("app.main:app", host=settings.host, port=settings.port)
