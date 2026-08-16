"""FastAPI 入口：API 路由 + 静态页面托管。"""

import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles

from app.config import get_settings
from app.deepseek_client import DeepSeekError
from app.pipeline import run_assessment, run_chat
from app.schemas import (
    SERVER_ERRORS,
    UPSTREAM_ERRORS,
    AssessmentResult,
    ChatRequest,
    ChatResponse,
    FormInput,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="登革热风险评估服务",
    description=(
        "基于问卷 + DeepSeek + 登革热逻辑回归模型（巴西 SINAN 2023–2025）"
        "的风险自评后端。评分为相对风险参考值，非感染概率。"
    ),
)


# 注意：API 路由必须先注册，静态目录最后挂载到 "/"，否则会遮住 API。


@app.post("/api/assess", response_model=AssessmentResult)
async def assess(form: FormInput) -> AssessmentResult:
    """接收问卷，返回风险评估结果。"""
    try:
        return await run_assessment(form)
    except DeepSeekError as exc:
        logger.error("上游 DeepSeek 服务错误：%s", exc)
        raise HTTPException(
            status_code=502,
            detail=UPSTREAM_ERRORS.get(form.language, UPSTREAM_ERRORS["zh-CN"]),
        ) from exc
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("评估流程发生未知错误")
        raise HTTPException(
            status_code=500,
            detail=SERVER_ERRORS.get(form.language, SERVER_ERRORS["zh-CN"]),
        ) from exc


@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    """就用户自己的评估结果做追问。无状态：上下文与历史由前端回传。

    错误提示按请求语言本地化——聊天窗口里冒出一句中文报错，对西语用户
    比没有回复更让人困惑。
    """
    try:
        return await run_chat(req)
    except DeepSeekError as exc:
        logger.error("追问对话上游错误：%s", exc)
        raise HTTPException(
            status_code=502, detail=UPSTREAM_ERRORS.get(req.language, UPSTREAM_ERRORS["zh-CN"])
        ) from exc
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("追问对话发生未知错误")
        raise HTTPException(
            status_code=500, detail=SERVER_ERRORS.get(req.language, SERVER_ERRORS["zh-CN"])
        ) from exc


@app.get("/api/health")
async def health() -> dict:
    """健康检查：同时确认三个模型已加载。"""
    from app.ml_model import get_model

    return {
        "status": "ok",
        "mock_mode": get_settings().mock_mode,
        "models": list(get_model().info()),
    }


# ---- 静态页面：static/ 目录由前端 agent 维护 ----
# check_dir=False：目录暂不存在时不在 import 阶段崩溃（目录可在启动后再创建）
_static_dir = Path(__file__).resolve().parent.parent / "static"
try:
    if not _static_dir.is_dir():
        logger.warning("静态目录暂不存在：%s，当前仅提供 API 接口", _static_dir)
    app.mount(
        "/",
        StaticFiles(directory=str(_static_dir), html=True, check_dir=False),
        name="static",
    )
except Exception:
    logger.exception("挂载静态目录失败（%s），仅提供 API 接口", _static_dir)


if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run("app.main:app", host=settings.host, port=settings.port)
