"""应用配置：从环境变量 / .env 文件读取。

.env 固定按本文件位置解析到 service/.env（本地与远程 /opt/jiayi/service/.env 一致），
不依赖进程启动目录——无论从项目根还是 service/ 目录启动 uvicorn 行为都相同。
环境变量的优先级始终高于 .env 文件（pydantic-settings 默认行为，测试依赖这一点）。
"""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# service/ 目录（app/ 的上一级）
_SERVICE_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """服务全局配置，字段名与 .env 中的环境变量一一对应（大小写不敏感）。"""

    # DeepSeek 配置
    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-chat"
    deepseek_timeout: float = 60.0  # DeepSeek 请求超时（秒）

    # ML 模型配置：为空或文件不存在时使用内置启发式假模型
    ml_model_path: str = ""

    # 评测数据回流：每次评估完成后把脱敏记录追加到该 JSONL 文件
    # （相对路径相对项目根目录解析；置空则关闭回流）
    eval_log_path: str = "data/assessments.jsonl"

    # 演示模式：true 时 DeepSeek 与 ML 模型都返回可信假数据
    mock_mode: bool = True

    # 服务监听
    host: str = "0.0.0.0"
    port: int = 8000

    model_config = SettingsConfigDict(
        env_file=str(_SERVICE_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    """全局单例配置（lru_cache 缓存；测试中可用 get_settings.cache_clear() 重置）。"""
    return Settings()
