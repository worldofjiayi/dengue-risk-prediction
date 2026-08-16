"""应用配置：从环境变量 / .env 文件读取。"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


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
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    """全局单例配置（lru_cache 缓存；测试中可用 get_settings.cache_clear() 重置）。"""
    return Settings()
