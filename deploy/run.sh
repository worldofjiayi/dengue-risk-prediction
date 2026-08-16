#!/usr/bin/env bash
# jiayi 流行病感染风险评估服务 - 启动脚本
# 用法: ./deploy/run.sh
# 环境变量 PORT 可覆盖监听端口(默认 8000), 通常由 systemd 的 EnvironmentFile(.env) 注入

set -euo pipefail

# 切换到脚本所在目录的上一级(即项目根目录)
cd "$(dirname "$(readlink -f "$0")")/.."

# 如果存在虚拟环境 .venv 则激活
if [ -f ".venv/bin/activate" ]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
fi

exec uvicorn app.main:app --host 0.0.0.0 --port "${PORT:-8000}" --workers 2
