#!/usr/bin/env bash
# 登革热风险自评服务 —— 手动启动脚本（调试用）
#
# 生产环境由 systemd 直接调用 uvicorn（见 deploy/jiayi.service），不经过本脚本。
# 本脚本用于在服务器上手动前台启动、观察输出，便于排查问题：
#
#     ./deploy/run.sh              # 用 .env 里的 PORT（缺省 8000）
#     PORT=8080 ./deploy/run.sh    # 临时换端口
#
# 注意：绑定 80 等特权端口时本脚本需要 sudo；systemd 那条路径靠
# AmbientCapabilities=CAP_NET_BIND_SERVICE 实现，不需要 root。

set -euo pipefail

# 切换到脚本所在目录的上一级（即项目根目录）
cd "$(dirname "$(readlink -f "$0")")/.."

# 载入 .env 中的 PORT（忽略注释与空行）
if [ -f ".env" ]; then
    PORT_FROM_ENV="$(grep -E '^\s*PORT=' .env | tail -1 | cut -d= -f2- | tr -d '[:space:]' || true)"
fi

PORT="${PORT:-${PORT_FROM_ENV:-8000}}"

exec ./.venv/bin/uvicorn app.main:app --host 0.0.0.0 --port "$PORT" --workers 2
