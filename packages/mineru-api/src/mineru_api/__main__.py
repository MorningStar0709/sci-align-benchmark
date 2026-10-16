"""MinerU API — CLI entry point for ``python -m mineru_api``.

Migrated from ``controlsci.api.medical_rag.__main__``.
"""

from __future__ import annotations

import argparse
import json
import sys

from mineru_core.settings import get_settings
from mineru_api.app import app


def main():
    parser = argparse.ArgumentParser(description="MinerU API Server")
    settings = get_settings()
    parser.add_argument("--port", type=int, default=settings.api_port, help=f"监听端口 (默认 {settings.api_port})")
    parser.add_argument("--host", default=settings.api_host, help=f"绑定地址 (默认 {settings.api_host})")
    parser.add_argument("--check", action="store_true", help="仅运行健康检查后退出")
    args = parser.parse_args()

    if args.check:
        from mineru_api.controllers.medical_rag import health
        result = health()
        print(json.dumps(result, ensure_ascii=False, indent=2))
        sys.exit(0 if result["status"] in {"ok", "public_evidence"} else 1)
    else:
        import uvicorn
        uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
