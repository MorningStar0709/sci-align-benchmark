"""MinerU FastAPI application — main entry point.

Migrated from ``controlsci.api.demo_endpoints`` and ``controlsci.api.medical_rag``.
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from mineru_api.controllers.health import router as health_router
from mineru_api.controllers.demo import router as demo_router
from mineru_api.controllers.medical_rag import router as medical_rag_router
from mineru_api.controllers.cloud_demo import router as cloud_demo_router

app = FastAPI(
    title="MinerU API Server",
    description="MinerU 科学文档评测与医学 RAG API",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,  # "*" + credentials=True is insecure (OWASP)
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router)
app.include_router(demo_router)
app.include_router(medical_rag_router)
app.include_router(cloud_demo_router)
