"""Shared Pydantic models used across MinerU API."""

from __future__ import annotations

from pydantic import BaseModel


class StatusResponse(BaseModel):
    status: str
    message: str = ""
    detail: str | None = None
