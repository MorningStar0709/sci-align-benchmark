"""MinerU HTTP client used by the workbench.

Migrated from ``controlsci.api.mineru_client``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import httpx


@dataclass
class MinerUParseResult:
    status: str
    parse_ok: bool
    detail: str
    task_id: str = ""
    markdown: str = ""


class MinerUClient:
    def __init__(self, base_url: str, timeout: int = 120):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def is_available(self, timeout: float = 0.35) -> bool:
        for path in ("/health", "/docs"):
            try:
                response = httpx.get(f"{self.base_url}{path}", timeout=timeout, trust_env=False)
                if response.status_code == 200:
                    return True
            except Exception:
                continue
        return False

    def parse_pdf(self, pdf_path: Path) -> MinerUParseResult:
        content = pdf_path.read_bytes()
        try:
            with httpx.Client(timeout=self.timeout, trust_env=False) as client:
                files = {"files": (pdf_path.name, content, "application/pdf")}
                response = client.post(f"{self.base_url}/file_parse", files=files)
                if response.status_code != 200:
                    return MinerUParseResult("failed", False, f"MinerU API returned HTTP {response.status_code}")

                parse_data = response.json()
                task_id = parse_data.get("task_id", "")
                if not task_id:
                    return MinerUParseResult("failed", False, "MinerU API 未返回 task_id")

                task_status = ""
                for _ in range(60):
                    time.sleep(1)
                    task_response = client.get(f"{self.base_url}/tasks/{task_id}", timeout=5)
                    if task_response.status_code != 200:
                        continue
                    task_status = task_response.json().get("status", "")
                    if task_status in {"completed", "failed"}:
                        break
                else:
                    return MinerUParseResult("timeout", False, "解析超时(60s)，任务仍在后台运行", task_id=task_id)

                if task_status != "completed":
                    return MinerUParseResult("failed", False, f"解析失败: {task_status}", task_id=task_id)

                result_response = client.get(f"{self.base_url}/tasks/{task_id}/result", timeout=10)
                if result_response.status_code != 200:
                    return MinerUParseResult("failed", False, f"结果读取失败 HTTP {result_response.status_code}", task_id=task_id)

                result = result_response.json()
                markdown = str(result.get("result", result))
                return MinerUParseResult("ok", True, f"解析完成，{len(markdown)} 字符 markdown", task_id=task_id, markdown=markdown)
        except httpx.TimeoutException:
            return MinerUParseResult("failed", False, "调用 MinerU API 超时")
        except Exception as exc:
            return MinerUParseResult("failed", False, str(exc)[:200])
