"""Health check providers — independent connectivity probes for each resource.

Extracted from resource_scheduler.py to separate health monitoring
from scheduling logic.
"""

import os
import subprocess
import time

import httpx

from mineru_core.settings import PROVIDERS
from mineru_agent.resource_config import ProviderHealth, ProviderStatus, ResourceStatus


class HealthCheck:
    TIMEOUT = 5

    @staticmethod
    def check_all() -> ResourceStatus:
        # Public API
        status = ResourceStatus()
        checks = [
            ("deepseek", HealthCheck._check_deepseek),
            ("mimo_vision", HealthCheck._check_mimo_vision),
            ("mimo_text", HealthCheck._check_mimo_text),
            ("minimax", HealthCheck._check_minimax),
            ("local_chat", HealthCheck._check_local_chat),
            ("local_compat", HealthCheck._check_local_compat),
            ("mineru", HealthCheck._check_mineru),
            ("gpu", HealthCheck._check_gpu),
        ]
        for name, fn in checks:
            h = fn()
            status.providers[name] = h
        status.available_providers = [n for n, h in status.providers.items()
                                       if h.status == ProviderStatus.AVAILABLE]
        status.degraded_providers = [n for n, h in status.providers.items()
                                      if h.status == ProviderStatus.DEGRADED]
        status.unavailable_providers = [n for n, h in status.providers.items()
                                         if h.status == ProviderStatus.UNAVAILABLE]
        status.checked_at = time.strftime("%Y-%m-%dT%H:%M:%S")
        return status

    @staticmethod
    def _check_deepseek() -> ProviderHealth:
        # Internal (dispatched via check_all)
        api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("DEEPSEEK_API_KEY", "")
        if not api_key:
            return ProviderHealth("deepseek", ProviderStatus.UNAVAILABLE, detail="API Key 未设置")
        t0 = time.time()
        try:
            resp = httpx.get(f"{PROVIDERS['deepseek'].base_url}/v1/models",
                             headers={"Authorization": f"Bearer {api_key}"},
                             timeout=HealthCheck.TIMEOUT)
            latency = int((time.time() - t0) * 1000)
            if resp.status_code == 200:
                return ProviderHealth("deepseek", ProviderStatus.AVAILABLE, latency, "连通正常")
            if resp.status_code == 401:
                return ProviderHealth("deepseek", ProviderStatus.UNAVAILABLE, latency, "API Key 无效 (401)")
            return ProviderHealth("deepseek", ProviderStatus.DEGRADED, latency,
                                  f"HTTP {resp.status_code}")
        except Exception as e:
            latency = int((time.time() - t0) * 1000)
            return ProviderHealth("deepseek", ProviderStatus.UNAVAILABLE, latency, str(e)[:100])

    @staticmethod
    def _check_mimo_vision() -> ProviderHealth:
        # Internal (dispatched via check_all)
        api_key = os.environ.get("MIMO_API_KEY", "")
        if not api_key:
            return ProviderHealth("mimo_vision", ProviderStatus.UNAVAILABLE, detail="MIMO_API_KEY 未设置")
        t0 = time.time()
        try:
            payload = {
                "model": "mimo-v2.5",
                "messages": [{"role": "user", "content": "ping"}],
                "max_tokens": 1,
                "thinking": {"type": "disabled"},
            }
            headers = {"api-key": api_key, "Content-Type": "application/json"}
            resp = httpx.post(f"{PROVIDERS['mimo'].base_url}/chat/completions",
                              json=payload, headers=headers, timeout=HealthCheck.TIMEOUT)
            latency = int((time.time() - t0) * 1000)
            if resp.status_code in (200, 400):
                return ProviderHealth("mimo_vision", ProviderStatus.AVAILABLE, latency, "连通正常")
            return ProviderHealth("mimo_vision", ProviderStatus.DEGRADED, latency,
                                  f"HTTP {resp.status_code}")
        except Exception as e:
            latency = int((time.time() - t0) * 1000)
            return ProviderHealth("mimo_vision", ProviderStatus.UNAVAILABLE, latency, str(e)[:100])

    @staticmethod
    def _check_mimo_text() -> ProviderHealth:
        # Internal (dispatched via check_all)
        api_key = os.environ.get("MIMO_API_KEY", "")
        if not api_key:
            return ProviderHealth("mimo_text", ProviderStatus.UNAVAILABLE, detail="MIMO_API_KEY 未设置")
        t0 = time.time()
        try:
            payload = {
                "model": "mimo-v2-flash",
                "messages": [{"role": "user", "content": "ping"}],
                "max_tokens": 1,
            }
            headers = {"api-key": api_key, "Content-Type": "application/json"}
            resp = httpx.post(f"{PROVIDERS['mimo'].base_url}/chat/completions",
                              json=payload, headers=headers, timeout=HealthCheck.TIMEOUT)
            latency = int((time.time() - t0) * 1000)
            if resp.status_code in (200, 400):
                return ProviderHealth("mimo_text", ProviderStatus.AVAILABLE, latency, "文本 API 连通正常")
            return ProviderHealth("mimo_text", ProviderStatus.DEGRADED, latency,
                                  f"HTTP {resp.status_code}")
        except Exception as e:
            latency = int((time.time() - t0) * 1000)
            return ProviderHealth("mimo_text", ProviderStatus.UNAVAILABLE, latency, str(e)[:100])

    @staticmethod
    def _check_minimax() -> ProviderHealth:
        # Internal (dispatched via check_all)
        api_key = os.environ.get("MINIMAX_API_KEY", "")
        if not api_key:
            return ProviderHealth("minimax", ProviderStatus.UNAVAILABLE, detail="MINIMAX_API_KEY 未设置")
        t0 = time.time()
        try:
            resp = httpx.get(f"{PROVIDERS['minimax'].base_url}/v1/messages",
                             headers={"x-api-key": api_key},
                             timeout=HealthCheck.TIMEOUT)
            latency = int((time.time() - t0) * 1000)
            if resp.status_code in (200, 400, 405):
                return ProviderHealth("minimax", ProviderStatus.AVAILABLE, latency, "连通正常")
            return ProviderHealth("minimax", ProviderStatus.DEGRADED, latency,
                                  f"HTTP {resp.status_code}")
        except Exception as e:
            latency = int((time.time() - t0) * 1000)
            return ProviderHealth("minimax", ProviderStatus.UNAVAILABLE, latency, str(e)[:100])

    @staticmethod
    def _check_local_chat() -> ProviderHealth:
        """Check llama.cpp chat server health via /health endpoint."""
        port = os.environ.get("LLAMA_CHAT_PORT", "8080")
        base = f"http://localhost:{port}"
        t0 = time.time()
        try:
            resp = httpx.get(f"{base}/health", timeout=HealthCheck.TIMEOUT)
            latency = int((time.time() - t0) * 1000)
            if resp.status_code == 200:
                return ProviderHealth("local_chat", ProviderStatus.AVAILABLE, latency,
                                      "llama.cpp server healthy")
            return ProviderHealth("local_chat", ProviderStatus.DEGRADED, latency,
                                  f"HTTP {resp.status_code}")
        except Exception as e:
            latency = int((time.time() - t0) * 1000)
            return ProviderHealth("local_chat", ProviderStatus.UNAVAILABLE, latency, str(e)[:100])

    @staticmethod
    def _check_local_compat() -> ProviderHealth:
        """Check llama.cpp OpenAI-compatible endpoint."""
        port = os.environ.get("LLAMA_CHAT_PORT", "8080")
        base = f"http://localhost:{port}"
        t0 = time.time()
        try:
            resp = httpx.get(f"{base}/v1/models", timeout=HealthCheck.TIMEOUT)
            latency = int((time.time() - t0) * 1000)
            if resp.status_code == 200:
                return ProviderHealth("local_compat", ProviderStatus.AVAILABLE, latency,
                                      "OpenAI-compatible endpoint available")
            return ProviderHealth("local_compat", ProviderStatus.DEGRADED, latency,
                                  f"HTTP {resp.status_code}")
        except Exception as e:
            latency = int((time.time() - t0) * 1000)
            return ProviderHealth("local_compat", ProviderStatus.UNAVAILABLE, latency, str(e)[:100])

    @staticmethod
    def _check_mineru() -> ProviderHealth:
        # Internal (dispatched via check_all)
        base = os.environ.get("MINERU_API_BASE", "")
        if not base:
            return ProviderHealth("mineru", ProviderStatus.UNAVAILABLE, detail="MINERU_API_BASE 未设置")
        t0 = time.time()
        try:
            resp = httpx.get(f"{base}/health", timeout=HealthCheck.TIMEOUT)
            latency = int((time.time() - t0) * 1000)
            if resp.status_code == 200:
                return ProviderHealth("mineru", ProviderStatus.AVAILABLE, latency, "服务健康")
            return ProviderHealth("mineru", ProviderStatus.DEGRADED, latency, f"HTTP {resp.status_code}")
        except Exception as e:
            latency = int((time.time() - t0) * 1000)
            return ProviderHealth("mineru", ProviderStatus.UNAVAILABLE, latency, str(e)[:100])

    @staticmethod
    def _check_gpu() -> ProviderHealth:
        # Internal (dispatched via check_all)
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode == 0:
                gpu_count = len([l for l in result.stdout.strip().split("\n") if l.strip()])
                return ProviderHealth("gpu", ProviderStatus.AVAILABLE, 0,
                                      f"{gpu_count} 个 GPU 可用")
            return ProviderHealth("gpu", ProviderStatus.UNAVAILABLE, detail="nvidia-smi 返回非零")
        except FileNotFoundError:
            return ProviderHealth("gpu", ProviderStatus.UNAVAILABLE, detail="nvidia-smi 未找到")
        except subprocess.TimeoutExpired:
            return ProviderHealth("gpu", ProviderStatus.UNAVAILABLE, detail="nvidia-smi 查询超时（GPU 驱动可能无响应）")
        except Exception as e:
            return ProviderHealth("gpu", ProviderStatus.UNAVAILABLE, detail=str(e)[:100])
