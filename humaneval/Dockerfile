# 默认走官方源；本机 WSL 内 registry-1.docker.io 直连被阻断（实测 0/5），
# 构建时用镜像站全名覆盖：
#   docker build --build-arg BASE_IMAGE=docker.m.daocloud.io/library/python:3.11-slim .
ARG BASE_IMAGE=python:3.11-slim
FROM ${BASE_IMAGE}

WORKDIR /app

# 只装采集服务要用的依赖。numpy/scipy/krippendorff 是分析侧的，进镜像只会把
# 一个"发链接给人填表"的服务撑大 200 MB 并多出一批可被 CVE 扫描到的面。
COPY requirements-server.txt ./
RUN pip install --no-cache-dir -r requirements-server.txt

COPY server/ ./server/
# 只拷渲染视图。analysis.json（agent 分）、raters.json（访问凭证）、manifest.json
# （溯源）都不进镜像：镜像一旦推到 registry，盲评与凭证同时失效。
# code 由运行时只读挂载注入，见 docker-compose.yml 的 HUMANEVAL_RATERS。
COPY pool/pool.json ./pool/pool.json

RUN useradd --create-home --uid 10001 humaneval \
 && mkdir -p /data && chown -R humaneval:humaneval /data /app

USER humaneval

ENV HUMANEVAL_DB=/data/humaneval.sqlite3 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

EXPOSE 8090

# /healthz 不需要凭证，只回题池身份；read_only 根文件系统下 /tmp 之外的写入都会失败，
# 所以探测走 python 内置 urllib 而不是 curl。
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
  CMD ["python", "-c", "import sys,urllib.request; r = urllib.request.urlopen('http://127.0.0.1:8090/healthz', timeout=4); sys.exit(0 if r.status == 200 else 1)"]

# --no-access-log：访问链接里就带 code，开 access log 等于把访问凭证写进日志流。
# --workers 1：db._WRITE_LOCK 是进程内的，多 worker 会退化成靠 SQLite 文件锁硬扛。
CMD ["uvicorn", "--factory", "server.app:create_app", \
     "--host", "0.0.0.0", "--port", "8090", "--no-access-log", "--workers", "1"]
