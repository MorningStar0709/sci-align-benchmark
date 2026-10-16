# humaneval — 人评采集服务

从自有境外 VPS 的 Docker 镜像恢复的完整源码。
> **脱敏（2026-10-03）**：本文件原版含主机别名、公网 IP 与 SSH 端口。三者合起来足以把这份代码关联到具体的人与机器，而它有可能随 artifact 仓对外发布 ⇒ 一律换成占位符，原值只留在本地部署笔记里。**git 历史里那份原文仍在**（本仓无 remote、无公开克隆，全量 bundle 在外盘），所以这不是「已经消失」，只是「现行面不再传播」。是否改写历史由作者决定。
迁移系统时丢失，2026-09-27 找回。

## 服务概述

FastAPI 人评采集服务，流程：知情同意 → 人口学 → 校准说明 → 逐题评分 → 完成。

**三条不可动摇的实现约束**（`server/app.py:1-8`）：

1. **盲评隔离**：渲染用的 `pool.json` 是只含展示字段的视图，agent 分不在采集服务读的文件里，模板写错也拿不到。分析视图在 `pool/analysis.json`，不进镜像。
2. **即时落盘**：每勾一题立即 upsert，"完成"按钮只置状态位；绝不依赖最后一次提交写库。
3. **题池锁定**：库里记的是"这份数据采自哪份题池 / 哪批 code"，分析侧启动即断言，不接受静默换池。

## 文件结构

```
humaneval/
├── server/                 # FastAPI 服务（727 行）
│   ├── app.py             # 主应用：路由、表单处理、盲评校验
│   ├── db.py              # SQLite 落盘：每题 upsert，完成置状态位
│   ├── anchors.py         # 评分维度与标签定义
│   ├── render.py          # 题目渲染（KaTeX 数学公式）
│   ├── pool_id.py         # 题池身份标识（SHA256）
│   ├── wording.py         # 界面文案
│   ├── templates/         # 6 页 HTML：landing/intake/calibrate/rate/done/base
│   └── static/            # CSS + KaTeX 字体与渲染器
├── pool/
│   ├── pool.json          # 题池（展示视图，31KB）
│   └── pool.json.sha256   # 题池校验
├── Dockerfile             # 安全加固：非 root 运行、read_only、cap_drop、no-new-privileges
├── docker-compose.yml     # 部署配置（含卷挂载与逃生门说明）
└── requirements-server.txt
```

## 部署

### 前置要求

- Docker + Docker Compose
- 评分者凭证文件 `raters.json`（**不进 git**，见下方「敏感信息」）

### 快速启动

```bash
cd humaneval/

# 1. 准备评分者凭证（36 个 code，格式见 secrets/raters.json 备份）
cp /path/to/raters.json secrets/raters.json

# 2. 构建并启动
docker compose up -d --build

# 3. 验证
curl http://127.0.0.1:8090/healthz
```

### 关键配置

| 环境变量 | 默认 | 说明 |
|:---------|:-----|:-----|
| `HUMANEVAL_RATERS` | `/run/raters.json` | 容器内凭证路径（只读挂载） |
| `HUMANEVAL_RATERS_HOST` | `./pool/raters.json` | 宿主机凭证路径 |
| `HUMANEVAL_ALLOW_POOL_MISMATCH` | `""`（关） | 逃生门：知情跨题池继续采时设非空 |
| `HUMANEVAL_POOL` | `HE_DIR/pool/pool.json` | 题池路径（测试用） |
| `HUMANEVAL_DB` | `/data/humaneval.sqlite3` | 数据库路径 |

### 换题池的后果

题池不能随手更换：挂一份新 `pool.json` 到旧卷上，服务会**直接拒启**而不是照常采集。

要换新库重采：
1. 换 docker compose project 名，或
2. `docker volume rm humaneval_humaneval_data`

确实要知情跨题池继续采，才用 `HUMANEVAL_ALLOW_POOL_MISMATCH` 逃生门。

## 敏感信息（不进 git）

| 文件 | 内容 | 恢复来源 |
|:-----|:-----|:---------|
| `secrets/raters.json` | 36 个评分者访问 code | <VPS> `/opt/humaneval/secrets/raters.json` |
| `.env` | 服务器路径配置 | <VPS> `/opt/humaneval/.env` |

本地开发时：
- `raters.json` 放 `secrets/` 或任意路径，通过 `HUMANEVAL_RATERS_HOST` 指向
- `.env` 按需修改，不要提交真实服务器路径

## 安全设计

- **盲评双门**：`_RENDER_KEYS` 白名单 + `_DISPLAY_FORBIDDEN` 黑名单，审计侧与被审计侧各写一份，防止一次改错拆两道门
- **凭证不烘镜像**：`raters.json` 运行时只读挂载，镜像推 registry 不泄露访问凭证
- **无 access log**：`--no-access-log`，访问链接里的 code 不写日志流
- **单 worker**：`db._WRITE_LOCK` 是进程内的，多 worker 退化成 SQLite 文件锁硬扛
- **非 root 运行**：uid 10001，`read_only` 根文件系统，`cap_drop: ALL`，`no-new-privileges`

## 什么随包、什么不随包

| 件 | 状态 |
|:--|:--|
| `server/`、`pipeline/`、`pool/pool.json`、`Dockerfile`、`docker-compose.yml` | 随包 |
| `secrets/raters.json`（评分者访问 code） | **不随包**：那是凭证。要自己跑，先 `pipeline/make_raters.py` 生成一份 |
| `.env`、`data/*.sqlite3` | 不随包：数据库在首次启动时自己建 |
| `analysis.json`（自动判官给的分数） | 不随包，且**刻意不烘进镜像**：评分者看到它就等于破了盲 |

起服务（无需 Docker，装 `requirements-server.txt` 即可）：

```bash
cd humaneval
HUMANEVAL_RATERS=secrets/raters.json HUMANEVAL_DB=/tmp/humaneval.sqlite3 \
  uvicorn --factory server.app:create_app --host 127.0.0.1 --port 8090 --no-access-log --workers 1
```

题池身份与 code 条数见 `GET /healthz`（不需要凭证）。

## 相关文档

- 题池构建：`pipeline/build_pool.py`
- 盲评审计：`pipeline/check_blinding.py`
- 评分者生成：`pipeline/make_raters.py`
