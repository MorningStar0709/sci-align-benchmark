"""人评采集服务：知情同意 → 人口学 → 校准说明 → 逐题评分 → 完成。

三条不可动摇的实现约束：
1. 渲染用的 pool.json 是**只含展示字段的视图**，agent 分根本不在采集服务读的文件里，
   模板写错也拿不到（分析视图在 pool/analysis.json，不进镜像）。
2. 每勾一题立即落盘（upsert），"完成"按钮只置状态位；绝不依赖最后一次提交写库。
3. 库里记的是"这份数据采自哪份题池 / 哪批 code"，分析侧启动即断言，不接受静默换池。
"""

import json
import os
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, Form, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from server import db, pool_id, wording
from server.anchors import DIM_NAMES, LABELS, SCORE_LEVELS
from server.render import to_html

SERVER_DIR = Path(__file__).resolve().parent
HE_DIR = SERVER_DIR.parent
# 题池与 code 都是运行时注入的：不烘进镜像、不写死路径，便于测试与部署各自指向
# 只读挂载点（env 可改，见 README 部署节）。
POOL_PATH = Path(os.environ.get("HUMANEVAL_POOL") or HE_DIR / "pool" / "pool.json")
RATERS_PATH = Path(os.environ.get("HUMANEVAL_RATERS") or HE_DIR / "pool" / "raters.json")
DEFAULT_DB = HE_DIR / "data" / "humaneval.sqlite3"

DEGREES = {"undergrad": "本科在读", "master": "硕士", "phd": "博士", "postdoc": "博士后",
           "engineer": "工程师", "other": "其他"}
FIELDS = {"control": "控制科学 / 自动化", "ee": "电气 / 电子", "cs": "计算机",
          "mech": "机械", "math": "数学", "otherSTEM": "其他理工", "nonSTEM": "非理工"}
PROJECTS = {"yes": "独立做过控制系统设计类课题", "partial": "参与过", "no": "没做过"}
_LEVEL_STRINGS = {str(v) for v in SCORE_LEVELS} | {f"{v:g}" for v in SCORE_LEVELS}


def _load_pool_items(pool_path: Path):
    if not pool_path.is_file():
        # is_file 而不是 exists：挂载一个不存在的文件时 Docker 会在宿主机上
        # 建一个空目录，exists() 为 True 但根本读不了。
        raise SystemExit(
            f"题池文件不存在或不是普通文件：{pool_path}。先跑 pipeline/build_pool.py，"
            "或在部署时用 HUMANEVAL_POOL 指向只读挂载的 pool.json")
    doc = json.loads(pool_path.read_text(encoding="utf-8"))
    hit = _DISPLAY_FORBIDDEN & {k for it in doc["items"] for k in it}
    if hit:
        # 接住“挂载错了文件 / 手改 pool.json 合并了分析视图”：盲评是唯一
        # 事后无法补救的失败模式，宁可起不来也不能把 agent 分送到页面上。
        raise SystemExit(f"{pool_path} 含禁止下发给采集服务的字段：{sorted(hit)}；"
                         "渲染视图不得含 agent 分与溯源字段，拒绝启动")
    return doc["items"], {i["item_id"]: i for i in doc["items"]}


def _load_codes(raters_path: Path):
    if not raters_path.is_file():
        raise SystemExit(
            f"code 清单不存在或不是普通文件：{raters_path}。它故意不烘进镜像（是访问凭证），"
            "请跑 pipeline/make_raters.py 生成，并用 HUMANEVAL_RATERS 指向"
            "一个只读挂载的文件（路径写错时 Docker 会建出空目录，就是现在这个报错）")
    doc = json.loads(raters_path.read_text(encoding="utf-8"))
    codes = {i["code"] for i in doc["items"]}
    if not codes:
        raise SystemExit(f"{raters_path} 里一个 code 也没有：没有可发链接，起来也是空服务")
    return codes


# 模板能看到的字段白名单。agent 分不在其中，也不在 pool.json 里——两道独立的门，
# 任何一道被拆掉 check_blinding --self-test 都会报（它注入 agent_score 并篡改白名单，
# 确认渲染层真的会被拦住）。
_RENDER_KEYS = ("item_id", "dimension")
_RENDER_HTML = {"question": "question_html", "expected_answer": "expected_html",
                "model_answer": "answer_html"}

# 与 pipeline/check_blinding.py 的同名集合故意各写一份：审计侧与被审计侧共用
# 一个常量的话，一次改错就能同时拆掉两道门。
_DISPLAY_FORBIDDEN = frozenset({
    "agent_score", "score", "reason", "issues", "origin_id", "difficulty_level",
    "judge1_score", "judge2_score", "avg_score", "model", "judge_model",
    "model_source", "source_file", "source_ref",
})


def _for_renderer(item: dict) -> dict:
    out = {k: item[k] for k in _RENDER_KEYS}
    out.update({dst: to_html(item[src]) for src, dst in _RENDER_HTML.items()})
    return out


def _bind_db(conn, pool_sha: str, code_set_sha: str) -> None:
    """把题池身份写进库；身份对不上时当场拒启，而不是等采集完再被分析侧拒报。

    换 seed 重抽题后服务照常起来、链接照常可填，采到第 10 人才在 `analyze.py`
    撞墙——那 10 个人的半小时就白花了。三种对不上分开处理：

    - 库已绑另一份题池**且已有评分**：拒。两份题池的评分合不到一起。
    - 库有评分但**没有** `pool_sha256`（这份库早于身份绑定上线）：也拒。现在补记
      等于替旧数据伪造出处，会把分析侧那道断言错报成“已绑定”——比不一致更隐蔽。
    - 库里一条评分都没有：没有可被污染的历史，直接改绑。否则一个测试残留的空壳
      就能把生产服务锁死。

    逃生门 `HUMANEVAL_ALLOW_POOL_MISMATCH=1`：知情继续时不报错，但已记录的身份
    一律不覆写——覆写就是把混池的库洗成单池。这个后果由开逃生门的人自己认。
    """
    stored = db.get_meta(conn, "pool_sha256")
    n_rating = db.count_ratings(conn)
    # 逃生门要显式开：`HUMANEVAL_ALLOW_POOL_MISMATCH=0` 按直觉就是“不开”，
    # 而 `bool(非空串)` 会把它读成开，所以 0/false/no 一律归到关。
    allowed = (os.environ.get("HUMANEVAL_ALLOW_POOL_MISMATCH") or "").strip().lower() \
        not in ("", "0", "false", "no")
    unbound = stored is None and n_rating > 0             # 有评分、无出处
    stale = stored not in (None, pool_sha) and n_rating > 0  # 有评分、但绑的是另一份题池

    if unbound and not allowed:
        raise SystemExit(
            f"库里有 {n_rating} 条评分却没记过 pool_sha256：这份库早于题池身份绑定上线，"
            "现在替它补记等于伪造出处（分析侧会错报为“已绑定”）。要么把 HUMANEVAL_DB "
            "指向新库重采，要么明确知情混采：HUMANEVAL_ALLOW_POOL_MISMATCH=1")
    if stale and not allowed:
        raise SystemExit(
            f"库记录的题池 {stored[:12]}… 与当前题池 {pool_sha[:12]}… 不是同一份，"
            f"而已收了 {n_rating} 条评分。继续采只会得到无法合并分析的数据：重抽题后请把 "
            "HUMANEVAL_DB 指向新库（已收数据作废但可审计），或知情继续时加 "
            "HUMANEVAL_ALLOW_POOL_MISMATCH=1")
    if unbound:
        print(f"⚠ 已按逃生门放行：库里 {n_rating} 条评分无题池出处，不补写 pool_sha256，"
              "分析侧仍会拒报（这是对的）")
    elif stale:
        print(f"⚠ 已放行跨题池继续采集：旧 {n_rating} 条评分绑 {stored[:12]}…，"
              f"与新题池 {pool_sha[:12]}… 的评分不能合并；分析侧必须 --ignore-pool-binding "
              "才出得了报告")
    elif stored is not None and stored != pool_sha:
        print(f"ℹ 空库改绑：pool_sha256 {stored[:12]}… → {pool_sha[:12]}…"
              "（库里 0 条评分，没有可被污染的历史）")

    # 两种情况下有意不写回身份：覆写会把无出处数据洗成有出处、把混池洗成单池。
    if not (unbound or stale):
        db.set_meta(conn, "pool_sha256", pool_sha)

    stored_codes = db.get_meta(conn, "code_set_sha256")
    if stored_codes is not None and stored_codes != code_set_sha:
        print(f"⚠ code 清单变了（{stored_codes[:12]}… → {code_set_sha[:12]}…）："
              "如果这是追加，旧链接仍有效；如果是换码，旧链接会 404")
    db.ensure_meta(conn, "code_set_sha256", code_set_sha)


def create_app(pool_path: Path = None, raters_path: Path = None) -> FastAPI:
    pool_path = Path(pool_path or POOL_PATH)
    raters_path = Path(raters_path or RATERS_PATH)
    app = FastAPI(title="humaneval", docs_url=None, redoc_url=None)
    app.mount("/static", StaticFiles(directory=str(SERVER_DIR / "static")), name="static")
    templates = Jinja2Templates(directory=str(SERVER_DIR / "templates"))

    items, item_index = _load_pool_items(pool_path)
    codes = _load_codes(raters_path)
    # 身份 = 题池语义内容的摘要，不是文件字节 sha（见 server/pool_id.py）。
    pool_sha = pool_id.display_sha(items)
    code_set_sha = pool_id.code_set_sha(codes)
    conn = db.connect(os.environ.get("HUMANEVAL_DB") or str(DEFAULT_DB))
    db.init_db(conn)
    _bind_db(conn, pool_sha, code_set_sha)

    def page(request, name, **context):
        context.setdefault("w", wording)
        context.setdefault("n_items", len(items))
        return templates.TemplateResponse(request=request, name=name, context=context)

    def require_known_code(code: str) -> None:
        if code not in codes:
            raise HTTPException(status_code=404)

    def rater_row(code: str):
        return db.get_rater(conn, code)

    def has_consent(code: str) -> bool:
        """知情同意的唯一可验证痕迹：过了 demographics 才会有 expertise。"""
        row = rater_row(code)
        return bool(row and row.get("expertise"))

    @app.get("/", response_class=HTMLResponse)
    def root(request: Request):
        """根路径落地页：给误入者一句说明（少一个被误判成"服务挂了"的 404）。"""
        return page(request, "landing.html")

    @app.get("/healthz")
    def healthz():
        """不依赖凭证的存活性探测：容器 healthcheck 与部署验收共用。"""
        return JSONResponse({"ok": True, "pool_sha256": pool_sha,
                             "n_items": len(items), "n_codes": len(codes)})

    @app.get("/r/{code}", response_class=HTMLResponse)
    def intake(request: Request, code: str):
        require_known_code(code)
        return page(request, "intake.html", code=code, degrees=DEGREES, fields=FIELDS,
                    projects=PROJECTS, already=rater_row(code) is not None)

    @app.post("/demographics")
    def save_demographics(code: Annotated[str, Form()],
                          degree: Annotated[str, Form()],
                          field: Annotated[str, Form()],
                          own_project: Annotated[str, Form()],
                          consented: Annotated[str, Form()]):
        require_known_code(code)
        if consented != "yes":
            raise HTTPException(status_code=422, detail="必须先同意知情说明")
        if degree not in DEGREES or field not in FIELDS or own_project not in PROJECTS:
            raise HTTPException(status_code=422, detail="人口学选项不合法")
        db.save_demographics(conn, code=code, degree=degree, field=field,
                             own_project=own_project)
        return RedirectResponse(url=f"/calibrate/{code}", status_code=303)

    @app.get("/calibrate/{code}", response_class=HTMLResponse)
    def calibrate(request: Request, code: str):
        require_known_code(code)
        if rater_row(code) is None:
            return RedirectResponse(url=f"/r/{code}", status_code=303)
        return page(request, "calibrate.html", code=code, labels=LABELS,
                    dim_names=DIM_NAMES, items_total=len(items))

    @app.get("/rate/{code}", response_class=HTMLResponse)
    def rate(request: Request, code: str):
        require_known_code(code)
        if rater_row(code) is None:
            return RedirectResponse(url=f"/r/{code}", status_code=303)
        rendered = [_for_renderer(it) for it in items]
        return page(request, "rate.html", code=code, items=rendered, labels=LABELS,
                    dim_names=DIM_NAMES, total=len(items))

    @app.post("/api/rating")
    def save_rating(code: Annotated[str, Form()], item_id: Annotated[str, Form()],
                    dimension: Annotated[str, Form()], score: Annotated[str, Form()],
                    duration_sec: Annotated[int, Form(ge=0, le=86400)],
                    client_seq: Annotated[int, Form(ge=0)] = 0,
                    truncated: Annotated[int, Form(ge=0, le=1)] = 0):
        """逐题即时落盘。truncated 由前端在"刷新续答、该题起始时刻未知"时置 1。

        duration_sec 上限 24h：分次作答允许一题横跨长暂停，上限只拦明显坏值
        （脚本乱写 / 时钟错乱），可信度由前端 truncated 标记与 analyze 的排除口径处理。
        """
        require_known_code(code)
        if not has_consent(code):
            raise HTTPException(status_code=403, detail="未完成知情同意")
        item = item_index.get(item_id)
        if item is None or item["dimension"] != dimension:
            raise HTTPException(status_code=422, detail="未知题目或维度不符")
        if score not in _LEVEL_STRINGS:
            raise HTTPException(status_code=422, detail="评分必须是五档之一")
        db.upsert_rating(conn, code=code, item_id=item_id, dimension=dimension,
                         score=float(score), duration_sec=duration_sec,
                         duration_truncated=truncated,
                         is_attention_check=bool(item.get("is_attention_check")),
                         client_seq=client_seq)
        return {"ok": True}

    @app.get("/api/rating")
    def read_ratings(code: Annotated[str, Query(min_length=1)]):
        require_known_code(code)
        return {"items": db.list_ratings(conn, code=code)}

    @app.post("/api/finish")
    def finish(code: Annotated[str, Form()]):
        require_known_code(code)
        if not has_consent(code):
            raise HTTPException(status_code=403, detail="未完成知情同意")
        db.mark_finished(conn, code)
        return {"ok": True}

    @app.get("/done/{code}", response_class=HTMLResponse)
    def done(request: Request, code: str):
        require_known_code(code)
        return page(request, "done.html", code=code,
                    saved=len(db.list_ratings(conn, code=code)))

    return app


# 不在此处建实例：`from server.app import create_app` 本不该打开生产库。
# 部署侧用 `uvicorn --factory server.app:create_app`（Dockerfile CMD 已同步）。

