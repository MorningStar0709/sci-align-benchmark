"""题池 / code 集的身份摘要：只由语义内容决定，与文件字节排版无关。

为什么不用 `sha256(文件字节)`：那条绑定在任何跨机场景下都会碎掉——

- 仓库 `core.autocrlf=true`，同一份 pool.json 在 Windows 工作区是 CRLF、在 Linux
  容器里是 LF，字节 sha 直接不同；
- 重新缩进、调整键顺序、删掉尾部空行都会改字节 sha，却没有一题的内容变过。

而这条摘要要回答的问题是"**这份数据采自哪份题池**"。身份一旦与换行符绑定，它既不能
跨机复现，也不能跨提交复用。文本级篡改本来另有防线：渲染视图与分析视图的逐字段一致性
（`build_pool.view_mismatches`）与 pool.json 字节里不得出现 `agent_score`。
"""

import hashlib
import json

_CORE = ("item_id", "dimension", "question", "expected_answer", "model_answer")


def _rows(items, fields, *, sort):
    """把题项压成只含指定字段的规范行；缺失字段用 None，不让 KeyError 冒充"不同"。"""
    seq = sorted(items, key=lambda it: str(it.get("item_id"))) if sort else list(items)
    return [{f: it.get(f) for f in fields} for it in seq]


def _digest(rows) -> str:
    canon = json.dumps(rows, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()


def display_sha(items) -> str:
    """渲染视图身份。**按文件顺序**取行：呈现顺序变了，评分者经历的就是另一份量表。"""
    return _digest(_rows(items, _CORE + ("is_attention_check",), sort=False))


def analysis_sha(items) -> str:
    """分析视图身份。按 item_id 排序：呈现顺序不影响"哪道题的 agent 分是多少"。"""
    return _digest(_rows(items, _CORE + ("agent_score", "origin_id",
                                         "difficulty_level"), sort=True))


def code_set_sha(codes) -> str:
    """本批发出的 code 集合身份（排序后取摘要，与生成顺序无关）。"""
    return _digest([{"code": c} for c in sorted(codes)])
