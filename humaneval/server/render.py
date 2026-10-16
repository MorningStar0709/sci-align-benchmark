"""含 LaTeX 的中文 Markdown -> 交给 KaTeX auto-render 处理的 HTML。

五处非常规处理各对应一个实测缺陷，删除任何一处都会**静默**改变评分者看到的
数学语义或打开注入面（KaTeX 对这些情况不报错也不警告），勿简化、勿"顺手清理"。
"""

import re

import markdown

# 扩展选择依据 60,398 条真实 model_answer 的测量：table 3.8%、fenced code 0.2%。
# 标题、列表、粗体、链接属 CommonMark 默认能力，不需扩展。arithmatex 必须在此列表内，
# 否则 \( \) 会被反斜杠转义规则吃掉。
_MD_EXTENSIONS = ["fenced_code", "tables", "pymdownx.arithmatex"]
_MD_CONFIG = {
    "pymdownx.arithmatex": {
        "generic": True,               # KaTeX 需要 generic 输出：\[..\] 与 \(..\)
        "preview": False,              # 仅对 mathjax 模式有意义，generic 下无副作用
        # inline_syntax 必须含 "dollar"：实测 60,398 条回答里 41.9% 用 $..$ 写行内
        # 公式（\) 形式仅 15.0%），关掉它会让近一半回答显示成字面 $T>0$。
        # $$ 被降级成行内的问题由 _normalize_math_blocks 处理，不靠禁用语法来规避。
        "inline_syntax": ["dollar", "round"],
        "block_syntax": ["dollar", "square", "begin"],
    }
}

_BLOCK_MATH = re.compile(r"\$\$(.+?)\$\$", re.S)
_MATH_REGION = re.compile(r"\$\$(.+?)\$\$|\\\((.+?)\\\)", re.S)

# 语料实测：现行题池（2026-09-10 定版）3 题各 1 处 \[..\]（H-02/H-08/H-09），均不是
# 空行隔离的独立块。arithmatex 的 square 语法只认块级起始位置，这些形态会被
# Python-Markdown 的反斜杠转义吃掉，评分者最终看到字面的方括号加原始 LaTeX。
# (?<!\\) 前导反斜杠排除 LaTeX 行距写法 \\[2pt]：否则非贪婪匹配会一路吞到
# 后面真正的 \]，紧随其后的合法 \[..\] 静默漏转。
_SQUARE_MATH = re.compile(r"(?<!\\)\\\[(.+?)\\\]", re.S)


def _promote_square_math(text: str) -> str:
    r"""把 \[..] 改写成 $$..$$，交给 _normalize_math_blocks 提升为独立块。

    两个已知陷阱，各有针对性防线：
    1. LaTeX 行距写法 \\[2pt] 的第二根反斜杠也"紧跟 ["，会被当成开界符
       （由 (?<!\\) 前导反斜杠排除）；
    2. 已在 $$..$$ 或 \(..\) 里的 \[..\] 一旦被改写，插入的 $$ 会与外层配对，
       把整块公式劈成两块——所以替换只在既有数学区间**之外**进行。

    \left[ .. \right] 不受影响：那两处 `[` / `]` 的前导字符是 "left"/"right"
    而非反斜杠，不匹配本正则。
    """

    def repl(match: re.Match) -> str:
        inner = match.group(1)
        # 兜底：inner 里若混进 $$（未闭合公式把本匹配卷了进来），整段原样返回——
        # 宁可漏转这一处，也不能把别的公式劈开。正常路径由下面的区间分段保证。
        return match.group(0) if "$$" in inner else f"$${inner}$$"

    parts, pos = [], 0
    for m in _MATH_REGION.finditer(text):
        parts.append(_SQUARE_MATH.sub(repl, text[pos:m.start()]))
        parts.append(m.group(0))
        pos = m.end()
    parts.append(_SQUARE_MATH.sub(repl, text[pos:]))
    return "".join(parts)


# (?<!\\) 避免二次包裹已是 \rank 的控制序列；(?<!\{) 避免把
# \operatorname{rank} 里的 rank 再包一层。
_BARE_RANK = re.compile(r"(?<!\\)(?<!\{)\brank(?=\s*\()")

# 注入面：Python-Markdown 默认**不消毒**，`<script>` / `<img onerror=..>` / `<!-- -->`
# 原样穿透（实测），而模板侧用 `| safe` 输出。待评回答是不可信的模型生成内容，
# 一段 HTML 就足以伪造 radio、改写评分者的选择并持有页面上的 rater code。
# 实测只需处理"标签形状"：裸 `<`（`c<d`、`x < y`）markdown 自己已经转成 `&lt;`。
_HTML_TAG = re.compile(r"</?[A-Za-z][^>]*>?|<!--.*?-->|<!\[CDATA\[.*?\]\]>|<![^>]*>", re.S)
# 代码区不做预转义：markdown 自己会转义 code 内容，预转义会变成 `&amp;lt;` 双重转义，
# 评分者看到字面的 `&lt;b&gt;`。
_CODE_REGION = re.compile(r"(`{3,}.*?`{3,}|`[^`\n]*`)", re.S)


def _neutralize_raw_html(text: str) -> str:
    r"""把代码区外的 HTML 标签形状转成可见文本（`&lt;b&gt;` 渲染回 `<b>`）。

    数学区**故意不保护**：实测 `$$x &lt; y$$` 经 arithmatex 后仍为 `&lt;`（不是
    `&amp;lt;`），浏览器解码后 KaTeX 拿到的就是 `<`，语义不变。
    这也让人机输入更对称：agent 读到的原始 token 里 `<b>` 就是字面文本。
    """
    parts = _CODE_REGION.split(text)
    for i, seg in enumerate(parts):
        if i % 2 == 1:          # 奇数段是被捕获的代码区，原样保留
            continue
        parts[i] = _HTML_TAG.sub(lambda m: m.group(0).replace("<", "&lt;"), seg)
    return "".join(parts)


def _normalize_math_blocks(text: str) -> str:
    """缺陷 1：块级 $$ 不独占一行时会被 Python-Markdown 降级成行内并残留裸 $。

    必须成对匹配。逐 occurrence 判断前后邻字符的写法会把开界符也当成闭界符，
    从而在 $$..$$ 内部插入空行劈开公式本身。
    """
    return _BLOCK_MATH.sub(lambda m: "\n\n" + m.group(0) + "\n\n", text)


def _promote_bare_operators(text: str) -> str:
    """缺陷 2：KaTeX 没有内置 \\rank，裸 rank(A) 会被渲染成斜体 r·a·n·k 相乘。

    只改写数学区间内的内容；假设是"英文 rank 紧跟左括号只会出现在数学上下文"，
    中文正文不满足该形状。\\det 本身是合法控制序列，不在此处理。
    """

    def repl(match: re.Match) -> str:
        is_block = match.group(1) is not None
        inner = match.group(1) if is_block else match.group(2)
        fixed = _BARE_RANK.sub(lambda _: r"\operatorname{rank}", inner)
        return f"$${fixed}$$" if is_block else f"\\({fixed}\\)"

    return _MATH_REGION.sub(repl, text)


def to_html(text: str) -> str:
    """把一段 Markdown 渲染成 HTML，数学定界符原样留给客户端 KaTeX。"""
    raw = _promote_bare_operators(
        _normalize_math_blocks(_promote_square_math(_neutralize_raw_html(str(text or "")))))
    return markdown.markdown(
        raw,
        extensions=_MD_EXTENSIONS,
        extension_configs=_MD_CONFIG,
        output_format="html",
    )
