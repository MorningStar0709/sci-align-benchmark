# Sci-Align Benchmark

[![ci](https://github.com/MorningStar0709/sci-align-benchmark/actions/workflows/ci.yml/badge.svg)](https://github.com/MorningStar0709/sci-align-benchmark/actions/workflows/ci.yml)

**一套会审计自己测量仪器的评测协议，以及「审计之后结论变了」的证据。** 1,979 道控制科学题、
双判盲评加第三模型仲裁、晋级标准在任何计分比较之前书面冻结，全部由一个人在受限算力下端到端实现。

[English · 英文说明](README.md) · [协议](PROTOCOL.zh-CN.md) · [本仓能复算什么](ARTIFACT.md) · 代码 Apache-2.0 / 数据 CC-BY-4.0

![Audit-the-Instrument 评测协议：语料、双盲出题、n=3 作答、双判、裁定，另有逐题审计与跨判官世代漂移重判](figures/protocol.png)

*测量仪器本身被当作实验对象：标准先冻结再开始评分，判官自己也要被审。*

---

## 为什么做这件事

无可参照答案的生成任务上，用 LLM 当判官已成了给后训练打分的主流做法。而这个仪器几乎从不被它
服务的那些研究审计：评分服务在评测世代之间一换，分数就漂；而单次生成只是从一个未被测量的分布里
抽出来的一次抽样。本项目把「评测」本身当成研究对象，而不是假定可用的工具。两篇预印本讲了它：

| | 论文 | 预印本 |
|:--|:--|:--|
| **B** | *Benchmarking Post-Training Under Constrained Compute: Protocols for Question Generation, Variance-Controlled Evaluation, and Judge Auditing* | [10.5281/zenodo.23409480](https://doi.org/10.5281/zenodo.23409480) |
| **A** | *What Determines the Direction of Post-Training Gains? A Preregistered Multi-Scale Study of Method, Scale, and Base-Model Identity* | [10.5281/zenodo.23409394](https://doi.org/10.5281/zenodo.23409394) |

B 篇给出协议，A 篇是这套协议的一次应用。本仓是两篇的实现之处，也是它们数字的可核查之处。

## 仓里有什么

| 题池 | 协议 | 证据 |
|:--|:--|:--|
| 1,979 道控制科学题，来自 361 个来源，每题带机器可读的许可桶位 | 双判盲评加第三模型仲裁 · 晋级标准在任何计分比较之前书面冻结 · n=3 重生成标定世代间方差 · 跨判官世代重判固定答案以检出漂移 | 237 份逐题评测件（约 3.2 万条判分）；件内印出的每个聚合量都能从它的逐题记录复算 |

![人工评分界面：一道题显示题目、标准答案与待评回答](figures/humaneval.png)

*`humaneval/` 里的人工评分界面，也是用来对照自动判官的那一套。评分者看到题目、标准答案与待评
回答，逐维度打分，可中断可分次完成；13 道题约需 35–55 分钟。界面为中文，与实际施测一致。*

## 数字概览

| | | 怎么核 |
|:--|--:|:--|
| 题池 | 1,979 题，361 个来源 | [`SOURCES.md`](SOURCES.md)，或下面配方 1 |
| 已判分的答案（逐题记录） | 约 3.2 万条 | 下面配方 4 |
| 逐题评测件 | 237 份 | [`release/public-eval-files.txt`](release/public-eval-files.txt) |
| 被评测的模型身份 | 81 个，其中 23 个在本仓可达 | **核不了**：证据在开发仓 |
| 算力 | 约 350–400 GPU 时 | **核不了**：证据在开发仓 |
| 无人值守管线墙钟 | 194 小时，12 段日志 | 数值随包（`provenance/unattended_walltime.json`）；复算会 fail closed（清单第 4 行）|
| 代码 | 约 10 万行；本仓自带评测与验证子集 | **核不了**：证据在开发仓 |
| 投入 | 4 个月 · 单人 · 一台笔记本加一台租用 A800 | — |

整个研究共判分 **约 7.1 万条答案，跨 545 份产物**；未随本仓发布的 399 份，恰好是两篇论文里任何
一张图都不读的那些。写「核不了」的行，逐项证据在**开发仓**里，而开发仓不公开：我们宁愿直说，
也不暗示你能在这里找到。[`ARTIFACT.md`](ARTIFACT.md) 逐条写明本仓能核与不能核的。

## 训了什么，以及它是在什么上被量的

后训练矩阵冻结在一份随包的登记件里：[`release/adapter_manifest.json`](release/adapter_manifest.json)。
它记下 **25 个臂**——两个域（control、medical）× 三个模型尺寸（4B、9B、Gemma）——其中 **6 个是基座/raw
基线，19 个是训练出来的 LoRA 适配器**，每个条目带着基座模型、适配器的 `sha256`，以及它喂给哪些评测
产物。同一份文件还留着 11 条 surface 适配器、实际用过的判官提示、以及评测冻结（标志与一条说明）。

训练跑在一台 RTX 5090 上；判分密集的那部分评测也用了租来的 A800 机时（见上面「数字概览」与
[`ARTIFACT.md`](ARTIFACT.md) 第 13 行）。

训练与编排的**脚本不随包**——它们带着具体实验的批次代号，并假定本地有权重与日志目录。随包的是上面这份
登记件、产出已发表数字的聚合与验证代码、以及逐题输出本身。适配器同样未发布，见
[`adapters/`](adapters/README.md)。

## 两条结论

**赢家取决于你把方法跑在哪个规模上。** 两种强化类方法在 4B 与 9B 之间互换了名次。4B 上
RLVR-C v2 领先 MGPO-EG，但两者都低于各自的 SFT 基线（相对增益 −0.9% 与 −5.1%）；到 9B，
MGPO-EG 反超，同为相对 SFT（+7.6% 对 +2.0%）。两个独立的泛化探针（换基座家族、换任务域）
都让增益的**符号**同样翻转，跟着基座身份走，而不是跟着方法走。

**判官自己漂了，而协议抓到了它。** 在后一个评分世代上重判同一批答案，控制域出现系统性上行
（已发表的聚合值是判分 +0.0606，11 条模型臂全部上移；本仓按同题配对复算得约 +0.06），医疗域
基本没有。正因为晋级标准在任何计分比较之前就已冻结，这次漂移只让四条预注册比较中的一条越过了线。

完整的分析、消融与已声明的局限在两篇预印本里。凡能在本仓复算的数字，[`ARTIFACT.md`](ARTIFACT.md)
会写明怎么复算：上面引的那组增益就是清单第 12 行，复算命令也印在那里。复算不出的，清单会说出来：
漂移聚合值是 26 条声明锚点中的一条，清单第 1 行说明它在本仓核不了。

## 工程纪律，逐条点名

下面每一条都随包，而且都是**可以看**的，不是让你相信的。

- **一条命令重算头条聚合量。** [`tools/verify_pack.py`](tools/verify_pack.py) 只用标准库：从每份 run
  自己的逐题记录重算它的 `overall_score`、把文件的 `sha256` 对回它喂的那份投影、打出一张 CHECK 表。
  全程不碰云端。
- **题池每题带书面审计理由**，不只是题面与答案：每题四个证据字段、合计约 **284 万字符**，其中约 60%
  引了判断所依据的来源段落。≥300 字符的逐字跨度换成占位；规则与计数见
  [`LICENSE-LEDGER.md`](LICENSE-LEDGER.md)。
- **评分器是照「泄漏不了」造的。** 见 [`humaneval/`](humaneval/README.md)：评分者的 code 运行时挂载、
  不烘进镜像；`--no-access-log`，因为链接本身就带着凭证；进程以 uid 10001 跑在只读根文件系统上、
  capability 全部丢弃；题池身份与数据库不符时拒绝服务，除非显式知情且把不符记档。
- **检查判的是「新出现的失败」，不是「全绿」。** [`scripts/run_provenance_gates.py`](scripts/run_provenance_gates.py)
  跑一个公开检出所能支撑的检查，只报相对基线**新出现的失败**。一支符合命名约定、却两张登记单里都
  没有的脚本，本身就是失败。
- **入口在缺输入时会停下来并说明原因。** 本仓自带的四个 `--check` 命令里，三个**带着理由**非零退出
  而不是抛一段栈，第四个返回 skip；[`tests/test_entry_smoke.py`](tests/test_entry_smoke.py) 像读者
  一样调用全部四个，因此「套件全绿、入口从没跑过」不可能出现。
- **每个读数都被分类，而不是被宣称。** [`ARTIFACT.md`](ARTIFACT.md) 把它们逐条归为
  *Recomputable*、*Registered only* 或 *Fails closed*，包括本仓无法验证的那些行。
- **构建是规则，不是手挑。** 这棵树由开发仓里的一个脚本生成，[`BUILD-CRITERIA.md`](BUILD-CRITERIA.md)
  记下准入规则与由此得到的文件集。

## 快速上手

```bash
uv sync --all-packages --all-extras
.venv/bin/python scripts/run_provenance_gates.py   # 检查纪律
.venv/bin/python -m pytest tests/ -q               # 67 passed, 4 skipped
```

然后读题池（[`benchmark/dataset/core_v3.json`](benchmark/dataset/core_v3.json)）、协议
（[`PROTOCOL.zh-CN.md`](PROTOCOL.zh-CN.md)）与逐条主张清单（[`ARTIFACT.md`](ARTIFACT.md)）。

### 自己核一遍

下面四条都能在克隆后直接跑，右边是应得的输出。

```bash
# 1. 题池与逐题许可桶位
#    → 1979 {'redistributable': 685, 'arxiv-only': 927, 'restricted': 241, 'textbook-unchecked': 126}
.venv/bin/python -c "import json;d=json.load(open('benchmark/dataset/core_v3.json'));print(len(d['questions']),d['meta']['license_tagging']['counts_by_bucket'])"

# 2. 上面引的十八格增益 → 4B +14.9/−0.9/−5.1，9B +13.0/+2.0/+7.6
.venv/bin/python -c "import sys;sys.path.insert(0,'papers/paper-a/figures/scripts');import surface_stats as s;[print(k,{a:round(b,1) for a,b in v.items() if a!='gens'}) for k,v in s.gains().items()]"

# 3. 一条随包的聚合件，对回它自己的源件 → True
.venv/bin/python -c "import json,hashlib,pathlib;d=json.load(open('release/evidence/medical_mgpo_9b_run1__medical_dual_rejudge.json'));print(hashlib.sha256(pathlib.Path(d['source']).read_bytes()).hexdigest()==d['source_sha256'])"

# 4. 逐题评测件，以及它们里面的逐题记录
#    → per-item files: 146 of 237 | answers: 31796
.venv/bin/python -c "
import json
rels=[l.strip() for l in open('release/public-eval-files.txt') if l.strip() and not l.startswith('#')]
docs=[json.load(open(r)) for r in rels]
docs=[d for d in docs if d.get('records')]
print('per-item files:', len(docs), 'of', len(rels), '| answers:', sum(len(d['records']) for d in docs))"
```

核不了的行（以及为什么核不了）都记在 [`ARTIFACT.md`](ARTIFACT.md) 上。

## 目录一览

```text
benchmark/dataset/     题池（1,979 题）、逐题许可桶位、JSON schema
benchmark/eval/        两篇论文用到的逐题评测件
provenance/            已登记的聚合件：模型身份、墙钟、主张锚点、工件索引
release/evidence/      主张清单所指向的逐实验聚合 JSON
packages/, scripts/    工具链里评测与验证的那一部分
humaneval/             人工评分器界面，用于对照判官
figures/               协议图与人工评分界面截图
```

## 不包含什么

- **正文源**：两篇论文已作为预印本发布（DOI 见上），不在本仓重复。`papers/` 下只有统计模块
  （增益面所依赖的那个，见清单第 12 行），没有任何正文文本。
- **训练语料与基座权重**：两者都不是我们能再分发的。
- **判官 API 记录**：判分背后的云端调用不在本仓；它们产出的逐题分数在本仓。
- **来源许可不允许再分发的引文**：这类题目的逐字来源摘录以固定占位替换，且每题带自己的许可桶位。
  计数见 [`LICENSE-LEDGER.md`](LICENSE-LEDGER.md)。
- **LoRA 适配器与题池镜像**：尚未发布，见 [`adapters/`](adapters/README.md)。

## 许可

代码 **Apache-2.0**；数据与内容 **CC-BY-4.0**。第三方来源保留其自身条款。
[`LICENSE.md`](LICENSE.md)、[`LICENSE-LEDGER.md`](LICENSE-LEDGER.md) 与
[`SOURCES.md`](SOURCES.md) 记录了哪些可再分发、哪些不可、来源怎么登记，并给出权利方的下架路径。

## 由来

本工件是 [ControlMind](https://github.com/MorningStar0709/ControlMind) 评测轨（Sci-Align）的**学术延续**。
ControlMind 是一个三轨系统（benchmark · data agent · medical RAG），获 2026 CICC 数据智能竞赛三等奖
（MinerU 与上海人工智能实验室主办）。相对那一轨，**题池从 500 扩到 1,979，判官从单判升级为双判**，
因此**本仓结果与那个榜不可直接比较**；ControlMind 的另外两轨不受本仓影响。

## 引用

含作者在内的元数据在 [`CITATION.cff`](CITATION.cff)；GitHub 会把它渲染成「Cite this repository」。
