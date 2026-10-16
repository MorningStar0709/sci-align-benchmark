# `unattended_walltime.json` — 入仓说明

**入仓时间**: 2026-10-01
**生成命令**: `.venv/bin/python scripts/collect_unattended_walltime.py`
**数据来源**: 仓库根 `logs/*.log` 里的 `[SUMMARY] 总耗时 <n> min` 行，**按值去重**
**现读数**: 21 份日志 → **12 段** → **11637.1 min = 194.0 h**

## 为什么收进仓

对外那句「无人值守管线累计墙钟 **194 h**」的唯一源原先只在
`logs/*.log`（git 跟踪数 = 0，未入仓）与 `models/**/*.log`（指向 `/mnt/rdata` 的
symlink、仓外）里 ⇒ **本机能算、干净检出算不出**。本件把摘要值提进仓，
做法同 E11（`e11_judge_ppl_result.json`）与 e12_e4（10-01 两次同型迁移）。

## 两个必须知道的事实

1. **`models/` 那一趟是多余的**。原口头复算命令 `rglob("*.log")` 同时递归 `models/`
   （NFS、GB 级）——**本仓 10-01 明令禁止无过滤递归搜索**（会把盘读爆、进程卡 `D` 态）。
   实测只扫 `logs/` 得到**完全相同**的 12 段 / 11637.1 min，`models/` 零贡献。
   现生成器**只扫给定目录**，测试 `test_generator_only_scans_the_dir_it_is_given` 钉住这点。
2. **别按文件直加**。同一次运行有 launch 与会话两份日志、`总耗时` 相同；
   直加得 337.7 h（虚报 74%）。本件按**值**去重，并把同值的日志名并在一段里。

## 干净检出上能做什么

- **能**：核件自洽——分段之和 = `total_minutes` = 194.0 h（`tests/test_unattended_walltime.py`）。
- **不能**：重扫。`logs/` 是本机产物，干净检出没有它；此时对照测试**明跳**（不假绿）。
  这与「可复算 vs 可重跑」是两件事的既定口径一致。

## 与「220+ 小时」的区别（勿混）

本件锚定的是**本地段小时级墙钟**。云端段是 AutoDL **计费日**（6.4 天，按天计费、
无小时级日志），两者**记账单位不同**，不可相加。对外应分开说：
**本地 194 h ＋ 云端 6.4 计费日**（见 `docs/reports/promo-numbers-with-provenance.md` §一 #1 #2）。

## 恢复方式

`git rm provenance/unattended_walltime.json`，并在本机重跑上面的生成命令即可再生
（前提是 `logs/` 仍在盘）。
