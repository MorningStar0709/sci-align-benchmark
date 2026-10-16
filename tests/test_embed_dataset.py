"""T1 embed_dataset 测试 — 提取口径 / 幂等续跑 / 原子写。

覆盖:
- extract: questions/messages 两种 kind、多 user 拼接、缺 user 报错、行号 id
- already_done: 产物缺失 / meta n 不匹配 / meta n 匹配 三态
- file_sha256 稳定性与 atomic_write 原子性（无 .tmp 残留）
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from scripts.embed_dataset import (
    PROJECT_ROOT,
    TextSource,
    already_done,
    atomic_write,
    extract,
    file_sha256,
)


class TestExtract:
    def test_questions_kind(self, tmp_path, monkeypatch):
        monkeypatch.setattr("scripts.embed_dataset.PROJECT_ROOT", tmp_path)
        qs = [{"id": "CS-EVO-00001", "question": "题一"},
              {"id": "CS-EVO-00002", "question": "题二"}]
        (tmp_path / "bank.json").write_text(json.dumps({"meta": {}, "questions": qs}), encoding="utf-8")
        src = TextSource("bank", "bank.json", "questions", "")
        texts, ids = extract(src)
        assert texts == ["题一", "题二"]
        assert ids == ["CS-EVO-00001", "CS-EVO-00002"]

    def test_messages_kind_with_line_number_ids(self, tmp_path, monkeypatch):
        monkeypatch.setattr("scripts.embed_dataset.PROJECT_ROOT", tmp_path)
        lines = [
            json.dumps({"messages": [{"role": "system", "content": "s"},
                                     {"role": "user", "content": "题甲"}]}, ensure_ascii=False),
            json.dumps({"messages": [{"role": "user", "content": "题乙"}]}, ensure_ascii=False),
        ]
        (tmp_path / "train.jsonl").write_text("\n".join(lines), encoding="utf-8")
        src = TextSource("train", "train.jsonl", "messages", "")
        texts, ids = extract(src)
        assert texts == ["题甲", "题乙"]
        assert ids == ["1", "2"]  # 1-based 行号

    def test_messages_kind_joins_multiple_user_parts(self, tmp_path, monkeypatch):
        monkeypatch.setattr("scripts.embed_dataset.PROJECT_ROOT", tmp_path)
        (tmp_path / "t.jsonl").write_text(
            json.dumps({"messages": [{"role": "user", "content": "题干"}, {"role": "user", "content": "附录"}]}),
            encoding="utf-8")
        src = TextSource("t", "t.jsonl", "messages", "")
        texts, _ = extract(src)
        assert texts == ["题干\n附录"]

    def test_messages_missing_user_raises(self, tmp_path, monkeypatch):
        monkeypatch.setattr("scripts.embed_dataset.PROJECT_ROOT", tmp_path)
        (tmp_path / "t.jsonl").write_text(
            json.dumps({"messages": [{"role": "system", "content": "s"}]}), encoding="utf-8")
        src = TextSource("t", "t.jsonl", "messages", "")
        with pytest.raises(ValueError, match="无 role=user"):
            extract(src)


class TestAlreadyDone:
    def test_missing_products_not_done(self, tmp_path, monkeypatch):
        monkeypatch.setattr("scripts.embed_dataset.PROJECT_ROOT", tmp_path)
        monkeypatch.setattr("scripts.embed_dataset.OUTPUT_DIR", tmp_path / "out")
        src = TextSource("g", "g.jsonl", "messages", "")
        (tmp_path / "g.jsonl").write_text(
            json.dumps({"messages": [{"role": "user", "content": "x"}]}), encoding="utf-8")
        done, _ = already_done(src)
        assert not done

    def test_meta_n_mismatch_not_done(self, tmp_path, monkeypatch):
        out = tmp_path / "out"
        monkeypatch.setattr("scripts.embed_dataset.PROJECT_ROOT", tmp_path)
        monkeypatch.setattr("scripts.embed_dataset.OUTPUT_DIR", out)
        out.mkdir()
        np.save(out / "embeddings_g.npy", np.zeros((1, 4)))
        (out / "g.meta.json").write_text(json.dumps({"n": 99}), encoding="utf-8")
        (tmp_path / "g.jsonl").write_text(
            json.dumps({"messages": [{"role": "user", "content": "x"}]}), encoding="utf-8")
        done, reason = already_done(TextSource("g", "g.jsonl", "messages", ""))
        assert not done and "99" in reason

    def test_meta_n_match_done(self, tmp_path, monkeypatch):
        out = tmp_path / "out"
        monkeypatch.setattr("scripts.embed_dataset.PROJECT_ROOT", tmp_path)
        monkeypatch.setattr("scripts.embed_dataset.OUTPUT_DIR", out)
        out.mkdir()
        np.save(out / "embeddings_g.npy", np.zeros((1, 4)))
        (out / "g.meta.json").write_text(json.dumps({"n": 1}), encoding="utf-8")
        (tmp_path / "g.jsonl").write_text(
            json.dumps({"messages": [{"role": "user", "content": "x"}]}), encoding="utf-8")
        done, _ = already_done(TextSource("g", "g.jsonl", "messages", ""))
        assert done


class TestSha256AndAtomic:
    def test_sha256_stable(self, tmp_path):
        p = tmp_path / "a.bin"
        p.write_bytes(b"data" * 1000)
        assert file_sha256(p) == file_sha256(p)

    def test_atomic_write_no_tmp_leftover(self, tmp_path):
        target = tmp_path / "out.json"
        atomic_write(target, lambda p: p.write_text('{"a": 1}', encoding="utf-8"))
        assert target.exists()
        assert json.loads(target.read_text(encoding="utf-8")) == {"a": 1}
        assert not list(tmp_path.glob("*.tmp"))


def test_project_root_is_workspace():
    assert (PROJECT_ROOT / "scripts" / "embed_dataset.py").exists()
