"""配置驱动的医学术语映射管理器，支持按领域和语言动态加载术语映射。

从 data/medical_terminology.json 加载术语映射配置，
替代原有的 _ZH_RETRIEVAL_TERMS 硬编码列表。
"""

from __future__ import annotations

import copy
import json
import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional

from mineru_core.paths import PROJECT_ROOT

logger = logging.getLogger("MedicalTerminology")

DEFAULT_CONFIG_NAME = "medical_terminology.json"
ENV_CONFIG_PATH_KEY = "MEDICAL_TERMINOLOGY_PATH"


class MedicalTerminologyManager:
    """医学术语映射管理器，支持配置驱动的术语查询和匹配。

    通过 JSON 配置文件加载术语映射，构建双向索引支持：
      - 中文术语 → 英文检索词
      - 术语组 / 同义词查询
      - 配置文件热更新（reload_config）
    """

    def __init__(self, config_path: Optional[Path] = None):
        self._config_path = config_path or self._default_config_path()
        self._zh_to_en: Dict[str, dict] = {}
        self._en_groups: Dict[str, dict] = {}
        self._loaded = False
        self._config_mtime: float = 0.0
        self._load_config()

    @staticmethod
    def _default_config_path() -> Path:
        env_path = os.environ.get(ENV_CONFIG_PATH_KEY)
        if env_path:
            return Path(env_path)
        return PROJECT_ROOT / "data" / DEFAULT_CONFIG_NAME

    def _load_config(self):
        if not self._config_path.exists():
            logger.error("术语配置文件不存在: %s，术语映射不可用", self._config_path)
            self._loaded = True
            return

        self._config_mtime = self._config_path.stat().st_mtime
        try:
            with open(self._config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
            self._build_indices(config)
            self._loaded = True
            logger.info(
                "已加载术语配置: %s，%d 个中文术语映射",
                self._config_path,
                len(self._zh_to_en),
            )
        except (json.JSONDecodeError, OSError) as e:
            logger.error("加载术语配置失败: %s", e)
            self._loaded = True

    def _build_indices(self, config: dict):
        for group in config.get("term_groups", []):
            group_id = group["group_id"]
            self._en_groups[group_id] = group

            for mapping in group.get("mappings", []):
                zh_term = mapping.get("zh")
                en_term = mapping.get("en")
                if not zh_term or not en_term:
                    logger.warning(
                        "跳过缺少 zh/en 的映射条目: group=%s, mapping=%s",
                        group_id,
                        mapping,
                    )
                    continue
                term_info = {
                    "en": en_term,
                    "type": mapping.get("type", "generic"),
                    "group_id": group_id,
                    "search_group": mapping.get("search_group", ""),
                    "synonyms": mapping.get("synonyms", []),
                }
                self._zh_to_en[zh_term] = term_info

                for synonym in mapping.get("synonyms", []):
                    if synonym not in self._zh_to_en:
                        self._zh_to_en[synonym] = {
                            **term_info,
                            "is_synonym": True,
                        }

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def find_matches(self, text: str) -> List[dict]:
        """在文本中查找匹配的医学术语，按文本位置排序返回。

        当前术语库规模 ~36 条时 O(n*m) 全表扫描可接受。
        若术语库扩展至数千条，应替换为 Aho-Corasick 自动机或 Trie 索引。
        """
        matches = []
        for zh_term, mapping in self._zh_to_en.items():
            pos = text.find(zh_term)
            if pos >= 0:
                matches.append(
                    {
                        "source": zh_term,
                        "rewrite": mapping["en"],
                        "kind": mapping["type"],
                        "pos": pos,
                        "group_id": mapping["group_id"],
                        "search_group": mapping["search_group"],
                        "is_synonym": mapping.get("is_synonym", False),
                    }
                )
        return sorted(matches, key=lambda x: x["pos"])

    def get_term_info(self, zh_term: str) -> Optional[dict]:
        return self._zh_to_en.get(zh_term)

    def get_group_terms(self, group_id: str) -> List[dict]:
        group = self._en_groups.get(group_id)
        return copy.deepcopy(group.get("mappings", [])) if group else []

    def reload_config(self):
        if self._config_path.exists():
            current_mtime = self._config_path.stat().st_mtime
            if current_mtime == self._config_mtime:
                return
        logger.warning("术语配置文件已不存在，将清空术语映射: %s", self._config_path)
        self._zh_to_en.clear()
        self._en_groups.clear()
        self._loaded = False
        self._load_config()


@lru_cache(maxsize=1)
def get_terminology_manager() -> MedicalTerminologyManager:
    """获取术语管理器单例，首次调用时加载配置文件。"""
    return MedicalTerminologyManager()
