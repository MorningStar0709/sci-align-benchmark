"""Shared constants for mineru_pipeline modules.

Single source of truth for FALLBACK_METADATA and CATEGORY_DIFFICULTY_MAP.
"""

FALLBACK_METADATA: dict[str, dict] = {
    "Computer_Controlled_Systems_Astrom": {
        "title": "Computer-Controlled Systems",
        "author": "Karl Johan Åström / Björn Wittenmark",
        "control_category": "数字控制, 现代控制",
        "source": "GitHub: evilsin/control-and-system-book",
        "copyright": "Open Access / Educational",
    },
    "Essentials_of_Robust_Control_Zhou_Doyle": {
        "title": "Essentials of Robust Control",
        "author": "Kemin Zhou / John C. Doyle",
        "control_category": "鲁棒控制, 现代控制",
        "source": "GitHub: GeorgeQLe/Textbooks-and-Papers",
        "copyright": "Open Access / Educational",
    },
    "控制之美": {
        "title": "控制之美",
        "author": "Unknown / To be verified",
        "control_category": "经典控制, 现代控制",
        "source": "中文教材 / Chinese Textbook",
        "copyright": "Educational / Copyright holder to be verified",
    },
    "控制理论导论_郭雷": {
        "title": "控制理论导论",
        "author": "郭雷",
        "control_category": "现代控制, 控制理论",
        "source": "中文教材 / Chinese Textbook",
        "copyright": "Educational / Copyright holder to be verified",
    },
    "最优控制理论与应用": {
        "title": "最优控制理论与应用",
        "author": "Unknown / To be verified",
        "control_category": "最优控制",
        "source": "中文教材 / Chinese Textbook",
        "copyright": "Educational / Copyright holder to be verified",
    },
    "线性系统理论_郑大钟": {
        "title": "线性系统理论",
        "author": "郑大钟",
        "control_category": "现代控制, 线性系统",
        "source": "中文教材 / Chinese Textbook",
        "copyright": "Educational / Copyright holder to be verified",
    },
    "智能控制": {
        "title": "智能控制",
        "author": "Unknown / To be verified",
        "control_category": "智能控制",
        "source": "中文教材 / Chinese Textbook",
        "copyright": "Educational / Copyright holder to be verified",
    },
    "自动控制原理题海与考研指导_胡寿松": {
        "title": "自动控制原理题海与考研指导",
        "author": "胡寿松",
        "control_category": "经典控制",
        "source": "中文教材 / Chinese Textbook",
        "copyright": "Educational / Copyright holder to be verified",
    },
    "自抗扰控制技术": {
        "title": "自抗扰控制技术",
        "author": "韩京清",
        "control_category": "自抗扰控制, 非线性控制",
        "source": "中文教材 / Chinese Textbook",
        "copyright": "Educational / Copyright holder to be verified",
    },
    "动态系统的反馈控制_Franklin": {
        "title": "动态系统的反馈控制",
        "author": "Gene F. Franklin / J. David Powell / Abbas Emami-Naeini",
        "control_category": "经典控制, 现代控制, 数字控制",
        "source": "中文教材 / Chinese Textbook",
        "copyright": "Educational / Copyright holder to be verified",
    },
    "先进PID控制MATLAB仿真": {
        "title": "先进PID控制MATLAB仿真",
        "author": "刘金琨",
        "control_category": "智能控制, PID控制",
        "source": "中文教材 / Chinese Textbook",
        "copyright": "Educational / Copyright holder to be verified",
    },
    "自动控制原理_胡寿松": {
        "title": "自动控制原理",
        "author": "胡寿松",
        "control_category": "经典控制, 现代控制",
        "source": "中文教材 / Chinese Textbook",
        "copyright": "Educational / Copyright holder to be verified",
    },
    "非线性系统_Khalil": {
        "title": "非线性系统（第三版）",
        "author": "Hassan K. Khalil",
        "control_category": "非线性控制",
        "source": "中文教材 / Chinese Textbook",
        "copyright": "Educational / Copyright holder to be verified",
    },
}

# Difficulty level mapping (unified from build_corpus / build_metadata_light / stats_corpus).
# L1=foundational, L2=intermediate, L3=advanced, L4=specialized
CATEGORY_DIFFICULTY_MAP: dict[str, str] = {
    "经典控制": "L1", "PID控制": "L1", "classical": "L1",
    "现代控制": "L2", "线性系统": "L2", "数字控制": "L2",
    "控制理论": "L2", "估计与定位": "L2", "modern": "L2", "digital": "L2",
    "multi_agent": "L2",
    "最优控制": "L3", "鲁棒控制": "L3", "自适应控制": "L3",
    "非线性控制": "L3", "智能控制": "L3", "导航制导与控制": "L3",
    "自抗扰控制": "L3", "模型预测控制": "L3",
    "nonlinear": "L3", "robust": "L3", "optimal": "L3",
    "adaptive": "L3", "intelligent": "L3", "mpc": "L3",
    # English aliases (from build_corpus)
    "非线性": "L3", "自适应": "L4", "多智能体": "L4",
}
