/* 评分页：KaTeX 渲染 + 逐题即时落盘。
 *
 * 这里的核心约束是"绝不让一次点击静默丢失"：保存失败会重试，仍失败就把该题
 * 标成待处理并让**再点一次该题就能重发**，而不是等最后一次提交统一写库。
 *
 * 计时口径（与 wording 里对参与者承诺的"每题作答耗时"一致）：
 * 一屏 13 题滚动呈现，没有"该题开始作答"的事件可锚，所以用「注意力基线」近似：
 * 基线 = 页面加载时刻，或上一次作答切换的时刻。第 k 题的耗时 = 本题切换时刻 -
 * 上一次切换时刻，即"读完并决策这一题所花的时间"，不是从打开页面算起的累计值。
 * 基线不可信的场合一律标 truncated=1，由服务端保留已有的可信耗时不被覆盖：
 *   - 刷新续答后已答过的题（起始时刻在上一个会话）；
 *   - 对已保存过的题回头改分（这次点击不是"作答"）。
 */
(() => {
  "use strict";

  const KATEX_OPTS = {
    delimiters: [
      { left: "\\[", right: "\\]", display: true },
      { left: "\\(", right: "\\)", display: false },
      { left: "$$", right: "$$", display: true },
    ],
    // 默认 true：一个解析失败的公式会抛 ParseError 打断整页。
    throwOnError: false,
    errorColor: "#cc0000",
    // 中文写进数学模式时默认 warn 会逐字符刷 console。
    strict: "ignore",
    ignoredClasses: ["no-math"],
    errorCallback: (msg) => console.warn("[katex]", msg),
  };

  const cards = [...document.querySelectorAll(".item")];
  if (!cards.length) {
    if (window.renderMathInElement) renderMathInElement(document.body, KATEX_OPTS);
    return;
  }

  const CODE = cards[0].dataset.code;
  const progress = document.getElementById("progress");
  const total = cards.length;
  const savedIds = new Set();
  const pending = new Map();      // itemId -> 最近一次想写但没落库的字段
  const seq = new Map();          // itemId -> 递增序号，服务端只接受不小于已存序号的写
  // trustStart: itemId -> 该题首次作答的可信起始时刻；null = 起始时刻不可信。
  const trustStart = new Map();
  let baseline = performance.now();

  for (const card of cards) trustStart.set(card.dataset.item, performance.now());

  function cardOf(itemId) {
    return cards.find((c) => c.dataset.item === itemId);
  }

  function mark(card, kind) {
    const el = card.querySelector('[data-role="save"]');
    card.classList.remove("done", "pending");
    if (kind) card.classList.add(kind);
    el.textContent = { done: "✓", pending: "!" }[kind] || "○";
  }

  function refreshProgress() {
    if (progress) progress.textContent = `已保存 ${savedIds.size} / ${total}`;
  }

  async function post(url, fields, attempt = 1) {
    const body = new FormData();
    for (const [k, v] of Object.entries(fields)) body.append(k, v);
    try {
      const res = await fetch(url, { method: "POST", body });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      return await res.json();
    } catch (err) {
      if (attempt <= 3) {
        await new Promise((r) => setTimeout(r, 1000 * 2 ** (attempt - 1)));
        return post(url, fields, attempt + 1);
      }
      throw err;
    }
  }

  function buildFields(itemId) {
    const card = cardOf(itemId);
    const picked = card.querySelector(".score-opt:checked");
    if (!picked) return null;
    const now = performance.now();
    const start = trustStart.get(itemId);
    const duration = start == null ? 0 : Math.max(0, Math.round((now - start) / 1000));
    const truncated = start == null ? 1 : 0;
    seq.set(itemId, (seq.get(itemId) || 0) + 1);
    return {
      code: CODE,
      item_id: itemId,
      dimension: card.dataset.dim,
      score: picked.value,
      duration_sec: duration,
      truncated: truncated,
      client_seq: seq.get(itemId),
    };
  }

  async function save(itemId) {
    const card = cardOf(itemId);
    const fields = buildFields(itemId);
    if (!fields) return;
    pending.set(itemId, fields);

    try {
      await post("/api/rating", fields);
      // 迟到的响应不能覆盖更新的选择：seq 已前进说明后面又有一次点击。
      if (pending.get(itemId) !== fields) return;
      pending.delete(itemId);
      savedIds.add(itemId);
      mark(card, "done");
      refreshProgress();
    } catch (err) {
      mark(card, "pending");
      warn(`${itemId} 保存失败，请再点一次该题的选项重试（不需要改档位）。`);
      console.error("[save]", err);
    }
  }

  let warnEl = null;
  function warn(text) {
    if (!warnEl) {
      warnEl = document.createElement("p");
      warnEl.className = "flash";
      document.querySelector(".final").prepend(warnEl);
    }
    warnEl.textContent = text;
  }

  async function restore() {
    try {
      const res = await fetch(`/api/rating?code=${encodeURIComponent(CODE)}`);
      const { items } = await res.json();
      for (const r of items) {
        const card = cardOf(r.item_id);
        if (!card) {
          console.warn("[restore] 库里有该 code 不认识的题目，跳过：", r.item_id);
          continue;
        }
        const radio = card.querySelector(`.score-opt[value="${r.score}"]`);
        if (!radio) continue;
        radio.checked = true;
        radio.closest(".opt").classList.add("checked");
        savedIds.add(r.item_id);
        mark(card, "done");
        trustStart.set(r.item_id, null);   // 起始时刻在上一个会话，不可信
      }
      refreshProgress();
    } catch (err) {
      console.error("[restore]", err);
      warn("未能读取已保存的进度，请检查网络后再继续。");
    }
  }

  for (const card of cards) {
    const itemId = card.dataset.item;
    for (const radio of card.querySelectorAll(".score-opt")) {
      radio.addEventListener("change", () => {
        for (const r of card.querySelectorAll(".score-opt")) {
          r.closest(".opt").classList.toggle("checked", r.checked);
        }
        // 本题的耗时从"上一次切换"起算；改分（已存过题）一律标不可信。
        const firstTime = !savedIds.has(itemId) && trustStart.get(itemId) != null;
        if (firstTime) trustStart.set(itemId, baseline);
        baseline = performance.now();
        if (!firstTime) trustStart.set(itemId, null);
        save(itemId);
      });
    }
    // 失败恢复：radio 对"再点同一个已选中项"不触发 change，所以整卡 click 也试一次。
    card.addEventListener("click", () => {
      if (pending.has(itemId)) save(itemId);
    });
  }

  window.addEventListener("beforeunload", (e) => {
    if (pending.size) {
      e.preventDefault();
      e.returnValue = "";
    }
  });

  const finish = document.getElementById("finish");
  if (finish) {
    finish.addEventListener("click", async () => {
      const missing = cards.filter((c) => !c.querySelector(".score-opt:checked"));
      if (missing.length) {
        warn(`还有 ${missing.length} 题未选择：` +
             missing.slice(0, 6).map((c) => c.dataset.item).join(", ") +
             (missing.length > 6 ? " …" : ""));
        missing[0].scrollIntoView({ behavior: "smooth", block: "center" });
        return;
      }
      if (pending.size) {
        // 提交前把未落库的题按当前选择重发一遍，不让"点过了但没存上"蒙混过关。
        warn(`正在重试 ${pending.size} 题…`);
        const ids = [...pending.keys()];
        for (const id of ids) await save(id);
        if (pending.size) {
          warn("还有未保存成功的题目，请联网后重试或按提示再点一次对应题目。");
          return;
        }
      }
      finish.disabled = true;
      try {
        await post("/api/finish", { code: CODE });
        window.location.href = `/done/${CODE}`;
      } catch (err) {
        finish.disabled = false;
        warn("提交失败，请稍后重试。");
      }
    });
  }

  // 三个脚本都是 defer 且按 katex → auto-render → 本文件 的顺序执行，
  // 运行到这里时 DOM 已解析完、renderMathInElement 已存在，不需要再等事件。
  if (window.renderMathInElement) renderMathInElement(document.body, KATEX_OPTS);
  restore();
})();
