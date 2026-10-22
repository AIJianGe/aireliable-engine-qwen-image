/* ============================================================
   Qwen-Image WebUI 逻辑（原生 JS，直连引擎 API）
   四区联动：设置区（模式/档位） ↔ 参数区（含输入图/视角控制） ↔ LoRA 区 ↔ 结果区（SSE + 历史）
   ============================================================ */
"use strict";

const API = "";  // 同源直连（引擎自身挂载 /ui）

// ---------- 档位显示名（内部标识 → 原始模型名，仅影响 UI 显示） ----------
const PRECISION_LABELS = {
  gguf_q8: "Qwen-Image · GGUF-Q8",
  q8_0: "Qwen-Image · GGUF-Q8",
};
const precisionLabel = (p) => PRECISION_LABELS[p] || p;

// ---------- 状态 ----------
const state = {
  mode: "t2i",
  images: [],        // [{ name, dataUrl, b64 }]
  loras: [],         // [{ name, weight, loaded }]
  generating: false,
  precisionTouched: false,  // 用户手动选定具体档位后，下拉框不再自动跟随实际档位
};

// ---------- DOM ----------
const $ = (id) => document.getElementById(id);

const MODE_HINTS = {
  t2i: "文生图：仅文字描述生成图像（Qwen-Image-2512，中文字符渲染强项）",
  edit: "指令编辑：1 张输入图 + 文字指令（Qwen-Image-Edit-2509，如\"把背景换成海滩\"；可选视角控制）",
  i2i: "多图参考：2-5 张参考图原生多图语义交互（如\"Picture 1 的女孩抱着 Picture 2 的猫\"；官方推荐 ≤3 张，更多指令遵循度下降）",
};

// 各模式推荐参数（Qwen-Image 官方推荐；true_cfg 为真 CFG 强度）
const MODE_RECOMMEND = {
  t2i:  { steps: 28, trueCfg: 4.0, note: "文生图官方推荐：28 步 · True CFG 4.0" },
  edit: { steps: 24, trueCfg: 4.0, note: "指令编辑：24 步收敛快，低 CFG 保留原图结构" },
  i2i:  { steps: 28, trueCfg: 4.0, note: "多图参考：28 步 · True CFG 4.0，prompt 建议 Picture 1/2/3 指代" },
};

// ============================================================
// 初始化
// ============================================================
document.addEventListener("DOMContentLoaded", () => {
  bindModeSeg();
  bindSizePresets();
  bindSliders();
  initViewControl();  // 视角控制（edit 模式）：拖拽圆盘 + 俯仰滑杆 + 景别
  bindRecoApply();
  bindLightbox();
  bindUpload();
  bindButtons();
  bindPrecisionSelect();
  refreshStatus();
  refreshCapabilities();
  refreshHistory();
  setInterval(refreshStatus, 15000);  // 状态轮询（空闲时）
});

// ============================================================
// 设置区：模式切换
// ============================================================
function bindModeSeg() {
  // 仅绑定模式段选择器（视角控制的景别按钮同为 .seg-btn，不能误绑）
  document.querySelectorAll("#mode-seg .seg-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll("#mode-seg .seg-btn").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      state.mode = btn.dataset.mode;
      $("mode-hint").textContent = MODE_HINTS[state.mode];
      $("image-field").style.display = state.mode === "t2i" ? "none" : "";
      $("view-field").style.display = state.mode === "edit" ? "" : "none";
      $("image-req").textContent = state.mode === "edit" ? "（必填 1 张）" : "（必填 2-5 张）";
      $("footer-mode").textContent = `模式：${btn.textContent}`;
      updateRecoBar();
    });
  });
  updateRecoBar();
}

// ============================================================
// 设置区：模式推荐参数条
// ============================================================
function updateRecoBar() {
  const reco = MODE_RECOMMEND[state.mode];
  if (!reco) { $("param-reco").style.display = "none"; return; }
  $("param-reco").style.display = "";
  $("param-reco-text").textContent =
    `📌 本模式推荐：步数 ${reco.steps} · True CFG ${reco.trueCfg} — ${reco.note}`;
}

function bindRecoApply() {
  $("btn-apply-reco").addEventListener("click", () => {
    const reco = MODE_RECOMMEND[state.mode];
    if (!reco) return;
    $("steps").value = reco.steps;
    $("true-cfg").value = reco.trueCfg;
    // 触发 input 事件同步数值显示（与 bindSliders 的监听保持一致）
    $("steps").dispatchEvent(new Event("input"));
    $("true-cfg").dispatchEvent(new Event("input"));
  });
}

// ============================================================
// 设置区：精度档位下拉
// ============================================================
function bindPrecisionSelect() {
  // 默认选中 GGUF Q8（manifest 默认档）；选回「跟随已加载档位」时
  // 恢复自动同步，选定具体档位后不再被状态轮询覆盖
  $("precision-select").addEventListener("change", (e) => {
    state.precisionTouched = e.target.value !== "";
  });
}

// ============================================================
// 参数区
// ============================================================
function bindSizePresets() {
  document.querySelectorAll(".preset").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".preset").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      $("width").value = btn.dataset.w;
      $("height").value = btn.dataset.h;
    });
  });
  // 默认高亮 1:1（Qwen 官方推荐 1328²）
  document.querySelector('.preset[data-w="1328"][data-h="1328"]').classList.add("active");
}

function bindSliders() {
  $("steps").addEventListener("input", (e) => { $("steps-val").textContent = e.target.value; });
  $("true-cfg").addEventListener("input", (e) => { $("true-cfg-val").textContent = e.target.value; });
}

// ============================================================
// 视角控制（edit 模式）：水平拖拽圆盘 + 俯仰竖直滑杆 + 景别三段
// 拖拽结果吸附到后端 ViewSpec 支持的档位（schemas.py Literal 枚举，契约零改动）
// ============================================================
const VIEW_H_LEVELS = [
  { key: "left90",  angle: -90, label: "左侧 90°" },
  { key: "left45",  angle: -45, label: "左侧 45°" },
  { key: "front",   angle: 0,   label: "正面" },
  { key: "right45", angle: 45,  label: "右侧 45°" },
  { key: "right90", angle: 90,  label: "右侧 90°" },
];
const VIEW_V_LEVELS = [
  { key: "up30",   pos: 0,     label: "仰视 30°" },
  { key: "level",  pos: 1 / 3, label: "平视" },
  { key: "down30", pos: 2 / 3, label: "俯视 30°" },
  { key: "down60", pos: 1,     label: "高角俯视" },
];
const view = { horizontal: "front", vertical: "level", distance: "medium" };

const DIAL_CX = 76, DIAL_CY = 86;  // 圆盘圆心（SVG viewBox 坐标，与 index.html 一致）
const PITCH_HANDLE_H = 14;         // 俯仰手柄高度（px，与 style.css 一致）

function initViewControl() {
  buildDialTicks();
  buildPitchLabels();
  bindDialDrag();
  bindPitchDrag();
  bindDistSeg();
  $("view-reset").addEventListener("click", () => {
    Object.assign(view, { horizontal: "front", vertical: "level", distance: "medium" });
    renderView();
  });
  renderView();
}

// 圆盘刻度：5 档位置在弧外侧（半径 53~59）
function buildDialTicks() {
  const g = $("dial-ticks");
  for (const lv of VIEW_H_LEVELS) {
    const rad = (lv.angle * Math.PI) / 180;
    const sin = Math.sin(rad), cos = Math.cos(rad);
    const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
    line.setAttribute("x1", (DIAL_CX + 53 * sin).toFixed(1));
    line.setAttribute("y1", (DIAL_CY - 53 * cos).toFixed(1));
    line.setAttribute("x2", (DIAL_CX + 59 * sin).toFixed(1));
    line.setAttribute("y2", (DIAL_CY - 59 * cos).toFixed(1));
    g.appendChild(line);
  }
}

// 俯仰标签：4 档右侧标注，与手柄中心对齐（轨道 96px，手柄中心 7 + pos*82）
function buildPitchLabels() {
  const box = $("pitch-labels");
  VIEW_V_LEVELS.forEach((lv) => {
    const span = document.createElement("span");
    span.textContent = lv.label;
    span.dataset.key = lv.key;
    span.style.top = `${(7 + lv.pos * 82 - 8).toFixed(1)}px`;
    box.appendChild(span);
  });
}

// 水平圆盘：指针拖拽（Pointer Events + 捕获），角度 → 最近档位吸附
function bindDialDrag() {
  const svg = $("dial-h");
  let dragging = false;
  const apply = (e) => {
    const rect = svg.getBoundingClientRect();
    const x = ((e.clientX - rect.left) / rect.width) * 152;
    const y = ((e.clientY - rect.top) / rect.height) * 112;
    let a = (Math.atan2(x - DIAL_CX, DIAL_CY - y) * 180) / Math.PI;  // 0=上方，右为正
    a = Math.max(-90, Math.min(90, a));
    view.horizontal = nearestLevel(VIEW_H_LEVELS, "angle", a).key;
    renderView();
  };
  svg.addEventListener("pointerdown", (e) => {
    dragging = true;
    try { svg.setPointerCapture(e.pointerId); } catch { /* 合成事件无活动指针，忽略 */ }
    apply(e);
  });
  svg.addEventListener("pointermove", (e) => { if (dragging) apply(e); });
  svg.addEventListener("pointerup", () => { dragging = false; });
  svg.addEventListener("pointercancel", () => { dragging = false; });
  svg.addEventListener("keydown", (e) => {
    const idx = VIEW_H_LEVELS.findIndex((l) => l.key === view.horizontal);
    if (e.key === "ArrowLeft" && idx > 0) {
      view.horizontal = VIEW_H_LEVELS[idx - 1].key; renderView(); e.preventDefault();
    } else if (e.key === "ArrowRight" && idx < VIEW_H_LEVELS.length - 1) {
      view.horizontal = VIEW_H_LEVELS[idx + 1].key; renderView(); e.preventDefault();
    }
  });
}

// 俯仰滑杆：竖直拖拽 → 4 档吸附
function bindPitchDrag() {
  const track = $("pitch-track");
  let dragging = false;
  const apply = (e) => {
    const rect = track.getBoundingClientRect();
    const usable = rect.height - PITCH_HANDLE_H;
    const pos = Math.max(0, Math.min(1, (e.clientY - rect.top - PITCH_HANDLE_H / 2) / usable));
    view.vertical = nearestLevel(VIEW_V_LEVELS, "pos", pos).key;
    renderView();
  };
  track.addEventListener("pointerdown", (e) => {
    dragging = true;
    try { track.setPointerCapture(e.pointerId); } catch { /* 合成事件无活动指针，忽略 */ }
    apply(e);
  });
  track.addEventListener("pointermove", (e) => { if (dragging) apply(e); });
  track.addEventListener("pointerup", () => { dragging = false; });
  track.addEventListener("pointercancel", () => { dragging = false; });
  track.addEventListener("keydown", (e) => {
    const idx = VIEW_V_LEVELS.findIndex((l) => l.key === view.vertical);
    if (e.key === "ArrowUp" && idx > 0) {
      view.vertical = VIEW_V_LEVELS[idx - 1].key; renderView(); e.preventDefault();
    } else if (e.key === "ArrowDown" && idx < VIEW_V_LEVELS.length - 1) {
      view.vertical = VIEW_V_LEVELS[idx + 1].key; renderView(); e.preventDefault();
    }
  });
}

// 景别三段：点击切换
function bindDistSeg() {
  $("view-d-seg").querySelectorAll(".seg-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      view.distance = btn.dataset.d;
      renderView();
    });
  });
}

function nearestLevel(levels, prop, value) {
  let best = levels[0], bestDist = Infinity;
  for (const lv of levels) {
    const d = Math.abs(lv[prop] - value);
    if (d < bestDist) { best = lv; bestDist = d; }
  }
  return best;
}

// 三控件统一渲染（水平圆盘 / 俯仰滑杆 / 景别按钮）
function renderView() {
  // 水平：相机图标绕主体旋转（顶部=正面，右=右侧视角），虚线针指向主体
  const h = VIEW_H_LEVELS.find((l) => l.key === view.horizontal) || VIEW_H_LEVELS[2];
  $("dial-cam").setAttribute("transform", `rotate(${h.angle} ${DIAL_CX} ${DIAL_CY})`);
  const rad = (h.angle * Math.PI) / 180;
  const sin = Math.sin(rad), cos = Math.cos(rad);
  const needle = $("dial-needle-line");
  needle.setAttribute("x1", (DIAL_CX + 26 * sin).toFixed(1));
  needle.setAttribute("y1", (DIAL_CY - 26 * cos).toFixed(1));
  needle.setAttribute("x2", (DIAL_CX + 8 * sin).toFixed(1));
  needle.setAttribute("y2", (DIAL_CY - 8 * cos).toFixed(1));
  $("view-h-val").textContent = h.label;
  $("dial-h").setAttribute("aria-valuetext", h.label);

  // 俯仰：手柄位置 + 标签高亮
  const v = VIEW_V_LEVELS.find((l) => l.key === view.vertical) || VIEW_V_LEVELS[1];
  $("pitch-handle").style.top = `${(7 + v.pos * 82).toFixed(1)}px`;
  $("view-v-val").textContent = v.label;
  $("pitch-track").setAttribute("aria-valuetext", v.label);
  $("pitch-labels").querySelectorAll("span").forEach((s) => {
    s.classList.toggle("active", s.dataset.key === v.key);
  });

  // 景别
  $("view-d-seg").querySelectorAll(".seg-btn").forEach((b) => {
    b.classList.toggle("active", b.dataset.d === view.distance);
  });
}

// 输入图上传（点击 + 拖拽）
function bindUpload() {
  const dz = $("dropzone");
  const fi = $("file-input");
  dz.addEventListener("click", () => fi.click());
  dz.addEventListener("dragover", (e) => { e.preventDefault(); dz.classList.add("dragover"); });
  dz.addEventListener("dragleave", () => dz.classList.remove("dragover"));
  dz.addEventListener("drop", (e) => {
    e.preventDefault();
    dz.classList.remove("dragover");
    addImages(e.dataTransfer.files);
  });
  fi.addEventListener("change", () => addImages(fi.files));
}

function addImages(fileList) {
  const max = state.mode === "edit" ? 1 : 5;
  for (const f of fileList) {
    if (!f.type.startsWith("image/")) continue;
    if (state.images.length >= max) {
      showError(`输入图已达上限（${state.mode === "edit" ? 1 : 5} 张），多余图片已忽略`);
      break;
    }
    const reader = new FileReader();
    reader.onload = (ev) => {
      state.images.push({ name: f.name, dataUrl: ev.target.result });
      renderThumbs();
    };
    reader.readAsDataURL(f);
  }
}

function renderThumbs() {
  const box = $("thumbs");
  box.innerHTML = "";
  state.images.forEach((img, i) => {
    const div = document.createElement("div");
    div.className = "thumb";
    div.innerHTML = `<img src="${img.dataUrl}"><button class="rm" title="移除">×</button>`;
    div.querySelector(".rm").addEventListener("click", () => {
      state.images.splice(i, 1);
      renderThumbs();
    });
    box.appendChild(div);
  });
}

// ============================================================
// LoRA 区
// ============================================================
async function refreshCapabilities() {
  try {
    const r = await fetch(`${API}/capabilities`);
    const cap = await r.json();
    renderLoraList(cap.loras || []);
  } catch {
    $("lora-status").textContent = "capabilities 获取失败（引擎未启动？）";
  }
}

function renderLoraList(names) {
  const box = $("lora-list");
  if (!names.length) {
    box.innerHTML = `<p class="empty">暂无可用 LoRA（放入 models/image/qwen-image/loras/ 后刷新页面）</p>`;
    return;
  }
  box.innerHTML = "";
  names.forEach((name) => {
    const item = document.createElement("div");
    item.className = "lora-item";
    item.innerHTML = `
      <input type="checkbox" data-name="${name}">
      <div>
        <div class="lora-name">${name}</div>
        <div class="lora-tag" data-tag>未挂载</div>
      </div>
      <div style="display:flex;align-items:center;gap:6px">
        <input type="range" min="0" max="1.5" step="0.05" value="0.8" data-w-slider>
        <span class="lora-w" data-w-val>0.8</span>
      </div>`;
    const slider = item.querySelector("[data-w-slider]");
    slider.addEventListener("input", () => { item.querySelector("[data-w-val]").textContent = slider.value; });
    box.appendChild(item);
  });
}

function bindButtons() {
  $("btn-refresh-status").addEventListener("click", refreshStatus);
  $("btn-apply-loras").addEventListener("click", applyLoras);
  $("btn-clear-loras").addEventListener("click", clearLoras);
  $("btn-generate").addEventListener("click", generate);
}

function selectedLoras() {
  const out = [];
  document.querySelectorAll(".lora-item").forEach((item) => {
    const cb = item.querySelector("input[type=checkbox]");
    if (cb.checked) {
      out.push({ name: cb.dataset.name, weight: parseFloat(item.querySelector("[data-w-val]").textContent) });
    }
  });
  return out;
}

async function applyLoras() {
  const target = selectedLoras();
  // 先卸载当前已挂但未选中的
  for (const item of document.querySelectorAll(".lora-item")) {
    const name = item.querySelector("input[type=checkbox]").dataset.name;
    const tag = item.querySelector("[data-tag]");
    if (tag.textContent === "已挂载" && !target.some((t) => t.name === name)) {
      await fetch(`${API}/lora/unload`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name }),
      });
      tag.textContent = "未挂载";
    }
  }
  // 再加载选中的
  for (const t of target) {
    try {
      const r = await fetch(`${API}/lora/load`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name: t.name, weight: t.weight }),
      });
      if (!r.ok) {
        const e = await r.json().catch(() => ({}));
        throw new Error(e.detail || r.statusText);
      }
      const item = document.querySelector(`.lora-item input[data-name="${CSS.escape(t.name)}"]`)?.closest(".lora-item");
      if (item) item.querySelector("[data-tag]").textContent = "已挂载";
    } catch (err) {
      showError(`LoRA「${t.name}」挂载失败：${err.message}`);
    }
  }
  markLoadedLoras();
  $("lora-status").textContent = `已应用 ${target.length} 个 LoRA`;
}

async function clearLoras() {
  const r = await fetch(`${API}/status`).catch(() => null);
  const st = r ? await r.json() : {};
  for (const name of st.loaded_loras || []) {
    await fetch(`${API}/lora/unload`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name }),
    });
  }
  document.querySelectorAll("[data-tag]").forEach((t) => (t.textContent = "未挂载"));
  document.querySelectorAll(".lora-item input[type=checkbox]").forEach((c) => (c.checked = false));
  $("lora-status").textContent = "已全部卸载";
}

async function markLoadedLoras() {
  const r = await fetch(`${API}/status`).catch(() => null);
  if (!r) return;
  const st = await r.json();
  const loaded = st.loaded_loras || [];
  document.querySelectorAll(".lora-item").forEach((item) => {
    const name = item.querySelector("input[type=checkbox]").dataset.name;
    item.querySelector("[data-tag]").textContent = loaded.includes(name) ? "已挂载" : "未挂载";
  });
}

// ============================================================
// 状态轮询
// ============================================================
async function refreshStatus() {
  if (state.generating) return;
  try {
    const r = await fetch(`${API}/status`);
    const st = await r.json();
    const badge = $("state-badge");
    badge.className = "badge " + (st.state || "");
    badge.textContent = {
      idle: "空闲", loading: "加载中", ready: "就绪",
      inferring: "推理中", error: "异常",
    }[st.state] || st.state;
    $("state-detail").textContent =
      `档位 ${precisionLabel((st.extra && st.extra.precision) || st.precision) || "—"} · 显存 ${st.vram_used || "—"} · LoRA ${st.loaded_loras?.length || 0} 个`;
    // 下拉框自动跟随实际档位（用户选定具体档位后不覆盖；引擎热切换后 UI 同步）
    if (!state.precisionTouched) {
      const cur = (st.extra && st.extra.precision) || st.precision;
      const sel = $("precision-select");
      if (cur && sel.value !== cur && [...sel.options].some((o) => o.value === cur)) {
        sel.value = cur;
      }
    }
    if (st.state === "ready") markLoadedLoras();
  } catch {
    $("state-badge").className = "badge";
    $("state-badge").textContent = "未连接";
    $("state-detail").textContent = "引擎不可达（127.0.0.1:7007）";
  }
}

// ============================================================
// 结果区：生成（SSE 流）
// ============================================================
async function generate() {
  if (state.generating) return;

  // 校验
  const prompt = $("prompt").value.trim();
  if (!prompt) { showError("请输入提示词"); return; }
  if (state.mode === "edit" && state.images.length !== 1) { showError("指令编辑模式需要恰好 1 张输入图"); return; }
  if (state.mode === "i2i" && (state.images.length < 2 || state.images.length > 5)) {
    showError("多图参考模式需要 2-5 张输入图（官方推荐 ≤3 张，超过指令遵循度下降）"); return;
  }

  const inputs = {
    prompt,
    mode: state.mode,
    width: parseInt($("width").value, 10) || null,
    height: parseInt($("height").value, 10) || null,
    num_inference_steps: parseInt($("steps").value, 10),
    true_cfg_scale: parseFloat($("true-cfg").value),
    seed: parseInt($("seed").value, 10),
  };
  // edit 模式视角控制（任一维度偏离默认才传，全默认 = 纯指令编辑）
  if (state.mode === "edit") {
    const v = { horizontal: view.horizontal, vertical: view.vertical, distance: view.distance };
    if (!(v.horizontal === "front" && v.vertical === "level" && v.distance === "medium")) {
      inputs.view = v;
    }
  }
  const precision = $("precision-select").value;
  if (precision) inputs.precision = precision;

  // 不携带 task_id：直连链路契约（engine.py 以 request.task_id 是否非空区分
  // gateway 转发/直连）。带自造 task_id 会被误判为转发链路 → 任务中心只落
  // created 不更新终态（状态永远停在 queued）。引擎会自动生成 task_id。
  const body = {
    inputs,
  };
  if (state.mode !== "t2i") {
    body.images = state.images.map((img) => {
      // dataUrl: data:image/png;base64,xxxx → b64_json
      const b64 = img.dataUrl.split(",")[1];
      return { b64_json: b64 };
    });
  }

  // UI 状态
  state.generating = true;
  $("btn-generate").disabled = true;
  $("btn-generate").textContent = "生成中…";
  $("progress-wrap").style.display = "flex";
  $("progress-fill").style.width = "0%";
  $("progress-text").textContent = "连接引擎…";
  $("error-box").style.display = "none";
  $("state-badge").className = "badge inferring";
  $("state-badge").textContent = "推理中";

  try {
    const resp = await fetch(`${API}/infer`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!resp.ok) {
      const e = await resp.json().catch(() => ({}));
      throw new Error(e.detail || `HTTP ${resp.status}`);
    }

    // 解析 SSE 流（fetch ReadableStream 手动读 data: 行）
    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buf = "";
    let finalResult = null;

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      const lines = buf.split("\n");
      buf = lines.pop();  // 末行可能不完整，留在缓冲
      for (const line of lines) {
        if (!line.startsWith("data:")) continue;
        let frame;
        try { frame = JSON.parse(line.slice(5)); } catch { continue; }
        handleFrame(frame);
        if (frame.stage === "completed" && frame.result) finalResult = frame.result;
      }
    }

    if (finalResult) {
      showResult(finalResult);
    } else {
      throw new Error("SSE 流结束但未收到结果帧");
    }
  } catch (err) {
    showError(`生成失败：${err.message}`);
  } finally {
    state.generating = false;
    $("btn-generate").disabled = false;
    $("btn-generate").textContent = "开始生成";
    $("progress-wrap").style.display = "none";
    refreshStatus();
    refreshHistory();
  }
}

function handleFrame(f) {
  const pct = Math.round((f.progress || 0) * 100);
  $("progress-fill").style.width = `${pct}%`;
  const stageText = {
    staging_images: "加载输入图",
    model_loading: "模型加载中",
    preparing: "准备中",
    denoising: "采样去噪",
    completed: "完成",
  }[f.stage] || f.stage;
  $("progress-text").textContent = `${stageText} ${pct}%${f.detail ? " · " + f.detail : ""}`;
  if (f.stage === "error") {
    throw new Error(f.error?.message || "引擎错误");
  }
}

function showResult(result) {
  const view = $("result-view");
  const url = result.image_url + "?t=" + Date.now();  // 防缓存
  view.innerHTML = `<img src="${url}" alt="生成结果" title="点击全屏查看">`;
  // 点击结果图 → 全屏
  view.querySelector("img").addEventListener("click", () => openLightbox(url));
  const meta = $("result-meta");
  meta.innerHTML = `
    <span class="meta-chip">模式 ${result.mode}</span>
    <span class="meta-chip">${result.width}×${result.height}</span>
    <span class="meta-chip">档位 ${result.precision}</span>
    <span class="meta-chip">seed ${result.seed}</span>
    <span class="meta-chip">${result.infer_seconds}s 采样 / ${result.duration_seconds}s 总计</span>
    <span class="meta-chip"><a href="${url}" download>下载</a></span>
    <span class="meta-chip meta-chip-btn" id="btn-fullscreen" title="全屏查看大图">🔍 全屏查看</span>`;
  $("btn-fullscreen").addEventListener("click", () => openLightbox(url));
}

// ============================================================
// 结果区 / 历史图：全屏查看（lightbox）
// ============================================================
function openLightbox(url) {
  $("lightbox-img").src = url;
  $("lightbox").style.display = "flex";
  document.body.style.overflow = "hidden";  // 锁定背景滚动
}

function closeLightbox() {
  $("lightbox").style.display = "none";
  $("lightbox-img").src = "";  // 释放大图内存
  document.body.style.overflow = "";
}

function bindLightbox() {
  $("lightbox").addEventListener("click", (e) => {
    if (e.target === $("lightbox") || e.target.id === "lightbox-close") closeLightbox();
  });
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && $("lightbox").style.display !== "none") closeLightbox();
  });
}

async function refreshHistory() {
  try {
    const r = await fetch(`${API}/history?page=1&page_size=24`);
    const h = await r.json();
    const grid = $("history-grid");
    grid.innerHTML = "";
    // 降级提示（gateway 不可达时历史视图不可用，生成功能不受影响）
    if (h.degraded) {
      grid.innerHTML = `<div class="history-degraded">平台网关不可达，暂时无法加载历史（生成功能不受影响）</div>`;
      return;
    }
    for (const item of h.items || []) {
      const div = document.createElement("div");
      div.className = "history-item";
      div.title = `${item.prompt?.slice(0, 60) || ""} · seed=${item.seed ?? "—"}\n单击载入结果区 · 双击全屏查看`;
      div.innerHTML = `<img src="${item.image_url}" loading="lazy"><span class="h-mode">${item.mode}</span>`;
      div.addEventListener("click", () => {
        $("result-view").innerHTML = `<img src="${item.image_url}" alt="${item.task_id}" title="点击全屏查看">`;
        // 点击结果区历史图也可全屏
        $("result-view").querySelector("img").addEventListener("click", () => openLightbox(item.image_url));
        $("result-meta").innerHTML = `
          <span class="meta-chip">模式 ${item.mode}</span>
          <span class="meta-chip">${item.width ?? "—"}×${item.height ?? "—"}</span>
          <span class="meta-chip">seed ${item.seed ?? "—"}</span>
          <span class="meta-chip">${new Date(item.created_at * 1000).toLocaleString()}</span>
          <span class="meta-chip meta-chip-btn" id="btn-fullscreen" title="全屏查看大图">🔍 全屏查看</span>`;
        $("btn-fullscreen").addEventListener("click", () => openLightbox(item.image_url));
      });
      // 双击历史缩略图 → 直接全屏
      div.addEventListener("dblclick", () => openLightbox(item.image_url));
      grid.appendChild(div);
    }
  } catch { /* 引擎未启动时忽略 */ }
}

// ---------- 错误提示（不自动消失，手动关闭） ----------
function showError(msg) {
  $("error-text").textContent = msg;
  $("error-box").style.display = "block";
}
