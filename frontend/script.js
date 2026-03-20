const fileInput = document.getElementById("fileInput");
const dropzone = document.getElementById("dropzone");
const uploadBtn = document.getElementById("uploadBtn");
const clearBtn = document.getElementById("clearBtn");
const statusEl = document.getElementById("status");
const previewEl = document.getElementById("preview");
const processedPreviewEl = document.getElementById("processedPreview");
const processedPreviewCardEl = document.getElementById("processedPreviewCard");
const previewModeSelect = document.getElementById("previewMode");
const tbody = document.getElementById("tbody");
const thead = document.getElementById("thead");
const addRowBtn = document.getElementById("addRowBtn");
const exportBtn = document.getElementById("exportBtn");
const downloadEl = document.getElementById("download");
const thresholdInput = document.getElementById("threshold");
const thresholdValueEl = document.getElementById("thresholdValue");
const fileMetaEl = document.getElementById("fileMeta");
const lowCountEl = document.getElementById("lowCount");
const riskColumnSelect = document.getElementById("riskColumn");
const resultTitleEl = document.getElementById("resultTitle");
const resultMetaEl = document.getElementById("resultMeta");

let selectedFile = null;
let currentHeaders = [];
let threshold = Number(thresholdInput?.value ?? 0.85);
let currentTitle = "";
let currentMeta = {};
let currentHeaderBoxes = {};
let currentProcessedPreview = "";
let currentHighlightShape = null;
let previewMode = previewModeSelect?.value || "original";

let previewImageEl = null;
let previewOverlayEl = null;
let processedPreviewImageEl = null;
let processedPreviewOverlayEl = null;

function setStatus(text, type = "") {
  statusEl.textContent = text;
  statusEl.classList.remove("ok", "error");
  if (type) statusEl.classList.add(type);
}

function setLoading(btn, loading) {
  if (!btn) return;
  if (loading) {
    btn.dataset.loading = "1";
  } else {
    delete btn.dataset.loading;
  }
}

function formatBytes(bytes) {
  const n = Number(bytes ?? 0);
  if (!Number.isFinite(n) || n <= 0) return "0 B";
  const units = ["B", "KB", "MB", "GB"];
  let value = n;
  let unitIndex = 0;
  while (value >= 1024 && unitIndex < units.length - 1) {
    value /= 1024;
    unitIndex += 1;
  }
  return `${value.toFixed(value >= 10 || unitIndex === 0 ? 0 : 1)} ${units[unitIndex]}`;
}

function setFileMeta(file) {
  if (!fileMetaEl) return;
  if (!file) {
    fileMetaEl.textContent = "";
    return;
  }
  fileMetaEl.textContent = `已选择：${file.name}（${formatBytes(file.size)}）`;
}

function syncPreviewModeUI() {
  const hasProcessedPreview = Boolean(currentProcessedPreview);
  if (processedPreviewCardEl) {
    processedPreviewCardEl.classList.toggle(
      "hidden",
      !(previewMode === "processed" && hasProcessedPreview)
    );
  }
}

function refreshLowConfidenceSummary() {
  if (!lowCountEl) return;
  const count = document.querySelectorAll(".cell-editable.low").length;
  lowCountEl.textContent = count > 0 ? `（当前标红：${count} 个）` : "";
}

function updateRowNumbers() {
  const rows = tbody.querySelectorAll("tr[data-row='1']");
  rows.forEach((tr, index) => {
    const cell = tr.querySelector("td[data-col='__index__']");
    if (cell) cell.textContent = String(index + 1);
  });
}

function renderMeta(meta = {}) {
  currentMeta = meta && typeof meta === "object" ? meta : {};
  if (!resultMetaEl) return;

  const entries = Object.entries(currentMeta).filter(([key, value]) => {
    if (key === "标题") return false;
    return String(value ?? "").trim() !== "";
  });

  if (entries.length === 0) {
    resultMetaEl.innerHTML = "";
    return;
  }

  resultMetaEl.innerHTML = entries
    .map(
      ([key, value]) => `
        <div class="meta-item">
          <span>${key}</span>
          <strong>${String(value)}</strong>
        </div>
      `
    )
    .join("");
}

function clearPreviewHighlight() {
  currentHighlightShape = null;
  if (previewOverlayEl) previewOverlayEl.innerHTML = "";
  if (processedPreviewOverlayEl) processedPreviewOverlayEl.innerHTML = "";
}

function renderImagePreview(container, src, alt, onLoad) {
  container.innerHTML = "";

  const stage = document.createElement("div");
  stage.className = "preview-stage";

  const img = document.createElement("img");
  img.src = src;
  img.alt = alt;
  if (onLoad) img.addEventListener("load", onLoad);

  const overlay = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  overlay.classList.add("preview-overlay");

  stage.appendChild(img);
  stage.appendChild(overlay);
  container.appendChild(stage);
  return { img, overlay };
}

function renderPreview(file) {
  previewEl.innerHTML = "";
  previewImageEl = null;
  previewOverlayEl = null;
  clearPreviewHighlight();

  if (!file) {
    previewEl.innerHTML = `<div class="preview-placeholder">等待上传</div>`;
    return;
  }

  const name = (file.name || "").toLowerCase();
  const url = URL.createObjectURL(file);

  if (name.endsWith(".png") || name.endsWith(".jpg") || name.endsWith(".jpeg")) {
    const rendered = renderImagePreview(previewEl, url, "原图预览", () => {
      if (currentHighlightShape && previewMode === "original") {
        showPreviewHighlight(currentHighlightShape);
      }
    });
    previewImageEl = rendered.img;
    previewOverlayEl = rendered.overlay;
    return;
  }

  if (name.endsWith(".svg")) {
    const obj = document.createElement("object");
    obj.type = "image/svg+xml";
    obj.data = url;
    previewEl.appendChild(obj);
    return;
  }

  if (name.endsWith(".xlsx")) {
    previewEl.innerHTML = `<div class="preview-placeholder">XLSX 无法直接预览，将直接解析表格数据</div>`;
    return;
  }

  previewEl.innerHTML = `<div class="preview-placeholder">暂不支持该文件预览</div>`;
}

function renderProcessedPreview(dataUrl) {
  currentProcessedPreview = dataUrl || "";
  if (!processedPreviewEl) return;

  processedPreviewEl.innerHTML = "";
  processedPreviewImageEl = null;
  processedPreviewOverlayEl = null;

  if (!dataUrl) {
    processedPreviewEl.innerHTML = `<div class="preview-placeholder">等待识别</div>`;
    syncPreviewModeUI();
    return;
  }

  const rendered = renderImagePreview(processedPreviewEl, dataUrl, "预处理后预览", () => {
    if (currentHighlightShape && previewMode === "processed") {
      showPreviewHighlight(currentHighlightShape);
    }
  });
  processedPreviewImageEl = rendered.img;
  processedPreviewOverlayEl = rendered.overlay;
  syncPreviewModeUI();
}

function drawPolygonOnPreview(targetContainer, targetImage, targetOverlay, points) {
  const containerWidth = targetContainer.clientWidth;
  const containerHeight = targetContainer.clientHeight;
  const naturalWidth = targetImage.naturalWidth || 0;
  const naturalHeight = targetImage.naturalHeight || 0;

  if (!containerWidth || !containerHeight || !naturalWidth || !naturalHeight) {
    targetOverlay.innerHTML = "";
    return;
  }

  const scale = Math.min(containerWidth / naturalWidth, containerHeight / naturalHeight);
  const displayWidth = naturalWidth * scale;
  const displayHeight = naturalHeight * scale;
  const offsetX = (containerWidth - displayWidth) / 2;
  const offsetY = (containerHeight - displayHeight) / 2;

  targetOverlay.setAttribute("viewBox", `0 0 ${containerWidth} ${containerHeight}`);
  targetOverlay.setAttribute("width", `${containerWidth}`);
  targetOverlay.setAttribute("height", `${containerHeight}`);
  targetOverlay.innerHTML = "";

  const polygon = document.createElementNS("http://www.w3.org/2000/svg", "polygon");
  const pointsAttr = points
    .map((point) => `${offsetX + point.x * displayWidth},${offsetY + point.y * displayHeight}`)
    .join(" ");
  polygon.setAttribute("points", pointsAttr);
  polygon.setAttribute("class", "preview-polygon");
  targetOverlay.appendChild(polygon);
}

function showPreviewHighlight(shape) {
  currentHighlightShape = shape || null;
  if (previewOverlayEl) previewOverlayEl.innerHTML = "";
  if (processedPreviewOverlayEl) processedPreviewOverlayEl.innerHTML = "";
  if (!shape) return;

  const targetImage = previewMode === "processed" ? processedPreviewImageEl : previewImageEl;
  const targetOverlay = previewMode === "processed" ? processedPreviewOverlayEl : previewOverlayEl;
  const targetContainer = previewMode === "processed" ? processedPreviewEl : previewEl;
  const pointsKey = previewMode === "processed" ? "processed_points" : "points";
  const points = Array.isArray(shape?.[pointsKey]) ? shape[pointsKey] : null;

  if (!targetImage || !targetOverlay || !targetContainer || !points || points.length === 0) {
    return;
  }

  drawPolygonOnPreview(targetContainer, targetImage, targetOverlay, points);
}

function applyLowConfidenceStyle(el, confidence) {
  el.classList.remove("low");
  if (confidence != null && confidence < threshold) {
    el.classList.add("low");
  }
}

function makeEditableCell(text, field, confidence, box) {
  const span = document.createElement("span");
  span.className = "cell-editable";
  span.contentEditable = "true";
  span.dataset.field = field;
  span.dataset.confidence = String(confidence ?? 1);
  if (box) span.dataset.box = JSON.stringify(box);
  span.textContent = text ?? "";

  span.addEventListener("input", () => {
    span.dataset.confidence = "1";
    applyLowConfidenceStyle(span, 1);
    refreshLowConfidenceSummary();
    updateRiskColumns({ preserveSelection: true });
  });

  applyLowConfidenceStyle(span, confidence);
  return span;
}

function addRow(values = {}, confidences = {}, boxes = {}) {
  const tr = document.createElement("tr");
  tr.dataset.row = "1";

  const tdIndex = document.createElement("td");
  tdIndex.className = "col-index";
  tdIndex.dataset.col = "__index__";
  tdIndex.textContent = "1";
  tr.appendChild(tdIndex);

  for (const header of currentHeaders) {
    const td = document.createElement("td");
    td.dataset.col = header;
    td.appendChild(
      makeEditableCell(
        String(values[header] ?? ""),
        header,
        Number(confidences[header] ?? 1),
        boxes[header] ?? null
      )
    );
    tr.appendChild(td);
  }

  const tdOp = document.createElement("td");
  tdOp.dataset.col = "__action__";
  const delBtn = document.createElement("button");
  delBtn.className = "btn";
  delBtn.textContent = "删除";
  delBtn.addEventListener("click", () => {
    tr.remove();
    if (tbody.querySelectorAll("tr[data-row='1']").length === 0) {
      const emptyRow = document.createElement("tr");
      emptyRow.className = "empty";
      emptyRow.innerHTML = `<td colspan="${currentHeaders.length + 2}">暂无数据</td>`;
      tbody.appendChild(emptyRow);
    }
    refreshLowConfidenceSummary();
    updateRowNumbers();
    updateRiskColumns({ preserveSelection: true });
    applyColumnFilter();
  });
  tdOp.appendChild(delBtn);
  tr.appendChild(tdOp);

  const empty = tbody.querySelector("tr.empty");
  if (empty) empty.remove();

  tbody.appendChild(tr);
  refreshLowConfidenceSummary();
  updateRowNumbers();
  updateRiskColumns({ preserveSelection: true });
  applyColumnFilter();
}

function clearTable() {
  thead.innerHTML = `
    <tr>
      <th class="col-index" data-col="__index__">序号</th>
      <th>内容</th>
      <th>操作</th>
    </tr>
  `;
  tbody.innerHTML = `
    <tr class="empty">
      <td colspan="3">尚无数据，请先上传并识别</td>
    </tr>
  `;
  currentHeaders = [];
  currentTitle = "";
  currentMeta = {};
  currentHeaderBoxes = {};
  currentProcessedPreview = "";
  if (resultTitleEl) resultTitleEl.textContent = "";
  if (resultMetaEl) resultMetaEl.innerHTML = "";
  if (processedPreviewEl) {
    processedPreviewEl.innerHTML = `<div class="preview-placeholder">等待识别</div>`;
  }
  processedPreviewImageEl = null;
  processedPreviewOverlayEl = null;
  syncPreviewModeUI();
  clearPreviewHighlight();
  refreshLowConfidenceSummary();
  updateRiskColumns({ preserveSelection: false });
}

function renderTranscript(data) {
  const headers = Array.isArray(data.headers) ? data.headers : [];
  const rows = Array.isArray(data.rows) ? data.rows : [];
  const title = typeof data.title === "string" ? data.title.trim() : "";
  const meta = data.meta && typeof data.meta === "object" ? data.meta : {};
  const headerBoxes = data.header_boxes && typeof data.header_boxes === "object" ? data.header_boxes : {};
  const processedPreview = typeof data.processed_preview === "string" ? data.processed_preview : "";

  if (headers.length === 0) {
    clearTable();
    setStatus("未识别到有效的表格结构", "error");
    return;
  }

  currentHeaders = headers;
  currentTitle = title;
  currentHeaderBoxes = headerBoxes;

  if (resultTitleEl) resultTitleEl.textContent = title || "";
  renderMeta(meta);
  renderProcessedPreview(processedPreview);

  let thHtml = `<th class="col-index" data-col="__index__">序号</th>`;
  for (const header of headers) {
    const box = headerBoxes[header] ? ` data-box='${JSON.stringify(headerBoxes[header])}'` : "";
    thHtml += `<th data-col="${header}"${box}>${header}</th>`;
  }
  thHtml += `<th data-col="__action__">操作</th>`;
  thead.innerHTML = `<tr>${thHtml}</tr>`;

  tbody.innerHTML = "";
  if (rows.length === 0) {
    tbody.innerHTML = `<tr class="empty"><td colspan="${headers.length + 2}">未识别到数据行</td></tr>`;
  } else {
    for (const row of rows) {
      addRow(row.values || {}, row.confidences || {}, row.boxes || {});
    }
  }

  exportBtn.disabled = false;
  downloadEl.innerHTML = "";
  refreshLowConfidenceSummary();
  updateRowNumbers();
  updateRiskColumns({ preserveSelection: false });
  applyColumnFilter();
}

async function uploadFile(file) {
  const form = new FormData();
  form.append("file", file);

  const res = await fetch("/upload", { method: "POST", body: form });
  if (!res.ok) {
    const detail = await res.json().catch(() => null);
    throw new Error(detail?.detail || `HTTP ${res.status}`);
  }
  return res.json();
}

async function exportExcel(payload) {
  const res = await fetch("/export", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    const detail = await res.json().catch(() => null);
    throw new Error(detail?.detail || `HTTP ${res.status}`);
  }
  return res.json();
}

function buildPayloadFromUI() {
  const rows = [...tbody.querySelectorAll("tr[data-row='1']")];
  return {
    title: currentTitle,
    meta: currentMeta,
    headers: currentHeaders,
    rows: rows.map((tr) => {
      const values = {};
      const confidences = {};
      const boxes = {};
      tr.querySelectorAll(".cell-editable").forEach((span) => {
        const field = span.dataset.field;
        values[field] = span.textContent.trim();
        confidences[field] = Number(span.dataset.confidence ?? 1);
        if (span.dataset.box) {
          boxes[field] = JSON.parse(span.dataset.box);
        }
      });
      return { values, confidences, boxes };
    }),
  };
}

function isSupportedFile(file) {
  const name = (file?.name || "").toLowerCase();
  return (
    name.endsWith(".png") ||
    name.endsWith(".jpg") ||
    name.endsWith(".jpeg") ||
    name.endsWith(".svg") ||
    name.endsWith(".xlsx")
  );
}

function setSelectedFile(file) {
  selectedFile = file;
  uploadBtn.disabled = !selectedFile;
  clearBtn.disabled = !selectedFile;
  exportBtn.disabled = true;
  downloadEl.innerHTML = "";
  setFileMeta(selectedFile);
  renderPreview(selectedFile);
  clearTable();
  if (selectedFile) {
    setStatus("已选择文件，点击“开始识别”进行处理");
  } else {
    setStatus("请选择 PNG/JPG/SVG/XLSX 文件");
  }
}

function updateRiskColumns({ preserveSelection }) {
  if (!riskColumnSelect) return;
  const currentValue = riskColumnSelect.value;
  const counts = new Map();

  currentHeaders.forEach((header) => counts.set(header, 0));
  tbody.querySelectorAll("tr[data-row='1']").forEach((tr) => {
    tr.querySelectorAll(".cell-editable").forEach((span) => {
      const field = span.dataset.field;
      const confidence = Number(span.dataset.confidence ?? 1);
      if (field && confidence < threshold) {
        counts.set(field, (counts.get(field) || 0) + 1);
      }
    });
  });

  const options = [];
  counts.forEach((count, header) => {
    if (count > 0) options.push({ header, count });
  });
  options.sort((a, b) => b.count - a.count);

  riskColumnSelect.innerHTML = `<option value="">全部列</option>`;
  options.forEach((item) => {
    const option = document.createElement("option");
    option.value = item.header;
    option.textContent = `${item.header}（${item.count}）`;
    riskColumnSelect.appendChild(option);
  });

  if (preserveSelection && currentValue) {
    const exists = [...riskColumnSelect.options].some((option) => option.value === currentValue);
    riskColumnSelect.value = exists ? currentValue : "";
  }

  applyColumnFilter();
}

function applyColumnFilter() {
  if (!riskColumnSelect) return;
  const target = riskColumnSelect.value;
  document.querySelectorAll("[data-col]").forEach((el) => {
    const col = el.dataset.col;
    if (!target || col === "__index__" || col === "__action__" || col === target) {
      el.classList.remove("col-hidden");
    } else {
      el.classList.add("col-hidden");
    }
  });
}

fileInput?.addEventListener("change", () => {
  const file = fileInput.files?.[0] || null;
  setSelectedFile(file);
});

dropzone?.addEventListener("click", () => fileInput.click());
dropzone?.addEventListener("keydown", (event) => {
  if (event.key === "Enter" || event.key === " ") {
    event.preventDefault();
    fileInput.click();
  }
});

dropzone?.addEventListener("dragover", (event) => {
  event.preventDefault();
  dropzone.classList.add("dragover");
});

dropzone?.addEventListener("dragleave", () => {
  dropzone.classList.remove("dragover");
});

dropzone?.addEventListener("drop", (event) => {
  event.preventDefault();
  dropzone.classList.remove("dragover");
  const file = event.dataTransfer?.files?.[0];
  if (!file) return;
  if (!isSupportedFile(file)) {
    setSelectedFile(null);
    setStatus("不支持的文件类型，仅支持 PNG/JPG/SVG/XLSX", "error");
    return;
  }
  setSelectedFile(file);
});

clearBtn?.addEventListener("click", () => {
  fileInput.value = "";
  setSelectedFile(null);
});

thresholdInput?.addEventListener("input", () => {
  threshold = Number(thresholdInput.value);
  if (thresholdValueEl) thresholdValueEl.textContent = threshold.toFixed(2);
  document.querySelectorAll(".cell-editable").forEach((el) => {
    applyLowConfidenceStyle(el, Number(el.dataset.confidence ?? 1));
  });
  refreshLowConfidenceSummary();
  updateRiskColumns({ preserveSelection: true });
});

previewModeSelect?.addEventListener("change", () => {
  previewMode = previewModeSelect.value || "original";
  syncPreviewModeUI();
  if (currentHighlightShape) {
    showPreviewHighlight(currentHighlightShape);
  } else {
    clearPreviewHighlight();
  }
});

uploadBtn?.addEventListener("click", async () => {
  if (!selectedFile) return;
  uploadBtn.disabled = true;
  clearBtn.disabled = true;
  setLoading(uploadBtn, true);
  setStatus("识别中，请稍候（首次加载 OCR 模型可能较慢）");

  try {
    const data = await uploadFile(selectedFile);
    renderTranscript(data);
    setStatus("识别完成，可在右侧表格中修改后导出", "ok");
  } catch (error) {
    setStatus(`识别失败：${error.message || error}`, "error");
    clearTable();
    exportBtn.disabled = true;
  } finally {
    uploadBtn.disabled = false;
    clearBtn.disabled = !selectedFile;
    setLoading(uploadBtn, false);
  }
});

addRowBtn?.addEventListener("click", () => {
  if (currentHeaders.length === 0) {
    alert("请先上传文件并识别出表头结构后再添加行");
    return;
  }
  addRow({}, {}, {});
});

exportBtn?.addEventListener("click", async () => {
  exportBtn.disabled = true;
  setLoading(exportBtn, true);
  downloadEl.innerHTML = "";
  setStatus("导出中");

  try {
    const res = await exportExcel(buildPayloadFromUI());
    if (!res.download_url) throw new Error("后端未返回下载链接");
    downloadEl.innerHTML = `导出成功：<a href="${res.download_url}" target="_blank" rel="noopener">点击下载 Excel</a>`;
    setStatus("导出完成", "ok");
  } catch (error) {
    setStatus(`导出失败：${error.message || error}`, "error");
  } finally {
    exportBtn.disabled = false;
    setLoading(exportBtn, false);
  }
});

riskColumnSelect?.addEventListener("change", () => {
  applyColumnFilter();
});

tbody?.addEventListener("click", (event) => {
  const target = event.target.closest(".cell-editable");
  if (!target) return;
  showPreviewHighlight(target.dataset.box ? JSON.parse(target.dataset.box) : null);
});

tbody?.addEventListener("focusin", (event) => {
  const target = event.target.closest(".cell-editable");
  if (!target) return;
  showPreviewHighlight(target.dataset.box ? JSON.parse(target.dataset.box) : null);
});

thead?.addEventListener("click", (event) => {
  const target = event.target.closest("th[data-box]");
  if (!target) return;
  showPreviewHighlight(target.dataset.box ? JSON.parse(target.dataset.box) : null);
});

window.addEventListener("resize", () => {
  if (currentHighlightShape) {
    showPreviewHighlight(currentHighlightShape);
  }
});

renderPreview(null);
clearTable();
setFileMeta(null);
if (thresholdValueEl) thresholdValueEl.textContent = threshold.toFixed(2);
syncPreviewModeUI();
