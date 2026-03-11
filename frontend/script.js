const fileInput = document.getElementById("fileInput");
const dropzone = document.getElementById("dropzone");
const uploadBtn = document.getElementById("uploadBtn");
const clearBtn = document.getElementById("clearBtn");
const statusEl = document.getElementById("status");
const previewEl = document.getElementById("preview");
const tbody = document.getElementById("tbody");
const thead = document.getElementById("thead");
const addRowBtn = document.getElementById("addRowBtn");
const exportBtn = document.getElementById("exportBtn");
const downloadEl = document.getElementById("download");
const thresholdInput = document.getElementById("threshold");
const thresholdValueEl = document.getElementById("thresholdValue");
const fileMetaEl = document.getElementById("fileMeta");
const lowCountEl = document.getElementById("lowCount");

let selectedFile = null;
let currentHeaders = []; // 存储当前表格的表头 ["姓名", "成绩", ...]
let threshold = Number(thresholdInput?.value ?? 0.85);

function setStatus(text, type) {
  statusEl.textContent = text;
  statusEl.classList.remove("ok", "error");
  if (type) statusEl.classList.add(type);
}

function setLoading(btn, loading) {
  if (!btn) return;
  if (loading) btn.dataset.loading = "1";
  else delete btn.dataset.loading;
}

function formatBytes(bytes) {
  const n = Number(bytes ?? 0);
  if (!Number.isFinite(n) || n <= 0) return "0 B";
  const units = ["B", "KB", "MB", "GB"];
  let v = n;
  let i = 0;
  while (v >= 1024 && i < units.length - 1) {
    v /= 1024;
    i += 1;
  }
  return `${v.toFixed(v >= 10 || i === 0 ? 0 : 1)} ${units[i]}`;
}

function setFileMeta(file) {
  if (!fileMetaEl) return;
  if (!file) {
    fileMetaEl.textContent = "";
    return;
  }
  fileMetaEl.textContent = `已选择：${file.name}（${formatBytes(file.size)}）`;
}

function refreshLowConfidenceSummary() {
  if (!lowCountEl) return;
  const cells = document.querySelectorAll(".cell-editable.low");
  const n = cells.length;
  lowCountEl.textContent = n > 0 ? `（当前标红：${n} 个）` : "";
}

function clearTable() {
  thead.innerHTML = `
    <tr>
      <th>内容</th>
      <th>操作</th>
    </tr>
  `;
  tbody.innerHTML = `
    <tr class="empty">
      <td colspan="2">尚无数据，请先上传并识别</td>
    </tr>
  `;
  currentHeaders = [];
  refreshLowConfidenceSummary();
}

function renderPreview(file) {
  previewEl.innerHTML = "";

  if (!file) {
    previewEl.innerHTML = `<div class="preview-placeholder">等待上传</div>`;
    return;
  }

  const name = (file.name || "").toLowerCase();
  const url = URL.createObjectURL(file);

  if (name.endsWith(".png") || name.endsWith(".jpg") || name.endsWith(".jpeg")) {
    const img = document.createElement("img");
    img.src = url;
    img.alt = "预览";
    previewEl.appendChild(img);
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

  previewEl.innerHTML = `<div class="preview-placeholder">不支持预览</div>`;
}

function applyLowConfidenceStyle(el, conf) {
  el.classList.remove("low");
  if (conf != null && conf < threshold) el.classList.add("low");
}

function makeEditableCell(text, field, confidence) {
  const span = document.createElement("span");
  span.className = "cell-editable";
  span.contentEditable = "true";
  span.dataset.field = field;
  span.dataset.confidence = String(confidence ?? 1);
  span.textContent = text ?? "";

  span.addEventListener("input", () => {
    span.dataset.confidence = "1";
    applyLowConfidenceStyle(span, 1);
    refreshLowConfidenceSummary();
  });

  applyLowConfidenceStyle(span, confidence);
  return span;
}

function addRow(values = {}, confidences = {}) {
  const tr = document.createElement("tr");
  tr.dataset.row = "1";

  // 根据 currentHeaders 生成对应的单元格
  for (const header of currentHeaders) {
    const td = document.createElement("td");
    const val = values[header] ?? "";
    const conf = confidences[header] ?? 1.0;
    
    td.appendChild(makeEditableCell(String(val), header, Number(conf)));
    tr.appendChild(td);
  }

  // 操作列
  const tdOp = document.createElement("td");
  const delBtn = document.createElement("button");
  delBtn.className = "btn";
  delBtn.textContent = "删除";
  delBtn.addEventListener("click", () => {
    tr.remove();
    if (tbody.querySelectorAll("tr[data-row='1']").length === 0) {
      // 如果删完了，显示空提示，但不清空表头
      const emptyRow = document.createElement("tr");
      emptyRow.className = "empty";
      const colSpan = currentHeaders.length + 1;
      emptyRow.innerHTML = `<td colspan="${colSpan}">暂无数据</td>`;
      tbody.appendChild(emptyRow);
    }
    refreshLowConfidenceSummary();
  });
  tdOp.appendChild(delBtn);
  tr.appendChild(tdOp);

  // 移除空提示行
  const empty = tbody.querySelector("tr.empty");
  if (empty) empty.remove();
  
  tbody.appendChild(tr);
  refreshLowConfidenceSummary();
}

function renderTranscript(data) {
  // data: { headers: [...], rows: [{values: {}, confidences: {}}, ...] }
  const headers = Array.isArray(data.headers) ? data.headers : [];
  const rows = Array.isArray(data.rows) ? data.rows : [];

  if (headers.length === 0) {
    clearTable();
    setStatus("未识别到有效的表格结构", "error");
    return;
  }

  currentHeaders = headers;

  // 渲染表头
  let thHtml = "";
  for (const h of headers) {
    thHtml += `<th>${h}</th>`;
  }
  thHtml += `<th>操作</th>`;
  thead.innerHTML = `<tr>${thHtml}</tr>`;

  // 渲染内容
  tbody.innerHTML = "";
  if (rows.length === 0) {
    const colSpan = headers.length + 1;
    tbody.innerHTML = `<tr class="empty"><td colspan="${colSpan}">未识别到数据行</td></tr>`;
  } else {
    for (const row of rows) {
      addRow(row.values, row.confidences);
    }
  }

  downloadEl.innerHTML = "";
  exportBtn.disabled = false;
  refreshLowConfidenceSummary();
}

async function uploadFile(file) {
  const form = new FormData();
  form.append("file", file);

  const res = await fetch("/upload", { method: "POST", body: form });
  if (!res.ok) {
    const detail = await res.json().catch(() => null);
    const msg = detail && detail.detail ? detail.detail : `HTTP ${res.status}`;
    throw new Error(msg);
  }
  return await res.json();
}

async function exportExcel(payload) {
  const res = await fetch("/export", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    const detail = await res.json().catch(() => null);
    const msg = detail && detail.detail ? detail.detail : `HTTP ${res.status}`;
    throw new Error(msg);
  }
  return await res.json();
}

function buildPayloadFromUI() {
  const rows = [...tbody.querySelectorAll("tr[data-row='1']")];
  const payloadRows = [];

  for (const tr of rows) {
    const values = {};
    const confidences = {};
    
    // 遍历每一个可编辑的单元格
    const editables = tr.querySelectorAll(".cell-editable");
    editables.forEach(span => {
      const field = span.dataset.field;
      const conf = Number(span.dataset.confidence ?? 1);
      values[field] = span.textContent.trim();
      confidences[field] = conf;
    });

    payloadRows.push({
      values,
      confidences
    });
  }

  return {
    headers: currentHeaders,
    rows: payloadRows
  };
}

fileInput.addEventListener("change", () => {
  const f = fileInput.files && fileInput.files[0] ? fileInput.files[0] : null;
  setSelectedFile(f);
});

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
  if (selectedFile) setStatus("已选择文件，点击“开始识别”进行处理", "");
  else setStatus("请选择 PNG/JPG/SVG/XLSX 文件", "");
}

dropzone?.addEventListener("click", () => fileInput.click());
dropzone?.addEventListener("keydown", (e) => {
  if (e.key === "Enter" || e.key === " ") {
    e.preventDefault();
    fileInput.click();
  }
});

dropzone?.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropzone.classList.add("dragover");
});
dropzone?.addEventListener("dragleave", () => dropzone.classList.remove("dragover"));
dropzone?.addEventListener("drop", (e) => {
  e.preventDefault();
  dropzone.classList.remove("dragover");
  const f = e.dataTransfer?.files?.[0];
  if (!f) return;
  if (!isSupportedFile(f)) {
    setSelectedFile(null);
    setStatus("不支持的文件类型：仅支持 PNG/JPG/SVG/XLSX", "error");
    return;
  }
  setSelectedFile(f);
});

clearBtn.addEventListener("click", () => {
  fileInput.value = "";
  setSelectedFile(null);
});

thresholdInput?.addEventListener("input", () => {
  threshold = Number(thresholdInput.value);
  if (thresholdValueEl) thresholdValueEl.textContent = threshold.toFixed(2);
  document.querySelectorAll(".cell-editable").forEach((el) => {
    const conf = Number(el.dataset.confidence ?? 1);
    applyLowConfidenceStyle(el, conf);
  });
  refreshLowConfidenceSummary();
});

uploadBtn.addEventListener("click", async () => {
  if (!selectedFile) return;
  uploadBtn.disabled = true;
  clearBtn.disabled = true;
  setLoading(uploadBtn, true);
  setStatus("识别中，请稍候…（首次加载 OCR 模型可能较慢）", "");

  try {
    const data = await uploadFile(selectedFile);
    renderTranscript(data);
    setStatus("识别完成，可在右侧表格中修改后导出", "ok");
  } catch (e) {
    setStatus(`识别失败：${e.message || e}`, "error");
    clearTable();
    exportBtn.disabled = true;
  } finally {
    uploadBtn.disabled = false;
    clearBtn.disabled = !selectedFile;
    setLoading(uploadBtn, false);
  }
});

addRowBtn.addEventListener("click", () => {
  if (currentHeaders.length === 0) {
    alert("请先上传文件并识别出表头结构后再添加行");
    return;
  }
  addRow({}, {});
});

exportBtn.addEventListener("click", async () => {
  exportBtn.disabled = true;
  setLoading(exportBtn, true);
  downloadEl.innerHTML = "";
  setStatus("导出中…", "");

  try {
    const payload = buildPayloadFromUI();
    const res = await exportExcel(payload);
    const url = res.download_url;
    if (!url) throw new Error("后端未返回下载链接");

    downloadEl.innerHTML = `导出成功：<a href="${url}" target="_blank" rel="noopener">点击下载 Excel</a>`;
    setStatus("导出完成", "ok");
  } catch (e) {
    setStatus(`导出失败：${e.message || e}`, "error");
  } finally {
    exportBtn.disabled = false;
    setLoading(exportBtn, false);
  }
});

renderPreview(null);
clearTable();
setFileMeta(null);
if (thresholdValueEl) thresholdValueEl.textContent = threshold.toFixed(2);
