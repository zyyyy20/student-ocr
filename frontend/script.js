const THRESHOLD = 0.85;

const fileInput = document.getElementById("fileInput");
const uploadBtn = document.getElementById("uploadBtn");
const statusEl = document.getElementById("status");
const previewEl = document.getElementById("preview");
const tbody = document.getElementById("tbody");
const thead = document.getElementById("thead");
const addRowBtn = document.getElementById("addRowBtn");
const exportBtn = document.getElementById("exportBtn");
const downloadEl = document.getElementById("download");

let selectedFile = null;
let currentHeaders = []; // 存储当前表格的表头 ["姓名", "成绩", ...]

function setStatus(text, type) {
  statusEl.textContent = text;
  statusEl.classList.remove("ok", "error");
  if (type) statusEl.classList.add(type);
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
  if (conf != null && conf < THRESHOLD) el.classList.add("low");
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
  });
  tdOp.appendChild(delBtn);
  tr.appendChild(tdOp);

  // 移除空提示行
  const empty = tbody.querySelector("tr.empty");
  if (empty) empty.remove();
  
  tbody.appendChild(tr);
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
  selectedFile = fileInput.files && fileInput.files[0] ? fileInput.files[0] : null;
  uploadBtn.disabled = !selectedFile;
  exportBtn.disabled = true;
  downloadEl.innerHTML = "";
  renderPreview(selectedFile);
  clearTable();
  if (selectedFile) setStatus("已选择文件，点击“开始识别”进行处理", "");
});

uploadBtn.addEventListener("click", async () => {
  if (!selectedFile) return;
  uploadBtn.disabled = true;
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
  }
});

renderPreview(null);
clearTable();
