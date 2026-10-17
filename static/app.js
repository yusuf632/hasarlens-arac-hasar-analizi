const imageInput = document.querySelector("#image-input");
const dropZone = document.querySelector("#drop-zone");
const emptyState = document.querySelector("#empty-state");
const previewState = document.querySelector("#preview-state");
const imagePreview = document.querySelector("#image-preview");
const fileName = document.querySelector("#file-name");
const removeImageButton = document.querySelector("#remove-image");
const analyzeButton = document.querySelector("#analyze-button");
const buttonLabel = analyzeButton.querySelector(".button-label");
const message = document.querySelector("#message");
const resultEmpty = document.querySelector("#result-empty");
const resultContent = document.querySelector("#result-content");

let selectedFile = null;
let previewUrl = null;

function showMessage(text, kind = "") {
  message.textContent = text;
  message.className = `message${kind ? ` ${kind}` : ""}`;
}

function clearResult() {
  resultContent.hidden = true;
  resultEmpty.hidden = false;
}

function setSelectedFile(file) {
  if (!file) return;

  const allowedTypes = ["image/jpeg", "image/png", "image/webp"];
  if (!allowedTypes.includes(file.type)) {
    showMessage("JPEG, PNG veya WEBP biçiminde bir fotoğraf seçin.", "error");
    return;
  }
  if (file.size > 10 * 1024 * 1024) {
    showMessage("Fotoğraf 10 MB sınırını aşıyor.", "error");
    return;
  }

  selectedFile = file;
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = URL.createObjectURL(file);
  imagePreview.src = previewUrl;
  fileName.textContent = file.name;
  emptyState.hidden = true;
  previewState.hidden = false;
  analyzeButton.disabled = false;
  analyzeButton.classList.remove("is-loading");
  buttonLabel.textContent = "Fotoğrafı analiz et";
  showMessage("");
  clearResult();
}

function removeSelectedFile() {
  selectedFile = null;
  imageInput.value = "";
  imagePreview.removeAttribute("src");
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = null;
  previewState.hidden = true;
  emptyState.hidden = false;
  analyzeButton.disabled = true;
  showMessage("");
  clearResult();
}

function formatProbability(value) {
  return `${Number(value).toLocaleString("tr-TR", {
    minimumFractionDigits: 1,
    maximumFractionDigits: 1,
  })}%`;
}

function renderResult(result) {
  const damaged = result.is_damaged;
  const status = document.querySelector("#result-status");
  status.classList.toggle("is-damaged", damaged);
  document.querySelector("#status-kicker").textContent = damaged
    ? "MODEL HASAR OLASILIĞI SAPTADI"
    : "MODEL HASAR OLASILIĞI SAPTAMADI";
  document.querySelector("#status-title").textContent = damaged
    ? "Olası hasar"
    : "Hasarsız görünüyor";
  document.querySelector("#result-region").textContent =
    result.region === "ÖN" ? "Ön bölüm" : "Arka bölüm";

  const confidence = Math.max(0, Math.min(100, Number(result.confidence)));
  document.querySelector("#confidence-value").textContent =
    formatProbability(confidence);
  document.querySelector("#confidence-bar").style.width = `${confidence}%`;

  const probabilityList = document.querySelector("#probability-list");
  probabilityList.replaceChildren();
  for (const [label, probability] of Object.entries(result.probabilities)) {
    const item = document.createElement("li");
    item.className = "probability-row";

    const name = document.createElement("span");
    name.textContent = label;
    const track = document.createElement("span");
    track.className = "probability-track";
    const bar = document.createElement("span");
    bar.style.width = `${Math.max(0, Math.min(100, Number(probability)))}%`;
    track.append(bar);
    const value = document.createElement("strong");
    value.textContent = formatProbability(probability);
    item.append(name, track, value);
    probabilityList.append(item);
  }

  const analyzedAt = new Date(result.analyzed_at);
  document.querySelector("#analyzed-at").textContent =
    `Analiz zamanı · ${analyzedAt.toLocaleString("tr-TR", {
      dateStyle: "medium",
      timeStyle: "short",
    })}`;
  resultEmpty.hidden = true;
  resultContent.hidden = false;
}

imageInput.addEventListener("change", () => {
  setSelectedFile(imageInput.files[0]);
});

dropZone.addEventListener("click", (event) => {
  if (event.target.closest("#remove-image")) return;
  imageInput.click();
});

dropZone.addEventListener("keydown", (event) => {
  if (event.target !== dropZone) return;
  if (event.key === "Enter" || event.key === " ") {
    event.preventDefault();
    imageInput.click();
  }
});

for (const eventName of ["dragenter", "dragover"]) {
  dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropZone.classList.add("is-dragging");
  });
}

for (const eventName of ["dragleave", "drop"]) {
  dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    if (eventName === "drop") setSelectedFile(event.dataTransfer.files[0]);
    dropZone.classList.remove("is-dragging");
  });
}

removeImageButton.addEventListener("click", (event) => {
  event.stopPropagation();
  removeSelectedFile();
});

analyzeButton.addEventListener("click", async () => {
  if (!selectedFile || analyzeButton.disabled) return;

  const formData = new FormData();
  formData.append("image", selectedFile);
  analyzeButton.disabled = true;
  analyzeButton.classList.add("is-loading");
  buttonLabel.textContent = "Fotoğraf analiz ediliyor";
  showMessage("");

  try {
    const response = await fetch("/api/analyze", {
      method: "POST",
      body: formData,
    });
    const result = await response.json();
    if (!response.ok) {
      throw new Error(result.error || "Fotoğraf analiz edilemedi.");
    }
    renderResult(result);
  } catch (error) {
    showMessage(
      error instanceof Error
        ? error.message
        : "Sunucuya ulaşılamadı. Uygulamanın çalıştığını kontrol edin.",
      "error",
    );
  } finally {
    analyzeButton.disabled = !selectedFile;
    analyzeButton.classList.remove("is-loading");
    buttonLabel.textContent = "Yeniden analiz et";
  }
});

window.addEventListener("beforeunload", () => {
  if (previewUrl) URL.revokeObjectURL(previewUrl);
});
