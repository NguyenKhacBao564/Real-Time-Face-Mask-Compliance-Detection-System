const video = document.getElementById("video");
const overlay = document.getElementById("overlay");
const ctx = overlay.getContext("2d");
const statusEl = document.getElementById("status");
const correctEl = document.getElementById("correct");
const incorrectEl = document.getElementById("incorrect");
const noMaskEl = document.getElementById("no-mask");
const fpsEl = document.getElementById("fps");
const latencyEl = document.getElementById("latency");
const violationLogEl = document.getElementById("violation-log");
const stageEl = document.querySelector(".stage");
const captureButton = document.getElementById("capture-button");
const captureStatusEl = document.getElementById("capture-status");

const captureCanvas = document.createElement("canvas");
const captureCtx = captureCanvas.getContext("2d");

const colors = {
  correct_mask: "#2fb36d",
  incorrect_mask: "#d9911b",
  no_mask: "#dc3b3b",
};

let socket;
let latestDetections = [];
let lastMessageAt = performance.now();
let smoothedFps = 0;

function resizeCanvas() {
  const videoWidth = video.videoWidth || 1;
  const videoHeight = video.videoHeight || 1;
  const stageWidth = stageEl.clientWidth;
  const stageHeight = stageEl.clientHeight;
  const scale = Math.min(stageWidth / videoWidth, stageHeight / videoHeight);
  const renderedWidth = Math.round(videoWidth * scale);
  const renderedHeight = Math.round(videoHeight * scale);
  const offsetX = Math.round((stageWidth - renderedWidth) / 2);
  const offsetY = Math.round((stageHeight - renderedHeight) / 2);

  overlay.width = videoWidth;
  overlay.height = videoHeight;
  overlay.style.width = `${renderedWidth}px`;
  overlay.style.height = `${renderedHeight}px`;
  overlay.style.left = `${offsetX}px`;
  overlay.style.top = `${offsetY}px`;
}

function drawDetections() {
  resizeCanvas();
  ctx.clearRect(0, 0, overlay.width, overlay.height);
  for (const detection of latestDetections) {
    const [x1, y1, x2, y2] = detection.bbox;
    const color = colors[detection.class_name] || "#ffffff";
    ctx.strokeStyle = color;
    ctx.lineWidth = 3;
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
    ctx.fillStyle = color;
    ctx.font = "16px system-ui, sans-serif";
    ctx.fillText(`${detection.class_name} ${detection.confidence}`, x1, Math.max(20, y1 - 8));
  }
}

function updateMetrics(payload) {
  const counts = payload.counts || {};
  correctEl.textContent = counts.correct_mask || 0;
  incorrectEl.textContent = counts.incorrect_mask || 0;
  noMaskEl.textContent = counts.no_mask || 0;
  fpsEl.textContent = smoothedFps.toFixed(1);
  latencyEl.textContent = `${payload.latency_ms || 0} ms`;
}

async function refreshViolationEvents() {
  try {
    const response = await fetch("/api/v1/events?limit=5");
    if (!response.ok) {
      return;
    }
    const payload = await response.json();
    const events = payload.events || [];
    violationLogEl.innerHTML = "";
    for (const event of events) {
      const item = document.createElement("li");
      const time = new Date(event.timestamp).toLocaleTimeString();
      const confidence = (event.confidence * 100).toFixed(0);
      const meta = document.createElement("span");
      meta.textContent = `${time} - ${event.label} ${confidence}%`;
      item.appendChild(meta);
      if (event.snapshot_path) {
        const link = document.createElement("a");
        link.href = `/api/v1/events/${event.event_id}/snapshot`;
        link.target = "_blank";
        link.rel = "noopener";
        link.textContent = "snapshot";
        link.style.marginLeft = "6px";
        item.appendChild(link);
      }
      violationLogEl.appendChild(item);
    }
  } catch (error) {
    // Non-fatal: panel will retry on the next tick.
  }
}

function sendFrame() {
  if (!socket || socket.readyState !== WebSocket.OPEN || !video.videoWidth) {
    return;
  }
  captureCanvas.width = video.videoWidth;
  captureCanvas.height = video.videoHeight;
  captureCtx.drawImage(video, 0, 0, captureCanvas.width, captureCanvas.height);
  captureCanvas.toBlob((blob) => {
    if (blob && socket.readyState === WebSocket.OPEN) {
      socket.send(blob);
    }
  }, "image/jpeg", 0.75);
}

function captureBlob() {
  if (!video.videoWidth) {
    return Promise.reject(new Error("Camera is not ready"));
  }
  captureCanvas.width = video.videoWidth;
  captureCanvas.height = video.videoHeight;
  captureCtx.drawImage(video, 0, 0, captureCanvas.width, captureCanvas.height);
  return new Promise((resolve, reject) => {
    captureCanvas.toBlob((blob) => {
      if (blob) {
        resolve(blob);
      } else {
        reject(new Error("Could not capture frame"));
      }
    }, "image/jpeg", 0.9);
  });
}

async function saveTestFrame() {
  captureButton.disabled = true;
  captureStatusEl.textContent = "Saving...";

  try {
    const blob = await captureBlob();
    const form = new FormData();
    form.append("file", blob, "webcam.jpg");
    form.append("label", document.getElementById("capture-label").value);
    form.append("subtype", document.getElementById("capture-subtype").value);
    form.append("lighting", document.getElementById("capture-lighting").value);
    form.append("occlusion", document.getElementById("capture-occlusion").value);
    form.append("blur", document.getElementById("capture-blur").value);
    form.append("reflection", document.getElementById("capture-reflection").value);
    form.append("notes", `prediction=${JSON.stringify(latestDetections)}`);

    const response = await fetch("/api/v1/dataset/capture", {
      method: "POST",
      body: form,
    });
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || "Capture failed");
    }
    captureStatusEl.textContent = `Saved ${payload.image_path}`;
  } catch (error) {
    captureStatusEl.textContent = error.message;
  } finally {
    captureButton.disabled = false;
  }
}

async function start() {
  const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
  video.srcObject = stream;

  const protocol = window.location.protocol === "https:" ? "wss" : "ws";
  socket = new WebSocket(`${protocol}://${window.location.host}/api/v1/ws/detect`);

  socket.addEventListener("open", () => {
    statusEl.textContent = "Connected";
    setInterval(sendFrame, 200);
  });

  socket.addEventListener("message", (event) => {
    const now = performance.now();
    const instantFps = 1000 / Math.max(1, now - lastMessageAt);
    smoothedFps = smoothedFps ? smoothedFps * 0.8 + instantFps * 0.2 : instantFps;
    lastMessageAt = now;

    const payload = JSON.parse(event.data);
    if (payload.error) {
      statusEl.textContent = payload.error;
      return;
    }
    latestDetections = payload.detections || [];
    updateMetrics(payload);
    drawDetections();
  });

  socket.addEventListener("close", () => {
    statusEl.textContent = "Disconnected";
  });
}

start().catch((error) => {
  statusEl.textContent = error.message;
});

captureButton.addEventListener("click", saveTestFrame);

refreshViolationEvents();
setInterval(refreshViolationEvents, 3000);
