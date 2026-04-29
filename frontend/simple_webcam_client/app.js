const video = document.getElementById("video");
const overlay = document.getElementById("overlay");
const ctx = overlay.getContext("2d");
const statusEl = document.getElementById("status");
const correctEl = document.getElementById("correct");
const incorrectEl = document.getElementById("incorrect");
const noMaskEl = document.getElementById("no-mask");
const latencyEl = document.getElementById("latency");

const captureCanvas = document.createElement("canvas");
const captureCtx = captureCanvas.getContext("2d");

const colors = {
  correct_mask: "#2fb36d",
  incorrect_mask: "#d9911b",
  no_mask: "#dc3b3b",
};

let socket;
let latestDetections = [];

function resizeCanvas() {
  overlay.width = video.videoWidth || overlay.clientWidth;
  overlay.height = video.videoHeight || overlay.clientHeight;
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
  latencyEl.textContent = `${payload.latency_ms || 0} ms`;
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

