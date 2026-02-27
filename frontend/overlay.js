// overlay.js — canvas bounding box renderer
// Canvas must be: position:absolute; inset:0; width:100%; height:100%; pointer-events:none

const canvas  = document.getElementById('overlay-canvas');
const ctx     = canvas.getContext('2d');
const COLORS  = { 1: '#ef4444', 2: '#f97316', 3: '#10b981', 4: '#6b7280' };
const LABELS  = { 1: 'VERY CLOSE', 2: 'NEARBY', 3: 'AHEAD', 4: 'FAR' };

function resizeCanvas() {
  const wrap = document.getElementById('camera-wrap');
  canvas.width  = wrap.clientWidth;
  canvas.height = wrap.clientHeight;
}
window.addEventListener('resize', resizeCanvas);
resizeCanvas();

// Scale factor: bbox coords are in original camera frame space (640×480).
// Backend detector.py maps detections back to original frame coords before sending.
function scaleCoord(val, model_dim, display_dim) {
  return (val / model_dim) * display_dim;
}
const MODEL_W = 640, MODEL_H = 480;   // original camera frame size (NOT inference size)

const overlay = {
  update(detections) {
    resizeCanvas();
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    this._drawZones();
    detections.forEach(d => this._drawBox(d));
  },
  clear() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
  },
  _drawZones() {
    ctx.strokeStyle = 'rgba(255,255,255,0.08)';
    ctx.lineWidth = 1;
    const w = canvas.width;
    const h = canvas.height;
    // Left | Center | Right zone dividers
    [0.33, 0.67].forEach(ratio => {
      ctx.beginPath();
      ctx.moveTo(w * ratio, 0);
      ctx.lineTo(w * ratio, h);
      ctx.stroke();
    });
    // Zone labels
    ctx.fillStyle = 'rgba(255,255,255,0.2)';
    ctx.font = 'bold 9px Inter, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('LEFT', w * 0.167, 14);
    ctx.fillText('AHEAD', w * 0.5, 14);
    ctx.fillText('RIGHT', w * 0.833, 14);
  },
  _drawBox(d) {
    const dw = canvas.width, dh = canvas.height;
    const x1 = scaleCoord(d.x1, MODEL_W, dw);
    const y1 = scaleCoord(d.y1, MODEL_H, dh);
    const x2 = scaleCoord(d.x2, MODEL_W, dw);
    const y2 = scaleCoord(d.y2, MODEL_H, dh);
    const color = COLORS[d.distance_level] || '#ffffff';
    const conf  = Math.round(d.confidence * 100);
    const label = `${d.class_name} · ${d.distance} · ${conf}%`;

    // Box
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

    // Filled label background
    ctx.font = 'bold 11px Inter, sans-serif';
    const textW = ctx.measureText(label).width;
    ctx.fillStyle = color;
    ctx.fillRect(x1, y1 - 20, textW + 10, 20);

    // Label text
    ctx.fillStyle = '#fff';
    ctx.fillText(label, x1 + 5, y1 - 6);
  }
};

window.overlay = overlay;
