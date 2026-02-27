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

// Scale factor: bbox coords are in original camera frame space.
// Backend sends frame_w / frame_h with every detection message.
function scaleCoord(val, model_dim, display_dim) {
  return (val / model_dim) * display_dim;
}

// ─── Nav arrow animation frame handle ─────────────────────────────
let _navArrowRAF = null;
let _findTargetRAF = null;

const overlay = {
  update(detections, frame_w, frame_h) {
    resizeCanvas();
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    this._drawZones();
    detections.forEach(d => this._drawBox(d, frame_w, frame_h));
  },
  clear() {
    if (_navArrowRAF)    { cancelAnimationFrame(_navArrowRAF);    _navArrowRAF    = null; }
    if (_findTargetRAF)  { cancelAnimationFrame(_findTargetRAF);  _findTargetRAF  = null; }
    ctx.clearRect(0, 0, canvas.width, canvas.height);
  },

  // ── NAVIGATE: animated arrow toward most dangerous object ────────
  drawNavArrow(detections, frame_w, frame_h) {
    if (!detections || detections.length === 0) return;

    // Pick most dangerous: lowest distance_level, tie-break by largest bbox area
    const sorted = [...detections].sort((a, b) => {
      if (a.distance_level !== b.distance_level) return a.distance_level - b.distance_level;
      const aArea = (a.x2 - a.x1) * (a.y2 - a.y1);
      const bArea = (b.x2 - b.x1) * (b.y2 - b.y1);
      return bArea - aArea;
    });
    const target = sorted[0];
    if (!target) return;

    // Cancel any previous animation
    if (_navArrowRAF) { cancelAnimationFrame(_navArrowRAF); _navArrowRAF = null; }

    const color = COLORS[target.distance_level] || '#ffffff';
    const label = `${target.class_name} · ${target.direction}`;

    const drawFrame = () => {
      resizeCanvas();
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      const dw = canvas.width, dh = canvas.height;
      // Target center (scaled)
      const tx = scaleCoord((target.x1 + target.x2) / 2, frame_w, dw);
      const ty = scaleCoord((target.y1 + target.y2) / 2, frame_h, dh);

      // Arrow origin: bottom-center of canvas
      const ox = dw / 2;
      const oy = dh - 40;

      // Pulse factor (0..1, ~1 Hz)
      const pulse = 0.7 + 0.3 * Math.sin(Date.now() / 400 * Math.PI);

      // Shaft
      const dx = tx - ox, dy = ty - oy;
      const len = Math.sqrt(dx * dx + dy * dy);
      const ux = dx / len, uy = dy / len;
      // Stop the shaft 60px before the target center
      const shaftEndX = tx - ux * 60;
      const shaftEndY = ty - uy * 60;

      ctx.save();
      ctx.shadowColor  = color;
      ctx.shadowBlur   = 18 * pulse;
      ctx.strokeStyle  = color;
      ctx.lineWidth    = 5 * pulse;
      ctx.lineCap      = 'round';
      ctx.globalAlpha  = 0.85 * pulse;
      ctx.beginPath();
      ctx.moveTo(ox, oy);
      ctx.lineTo(shaftEndX, shaftEndY);
      ctx.stroke();

      // Arrowhead triangle
      const headLen = 28 * pulse;
      const angle   = Math.atan2(uy, ux);
      ctx.fillStyle   = color;
      ctx.beginPath();
      ctx.moveTo(tx - ux * 20, ty - uy * 20);
      ctx.lineTo(
        tx - ux * 20 - headLen * Math.cos(angle - Math.PI / 5),
        ty - uy * 20 - headLen * Math.sin(angle - Math.PI / 5)
      );
      ctx.lineTo(
        tx - ux * 20 - headLen * Math.cos(angle + Math.PI / 5),
        ty - uy * 20 - headLen * Math.sin(angle + Math.PI / 5)
      );
      ctx.closePath();
      ctx.fill();

      // Label near arrowhead
      ctx.globalAlpha  = 1;
      ctx.shadowBlur   = 0;
      ctx.font         = 'bold 13px Inter, sans-serif';
      ctx.textAlign    = 'center';
      ctx.fillStyle    = color;
      ctx.fillText(label, tx, ty - 28);

      ctx.restore();
      _navArrowRAF = requestAnimationFrame(drawFrame);
    };

    drawFrame();
  },

  // ── FIND: animated target ring on found object ───────────────────
  drawFindTarget(detection, frame_w, frame_h) {
    if (!detection) return;

    if (_findTargetRAF) { cancelAnimationFrame(_findTargetRAF); _findTargetRAF = null; }

    const drawFrame = () => {
      resizeCanvas();
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      const dw = canvas.width, dh = canvas.height;
      const cx = scaleCoord((detection.x1 + detection.x2) / 2, frame_w, dw);
      const cy = scaleCoord((detection.y1 + detection.y2) / 2, frame_h, dh);
      const bw = scaleCoord(detection.x2 - detection.x1, frame_w, dw);
      const bh = scaleCoord(detection.y2 - detection.y1, frame_h, dh);
      const baseR = Math.max(bw, bh) / 2 + 12;

      const t     = Date.now() / 600;
      const pulse = 0.5 + 0.5 * Math.sin(t * Math.PI);

      ctx.save();
      // Draw 3 concentric pulsing rings
      for (let i = 0; i < 3; i++) {
        const r = baseR + i * 16 + pulse * 10;
        const alpha = (1 - i * 0.28) * (0.4 + 0.6 * pulse);
        ctx.strokeStyle = `rgba(16, 185, 129, ${alpha})`;   // emerald
        ctx.lineWidth   = 3 - i * 0.6;
        ctx.shadowColor = '#10b981';
        ctx.shadowBlur  = 14 * pulse;
        ctx.beginPath();
        ctx.arc(cx, cy, r, 0, Math.PI * 2);
        ctx.stroke();
      }

      // Crosshair lines
      ctx.strokeStyle  = 'rgba(16,185,129,0.75)';
      ctx.lineWidth    = 2;
      ctx.shadowBlur   = 8;
      const ch = baseR * 0.45;
      [[0,-1],[0,1],[-1,0],[1,0]].forEach(([dx,dy]) => {
        ctx.beginPath();
        ctx.moveTo(cx + dx * (baseR - ch), cy + dy * (baseR - ch));
        ctx.lineTo(cx + dx * (baseR + 8),  cy + dy * (baseR + 8));
        ctx.stroke();
      });

      // Label
      ctx.shadowBlur  = 0;
      ctx.font        = 'bold 14px Inter, sans-serif';
      ctx.textAlign   = 'center';
      ctx.fillStyle   = '#10b981';
      ctx.fillText(`FOUND: ${detection.class_name.toUpperCase()}`, cx, cy - baseR - 20);
      ctx.restore();

      _findTargetRAF = requestAnimationFrame(drawFrame);
    };

    drawFrame();
  },

  _drawZones() {
    ctx.strokeStyle = 'rgba(255,255,255,0.08)';
    ctx.lineWidth = 1;
    const w = canvas.width;
    const h = canvas.height;
    // Zone dividers match spatial.py thresholds: 20%, 40%, 60%, 75%
    // Zones: far-left | left | ahead | right | far-right
    [0.20, 0.40, 0.60, 0.75].forEach(ratio => {
      ctx.beginPath();
      ctx.moveTo(w * ratio, 0);
      ctx.lineTo(w * ratio, h);
      ctx.stroke();
    });
    // Zone labels
    ctx.fillStyle = 'rgba(255,255,255,0.2)';
    ctx.font = 'bold 9px Inter, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('FAR-L', w * 0.10, 14);
    ctx.fillText('LEFT',  w * 0.30, 14);
    ctx.fillText('AHEAD', w * 0.50, 14);
    ctx.fillText('RIGHT', w * 0.675, 14);
    ctx.fillText('FAR-R', w * 0.875, 14);
  },
  _drawBox(d, frame_w, frame_h) {
    const dw = canvas.width, dh = canvas.height;
    const x1 = scaleCoord(d.x1, frame_w, dw);
    const y1 = scaleCoord(d.y1, frame_h, dh);
    const x2 = scaleCoord(d.x2, frame_w, dw);
    const y2 = scaleCoord(d.y2, frame_h, dh);
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

