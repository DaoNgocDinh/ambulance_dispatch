/**
 * SMART DISPATCH COMMAND CENTER - FRONTEND CLIENT
 * Interactive HTML5 Canvas Simulation, Realtime Playback, Web Audio & MDP Analytics
 */

// Global State
const state = {
  mapData: null,
  simState: null,
  isPlaying: false,
  playTimer: null,
  speedDelay: 400, // ms per step
  autoDispatch: true,
  selectedAmbulanceId: null,
  hoveredEntity: null,
  clickedEntity: null,
  soundEnabled: true,
  logFilter: 'all',

  // Viewport / Camera Transform
  view: {
    panX: 0,
    panY: 0,
    zoom: 1.0,
    isDragging: false,
    dragStartX: 0,
    dragStartY: 0,
    minZoom: 0.4,
    maxZoom: 5.0,
  },

  // Animation timestamp
  animTime: 0,
};

// Web Audio API Synthesizer
class SoundEffects {
  constructor() {
    this.ctx = null;
  }

  init() {
    if (!this.ctx) {
      const AudioCtx = window.AudioContext || window.webkitAudioContext;
      if (AudioCtx) this.ctx = new AudioCtx();
    }
    if (this.ctx && this.ctx.state === 'suspended') {
      this.ctx.resume();
    }
  }

  playBeep(freq, type, duration, gainVal = 0.05) {
    if (!state.soundEnabled) return;
    try {
      this.init();
      if (!this.ctx) return;
      const osc = this.ctx.createOscillator();
      const gain = this.ctx.createGain();
      osc.type = type;
      osc.frequency.setValueAtTime(freq, this.ctx.currentTime);
      gain.gain.setValueAtTime(gainVal, this.ctx.currentTime);
      gain.gain.exponentialRampToValueAtTime(0.0001, this.ctx.currentTime + duration);
      osc.connect(gain);
      gain.connect(this.ctx.destination);
      osc.start();
      osc.stop(this.ctx.currentTime + duration);
    } catch (e) {
      // Audio might be blocked by browser policy until user gesture
    }
  }

  callGenerated() {
    this.playBeep(880, 'sine', 0.15, 0.04);
    setTimeout(() => this.playBeep(1100, 'sine', 0.2, 0.04), 160);
  }

  ambulanceDispatched() {
    this.playBeep(523.25, 'triangle', 0.12, 0.05);
    setTimeout(() => this.playBeep(659.25, 'triangle', 0.18, 0.05), 120);
  }

  callCompleted() {
    this.playBeep(587.33, 'sine', 0.1, 0.04);
    setTimeout(() => this.playBeep(880, 'sine', 0.25, 0.05), 100);
  }

  callFailed() {
    this.playBeep(220, 'sawtooth', 0.35, 0.06);
  }
}

const sfx = new SoundEffects();

// DOM Elements
const dom = {
  canvas: document.getElementById('map-canvas'),
  canvasWrapper: document.getElementById('canvas-wrapper'),
  btnPlay: document.getElementById('btn-play'),
  playText: document.getElementById('play-text'),
  btnStep: document.getElementById('btn-step'),
  btnStep10: document.getElementById('btn-step10'),
  btnReset: document.getElementById('btn-reset'),
  speedSlider: document.getElementById('speed-slider'),
  speedValText: document.getElementById('speed-val-text'),
  btnSound: document.getElementById('btn-sound'),
  soundIcon: document.getElementById('sound-icon'),
  btnExportLog: document.getElementById('btn-export-log'),
  btnConfig: document.getElementById('btn-config'),

  // Map tools
  btnZoomIn: document.getElementById('btn-zoom-in'),
  btnZoomOut: document.getElementById('btn-zoom-out'),
  btnZoomFit: document.getElementById('btn-zoom-fit'),

  // Mode
  modeAuto: document.getElementById('mode-auto'),
  modeManual: document.getElementById('mode-manual'),
  manualBanner: document.getElementById('manual-banner'),
  btnCancelManual: document.getElementById('btn-cancel-manual'),

  // Header Displays
  clockMinute: document.getElementById('clock-minute'),
  clockMax: document.getElementById('clock-max'),
  timeProgressBar: document.getElementById('time-progress-bar'),
  simStatusBadge: document.getElementById('sim-status-badge'),
  simStatusText: document.getElementById('sim-status-text'),
  kpiServed: document.getElementById('kpi-served'),
  kpiPerfect: document.getElementById('kpi-perfect'),
  kpiFailed: document.getElementById('kpi-failed'),
  kpiReward: document.getElementById('kpi-reward'),

  // Panels
  pendingBadge: document.getElementById('pending-badge'),
  slotsContainer: document.getElementById('slots-container'),
  fleetCount: document.getElementById('fleet-count'),
  idleBadge: document.getElementById('idle-badge'),
  ambulanceList: document.getElementById('ambulance-list'),

  // MDP Tab
  mdpTotalReward: document.getElementById('mdp-total-reward'),
  rWait: document.getElementById('r-wait'),
  rFail: document.getElementById('r-fail'),
  rDist: document.getElementById('r-dist'),
  rBonus: document.getElementById('r-bonus'),
  rStep: document.getElementById('r-step'),
  mdpPotentialActions: document.getElementById('mdp-potential-actions'),
  mdpValidActions: document.getElementById('mdp-valid-actions'),
  mdpActionId: document.getElementById('mdp-action-id'),
  mdpDecodedAction: document.getElementById('mdp-decoded-action'),
  successRateBar: document.getElementById('success-rate-bar'),
  successRateText: document.getElementById('success-rate-text'),
  hospitalList: document.getElementById('hospital-list'),

  // Log Tab
  logText: document.getElementById('log-text'),
  logContainer: document.getElementById('log-container'),
  btnCopyLog: document.getElementById('btn-copy-log'),

  // Popup
  inspectorPopup: document.getElementById('inspector-popup'),
  popupTitle: document.getElementById('popup-title'),
  popupBody: document.getElementById('popup-body'),
  popupClose: document.getElementById('popup-close'),

  // Config Modal
  configModal: document.getElementById('config-modal'),
  modalCloseBtn: document.getElementById('modal-close-btn'),
  btnModalCancel: document.getElementById('btn-modal-cancel'),
  btnModalSave: document.getElementById('btn-modal-save'),
  cfgAmbulances: document.getElementById('cfg-ambulances'),
  cfgLambda: document.getElementById('cfg-lambda'),
  cfgMaxPending: document.getElementById('cfg-max-pending'),
  cfgMaxTime: document.getElementById('cfg-max-time'),
  cfgOvertime: document.getElementById('cfg-overtime'),
  cfgSeed: document.getElementById('cfg-seed'),
};

const ctx = dom.canvas.getContext('2d');

// Speed mapping
const SPEED_DELAYS = [1500, 800, 400, 200, 100, 40];
const SPEED_LABELS = ['0.5x', '1x', '2x', '5x', '10x', 'Max'];

// =========================================================================
// INITIALIZATION
// =========================================================================
async function init() {
  setupEventListeners();
  resizeCanvas();
  window.addEventListener('resize', resizeCanvas);

  try {
    const mapRes = await fetch('/api/map');
    state.mapData = await mapRes.json();

    const stateRes = await fetch('/api/state');
    state.simState = await stateRes.json();

    renderHospitalsList();
    fitMapToCanvas();
    updateUI();

    // Start animation loop for pulse & sirens
    requestAnimationFrame(renderLoop);
  } catch (err) {
    console.error('Lỗi khi tải dữ liệu khởi tạo:', err);
  }
}

// =========================================================================
// CANVAS COORDINATE TRANSFORMATIONS
// =========================================================================
function resizeCanvas() {
  const rect = dom.canvasWrapper.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  dom.canvas.width = rect.width * dpr;
  dom.canvas.height = rect.height * dpr;
  ctx.resetTransform();
  ctx.scale(dpr, dpr);
}

function worldToScreen(wx, wy) {
  if (!state.mapData) return { x: 0, y: 0 };
  const bounds = state.mapData.bounds;
  const rect = dom.canvasWrapper.getBoundingClientRect();
  const padding = 50;

  const wWidth = bounds.max_x - bounds.min_x;
  const wHeight = bounds.max_y - bounds.min_y;

  const baseScale = Math.min(
    (rect.width - padding * 2) / wWidth,
    (rect.height - padding * 2) / wHeight
  );

  const effScale = baseScale * state.view.zoom;

  // Inverted Y: world Y grows upwards, canvas Y grows downwards
  const sx = rect.width / 2 + (wx - (bounds.min_x + wWidth / 2)) * effScale + state.view.panX;
  const sy = rect.height / 2 - (wy - (bounds.min_y + wHeight / 2)) * effScale + state.view.panY;

  return { x: sx, y: sy, scale: effScale };
}

function screenToWorld(sx, sy) {
  if (!state.mapData) return { x: 0, y: 0 };
  const bounds = state.mapData.bounds;
  const rect = dom.canvasWrapper.getBoundingClientRect();
  const padding = 50;

  const wWidth = bounds.max_x - bounds.min_x;
  const wHeight = bounds.max_y - bounds.min_y;

  const baseScale = Math.min(
    (rect.width - padding * 2) / wWidth,
    (rect.height - padding * 2) / wHeight
  );

  const effScale = baseScale * state.view.zoom;

  const wx = bounds.min_x + wWidth / 2 + (sx - rect.width / 2 - state.view.panX) / effScale;
  const wy = bounds.min_y + wHeight / 2 - (sy - rect.height / 2 - state.view.panY) / effScale;

  return { x: wx, y: wy };
}

function fitMapToCanvas() {
  state.view.panX = 0;
  state.view.panY = 0;
  state.view.zoom = 1.0;
}

// =========================================================================
// CANVAS RENDERING
// =========================================================================
function renderLoop(timestamp) {
  state.animTime = timestamp;
  drawScene();
  requestAnimationFrame(renderLoop);
}

function drawScene() {
  if (!state.mapData || !state.simState) return;

  const rect = dom.canvasWrapper.getBoundingClientRect();
  ctx.clearRect(0, 0, rect.width, rect.height);

  // 1. Draw directed roads (172 directed edges with lane offset & arrows)
  drawRoadNetwork();

  // 2. Draw 80 nodes
  drawNodes();

  // 3. Draw 4 Hospitals
  drawHospitals();

  // 4. Draw Active Emergency Calls
  drawCalls();

  // 5. Draw Ambulance routes
  drawAmbulanceRoutes();

  // 6. Draw Ambulances
  drawAmbulances();
}

function drawRoadNetwork() {
  const edges = state.mapData.edges;
  const congestionMap = state.simState.edge_congestion || {};
  const isHoveredEdge = state.hoveredEntity && state.hoveredEntity.type === 'edge';

  edges.forEach((edge) => {
    const uPos = worldToScreen(edge.x1, edge.y1);
    const vPos = worldToScreen(edge.x2, edge.y2);

    const dx = vPos.x - uPos.x;
    const dy = vPos.y - uPos.y;
    const dist = Math.hypot(dx, dy);
    if (dist < 1) return;

    // Perpendicular normal vector for lane offset
    const nx = -dy / dist;
    const ny = dx / dist;

    // Offset lanes to the right of travel direction
    const laneOffset = 3.5 * Math.max(0.6, Math.min(2.0, state.view.zoom));
    const startX = uPos.x + nx * laneOffset;
    const startY = uPos.y + ny * laneOffset;
    const endX = vPos.x + nx * laneOffset;
    const endY = vPos.y + ny * laneOffset;

    // Congestion color
    const cong = congestionMap[edge.edge_id] || edge.congestion;
    let color = '#10b981'; // Green
    if (cong === 2) color = '#f59e0b'; // Amber
    if (cong === 3) color = '#ef4444'; // Red

    const isHovered = isHoveredEdge && state.hoveredEntity.data.edge_id === edge.edge_id;

    ctx.save();
    ctx.beginPath();
    ctx.moveTo(startX, startY);
    ctx.lineTo(endX, endY);
    ctx.strokeStyle = isHovered ? '#60a5fa' : color;
    ctx.lineWidth = isHovered ? 4.5 : 2.5;
    ctx.lineCap = 'round';
    ctx.shadowColor = isHovered ? '#3b82f6' : (cong === 3 ? 'rgba(239, 68, 68, 0.4)' : 'transparent');
    ctx.shadowBlur = isHovered ? 10 : (cong === 3 ? 6 : 0);
    ctx.stroke();

    // Directional Arrow along the lane
    if (dist > 30) {
      const arrowPosRatio = 0.55;
      const ax = startX + (endX - startX) * arrowPosRatio;
      const ay = startY + (endY - startY) * arrowPosRatio;
      const angle = Math.atan2(endY - startY, endX - startX);

      const arrowSize = 6 * Math.max(0.7, Math.min(1.8, state.view.zoom));
      ctx.beginPath();
      ctx.moveTo(ax, ay);
      ctx.lineTo(
        ax - arrowSize * Math.cos(angle - Math.PI / 6),
        ay - arrowSize * Math.sin(angle - Math.PI / 6)
      );
      ctx.lineTo(
        ax - arrowSize * Math.cos(angle + Math.PI / 6),
        ay - arrowSize * Math.sin(angle + Math.PI / 6)
      );
      ctx.closePath();
      ctx.fillStyle = isHovered ? '#93c5fd' : color;
      ctx.fill();
    }
    ctx.restore();
  });
}

function drawNodes() {
  const nodes = state.mapData.nodes;
  const isHoveredNode = state.hoveredEntity && state.hoveredEntity.type === 'node';

  nodes.forEach((node) => {
    const pos = worldToScreen(node.x, node.y);
    const isCentral = node.is_central;
    const isHovered = isHoveredNode && state.hoveredEntity.data.id === node.id;

    const radius = (isCentral ? 6.5 : 4.5) * Math.max(0.6, Math.min(1.6, state.view.zoom));

    ctx.save();
    ctx.beginPath();
    ctx.arc(pos.x, pos.y, radius, 0, Math.PI * 2);

    if (isHovered) {
      ctx.fillStyle = '#60a5fa';
      ctx.shadowColor = '#3b82f6';
      ctx.shadowBlur = 12;
    } else if (isCentral) {
      ctx.fillStyle = '#8b5cf6';
      ctx.shadowColor = 'rgba(139, 92, 246, 0.6)';
      ctx.shadowBlur = 6;
    } else {
      ctx.fillStyle = '#334155';
    }

    ctx.fill();
    ctx.lineWidth = 1.5;
    ctx.strokeStyle = isCentral ? '#c084fc' : '#64748b';
    ctx.stroke();

    // Node label if zoom is sufficient
    if (state.view.zoom > 0.8 || isCentral || isHovered) {
      ctx.fillStyle = isCentral ? '#f3e8ff' : '#94a3b8';
      ctx.font = `${Math.max(9, Math.round(10 * state.view.zoom))}px 'JetBrains Mono', monospace`;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(node.id, pos.x, pos.y - radius - 7);
    }
    ctx.restore();
  });
}

function drawHospitals() {
  const hospitals = state.mapData.hospitals;
  const isHoveredHosp = state.hoveredEntity && state.hoveredEntity.type === 'hospital';

  hospitals.forEach((hosp) => {
    const pos = worldToScreen(hosp.coord[0], hosp.coord[1]);
    const isHovered = isHoveredHosp && state.hoveredEntity.data.hospital_id === hosp.hospital_id;

    const size = 16 * Math.max(0.8, Math.min(2.0, state.view.zoom));

    ctx.save();
    // Red badge
    ctx.beginPath();
    ctx.arc(pos.x, pos.y, size / 2, 0, Math.PI * 2);
    ctx.fillStyle = isHovered ? '#dc2626' : '#ef4444';
    ctx.shadowColor = 'rgba(239, 68, 68, 0.8)';
    ctx.shadowBlur = 10;
    ctx.fill();
    ctx.lineWidth = 2;
    ctx.strokeStyle = '#fff';
    ctx.stroke();

    // White cross '+'
    ctx.fillStyle = '#fff';
    const crossW = size * 0.22;
    const crossL = size * 0.6;
    ctx.fillRect(pos.x - crossW / 2, pos.y - crossL / 2, crossW, crossL);
    ctx.fillRect(pos.x - crossL / 2, pos.y - crossW / 2, crossL, crossW);

    // Label
    ctx.fillStyle = '#ffffff';
    ctx.font = `bold ${Math.max(10, Math.round(11 * state.view.zoom))}px Inter, sans-serif`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    ctx.shadowColor = '#000';
    ctx.shadowBlur = 4;
    ctx.fillText(`BV ${hosp.hospital_id}`, pos.x, pos.y + size / 2 + 3);
    ctx.restore();
  });
}

function drawCalls() {
  const calls = state.simState.calls || [];
  const pulse = (Math.sin(state.animTime / 180) + 1) / 2; // 0..1 pulse

  calls.forEach((call) => {
    // Only draw active calls (pending, waiting, at_scene)
    if (!['Yêu cầu xử lý', 'Đợi xe cứu thương', 'Đang xử lý'].includes(call.status)) return;

    const pos = worldToScreen(call.coord[0], call.coord[1]);
    const isPending = call.status === 'Yêu cầu xử lý';

    let sevColor = '#fbbf24'; // Level 1
    if (call.severity === 2) sevColor = '#fb923c'; // Level 2
    if (call.severity === 3) sevColor = '#ef4444'; // Level 3

    ctx.save();

    // Pulsing radar ripple for pending calls
    if (isPending) {
      const rippleRadius = (12 + pulse * 18) * Math.max(0.8, state.view.zoom);
      ctx.beginPath();
      ctx.arc(pos.x, pos.y, rippleRadius, 0, Math.PI * 2);
      ctx.strokeStyle = sevColor;
      ctx.lineWidth = 2;
      ctx.globalAlpha = 1 - pulse;
      ctx.stroke();
      ctx.globalAlpha = 1.0;
    }

    // Call badge circle
    const baseRadius = 11 * Math.max(0.8, Math.min(1.8, state.view.zoom));
    ctx.beginPath();
    ctx.arc(pos.x, pos.y, baseRadius, 0, Math.PI * 2);
    ctx.fillStyle = sevColor;
    ctx.shadowColor = sevColor;
    ctx.shadowBlur = 12;
    ctx.fill();
    ctx.lineWidth = 2;
    ctx.strokeStyle = '#fff';
    ctx.stroke();

    // Text: Severity or Call ID
    ctx.fillStyle = '#0f172a';
    ctx.font = `bold ${Math.max(9, Math.round(10 * state.view.zoom))}px 'JetBrains Mono', monospace`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.shadowBlur = 0;
    ctx.fillText(`C${call.id}`, pos.x, pos.y);

    // Countdown / status pill
    ctx.fillStyle = '#f8fafc';
    ctx.font = `600 ${Math.max(9, Math.round(10 * state.view.zoom))}px Inter, sans-serif`;
    ctx.shadowColor = '#000';
    ctx.shadowBlur = 4;
    const tag = isPending ? `⏳${call.time_to_deadline}p` : `🚑 #${call.assigned_ambulance_id || ''}`;
    ctx.fillText(tag, pos.x, pos.y - baseRadius - 6);

    ctx.restore();
  });
}

function drawAmbulanceRoutes() {
  const ambulances = state.simState.ambulances || [];

  ambulances.forEach((amb) => {
    if (!amb.route_coords || amb.route_coords.length < 2) return;
    if (amb.status === 'Rảnh') return;

    ctx.save();
    ctx.beginPath();

    const startPos = worldToScreen(amb.route_coords[0][0], amb.route_coords[0][1]);
    ctx.moveTo(startPos.x, startPos.y);

    for (let i = 1; i < amb.route_coords.length; i++) {
      const pt = worldToScreen(amb.route_coords[i][0], amb.route_coords[i][1]);
      ctx.lineTo(pt.x, pt.y);
    }

    let routeColor = '#f59e0b';
    if (amb.status === 'Đang tới bệnh viện') routeColor = '#f43f5e';
    if (amb.status === 'Xử lý tại cuộc gọi') routeColor = '#a855f7';

    ctx.strokeStyle = routeColor;
    ctx.lineWidth = 2.5;
    ctx.setLineDash([6, 6]);
    ctx.lineDashOffset = -state.animTime / 60; // Animated moving dashes!
    ctx.shadowColor = routeColor;
    ctx.shadowBlur = 8;
    ctx.stroke();
    ctx.restore();
  });
}

function drawAmbulances() {
  const ambulances = state.simState.ambulances || [];
  const pulse = (Math.sin(state.animTime / 150) + 1) / 2;

  ambulances.forEach((amb) => {
    const pos = worldToScreen(amb.coord[0], amb.coord[1]);
    const isSelected = state.selectedAmbulanceId === amb.id;
    const isMoving = ['Đang đến cuộc gọi', 'Đang tới bệnh viện'].includes(amb.status);

    let statusColor = '#06b6d4'; // Cyan (Idle)
    if (amb.status === 'Đang đến cuộc gọi') statusColor = '#f59e0b'; // Amber
    if (amb.status === 'Xử lý tại cuộc gọi') statusColor = '#a855f7'; // Purple
    if (amb.status === 'Đang tới bệnh viện') statusColor = '#f43f5e'; // Rose
    if (amb.status === 'Xử lý tại bệnh viện') statusColor = '#10b981'; // Green

    ctx.save();

    // Siren pulse wave when responding
    if (isMoving) {
      const sirenRadius = (14 + pulse * 12) * Math.max(0.8, state.view.zoom);
      ctx.beginPath();
      ctx.arc(pos.x, pos.y, sirenRadius, 0, Math.PI * 2);
      ctx.strokeStyle = statusColor;
      ctx.lineWidth = 2.5;
      ctx.globalAlpha = 1 - pulse;
      ctx.stroke();
      ctx.globalAlpha = 1.0;
    }

    // Selection highlight ring
    if (isSelected) {
      ctx.beginPath();
      ctx.arc(pos.x, pos.y, 22 * state.view.zoom, 0, Math.PI * 2);
      ctx.strokeStyle = '#3b82f6';
      ctx.lineWidth = 3;
      ctx.shadowColor = '#3b82f6';
      ctx.shadowBlur = 14;
      ctx.stroke();
    }

    // Ambulance Badge
    const ambRadius = 13 * Math.max(0.8, Math.min(1.8, state.view.zoom));
    ctx.beginPath();
    ctx.arc(pos.x, pos.y, ambRadius, 0, Math.PI * 2);
    ctx.fillStyle = statusColor;
    ctx.shadowColor = statusColor;
    ctx.shadowBlur = 10;
    ctx.fill();
    ctx.lineWidth = 2;
    ctx.strokeStyle = '#ffffff';
    ctx.stroke();

    // Vehicle Icon
    ctx.font = `${Math.max(12, Math.round(13 * state.view.zoom))}px Inter, sans-serif`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.shadowBlur = 0;
    ctx.fillText('🚑', pos.x, pos.y - 1);

    // Ambulance Number tag
    ctx.fillStyle = '#ffffff';
    ctx.font = `bold ${Math.max(10, Math.round(10 * state.view.zoom))}px 'JetBrains Mono', monospace`;
    ctx.shadowColor = '#000';
    ctx.shadowBlur = 5;
    ctx.fillText(`Xe ${amb.id}`, pos.x, pos.y + ambRadius + 7);

    ctx.restore();
  });
}

// =========================================================================
// UI UPDATES & DOM BINDINGS
// =========================================================================
function updateUI() {
  if (!state.simState) return;

  const s = state.simState;
  const stats = s.stats;

  // Header Clocks & KPIs
  dom.clockMinute.textContent = s.current_minute;
  dom.clockMax.textContent = s.max_time;

  const progressPct = Math.min(100, (s.current_minute / s.max_time) * 100);
  dom.timeProgressBar.style.width = `${progressPct}%`;

  if (s.current_minute > s.max_time) {
    dom.simStatusText.textContent = `QUÁ GIỜ (${s.current_minute - s.max_time}p)`;
    dom.simStatusBadge.style.color = '#f59e0b';
    dom.simStatusBadge.style.borderColor = 'rgba(245, 158, 11, 0.4)';
  } else if (s.done) {
    dom.simStatusText.textContent = s.is_truncated ? 'DỪNG (TRUNCATED)' : 'HOÀN TẤT';
    dom.simStatusBadge.style.color = '#38bdf8';
  } else if (state.isPlaying) {
    dom.simStatusText.textContent = 'ĐANG CHẠY';
    dom.simStatusBadge.style.color = '#34d399';
  } else {
    dom.simStatusText.textContent = 'TẠM DỪNG';
    dom.simStatusBadge.style.color = '#94a3b8';
  }

  dom.kpiServed.textContent = stats.served_calls;
  dom.kpiPerfect.textContent = stats.perfect_calls;
  dom.kpiFailed.textContent = stats.failed_calls;
  dom.kpiReward.textContent = stats.total_reward.toFixed(2);
  dom.kpiReward.className = `kpi-num ${stats.total_reward < 0 ? 'text-danger' : 'text-success'}`;

  // Call Slots
  renderCallSlots(s.slots || []);

  // Ambulances List
  renderAmbulancesList(s.ambulances || []);

  // MDP Metrics
  dom.mdpTotalReward.textContent = stats.total_reward.toFixed(2);
  dom.rWait.textContent = stats.reward_breakdown.wait_penalty > 0 ? `-${stats.reward_breakdown.wait_penalty.toFixed(2)}` : '0.00';
  dom.rFail.textContent = stats.reward_breakdown.failure_penalty > 0 ? `-${stats.reward_breakdown.failure_penalty.toFixed(2)}` : '0.00';
  dom.rDist.textContent = stats.reward_breakdown.distance_penalty > 0 ? `-${stats.reward_breakdown.distance_penalty.toFixed(2)}` : '0.00';
  dom.rBonus.textContent = stats.reward_breakdown.completion_bonus > 0 ? `+${stats.reward_breakdown.completion_bonus.toFixed(2)}` : '0.00';
  dom.rStep.textContent = stats.last_step_reward.toFixed(2);

  if (state.mapData) {
    dom.mdpPotentialActions.textContent = state.mapData.num_potential_actions;
  }
  dom.mdpValidActions.textContent = `${s.valid_action_count} / ${state.mapData ? state.mapData.num_potential_actions : 229}`;
  dom.mdpActionId.textContent = `#${s.last_action_id}`;
  dom.mdpDecodedAction.textContent = s.last_decoded_action && s.last_decoded_action.length
    ? JSON.stringify(s.last_decoded_action)
    : '[] (Không phân công)';

  dom.successRateBar.style.width = `${stats.success_rate}%`;
  dom.successRateText.textContent = `${stats.success_rate}% (${stats.served_calls}/${stats.served_calls + stats.failed_calls})`;

  // Logs
  if (s.recent_log) {
    updateLogView(s.recent_log);
  }
}

function renderCallSlots(slots) {
  dom.slotsContainer.innerHTML = '';
  let occupiedCount = 0;

  slots.forEach((slot, idx) => {
    const card = document.createElement('div');
    if (slot) {
      occupiedCount++;
      card.className = 'slot-card occupied';
      let sevBadge = `<span class="call-badge sev-${slot.severity}">Mức ${slot.severity}</span>`;
      card.innerHTML = `
        <div class="slot-header">
          <span class="slot-name">SLOT ${idx} - Cuộc gọi #${slot.call_id}</span>
          ${sevBadge}
        </div>
        <div class="slot-details">
          <div>📍 Tuyến: N${slot.edge[0]} ➔ N${slot.edge[1]} (Cung #${slot.edge_id})</div>
          <div>⏱ Chờ: ${slot.wait_time}p | ⏳ Hạn còn: <strong>${slot.time_to_deadline}p</strong></div>
        </div>
      `;

      card.addEventListener('click', () => {
        if (!state.autoDispatch && state.selectedAmbulanceId) {
          executeManualDispatch(state.selectedAmbulanceId, slot.call_id);
        } else {
          showEntityPopup('call', slot);
        }
      });
    } else {
      card.className = 'slot-card empty';
      card.innerHTML = `
        <div class="slot-header">
          <span class="slot-name">SLOT ${idx} (Trống)</span>
          <span class="badge">Sẵn sàng</span>
        </div>
        <div class="slot-details">Chưa có cuộc gọi PENDING</div>
      `;
    }
    dom.slotsContainer.appendChild(card);
  });

  dom.pendingBadge.textContent = `${occupiedCount}/3 Pending`;
}

function renderAmbulancesList(ambulances) {
  dom.ambulanceList.innerHTML = '';
  dom.fleetCount.textContent = ambulances.length;

  let idleCount = 0;

  ambulances.forEach((amb) => {
    if (amb.status === 'Rảnh') idleCount++;

    const card = document.createElement('div');
    const isSelected = state.selectedAmbulanceId === amb.id;
    card.className = `amb-card ${getAmbulanceStatusClass(amb.status)} ${isSelected ? 'selected-for-dispatch' : ''}`;

    let targetInfo = 'Rảnh rỗi';
    if (amb.target_name) {
      targetInfo = `🎯 Đích: ${amb.target_name}`;
    }

    card.innerHTML = `
      <div class="amb-header">
        <div class="amb-title">🚑 Xe ${amb.id}</div>
        <span class="amb-status-tag">${amb.status}</span>
      </div>
      <div class="amb-info-grid">
        <div>📍 Vị trí: N${amb.current_edge[0]}➔N${amb.current_edge[1]} (+${amb.distance_from_start}km)</div>
        <div>⚡ Tốc độ: ${amb.current_speed} km/h</div>
        <div class="full-span">${targetInfo}</div>
        <div class="full-span">🛣 Quãng đường đã chạy: ${amb.total_distance_traveled} km</div>
      </div>
    `;

    card.addEventListener('click', () => {
      if (!state.autoDispatch) {
        if (amb.status === 'Rảnh') {
          state.selectedAmbulanceId = isSelected ? null : amb.id;
          renderAmbulancesList(state.simState.ambulances);
          updateManualBanner();
        } else {
          alert(`Xe ${amb.id} đang bận (${amb.status}), chỉ có thể phân công xe đang Rảnh!`);
        }
      } else {
        showEntityPopup('ambulance', amb);
      }
    });

    dom.ambulanceList.appendChild(card);
  });

  dom.idleBadge.textContent = `${idleCount}/${ambulances.length} Rảnh`;
}

function getAmbulanceStatusClass(status) {
  switch (status) {
    case 'Rảnh': return 'status-idle';
    case 'Đang đến cuộc gọi': return 'status-enroute';
    case 'Xử lý tại cuộc gọi': return 'status-scene';
    case 'Đang tới bệnh viện': return 'status-hospital';
    case 'Xử lý tại bệnh viện': return 'status-handover';
    default: return '';
  }
}

function renderHospitalsList() {
  if (!state.mapData) return;
  dom.hospitalList.innerHTML = '';

  state.mapData.hospitals.forEach((hosp) => {
    const item = document.createElement('div');
    item.className = 'hosp-item';
    item.innerHTML = `
      <div class="hosp-name">
        <span class="hosp-cross">✚</span>
        <span>BV ${hosp.hospital_id}: ${hosp.name}</span>
      </div>
      <div class="hosp-meta">
        <div>Cung N${hosp.from_node}➔N${hosp.to_node}</div>
        <div>Bàn giao: ${hosp.handover_time} phút</div>
      </div>
    `;
    item.addEventListener('click', () => showEntityPopup('hospital', hosp));
    dom.hospitalList.appendChild(item);
  });
}

function updateLogView(logText) {
  dom.logText.textContent = logText;
  dom.logContainer.scrollTop = 0;
}

// =========================================================================
// SIMULATION CONTROL API CALLS
// =========================================================================
async function stepSimulation(count = 1) {
  try {
    let res;
    if (count === 1) {
      res = await fetch('/api/step', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ auto_dispatch: state.autoDispatch }),
      });
    } else {
      res = await fetch('/api/step_multi', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ count, auto_dispatch: state.autoDispatch }),
      });
    }

    const prevState = state.simState;
    state.simState = await res.json();

    // Check events for audio feedback
    if (prevState && state.simState) {
      if (state.simState.stats.served_calls > prevState.stats.served_calls) {
        sfx.callCompleted();
      }
      if (state.simState.stats.failed_calls > prevState.stats.failed_calls) {
        sfx.callFailed();
      }
      if (state.simState.calls.length > prevState.calls.length) {
        sfx.callGenerated();
      }
      if (state.simState.last_decoded_action && state.simState.last_decoded_action.length > 0) {
        sfx.ambulanceDispatched();
      }
    }

    updateUI();

    if (state.simState.done && state.isPlaying) {
      stopPlayback();
    }
  } catch (err) {
    console.error('Lỗi khi thực hiện bước mô phỏng:', err);
    stopPlayback();
  }
}

async function executeManualDispatch(ambulanceId, callId) {
  try {
    const res = await fetch('/api/manual_dispatch', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ ambulance_id: ambulanceId, call_id: callId }),
    });

    state.simState = await res.json();
    sfx.ambulanceDispatched();

    state.selectedAmbulanceId = null;
    updateManualBanner();
    updateUI();
  } catch (err) {
    console.error('Lỗi khi phân công thủ công:', err);
  }
}

async function resetSimulation(config = null) {
  stopPlayback();
  try {
    const res = await fetch('/api/reset', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ config }),
    });
    state.simState = await res.json();
    state.selectedAmbulanceId = null;
    updateManualBanner();
    updateUI();
  } catch (err) {
    console.error('Lỗi khi reset mô phỏng:', err);
  }
}

// Playback Loop
function startPlayback() {
  if (state.simState && state.simState.done) {
    resetSimulation();
  }
  state.isPlaying = true;
  dom.btnPlay.className = 'btn btn-primary';
  dom.btnPlay.style.background = 'linear-gradient(135deg, #10b981, #059669)';
  dom.playText.textContent = 'Tạm dừng';

  const runStep = async () => {
    if (!state.isPlaying) return;
    await stepSimulation(1);
    if (state.isPlaying && !state.simState.done) {
      state.playTimer = setTimeout(runStep, state.speedDelay);
    }
  };

  runStep();
}

function stopPlayback() {
  state.isPlaying = false;
  if (state.playTimer) {
    clearTimeout(state.playTimer);
    state.playTimer = null;
  }
  dom.btnPlay.className = 'btn btn-primary';
  dom.btnPlay.style.background = '';
  dom.playText.textContent = 'Chạy';
  updateUI();
}

function togglePlayback() {
  if (state.isPlaying) {
    stopPlayback();
  } else {
    startPlayback();
  }
}

// =========================================================================
// EVENT LISTENERS & POPUPS
// =========================================================================
function setupEventListeners() {
  // Playback controls
  dom.btnPlay.addEventListener('click', togglePlayback);
  dom.btnStep.addEventListener('click', () => stepSimulation(1));
  dom.btnStep10.addEventListener('click', () => stepSimulation(10));
  dom.btnReset.addEventListener('click', () => resetSimulation());

  // Speed slider
  dom.speedSlider.addEventListener('input', (e) => {
    const idx = parseInt(e.target.value, 10) - 1;
    state.speedDelay = SPEED_DELAYS[idx];
    dom.speedValText.textContent = SPEED_LABELS[idx];
  });

  // Sound toggle
  dom.btnSound.addEventListener('click', () => {
    state.soundEnabled = !state.soundEnabled;
    dom.soundIcon.textContent = state.soundEnabled ? '🔊' : '🔇';
  });

  // Export Log
  dom.btnExportLog.addEventListener('click', () => {
    window.open('/api/log_export', '_blank');
  });

  // Dispatch mode toggle
  dom.modeAuto.addEventListener('click', () => {
    state.autoDispatch = true;
    dom.modeAuto.classList.add('active');
    dom.modeManual.classList.remove('active');
    state.selectedAmbulanceId = null;
    updateManualBanner();
  });

  dom.modeManual.addEventListener('click', () => {
    state.autoDispatch = false;
    dom.modeManual.classList.add('active');
    dom.modeAuto.classList.remove('active');
    updateManualBanner();
  });

  dom.btnCancelManual.addEventListener('click', () => {
    state.selectedAmbulanceId = null;
    updateManualBanner();
    renderAmbulancesList(state.simState.ambulances);
  });

  // Map Zoom & Pan
  dom.btnZoomIn.addEventListener('click', () => {
    state.view.zoom = Math.min(state.view.maxZoom, state.view.zoom * 1.25);
  });
  dom.btnZoomOut.addEventListener('click', () => {
    state.view.zoom = Math.max(state.view.minZoom, state.view.zoom / 1.25);
  });
  dom.btnZoomFit.addEventListener('click', fitMapToCanvas);

  // Canvas Mouse Interactions
  dom.canvas.addEventListener('mousedown', (e) => {
    state.view.isDragging = true;
    state.view.dragStartX = e.clientX - state.view.panX;
    state.view.dragStartY = e.clientY - state.view.panY;
  });

  window.addEventListener('mouseup', () => {
    state.view.isDragging = false;
  });

  window.addEventListener('mousemove', (e) => {
    if (state.view.isDragging) {
      state.view.panX = e.clientX - state.view.dragStartX;
      state.view.panY = e.clientY - state.view.dragStartY;
    }
  });

  dom.canvas.addEventListener('wheel', (e) => {
    e.preventDefault();
    const zoomFactor = e.deltaY < 0 ? 1.15 : 0.85;
    const newZoom = Math.max(state.view.minZoom, Math.min(state.view.maxZoom, state.view.zoom * zoomFactor));

    // Zoom towards mouse position
    const rect = dom.canvasWrapper.getBoundingClientRect();
    const mx = e.clientX - rect.left - rect.width / 2;
    const my = e.clientY - rect.top - rect.height / 2;

    state.view.panX -= (mx - state.view.panX) * (zoomFactor - 1);
    state.view.panY -= (my - state.view.panY) * (zoomFactor - 1);
    state.view.zoom = newZoom;
  }, { passive: false });

  // Canvas Click Detection (Ambulance, Call, Hospital, Node)
  dom.canvas.addEventListener('click', (e) => {
    const rect = dom.canvasWrapper.getBoundingClientRect();
    const clickX = e.clientX - rect.left;
    const clickY = e.clientY - rect.top;

    const hit = findEntityAt(clickX, clickY);
    if (hit) {
      if (!state.autoDispatch && hit.type === 'call' && state.selectedAmbulanceId) {
        executeManualDispatch(state.selectedAmbulanceId, hit.data.id);
      } else if (!state.autoDispatch && hit.type === 'ambulance' && hit.data.status === 'Rảnh') {
        state.selectedAmbulanceId = hit.data.id;
        updateManualBanner();
        renderAmbulancesList(state.simState.ambulances);
      } else {
        showEntityPopup(hit.type, hit.data);
      }
    } else {
      closeEntityPopup();
    }
  });

  dom.popupClose.addEventListener('click', closeEntityPopup);

  // Tabs
  document.querySelectorAll('.tab-btn').forEach((btn) => {
    btn.addEventListener('click', (e) => {
      document.querySelectorAll('.tab-btn').forEach((b) => b.classList.remove('active'));
      document.querySelectorAll('.tab-content').forEach((c) => c.classList.remove('active'));

      btn.classList.add('active');
      const targetId = btn.dataset.tab;
      document.getElementById(targetId).classList.add('active');
    });
  });

  // Copy Log
  dom.btnCopyLog.addEventListener('click', () => {
    navigator.clipboard.writeText(dom.logText.textContent);
    alert('Đã sao chép nhật ký phút hiện tại vào clipboard!');
  });

  // Config Modal
  dom.btnConfig.addEventListener('click', () => {
    if (state.simState) {
      dom.cfgAmbulances.value = state.mapData.config.num_ambulances;
      dom.cfgLambda.value = state.mapData.config.poisson_lambda;
      dom.cfgMaxPending.value = state.mapData.config.max_pending_calls;
      dom.cfgMaxTime.value = state.mapData.config.max_time;
      dom.cfgOvertime.value = state.mapData.config.max_overtime;
      dom.cfgSeed.value = state.mapData.config.seed;
    }
    dom.configModal.style.display = 'flex';
  });

  dom.modalCloseBtn.addEventListener('click', () => dom.configModal.style.display = 'none');
  dom.btnModalCancel.addEventListener('click', () => dom.configModal.style.display = 'none');

  dom.btnModalSave.addEventListener('click', () => {
    const newCfg = {
      num_ambulances: parseInt(dom.cfgAmbulances.value, 10),
      poisson_lambda: parseFloat(dom.cfgLambda.value),
      max_pending_calls: parseInt(dom.cfgMaxPending.value, 10),
      max_time: parseInt(dom.cfgMaxTime.value, 10),
      max_overtime: parseInt(dom.cfgOvertime.value, 10),
      seed: parseInt(dom.cfgSeed.value, 10),
    };
    dom.configModal.style.display = 'none';
    resetSimulation(newCfg);
  });

  // Keyboard Shortcuts (Space for Play/Pause, Arrow Right for Step)
  window.addEventListener('keydown', (e) => {
    if (e.target.tagName === 'INPUT') return;
    if (e.code === 'Space') {
      e.preventDefault();
      togglePlayback();
    } else if (e.code === 'ArrowRight') {
      e.preventDefault();
      stepSimulation(1);
    }
  });
}

function updateManualBanner() {
  if (state.autoDispatch) {
    dom.manualBanner.style.display = 'none';
  } else {
    dom.manualBanner.style.display = 'flex';
    if (state.selectedAmbulanceId) {
      document.getElementById('manual-instruction').textContent =
        `Đã chọn Xe ${state.selectedAmbulanceId}. Hãy CLICK vào một cuộc gọi (hoặc Slot) để phân công!`;
    } else {
      document.getElementById('manual-instruction').textContent =
        'Chế độ thủ công: Hãy CLICK chọn 1 xe cứu thương đang RẢNH (IDLE).';
    }
  }
}

// Find entity at screen coordinates
function findEntityAt(sx, sy) {
  if (!state.mapData || !state.simState) return null;

  // 1. Check Ambulances
  for (const amb of state.simState.ambulances || []) {
    const pos = worldToScreen(amb.coord[0], amb.coord[1]);
    if (Math.hypot(sx - pos.x, sy - pos.y) <= 18) {
      return { type: 'ambulance', data: amb };
    }
  }

  // 2. Check Calls
  for (const call of state.simState.calls || []) {
    const pos = worldToScreen(call.coord[0], call.coord[1]);
    if (Math.hypot(sx - pos.x, sy - pos.y) <= 18) {
      return { type: 'call', data: call };
    }
  }

  // 3. Check Hospitals
  for (const hosp of state.mapData.hospitals || []) {
    const pos = worldToScreen(hosp.coord[0], hosp.coord[1]);
    if (Math.hypot(sx - pos.x, sy - pos.y) <= 18) {
      return { type: 'hospital', data: hosp };
    }
  }

  // 4. Check Nodes
  for (const node of state.mapData.nodes || []) {
    const pos = worldToScreen(node.x, node.y);
    if (Math.hypot(sx - pos.x, sy - pos.y) <= 10) {
      return { type: 'node', data: node };
    }
  }

  return null;
}

function showEntityPopup(type, data) {
  dom.inspectorPopup.style.display = 'block';

  if (type === 'ambulance') {
    dom.popupTitle.textContent = `🚑 Xe Cứu Thương ${data.id}`;
    dom.popupBody.innerHTML = `
      <div><strong>Trạng thái:</strong> ${data.status}</div>
      <div><strong>Vị trí:</strong> N${data.current_edge[0]} ➔ N${data.current_edge[1]} (+${data.distance_from_start} km)</div>
      <div><strong>Vận tốc hiện tại:</strong> ${data.current_speed} km/h</div>
      <div><strong>Mục tiêu:</strong> ${data.target_name || 'Không'}</div>
      <div><strong>Bệnh viện đích:</strong> ${data.assigned_hospital_id ? 'BV ' + data.assigned_hospital_id : 'Chưa có'}</div>
      <div><strong>Thời gian sơ cứu còn lại:</strong> ${data.scene_time_remaining} phút</div>
      <div><strong>Thời gian bàn giao còn lại:</strong> ${data.hospital_time_remaining} phút</div>
      <div><strong>Tổng quãng đường:</strong> ${data.total_distance_traveled} km</div>
    `;
  } else if (type === 'call') {
    dom.popupTitle.textContent = `📞 Cuộc Gọi Cấp Cứu #${data.id}`;
    dom.popupBody.innerHTML = `
      <div><strong>Mức độ nghiêm trọng:</strong> Mức ${data.severity}</div>
      <div><strong>Trạng thái:</strong> ${data.status}</div>
      <div><strong>Vị trí:</strong> Cung N${data.edge[0]} ➔ N${data.edge[1]} (+${data.distance_from_start} km)</div>
      <div><strong>Thời điểm phát sinh:</strong> Phút thứ ${data.created_time}</div>
      <div><strong>Thời gian cho phép:</strong> ${data.failure_time} phút</div>
      <div><strong>Thời gian sơ cứu cần thiết:</strong> ${data.processing_time} phút</div>
      <div><strong>Thời gian đã chờ:</strong> ${data.wait_time} phút</div>
      <div><strong>Hạn chót còn lại:</strong> ${data.time_to_deadline} phút</div>
      <div><strong>Xe cứu thương phân công:</strong> ${data.assigned_ambulance_id ? 'Xe ' + data.assigned_ambulance_id : 'Chưa phân công'}</div>
    `;
  } else if (type === 'hospital') {
    dom.popupTitle.textContent = `🏥 Bệnh Viện ${data.hospital_id}`;
    dom.popupBody.innerHTML = `
      <div><strong>Tên cơ sở:</strong> ${data.name}</div>
      <div><strong>Vị trí:</strong> Cung có hướng N${data.from_node} ➔ N${data.to_node}</div>
      <div><strong>Khoảng cách từ điểm đi:</strong> +${data.distance_from_start} km (Tuyến dài ${data.road_length} km)</div>
      <div><strong>Thời gian gửi/bàn giao:</strong> ${data.handover_time} phút</div>
    `;
  } else if (type === 'node') {
    dom.popupTitle.textContent = `📍 Nút Giao Thông #${data.id}`;
    dom.popupBody.innerHTML = `
      <div><strong>Khu vực:</strong> ${data.is_central ? 'Trung tâm thành phố' : 'Ngoại ô'}</div>
      <div><strong>Kiểu giao lộ:</strong> ${data.type} (Bậc ${data.degree})</div>
      <div><strong>Tọa độ:</strong> (${data.x.toFixed(2)}, ${data.y.toFixed(2)}) km</div>
      <div><strong>Các nút liên kết kề:</strong> ${data.neighbors.join(', ')}</div>
    `;
  }
}

function closeEntityPopup() {
  dom.inspectorPopup.style.display = 'none';
}

// Start app
window.addEventListener('DOMContentLoaded', init);

