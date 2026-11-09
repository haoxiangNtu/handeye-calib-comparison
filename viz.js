import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { CSS2DRenderer, CSS2DObject } from 'three/addons/renderers/CSS2DRenderer.js';
import * as M from './calib-math.js';
import * as S from './sim.js';

THREE.Object3D.DEFAULT_UP.set(0, 0, 1);

const $ = (id) => document.getElementById(id);
const host = $('viz');
const state = { method: 'A', seed: 1, n: 20, current: 0, magnify: 20, showAll: true, showEst: true, showLabels: true };

const params = {
  A: { depthMm: 4, rotDeg: 1, sizePct: 0 },
  B: { spotMm: 1, scalePct: 0, fkMm: 1 },
};
const sliderDefs = {
  A: [
    ['depthMm', 'PnP 深度噪声', 0, 15, 0.5, 'mm'],
    ['rotDeg', 'PnP 姿态噪声', 0, 5, 0.1, '°'],
    ['sizePct', 'Tag 尺寸误差', -5, 5, 0.1, '%'],
  ],
  B: [
    ['spotMm', '光斑三角化噪声', 0, 5, 0.1, 'mm'],
    ['scalePct', '双目尺度误差', -3, 3, 0.1, '%'],
    ['fkMm', 'FK / 激光射线误差', 0, 5, 0.1, 'mm'],
  ],
};

let poses, noise;
function regenerate() {
  poses = { A: S.generateTagPoses(state.seed), B: S.generateLaserPoses(state.seed) };
  noise = S.generateNoise(state.seed);
}
regenerate();
if (location.hash === '#B') state.method = 'B';

// ---------- three.js setup ----------
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
host.appendChild(renderer.domElement);
const labels = new CSS2DRenderer();
labels.domElement.className = 'viz-labels';
host.appendChild(labels.domElement);

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(40, 1, 0.01, 20);
const HOME = [[1.45, 0.80, 1.05], [0.34, 0.0, 0.14]];
camera.position.set(...HOME[0]);
const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(...HOME[1]);
controls.enableDamping = true;
controls.update();

scene.add(new THREE.AmbientLight(0xffffff, 0.75));
const sun = new THREE.DirectionalLight(0xffffff, 0.9);
sun.position.set(1, -1, 2);
scene.add(sun);

let world = new THREE.Group();
scene.add(world);

function colors() {
  const cs = getComputedStyle(document.documentElement);
  const v = (n) => new THREE.Color(cs.getPropertyValue(n).trim());
  return { bg: v('--card'), fg: v('--fg'), muted: v('--muted'), line: v('--line'), a: v('--a'), b: v('--b'), est: v('--est'), laser: v('--laser') };
}

const vec = (p) => new THREE.Vector3(p[0], p[1], p[2]);
function poseMatrix(P) {
  const m = new THREE.Matrix4();
  m.set(P.R[0][0], P.R[0][1], P.R[0][2], P.t[0], P.R[1][0], P.R[1][1], P.R[1][2], P.t[1],
    P.R[2][0], P.R[2][1], P.R[2][2], P.t[2], 0, 0, 0, 1);
  return m;
}
function placed(obj, P) { obj.matrixAutoUpdate = false; obj.matrix.copy(poseMatrix(P)); return obj; }

function line(points, color, { dashed = false, opacity = 1, width } = {}) {
  const g = new THREE.BufferGeometry().setFromPoints(points.map(vec));
  const mat = dashed
    ? new THREE.LineDashedMaterial({ color, dashSize: 0.018, gapSize: 0.012, transparent: opacity < 1, opacity })
    : new THREE.LineBasicMaterial({ color, transparent: opacity < 1, opacity });
  const l = new THREE.Line(g, mat);
  if (dashed) l.computeLineDistances();
  return l;
}
function rod(a, b, radius, color, opacity = 1) {
  const A = vec(a), B = vec(b), dir = B.clone().sub(A), len = dir.length();
  const m = new THREE.Mesh(new THREE.CylinderGeometry(radius, radius, len, 16),
    new THREE.MeshStandardMaterial({ color, roughness: 0.6, transparent: opacity < 1, opacity }));
  m.position.copy(A.clone().add(B).multiplyScalar(0.5));
  m.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir.normalize());
  return m;
}
function ball(p, r, color) {
  const m = new THREE.Mesh(new THREE.SphereGeometry(r, 16, 12), new THREE.MeshStandardMaterial({ color }));
  m.position.copy(vec(p));
  return m;
}
function axes(len = 0.05, opacity = 1) {
  const g = new THREE.Group();
  [[0xd9534f, [len, 0, 0]], [0x4caf50, [0, len, 0]], [0x3f7fd9, [0, 0, len]]].forEach(([c, e]) => g.add(line([[0, 0, 0], e], c, { opacity })));
  return g;
}
function label(text, p, cls = '') {
  if (!state.showLabels) return new THREE.Group();
  const div = document.createElement('div');
  div.className = 'viz-lbl ' + cls;
  div.textContent = text;
  const o = new CSS2DObject(div);
  o.position.copy(vec(p));
  return o;
}
function midpoint(a, b, lift = 0.015, f = 0.5) { return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f + lift]; }

// Camera body + frustum, drawn in the optical frame (+z forward).
function cameraGlyph(color, opacity = 1, stereo = false) {
  const g = new THREE.Group();
  const w = 0.045, h = 0.028, d = 0.07;
  const c = [[-w, -h, d], [w, -h, d], [w, h, d], [-w, h, d]];
  c.forEach((p) => g.add(line([[0, 0, 0], p], color, { opacity })));
  g.add(line([...c, c[0]], color, { opacity }));
  const box = new THREE.Mesh(new THREE.BoxGeometry(stereo ? S.BASELINE + 0.04 : 0.06, 0.035, 0.03),
    new THREE.MeshStandardMaterial({ color, transparent: true, opacity: 0.35 * opacity }));
  box.position.set(stereo ? S.BASELINE / 2 : 0, 0, -0.016);
  g.add(box);
  return g;
}

function tagGlyph(color, opacity) {
  const h = S.TAG_SIZE / 2, g = new THREE.Group();
  const quad = new THREE.Mesh(new THREE.PlaneGeometry(S.TAG_SIZE, S.TAG_SIZE),
    new THREE.MeshStandardMaterial({ color, side: THREE.DoubleSide, transparent: true, opacity: 0.25 * opacity }));
  g.add(quad);
  g.add(line([[-h, -h, 0], [h, -h, 0], [h, h, 0], [-h, h, 0], [-h, -h, 0]], color, { opacity }));
  // a small inner square, like a tag code
  const q = h * 0.45;
  g.add(line([[-q, -q, 0], [q, -q, 0], [q, q, 0], [-q, q, 0], [-q, -q, 0]], color, { opacity }));
  return g;
}

// Very small 2-link arm reaching the wrist point above the flange.
function arm(flange, color) {
  const g = new THREE.Group();
  const shoulder = [0, 0, 0.24], L1 = 0.34, L2 = 0.34;
  const zf = M.col(flange.R, 2);
  const wrist = M.sub(flange.t, M.scale(zf, 0.09));
  const horiz = M.unit([wrist[0], wrist[1], 0]);
  const r = Math.hypot(wrist[0], wrist[1]), dz = wrist[2] - shoulder[2];
  const D = Math.min(Math.hypot(r, dz), L1 + L2 - 1e-3);
  const a = Math.atan2(dz, r), b = Math.acos((L1 * L1 + D * D - L2 * L2) / (2 * L1 * D));
  const elbow = M.add(shoulder, M.add(M.scale(horiz, L1 * Math.cos(a + b)), [0, 0, L1 * Math.sin(a + b)]));
  g.add(rod([0, 0, 0], shoulder, 0.045, color));
  g.add(rod(shoulder, elbow, 0.032, color));
  g.add(rod(elbow, wrist, 0.028, color));
  g.add(rod(wrist, flange.t, 0.022, color));
  [shoulder, elbow, wrist].forEach((p) => g.add(ball(p, 0.04, color)));
  const disk = rod(M.sub(flange.t, M.scale(zf, 0.006)), flange.t, 0.035, color);
  g.add(disk);
  return g;
}

function magnifiedPose(Yest, Ytrue, k) {
  const dR = M.matMul(Yest.R, M.T(Ytrue.R));
  const w = M.logRot(dR);
  return M.pose(M.matMul(M.rotvec(M.scale(w, k)), Ytrue.R), M.add(Ytrue.t, M.scale(M.sub(Yest.t, Ytrue.t), k)));
}

// ---------- scene build ----------
function build() {
  const C = colors();
  scene.background = C.bg;
  // CSS2D labels keep their DOM nodes unless removed explicitly; dispose GPU resources too.
  world.traverse((o) => {
    if (o.isCSS2DObject) o.element.remove();
    if (o.geometry) o.geometry.dispose();
    if (o.material) o.material.dispose();
  });
  scene.remove(world);
  world = new THREE.Group();
  scene.add(world);
  const isA = state.method === 'A';
  const n = state.n;
  const cur = Math.min(state.current, n - 1);
  const Ytrue = S.trueCamera();
  const result = isA ? S.solveTag(poses.A, noise, n, params.A) : S.solveLaser(poses.B, noise, n, params.B);
  const accent = isA ? C.a : C.b;

  // Table, grid and board.
  const table = new THREE.Mesh(new THREE.PlaneGeometry(1.4, 1.2), new THREE.MeshStandardMaterial({ color: C.line, transparent: true, opacity: 0.35 }));
  table.position.set(0.35, 0.05, -0.002);
  world.add(table);
  const grid = new THREE.GridHelper(1.4, 14, C.line, C.line);
  grid.rotation.x = Math.PI / 2;
  grid.position.set(0.35, 0.05, -0.001);
  world.add(grid);
  const bw = S.BOARD.x[1] - S.BOARD.x[0], bh = S.BOARD.y[1] - S.BOARD.y[0];
  const board = new THREE.Mesh(new THREE.PlaneGeometry(bw, bh),
    new THREE.MeshStandardMaterial({ color: 0xffffff, transparent: true, opacity: isA ? 0.35 : 0.95 }));
  board.position.set((S.BOARD.x[0] + S.BOARD.x[1]) / 2, (S.BOARD.y[0] + S.BOARD.y[1]) / 2, 0.0005);
  world.add(board);
  if (!isA) world.add(label('标定板（被动）', [S.BOARD.x[1], S.BOARD.y[1], 0.01], 'muted'));

  // Base frame and camera post.
  world.add(axes(0.12));
  world.add(label('base', [-0.05, -0.09, 0.0]));
  world.add(rod([Ytrue.t[0], Ytrue.t[1] - 0.03, 0], [Ytrue.t[0], Ytrue.t[1] - 0.03, Ytrue.t[2] - 0.03], 0.012, C.muted));

  // True camera.
  const camGlyph = placed(cameraGlyph(C.fg, 1, !isA), Ytrue);
  world.add(camGlyph);
  world.add(placed(axes(0.05), Ytrue));
  world.add(label(isA ? '相机（单目）' : '双目相机', M.add(Ytrue.t, [0, -0.02, 0.06])));

  // Estimated camera (error magnified).
  if (state.showEst) {
    const P = magnifiedPose(result.Yest, Ytrue, state.magnify);
    world.add(placed(cameraGlyph(C.est, 0.9, !isA), P));
    world.add(label(`估计相机（误差 ×${state.magnify}）`, M.add(P.t, [0, 0.02, -0.05]), 'est'));
  }

  // Arm at the current sample.
  const sample = isA ? poses.A[cur] : poses.B[cur];
  world.add(arm(sample.flange, C.line.clone().lerp(C.muted, 0.5)));
  world.add(placed(axes(0.06), sample.flange));
  world.add(label('flange', M.add(sample.flange.t, [0.04, 0.02, 0.05])));

  // Known transform base -> flange (FK).
  world.add(line([[0, 0, 0], sample.flange.t], C.fg, { opacity: 0.8 }));
  world.add(label('FK(q) 已知', midpoint([0, 0, 0], sample.flange.t, 0.02, 0.35), 'muted'));
  // Unknown transform base -> camera.
  world.add(line([[0, 0, 0], Ytrue.t], accent, { dashed: true }));
  world.add(label(isA ? 'Y = T_base_cam（未知）' : 'T_base_cam（未知）', midpoint([0, 0, 0], Ytrue.t, -0.03, 0.55), isA ? 'a' : 'b'));

  if (isA) {
    // All tag poses (faint), current tag highlighted.
    if (state.showAll) for (let i = 0; i < n; i++) if (i !== cur) world.add(placed(tagGlyph(C.a, 0.14), poses.A[i].tag));
    world.add(placed(tagGlyph(C.a, 1), sample.tag));
    world.add(placed(axes(0.05), sample.tag));
    world.add(label('AprilTag', M.add(sample.tag.t, [0, 0, 0.08]), 'a'));
    world.add(line([sample.flange.t, sample.tag.t], C.a, { dashed: true }));
    world.add(label('X（未知）', midpoint(sample.flange.t, sample.tag.t, 0.02), 'a'));
    world.add(line([Ytrue.t, sample.tag.t], C.fg));
    world.add(label('B：PnP 6D 位姿', midpoint(Ytrue.t, sample.tag.t, -0.025, 0.4), 'muted'));
  } else {
    const spot = sample.spot;
    const origin = M.apply(sample.flange, S.LASER_O);
    if (state.showAll) for (let i = 0; i < n; i++) if (i !== cur) world.add(ball(poses.B[i].spot, 0.006, C.laser));
    world.add(rod(origin, spot, 0.0025, C.laser));
    world.add(ball(spot, 0.01, C.laser));
    world.add(label(`激光 d = ${(sample.d * 1000).toFixed(0)} mm`, midpoint(origin, spot, 0), 'b'));
    world.add(label('光斑 p', M.add(spot, [0, 0.03, 0.02]), 'b'));
    const left = Ytrue.t, right = M.add(Ytrue.t, M.scale(M.col(Ytrue.R, 0), S.BASELINE));
    world.add(line([left, spot], C.fg, { opacity: 0.7 }));
    world.add(line([right, spot], C.fg, { opacity: 0.7 }));
    world.add(label('左右目三角化', midpoint(right, spot, 0.02, 0.35), 'muted'));
  }

  // Test points used for the error readout.
  world.add(ball(S.TEST_ON_BOARD, 0.007, C.fg));
  world.add(ball(S.TEST_ABOVE, 0.007, C.fg));
  world.add(line([S.TEST_ON_BOARD, S.TEST_ABOVE], C.muted, { dashed: true, opacity: 0.6 }));
  world.add(label('测试点 ①', M.add(S.TEST_ON_BOARD, [0.03, 0, 0.01]), 'muted'));
  world.add(label('测试点 ②', M.add(S.TEST_ABOVE, [0.03, 0, 0.01]), 'muted'));

  showMetrics(result);
}

function showMetrics(r) {
  const isA = state.method === 'A';
  $('m-res').textContent = r.residualMm.toFixed(2);
  $('m-res-label').textContent = isA ? '闭环平移残差 RMS' : '点配准残差 RMS';
  $('m-pos').textContent = r.camPosMm.toFixed(2);
  $('m-rot').textContent = r.camRotDeg.toFixed(2);
  $('m-on').textContent = r.onBoardMm.toFixed(2);
  $('m-above').textContent = r.aboveMm.toFixed(2);
}

// ---------- controls ----------
function renderSliders() {
  const box = $('noise-sliders');
  box.innerHTML = '';
  for (const [key, text, min, max, step, unit] of sliderDefs[state.method]) {
    const row = document.createElement('label');
    row.className = 'ctl';
    row.innerHTML = `<span>${text}</span><input type="range" min="${min}" max="${max}" step="${step}" value="${params[state.method][key]}"><output>${params[state.method][key]} ${unit}</output>`;
    const input = row.querySelector('input'), out = row.querySelector('output');
    input.addEventListener('input', () => {
      params[state.method][key] = parseFloat(input.value);
      out.textContent = `${input.value} ${unit}`;
      build();
    });
    box.appendChild(row);
  }
}

function bindRange(id, key, fmt = (v) => v) {
  const input = $(id), out = $(id + '-out');
  const sync = () => { out.textContent = fmt(state[key]); };
  input.addEventListener('input', () => {
    state[key] = parseFloat(input.value);
    if (key === 'n') { $('cur').max = state.n; if (state.current >= state.n) { state.current = state.n - 1; $('cur').value = state.n; $('cur-out').textContent = state.n; } }
    sync();
    build();
  });
  sync();
}

document.querySelectorAll('[data-method]').forEach((btn) => btn.addEventListener('click', () => {
  state.method = btn.dataset.method;
  document.querySelectorAll('[data-method]').forEach((b) => b.setAttribute('aria-pressed', String(b === btn)));
  host.dataset.method = state.method;
  renderSliders();
  build();
}));
bindRange('n', 'n');
$('cur').addEventListener('input', () => { state.current = parseInt($('cur').value, 10) - 1; $('cur-out').textContent = $('cur').value; build(); });
bindRange('mag', 'magnify', (v) => `×${v}`);
$('show-all').addEventListener('change', (e) => { state.showAll = e.target.checked; build(); });
$('show-labels').addEventListener('change', (e) => { state.showLabels = e.target.checked; build(); });
$('show-est').addEventListener('change', (e) => { state.showEst = e.target.checked; build(); });
$('reseed').addEventListener('click', () => { state.seed += 1; regenerate(); build(); });
$('reset-view').addEventListener('click', () => { camera.position.set(...HOME[0]); controls.target.set(...HOME[1]); controls.update(); });

let playing = null;
$('play').addEventListener('click', () => {
  if (playing) { clearInterval(playing); playing = null; $('play').textContent = '▶ 逐个播放样本'; return; }
  $('play').textContent = '❚❚ 暂停';
  playing = setInterval(() => {
    state.current = (state.current + 1) % state.n;
    $('cur').value = state.current + 1; $('cur-out').textContent = state.current + 1;
    build();
  }, 900);
});

// ---------- sizing / theme / loop ----------
function resize() {
  const w = host.clientWidth, h = Math.max(360, Math.min(560, Math.round(w * 0.62)));
  renderer.setSize(w, h);
  labels.setSize(w, h);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}
new ResizeObserver(resize).observe(host);
window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', build);
resize();
if (state.method === 'B') document.querySelector('[data-method="B"]').click();
renderSliders();
build();
(function loop() {
  requestAnimationFrame(loop);
  controls.update();
  renderer.render(scene, camera);
  labels.render(scene, camera);
})();
