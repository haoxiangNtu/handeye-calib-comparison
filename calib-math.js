// Small rigid-geometry toolkit for the interactive demo.
// Rotations are 3x3 row-major arrays, poses are {R, t}, vectors are [x, y, z].

export const I3 = () => [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
export const add = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
export const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
export const scale = (a, s) => [a[0] * s, a[1] * s, a[2] * s];
export const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
export const norm = (a) => Math.sqrt(dot(a, a));
export const unit = (a) => scale(a, 1 / norm(a));
export const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];

export function matMul(A, B) {
  const C = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
  for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++)
    for (let k = 0; k < 3; k++) C[i][j] += A[i][k] * B[k][j];
  return C;
}
export const T = (A) => [[A[0][0], A[1][0], A[2][0]], [A[0][1], A[1][1], A[2][1]], [A[0][2], A[1][2], A[2][2]]];
export const mulVec = (A, v) => [dot(A[0], v), dot(A[1], v), dot(A[2], v)];
export const col = (A, k) => [A[0][k], A[1][k], A[2][k]];
export const fromCols = (x, y, z) => [[x[0], y[0], z[0]], [x[1], y[1], z[1]], [x[2], y[2], z[2]]];

export const pose = (R, t) => ({ R, t });
export const compose = (A, B) => ({ R: matMul(A.R, B.R), t: add(mulVec(A.R, B.t), A.t) });
export const inv = (A) => { const Rt = T(A.R); return { R: Rt, t: scale(mulVec(Rt, A.t), -1) }; };
export const apply = (A, p) => add(mulVec(A.R, p), A.t);

// Rotation from a rotation vector (axis * angle, radians).
export function rotvec(w) {
  const th = norm(w);
  if (th < 1e-12) return I3();
  const k = scale(w, 1 / th), c = Math.cos(th), s = Math.sin(th), v = 1 - c;
  const [x, y, z] = k;
  return [
    [c + x * x * v, x * y * v - z * s, x * z * v + y * s],
    [y * x * v + z * s, c + y * y * v, y * z * v - x * s],
    [z * x * v - y * s, z * y * v + x * s, c + z * z * v],
  ];
}

export function rotAngleDeg(A, B) {
  const R = matMul(T(A), B);
  const c = Math.min(1, Math.max(-1, (R[0][0] + R[1][1] + R[2][2] - 1) / 2));
  return Math.acos(c) * 180 / Math.PI;
}

// Rotation taking unit vector a onto unit vector b.
export function alignVectors(a, b) {
  const v = cross(a, b), c = dot(a, b);
  if (c < -0.999999) {
    const axis = unit(Math.abs(a[0]) < 0.9 ? cross(a, [1, 0, 0]) : cross(a, [0, 1, 0]));
    return rotvec(scale(axis, Math.PI));
  }
  const s = norm(v);
  if (s < 1e-12) return I3();
  return rotvec(scale(v, Math.atan2(s, c) / s));
}

// Symmetric Jacobi eigen-decomposition; returns {values, vectors (columns)}.
function jacobiEigen(Ain) {
  const n = Ain.length;
  const A = Ain.map((r) => r.slice());
  const V = Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => (i === j ? 1 : 0)));
  for (let sweep = 0; sweep < 100; sweep++) {
    let off = 0;
    for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) off += A[p][q] * A[p][q];
    if (off < 1e-24) break;
    for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) {
      if (Math.abs(A[p][q]) < 1e-30) continue;
      const theta = (A[q][q] - A[p][p]) / (2 * A[p][q]);
      const t = Math.sign(theta || 1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
      const c = 1 / Math.sqrt(t * t + 1), s = t * c;
      for (let k = 0; k < n; k++) {
        const akp = A[k][p], akq = A[k][q];
        A[k][p] = c * akp - s * akq; A[k][q] = s * akp + c * akq;
      }
      for (let k = 0; k < n; k++) {
        const apk = A[p][k], aqk = A[q][k];
        A[p][k] = c * apk - s * aqk; A[q][k] = s * apk + c * aqk;
      }
      for (let k = 0; k < n; k++) {
        const vkp = V[k][p], vkq = V[k][q];
        V[k][p] = c * vkp - s * vkq; V[k][q] = s * vkp + c * vkq;
      }
    }
  }
  return { values: A.map((r, i) => r[i]), vectors: V };
}

// Horn's quaternion method: rotation R minimising sum w_i |R a_i - b_i|^2.
export function bestRotation(as, bs, ws) {
  const S = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
  as.forEach((a, i) => {
    const w = ws ? ws[i] : 1, b = bs[i];
    for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) S[r][c] += w * a[r] * b[c];
  });
  const [[xx, xy, xz], [yx, yy, yz], [zx, zy, zz]] = S;
  const N = [
    [xx + yy + zz, yz - zy, zx - xz, xy - yx],
    [yz - zy, xx - yy - zz, xy + yx, zx + xz],
    [zx - xz, xy + yx, -xx + yy - zz, yz + zy],
    [xy - yx, zx + xz, yz + zy, -xx - yy + zz],
  ];
  const { values, vectors } = jacobiEigen(N);
  let best = 0;
  for (let i = 1; i < 4; i++) if (values[i] > values[best]) best = i;
  const [w, x, y, z] = vectors.map((r) => r[best]);
  return [
    [w * w + x * x - y * y - z * z, 2 * (x * y - w * z), 2 * (x * z + w * y)],
    [2 * (x * y + w * z), w * w - x * x + y * y - z * z, 2 * (y * z - w * x)],
    [2 * (x * z - w * y), 2 * (y * z + w * x), w * w - x * x - y * y + z * z],
  ];
}

// Chordal mean of rotation matrices.
export function meanRotation(Rs) {
  const as = [], bs = [];
  for (const R of Rs) for (let k = 0; k < 3; k++) { as.push(col(I3(), k)); bs.push(col(R, k)); }
  return bestRotation(as, bs);
}

// Weighted Kabsch: pose P with dst ~= P * src.
export function rigidFit(src, dst, ws) {
  const w = ws || src.map(() => 1);
  const W = w.reduce((s, v) => s + v, 0);
  let cs = [0, 0, 0], cd = [0, 0, 0];
  src.forEach((p, i) => { cs = add(cs, scale(p, w[i] / W)); cd = add(cd, scale(dst[i], w[i] / W)); });
  const R = bestRotation(src.map((p) => sub(p, cs)), dst.map((p) => sub(p, cd)), w);
  return { R, t: sub(cd, mulVec(R, cs)) };
}

// A_i X = Y B_i by alternating closed-form updates (rotation chordal means + linear translation).
export function solveAXYB(As, Bs, iters = 60) {
  let X = { R: I3(), t: [0, 0, 0] }, Y = null;
  const n = As.length;
  for (let it = 0; it < iters; it++) {
    const RY = meanRotation(As.map((A, i) => matMul(matMul(A.R, X.R), T(Bs[i].R))));
    let tY = [0, 0, 0];
    As.forEach((A, i) => { tY = add(tY, scale(sub(add(mulVec(A.R, X.t), A.t), mulVec(RY, Bs[i].t)), 1 / n)); });
    Y = { R: RY, t: tY };
    const RX = meanRotation(As.map((A, i) => matMul(matMul(T(A.R), Y.R), Bs[i].R)));
    let tX = [0, 0, 0];
    As.forEach((A, i) => { tX = add(tX, scale(mulVec(T(A.R), sub(add(mulVec(Y.R, Bs[i].t), Y.t), A.t)), 1 / n)); });
    X = { R: RX, t: tX };
  }
  return refineAXYB(As, Bs, X, Y);
}

// Rotation matrix -> rotation vector.
export function logRot(R) {
  const c = Math.min(1, Math.max(-1, (R[0][0] + R[1][1] + R[2][2] - 1) / 2));
  const th = Math.acos(c);
  const v = [R[2][1] - R[1][2], R[0][2] - R[2][0], R[1][0] - R[0][1]];
  if (th < 1e-9) return scale(v, 0.5);
  return scale(v, th / (2 * Math.sin(th)));
}

function solveLinear(A, b) {
  const n = b.length, M = A.map((r, i) => [...r, b[i]]);
  for (let c = 0; c < n; c++) {
    let p = c;
    for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r;
    [M[c], M[p]] = [M[p], M[c]];
    for (let r = 0; r < n; r++) if (r !== c) {
      const f = M[r][c] / M[c][c];
      for (let k = c; k <= n; k++) M[r][k] -= f * M[c][k];
    }
  }
  return M.map((r, i) => r[n] / r[i]);
}

const perturb = (P, xi) => ({ R: matMul(rotvec(xi.slice(0, 3)), P.R), t: add(xi.slice(3), mulVec(rotvec(xi.slice(0, 3)), P.t)) });

// Weighted Gauss-Newton on SE(3) for A_i X = Y B_i (weights: 3 deg / 1 cm, as in extr_calib.py).
export function refineAXYB(As, Bs, X, Y, iters = 30) {
  const wr = 1 / (3 * Math.PI / 180), wt = 1 / 0.01;
  const resid = (X, Y) => {
    const r = [];
    As.forEach((A, i) => {
      const E = compose(inv(compose(A, X)), compose(Y, Bs[i]));
      r.push(...scale(logRot(E.R), wr), ...scale(E.t, wt));
    });
    return r;
  };
  for (let it = 0; it < iters; it++) {
    const r0 = resid(X, Y), h = 1e-7, J = [];
    for (let k = 0; k < 12; k++) {
      const xi = [0, 0, 0, 0, 0, 0]; xi[k % 6] = h;
      const r1 = k < 6 ? resid(perturb(X, xi), Y) : resid(X, perturb(Y, xi));
      J.push(r1.map((v, i) => (v - r0[i]) / h));
    }
    const H = J.map((a) => J.map((b) => a.reduce((s, v, i) => s + v * b[i], 0)));
    H.forEach((row, i) => { row[i] += 1e-9; });
    const g = J.map((a) => -a.reduce((s, v, i) => s + v * r0[i], 0));
    const d = solveLinear(H, g);
    X = perturb(X, d.slice(0, 6)); Y = perturb(Y, d.slice(6));
    if (Math.hypot(...d) < 1e-12) break;
  }
  return { X, Y };
}

// Deterministic RNG helpers.
export function makeRng(seed) {
  let s = seed >>> 0;
  const uni = () => {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  const gauss = () => {
    let u = 0, v = 0;
    while (u === 0) u = uni();
    while (v === 0) v = uni();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  };
  return { uni, gauss, range: (a, b) => a + (b - a) * uni(), vec: (s) => [gauss() * s, gauss() * s, gauss() * s] };
}
