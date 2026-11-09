// Simulated eye-to-hand setup and both calibration pipelines.
// Units: metres and radians. Frame "base" is z-up.
import * as M from './calib-math.js';

// Ground-truth camera: on a post, looking at the work area (OpenCV optical frame: +z forward, +x right, +y down).
export const CAM_POS = [0.50, -0.31, 0.42];
export const LOOK_AT = [0.50, 0.12, 0.02];
export const BASELINE = 0.12;
export function trueCamera() {
  const z = M.unit(M.sub(LOOK_AT, CAM_POS));
  const x = M.unit(M.cross(z, [0, 0, 1]));
  const y = M.cross(z, x);
  return M.pose(M.fromCols(x, y, z), CAM_POS.slice());
}

// Method A: AprilTag stuck on the back of the hand (unknown X = flange -> tag).
export const TAG_SIZE = 0.12;
export const X_TRUE = M.pose(M.rotvec([0, Math.PI / 2, 0]), [0.04, 0.0, 0.06]);

// Method B: laser ray in the flange frame (values from the laser_calib project).
export const LASER_O = [-0.0095, -0.0014, 0.107];
export const LASER_U = M.unit([-0.0302, 0.0034, 0.9995]);

// Board on the table.
export const BOARD = { x: [0.33, 0.67], y: [-0.06, 0.32], z: 0.0 };
export const TEST_ON_BOARD = [0.50, 0.13, 0.0];
export const TEST_ABOVE = [0.50, 0.13, 0.30];

const NSAMPLES = 30;

function randomRoll(rng, axis) { return M.rotvec(M.scale(axis, rng.range(-Math.PI, Math.PI))); }
function randomTilt(rng, maxDeg) {
  const a = rng.range(0, 2 * Math.PI), ang = rng.range(0, maxDeg) * Math.PI / 180;
  return M.rotvec([Math.cos(a) * ang, Math.sin(a) * ang, 0]);
}

export function generateTagPoses(seed) {
  const rng = M.makeRng(seed);
  const cam = trueCamera();
  const out = [];
  for (let i = 0; i < NSAMPLES; i++) {
    const tagPos = [rng.range(0.40, 0.62), rng.range(-0.02, 0.24), rng.range(0.18, 0.36)];
    // Tag z points away from the camera (tag faces the camera), tilted up to 35 deg, random roll.
    const zt = M.unit(M.sub(tagPos, cam.t));
    const R0 = M.alignVectors([0, 0, 1], zt);
    const Rtag = M.matMul(R0, M.matMul(randomTilt(rng, 35), randomRoll(rng, [0, 0, 1])));
    const tag = M.pose(Rtag, tagPos);
    const flange = M.compose(tag, M.inv(X_TRUE));
    out.push({ flange, tag });
  }
  return out;
}

export function generateLaserPoses(seed) {
  const rng = M.makeRng(seed + 7);
  const out = [];
  for (let i = 0; i < NSAMPLES; i++) {
    const spot = [rng.range(BOARD.x[0] + 0.03, BOARD.x[1] - 0.03), rng.range(BOARD.y[0] + 0.03, BOARD.y[1] - 0.03), BOARD.z];
    const d = rng.range(0.12, 0.28);
    // Laser points downwards, tilted up to 15 deg.
    const uw = M.mulVec(randomTilt(rng, 15), [0, 0, -1]);
    const R0 = M.alignVectors(LASER_U, uw);
    const Rf = M.matMul(randomRoll(rng, uw), R0);
    const tf = M.sub(M.sub(spot, M.scale(uw, d)), M.mulVec(Rf, LASER_O));
    out.push({ flange: M.pose(Rf, tf), spot, d });
  }
  return out;
}

// Noise draws are separate from poses so that moving a noise slider does not reshuffle the poses.
export function generateNoise(seed) {
  const rng = M.makeRng(seed * 31 + 3);
  return Array.from({ length: NSAMPLES }, () => ({ a: rng.vec(1), b: rng.vec(1), c: rng.vec(1) }));
}

// measure: how the camera would measure a 3D point given in the true camera frame.
function evaluate(Yest, Ytrue, measure = (p) => p) {
  const map = (p) => M.apply(Yest, measure(M.apply(M.inv(Ytrue), p)));
  return {
    camPosMm: M.norm(M.sub(Yest.t, Ytrue.t)) * 1000,
    camRotDeg: M.rotAngleDeg(Yest.R, Ytrue.R),
    onBoardMm: M.norm(M.sub(map(TEST_ON_BOARD), TEST_ON_BOARD)) * 1000,
    aboveMm: M.norm(M.sub(map(TEST_ABOVE), TEST_ABOVE)) * 1000,
  };
}

// params: depthMm (PnP depth noise), rotDeg (PnP rotation noise), sizePct (tag size error).
export function solveTag(poses, noise, n, params) {
  const Y = trueCamera();
  const As = [], Bs = [];
  for (let i = 0; i < n; i++) {
    const { flange, tag } = poses[i];
    const Btrue = M.compose(M.inv(Y), tag);
    // PnP: depth (along the viewing ray) is much less certain than lateral position.
    const ray = M.unit(Btrue.t);
    const lateral = M.sub(noise[i].a, M.scale(ray, M.dot(noise[i].a, ray)));
    const dt = M.add(M.scale(ray, noise[i].a[0] * params.depthMm / 1000), M.scale(lateral, params.depthMm / 4000));
    const t = M.scale(M.add(Btrue.t, dt), 1 + params.sizePct / 100);
    const R = M.matMul(M.rotvec(M.scale(noise[i].b, params.rotDeg * Math.PI / 180)), Btrue.R);
    As.push(flange); Bs.push(M.pose(R, t));
  }
  const { X, Y: Yest } = M.solveAXYB(As, Bs);
  const resid = As.map((A, i) => M.norm(M.compose(M.inv(M.compose(A, X)), M.compose(Yest, Bs[i])).t) * 1000);
  return { Yest, X, observed: Bs, residualMm: Math.sqrt(resid.reduce((s, v) => s + v * v, 0) / n), ...evaluate(Yest, Y) };
}

// params: spotMm (triangulated point noise), scalePct (stereo scale error), fkMm (robot/laser-ray error).
export function solveLaser(poses, noise, n, params) {
  const Y = trueCamera();
  const pc = [], pb = [];
  for (let i = 0; i < n; i++) {
    const { flange, spot, d } = poses[i];
    const camPt = M.apply(M.inv(Y), spot);
    // Stereo depth noise grows along the viewing ray; scale error stretches the whole point cloud.
    const ray = M.unit(camPt);
    const lateral = M.sub(noise[i].a, M.scale(ray, M.dot(noise[i].a, ray)));
    const dn = M.add(M.scale(ray, noise[i].a[0] * params.spotMm * 2 / 1000), M.scale(lateral, params.spotMm / 1000));
    pc.push(M.scale(M.add(camPt, dn), 1 + params.scalePct / 100));
    const pred = M.apply(flange, M.add(LASER_O, M.scale(LASER_U, d)));
    pb.push(M.add(pred, M.scale(noise[i].c, params.fkMm / 1000)));
  }
  const Yest = M.rigidFit(pc, pb);
  const resid = pc.map((p, i) => M.norm(M.sub(M.apply(Yest, p), pb[i])) * 1000);
  const measure = (p) => M.scale(p, 1 + params.scalePct / 100);
  return { Yest, observed: pc, residualMm: Math.sqrt(resid.reduce((s, v) => s + v * v, 0) / n), ...evaluate(Yest, Y, measure) };
}
