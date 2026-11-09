#!/usr/bin/env python3
"""Decoupled diagnostics on already-recorded laser_calib samples. Offline: reads JSON, never touches hardware.

Each check uses only ONE of the two measuring systems, so its result cannot be absorbed by the other:
  camera only        planarity of the triangulated laser spots
  robot + laser only planarity of FK * (o + d u); camera not involved
  controller vs URDF is the difference a fixed set of kinematic parameters?
and two coupled comparisons that show why the layers must be separated:
  camera vs robot    relative metric scale
  solver vs solver   how far two solvers disagree on the same 28 samples

Usage: decoupled_dryrun.py <laser_calib root> <out.json>
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares, minimize
from scipy.spatial.transform import Rotation

import kin

ROOT, OUT = Path(sys.argv[1]), Path(sys.argv[2])
REC = ROOT / "data/recalibration/20261109_camera_shift_recalibration"
F, B, CX, CY = 729.2711917567123, 0.12111325211874005, 933.8052520751953, 526.6510887145996  # rectified stereo model
chain = kin.load_chain(ROOT / "assets/robot_model/xarm7_laser.urdf")
load = lambda p: json.loads(Path(p).read_text())
rms = lambda x: float(np.sqrt(np.mean(np.square(x))))

train, val = load(REC / "train_complete_strict.json"), load(REC / "final_validation_8.json")
far = [s for s in load(REC / "train_with_far_strict.json") if s["id"] not in {t["id"] for t in train}]
active = load(ROOT / "calibration/active/final_robot_camera_laser_calibration.json")
v2 = load(ROOT / "calibration/candidates/near_planar_joint_v2_20261109.json")
ray_of = lambda m: (np.array(m["laser_ray_in_flange"]["origin_m"]), np.array(m["laser_ray_in_flange"]["direction_unit"]))


def arrays(samples):
    q = np.array([s["robot"]["before"]["joints"] for s in samples])
    T = np.array([kin.pose_to_matrix(s["robot"]["before"]["pose"]) for s in samples])
    d = np.array([s["laser"]["mean_mm"] for s in samples]) / 1000.0
    pc = np.array([s["vision"]["point_camera_m"] for s in samples])
    obs = np.array([[s["vision"]["left"]["x"], s["vision"]["left"]["y"], s["vision"]["right"]["x"]] for s in samples])
    return q, T, d, pc, obs


def spots(T, d, o, u):
    return np.einsum("nij,nj->ni", T[:, :3, :3], o[None] + d[:, None] * u[None]) + T[:, :3, 3]


def fit_plane(P):
    c = P.mean(0)
    n = np.linalg.svd(P - c)[2][-1]
    n = n if n[2] > 0 else -n
    return n, float(n @ c)


def kabsch(A, Bp, with_scale=False):
    ca, cb = A.mean(0), Bp.mean(0)
    U, S, Vt = np.linalg.svd((A - ca).T @ (Bp - cb))
    Dm = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
    Rm = Vt.T @ Dm @ U.T
    s = (S * np.diag(Dm)).sum() / ((A - ca) ** 2).sum() if with_scale else 1.0
    return s, Rm, cb - s * Rm @ ca


def triangulate(obs, ef=0.0, eb=0.0):
    f, b = F * (1 + ef), B * (1 + eb)
    z = f * b / (obs[:, 0] - obs[:, 2])
    return np.c_[(obs[:, 0] - CX) * z / f, (obs[:, 1] - CY) * z / f, z]


out = {"samples": {"train": len(train), "validation": len(val), "far": len(far)}}
SETS = {"train28": arrays(train), "val8": arrays(val), "far7": arrays(far)}

# ---- controller pose vs nominal URDF ------------------------------------------------------------------------------
seen, Q, P = set(), [], []
for f in (REC / "raw_samples_with_far.json", ROOT / "data/primary/processed/post_camera_move_samples_only.json"):
    for s in load(f):
        r = s.get("robot", {}).get("before")
        if not r or r.get("joints") is None:
            continue
        key = tuple(np.round(r["joints"], 5))
        if key not in seen:
            seen.add(key); Q.append(r["joints"]); P.append(r["pose"])
Q = np.array(Q); Tc = np.array([kin.pose_to_matrix(p) for p in P])


def se3(x):
    T = np.eye(4); T[:3, :3] = Rotation.from_rotvec(x[:3]).as_matrix(); T[:3, 3] = x[3:6]
    return T


def pose_residual(x, q, Tref):
    Tm = se3(x[28:34]) @ kin.fk(chain, q, dtheta=x[:7], dd=x[7:14], da=x[14:21], dalpha=x[21:28]) @ se3(x[34:40])
    dt = (Tm[:, :3, 3] - Tref[:, :3, 3]) * 1000.0
    dR = kin.rotvec_of(np.einsum("nij,nkj->nik", Tm[:, :3, :3], Tref[:, :3, :3]))
    return dt, dR


def fit_pose_model(free, q, Tref):
    def cost(z):
        x = np.zeros(40); x[free] = z
        dt, dR = pose_residual(x, q, Tref)
        return np.r_[dt.ravel(), dR.ravel() * 500.0, 1e-4 * z]   # tiny ridge: the 40-parameter set is redundant
    z = least_squares(cost, np.zeros(len(free)), x_scale="jac", xtol=1e-15, ftol=1e-15, gtol=1e-15).x
    x = np.zeros(40); x[free] = z
    return x


def pose_stats(x, q, Tref):
    dt, dR = pose_residual(x, q, Tref)
    return {"translation_rms_mm": rms(np.linalg.norm(dt, axis=1)), "rotation_rms_deg": float(np.degrees(rms(np.linalg.norm(dR, axis=1))))}


half = np.random.default_rng(0).permutation(len(Q)); a, b = half[: len(Q) // 2], half[len(Q) // 2:]
out["controller_vs_urdf"] = {
    "robot_configurations": len(Q),
    "nominal_urdf": pose_stats(np.zeros(40), Q, Tc),
    "fit_7_joint_zeros": pose_stats(fit_pose_model(list(range(7)), Q, Tc), Q, Tc),
    "fit_28_dh": pose_stats(fit_pose_model(list(range(28)), Q, Tc), Q, Tc),
    "fit_28_dh_plus_base_and_flange_frames": pose_stats(fit_pose_model(list(range(40)), Q, Tc), Q, Tc),
    "fit_on_half_tested_on_other_half": pose_stats(fit_pose_model(list(range(40)), Q[a], Tc[a]), Q[b], Tc[b]),
}

# ---- two solvers, same 28 samples -----------------------------------------------------------------------------------
Ta, Tv = np.array(active["T_base_camera"]["matrix"]), np.array(v2["T_base_camera"]["matrix"])
(oa, ua), (ov, uv) = ray_of(active), ray_of(v2)
out["two_solvers_same_28_samples"] = {
    "camera_position_difference_mm": float(np.linalg.norm(Tv[:3, 3] - Ta[:3, 3]) * 1000),
    "camera_orientation_difference_deg": float(np.degrees(np.linalg.norm(Rotation.from_matrix(Tv[:3, :3] @ Ta[:3, :3].T).as_rotvec()))),
    "laser_direction_difference_deg": float(np.degrees(np.arccos(ua @ uv))),
    "laser_origin_difference_mm": float(np.linalg.norm(oa - ov) * 1000),
    "fit_rms_mm": {"active": active["fit"]["rms_point_mm"], "v2_raw_3d": v2["fit"]["raw_3d_rms_mm"]},
}

# ---- camera only -----------------------------------------------------------------------------------------------------
q, T, d, pc, obs = SETS["train28"]
n, c = fit_plane(pc)
r = (pc @ n - c) * 1000
depth_per_px = pc[:, 2] ** 2 / (F * B) * 1000
out["camera_only_planarity"] = {
    "train28_rms_mm": rms(r), "train28_max_mm": float(np.abs(r).max()),
    "held_out_vs_train_plane_rms_mm": {k: rms((SETS[k][3] @ n - c) * 1000) for k in ("val8", "far7")},
    "depth_mm_per_disparity_px": [float(depth_per_px.min()), float(depth_per_px.max())],
    "equivalent_disparity_error_px": rms(r / (depth_per_px * abs(n[2]))),
}

# ---- robot + laser only ---------------------------------------------------------------------------------------------
def plate_fit(free_origin, joints, prior_deg=0.2, sig_mm=0.5):
    """Camera-free fit of plate (3) + ray direction (2) [+ ray origin (3)] [+ joint zeros J2-J6 (5), weak prior]."""
    o0, u0 = ray_of(v2)
    e1 = np.cross(u0, [1, 0, 0]); e1 /= np.linalg.norm(e1); e2 = np.cross(u0, e1)
    n0, c0 = fit_plane(spots(T, d, o0, u0))

    def unpack(x):
        nn = np.array([x[0], x[1], 1.0]); nn /= np.linalg.norm(nn)
        u = u0 + x[3] * e1 + x[4] * e2; u /= np.linalg.norm(u)
        k = 5
        o = o0.copy()
        if free_origin:
            o = o0 + x[k:k + 3]; k += 3
        dth = np.zeros(7)
        if joints:
            dth[1:6] = x[k:k + 5]
        return nn, x[2], o, u, dth

    def corrected(Tm, qm, dth):
        # joint-zero correction applied on top of the controller pose through the URDF differential
        return np.einsum("nij,njk,nkl->nil", kin.fk(chain, qm, dtheta=dth), np.linalg.inv(kin.fk(chain, qm)), Tm) if joints else Tm

    def cost(x, qm, Tm, dm):
        nn, cc, o, u, dth = unpack(x)
        rr = (spots(corrected(Tm, qm, dth), dm, o, u) @ nn - cc) * 1000
        return np.r_[rr, np.degrees(dth[1:6]) / prior_deg * sig_mm] if joints else rr

    x0 = np.zeros(5 + 3 * free_origin + 5 * joints); x0[0], x0[1], x0[2] = n0[0] / n0[2], n0[1] / n0[2], c0
    sol = least_squares(cost, x0, args=(q, T, d), x_scale="jac", xtol=1e-14, ftol=1e-14, gtol=1e-14)
    nn, cc, o, u, dth = unpack(sol.x)
    res = {"parameters": len(x0), "train_rms_mm": rms(sol.fun[: len(q)])}
    for k in ("val8", "far7"):
        q2, T2, d2 = SETS[k][:3]
        res[f"{k}_rms_mm"] = rms((spots(corrected(T2, q2, dth), d2, o, u) @ nn - cc) * 1000)
    loo = []
    for i in range(len(q)):
        m = np.arange(len(q)) != i
        ni, ci, oi, ui, di = unpack(least_squares(cost, sol.x, args=(q[m], T[m], d[m]), x_scale="jac").x)
        loo.append((spots(corrected(T[i:i + 1], q[i:i + 1], di), d[i:i + 1], oi, ui) @ ni - ci)[0] * 1000)
    res["leave_one_out_rms_mm"] = rms(loo)
    if joints:
        cov = np.linalg.pinv(sol.jac.T @ sol.jac) * sig_mm ** 2
        res["joint_zero_J2_J6_deg"] = np.degrees(dth[1:6]).tolist()
        res["joint_zero_posterior_std_deg"] = np.degrees(np.sqrt(np.diag(cov))[-5:]).tolist()
        res["joint_zero_prior_std_deg"] = prior_deg
    return res


tilt = np.degrees(np.arccos(np.clip(-(T[:, :3, :3] @ ray_of(v2)[1])[:, 2], -1, 1)))
fixed = {}
for name, model in (("active", active), ("v2", v2)):
    o, u = ray_of(model)
    for kname, Tk in (("controller_pose", T), ("urdf_fk", kin.fk(chain, q))):
        Pp = spots(Tk, d, o, u); n_, c_ = fit_plane(Pp)
        fixed[f"{kname}__ray_{name}"] = rms((Pp @ n_ - c_) * 1000)
out["robot_laser_only_planarity"] = {
    "pose_diversity": {"laser_tilt_deg": [float(tilt.min()), float(tilt.max())], "range_mm": [float(d.min() * 1000), float(d.max() * 1000)]},
    "ray_as_given_train28_rms_mm": fixed,
    "fit_plate_and_ray_direction": plate_fit(False, False),
    "fit_plate_and_full_ray": plate_fit(True, False),
    "fit_plate_ray_and_joint_zeros": plate_fit(True, True),
}

# fixed flange orientation scans (afternoon sessions): how flat is the existing board as seen by robot + laser?
flat = {}
for sess in ("20261109_fixed_grid_240mm_scan", "20261109_quadratic_correction_240mm"):
    plan, rep = load(ROOT / f"data/validation/{sess}/plan.json"), load(ROOT / f"data/validation/{sess}/report.json")
    plan = plan if isinstance(plan, list) else plan.get("targets", plan.get("points"))
    pose = {p["name"]: p["pose"] for p in plan}
    pts = [p for p in rep["points"] if p["name"] in pose]
    Tg = np.array([kin.pose_to_matrix(pose[p["name"]]) for p in pts]); dg = np.array([p["distance_mm"] for p in pts]) / 1000
    Pg = spots(Tg, dg, *ray_of(v2)); n_, c_ = fit_plane(Pg); rg = (Pg @ n_ - c_) * 1000
    flat[sess] = {"points": len(pts), "area_mm": [float(np.ptp(Pg[:, 0]) * 1000), float(np.ptp(Pg[:, 1]) * 1000)], "plane_rms_mm": rms(rg), "plane_max_mm": float(np.abs(rg).max())}
out["robot_laser_only_planarity"]["fixed_orientation_scans"] = flat

# ---- camera vs robot + laser: relative scale ------------------------------------------------------------------------
def scale_block(samples, model):
    q_, T_, d_, pc_, obs_ = arrays(samples)
    Pb = spots(T_, d_, *ray_of(model))
    after = lambda ef, eb, i=slice(None): (lambda X: rms(np.linalg.norm(X @ kabsch(X, Pb[i])[1].T + kabsch(X, Pb[i])[2] - Pb[i], axis=1) * 1000))(triangulate(obs_[i], ef, eb))
    r1 = minimize(lambda x: after(0, x[0]), [0.0], method="Nelder-Mead", options=dict(xatol=1e-6, fatol=1e-9))
    r2 = minimize(lambda x: after(x[0], x[1]), [0.0, 0.0], method="Nelder-Mead", options=dict(xatol=1e-6, fatol=1e-9))
    rng = np.random.default_rng(5); b1, b2 = [], []
    for _ in range(300):
        i = rng.integers(0, len(Pb), len(Pb))
        b1.append(minimize(lambda x: after(0, x[0], i), r1.x, method="Nelder-Mead").x[0])
        b2.append(minimize(lambda x: after(x[0], x[1], i), r2.x, method="Nelder-Mead").x)
    b2 = np.array(b2)
    return {"rigid_rms_mm": after(0, 0),
            "uniform_scale": {"camera_scale_error_pct": float(-r1.x[0] * 100), "bootstrap_std_pct": float(np.std(b1) * 100), "rms_mm": float(r1.fun)},
            "depth_and_lateral": {"depth_only_pct": float(-r2.x[0] * 100), "depth_only_std_pct": float(b2[:, 0].std() * 100),
                                  "uniform_pct": float(-r2.x[1] * 100), "uniform_std_pct": float(b2[:, 1].std() * 100), "rms_mm": float(r2.fun)}}


out["camera_vs_robot_scale"] = {f"{sname}__ray_{mname}": scale_block(S_, m) for sname, S_ in (("train28", train), ("all43", train + val + far))
                                for mname, m in (("active", active), ("v2", v2))}
OUT.write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
