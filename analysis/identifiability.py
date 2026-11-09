#!/usr/bin/env python3
"""Linearised identifiability of error sources for the two hand-eye methods, on real recorded poses.

Read-only: loads recorded samples, never touches hardware.
Each method's residual is written as a function of a full error-parameter vector (camera extrinsic, robot base
offset, joint zero offsets, method-specific terms). The Jacobian at the nominal solution, whitened by measurement
noise and scaled by a physical unit per parameter, tells us which errors the data can separate.
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path.home() / "projects/laser_calib"
sys.path.insert(0, str(ROOT / "src/algorithms"))
from solve_post_camera_move_latest_only import forward_kinematics, load_chain, valid_sample  # noqa: E402

chain = load_chain(ROOT / "assets/robot_model/xarm7_laser.urdf")
active = json.loads((ROOT / "calibration/active/final_robot_camera_laser_calibration.json").read_text())
MM, DEG01, PCT01 = 1e-3, np.radians(0.1), 1e-3


def rot(w):
    return Rotation.from_rotvec(w).as_matrix()


def perturb(T, w, t):
    out = np.eye(4)
    out[:3, :3] = rot(w) @ T[:3, :3]
    out[:3, 3] = rot(w) @ T[:3, 3] + t
    return out


def logT(T):
    return np.r_[Rotation.from_matrix(T[:3, :3]).as_rotvec(), T[:3, 3]]


# ---------------- Method B: laser ----------------
def laser_problem():
    src = json.loads((ROOT / "data/primary/processed/post_camera_move_samples_only.json").read_text())
    samples = [s for s in src if valid_sample(s)]
    q = np.array([s["robot"]["before"]["joints"] for s in samples])
    pc = np.array([s["vision"]["point_camera_m"] for s in samples])
    d = np.array([s["laser"]["mean_mm"] for s in samples]) / 1000
    Y0 = np.array(active["T_base_camera"]["matrix"])
    o0 = np.array(active["laser_ray_in_flange"]["origin_m"])
    u0 = np.array(active["laser_ray_in_flange"]["direction_unit"])
    # identifiability depends on pose geometry only: replace measurements by model-consistent ones
    pb_nom = np.array([forward_kinematics(chain, q[i])[:3, :3] @ (o0 + d[i] * u0) + forward_kinematics(chain, q[i])[:3, 3] for i in range(len(q))])
    pc = (pb_nom - Y0[:3, 3]) @ Y0[:3, :3]
    e1 = np.cross(u0, [1, 0, 0]); e1 /= np.linalg.norm(e1); e2 = np.cross(u0, e1)
    names = ([f"相机外参 r{a}" for a in "xyz"] + [f"相机外参 t{a}" for a in "xyz"]
             + [f"基座偏差 r{a}" for a in "xyz"] + [f"基座偏差 t{a}" for a in "xyz"]
             + [f"关节{j} 零位" for j in range(1, 8)]
             + [f"激光原点 {a}" for a in "xyz"] + ["激光方向 a", "激光方向 b", "双目尺度", "测距零点"])
    units = [DEG01] * 3 + [MM] * 3 + [DEG01] * 3 + [MM] * 3 + [DEG01] * 7 + [MM] * 3 + [DEG01] * 2 + [PCT01, 0.1 * MM]

    def residual(p):
        Y = perturb(Y0, p[0:3], p[3:6])
        Tb = perturb(np.eye(4), p[6:9], p[9:12])
        dq = p[12:19]
        o = o0 + p[19:22]
        u = u0 + p[22] * e1 + p[23] * e2; u /= np.linalg.norm(u)
        s, dz = p[24], p[25]
        r = []
        for i in range(len(q)):
            F = Tb @ forward_kinematics(chain, q[i] + dq)
            pb = F[:3, :3] @ (o + (d[i] + dz) * u) + F[:3, 3]
            pcam = Y[:3, :3] @ (pc[i] * (1 + s)) + Y[:3, 3]
            r.append(pcam - pb)
        return np.concatenate(r)

    sigma = np.full(3 * len(q), 1.0 * MM)  # ~1 mm per axis triangulation noise
    kinds = ["mm" if "t" == n.split()[-1][0] or n.startswith("激光原点") else "0.1°" for n in names]
    kinds[24], kinds[25] = "0.1%", "0.1 mm"
    return dict(name="B 激光", n=len(q), names=names, units=np.array(units), residual=residual, sigma=sigma,
                kinds=kinds, fitted=list(range(0, 6)) + list(range(19, 24)), gauge={6, 7, 8, 9, 10, 11, 12, 18, 25})


# ---------------- Method A: AprilTag ----------------
def tag_problem():
    files = sorted((ROOT / "data/cross_validation/single_tag").glob("train_*.json"))
    rec = [json.loads(f.read_text()) for f in files]
    q = np.array([r["joints_radians"] for r in rec])
    B = np.array([r["B"] for r in rec])
    A = np.array([forward_kinematics(chain, qi) for qi in q])
    # nominal X, Y from closed-loop least squares (good enough for a linearisation point)
    from scipy.optimize import least_squares

    def loop(p):
        X = perturb(np.eye(4), p[:3], p[3:6]); Y = perturb(np.eye(4), p[6:9], p[9:12])
        return np.concatenate([logT(np.linalg.inv(A[i] @ X) @ Y @ B[i]) * [1, 1, 1, 10, 10, 10] for i in range(len(A))])
    best = None
    for seed in range(12):
        x0 = np.random.default_rng(seed).normal(0, 1.5, 12); x0[3:6] *= 0.05; x0[9:12] = np.random.default_rng(seed).normal(0, 0.5, 3)
        sol = least_squares(loop, x0)
        if best is None or sol.cost < best.cost:
            best = sol
    X0 = perturb(np.eye(4), best.x[:3], best.x[3:6]); Y0 = perturb(np.eye(4), best.x[6:9], best.x[9:12])
    B = np.array([np.linalg.inv(Y0) @ A[i] @ X0 for i in range(len(A))])  # model-consistent observations
    names = ([f"相机外参 r{a}" for a in "xyz"] + [f"相机外参 t{a}" for a in "xyz"]
             + [f"基座偏差 r{a}" for a in "xyz"] + [f"基座偏差 t{a}" for a in "xyz"]
             + [f"关节{j} 零位" for j in range(1, 8)]
             + [f"Tag 安装 r{a}" for a in "xyz"] + [f"Tag 安装 t{a}" for a in "xyz"] + ["Tag 尺寸/焦距尺度"])
    units = [DEG01] * 3 + [MM] * 3 + [DEG01] * 3 + [MM] * 3 + [DEG01] * 7 + [DEG01] * 3 + [MM] * 3 + [PCT01]

    def residual(p):
        Y = perturb(Y0, p[0:3], p[3:6])
        Tb = perturb(np.eye(4), p[6:9], p[9:12])
        dq = p[12:19]
        X = perturb(X0, p[19:22], p[22:25])
        s = p[25]
        r = []
        for i in range(len(q)):
            Bi = B[i].copy(); Bi[:3, 3] *= (1 + s)
            r.append(logT(np.linalg.inv(Tb @ forward_kinematics(chain, q[i] + dq) @ X) @ Y @ Bi))
        return np.concatenate(r)

    # single 12 cm tag, mono PnP: ~0.5 deg rotation, ~2 mm translation noise
    sigma = np.tile(np.r_[[np.radians(0.5)] * 3, [2 * MM] * 3], len(q))
    kinds = ["mm" if "t" == n.split()[-1][0] else "0.1°" for n in names]
    kinds[25] = "0.1%"
    return dict(name="A AprilTag", n=len(q), names=names, units=np.array(units), residual=residual, sigma=sigma,
                kinds=kinds, fitted=list(range(0, 6)) + list(range(19, 25)), gauge={6, 7, 8, 9, 10, 11, 12, 18},
                loop_rms_mm=float(np.sqrt(np.mean(best.fun.reshape(-1, 6)[:, 3:] ** 2)) / 10 * 1000))


def analyse(P):
    names, units = P["names"], P["units"]
    k = len(names)
    r0 = P["residual"](np.zeros(k))
    J = np.zeros((len(r0), k))
    for j in range(k):
        h = units[j] * 1e-3
        e = np.zeros(k); e[j] = h
        J[:, j] = (P["residual"](e) - P["residual"](-e)) / (2 * h) * units[j]  # response to one "unit" of error
    Jw = J / P["sigma"][:, None]
    U, S, Vt = np.linalg.svd(Jw, full_matrices=False)
    null = [i for i, s in enumerate(S) if s < 1e-6 * S[0]]
    out = {"method": P["name"], "samples": P["n"], "params": k, "rank": int(k - len(null)),
           "singular_values": S.tolist(), "null_space": []}
    for i in null:
        v = Vt[i]; order = np.argsort(-np.abs(v))[:5]
        out["null_space"].append([(names[j], round(float(v[j]), 3)) for j in order if abs(v[j]) > 0.05])
    # Gauge fix: base offset and joint 1 / joint 7 are removed; the rest is estimated jointly.
    gauge = P["gauge"]
    keep = [j for j in range(k) if j not in gauge]
    H = Jw[:, keep].T @ Jw[:, keep]
    cov = np.linalg.pinv(H)
    std = np.sqrt(np.diag(cov))
    corr = cov / np.outer(std, std)
    rows = []
    for a, j in enumerate(keep):
        c = corr[a].copy(); c[a] = 0
        b = int(np.argmax(np.abs(c)))
        rows.append({"param": names[j], "std_units": float(std[a]), "unit": P["kinds"][j],
                     "most_correlated": names[keep[b]], "corr": float(c[b])})
    out["joint_estimate"] = rows
    # Same, but camera extrinsic + method terms only (what the projects actually solve)
    out["cond_after_gauge"] = float(np.linalg.cond(Jw[:, keep]))
    # Absorption: fit only what the project fits; how much of each other error stays visible in residuals?
    Fm = Jw[:, P["fitted"]]
    Q, _ = np.linalg.qr(Fm)
    vis = []
    for j in range(k):
        if j in P["fitted"]:
            continue
        col = Jw[:, j]
        left = col - Q @ (Q.T @ col)
        nrm = np.linalg.norm(col)
        # residual RMS (in sigma) left by one unit of this error, per sample
        vis.append({"param": names[j], "unit": P["kinds"][j],
                    "visible_fraction": float(np.linalg.norm(left) / nrm) if nrm > 0 else 0.0,
                    "rms_left_sigma": float(np.linalg.norm(left) / np.sqrt(P["n"]))})
    out["absorption"] = vis
    if "loop_rms_mm" in P:
        out["tag_loop_rms_mm"] = P["loop_rms_mm"]
    return out


def unit_name(u):
    return {MM: "mm", DEG01: "0.1°", PCT01: "0.1%", 0.1 * MM: "0.1 mm"}.get(u, "?")


if __name__ == "__main__":
    res = [analyse(laser_problem()), analyse(tag_problem())]
    print(json.dumps(res, ensure_ascii=False, indent=1))
