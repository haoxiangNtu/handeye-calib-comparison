#!/usr/bin/env python3
"""Polynomial compensation versus a layered physical model, on the same laser-targeting data. Offline; no hardware.

Each targeting record is: the camera picked a target pixel, the robot aimed the laser at the corresponding board point
with the calibration file's laser ray, the camera measured where the spot landed. The colleague fits the planar error
with a 44-coefficient polynomial in (x, y, laser distance, tilt x, tilt y, roll). Here the same records are explained
with 9 physical numbers instead:

  laser ray direction (2) + lateral origin (2)      robot + laser (plate constraint on laser readings, then refined)
  surface-height offset along the ray (1)          laser zero / origin along the ray
  board <-> base similarity (4)                    the link between the printed board and the robot base
  pixel -> board homography (8, not counted)       camera only, from the registration of the printed AprilGrid

Usage: compensation_vs_layered.py <laser_calib root> <config.json> <out.json>
The config lists, relative to the root, the targeting reports, the calibration file used to aim, the board
registration and the rectified camera matrix used by that calibration. The data stays in the colleague's project.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

import kin

ROOT, CFG, OUT = Path(sys.argv[1]), json.loads(Path(sys.argv[2]).read_text()), Path(sys.argv[3])
rms2 = lambda e: float(np.sqrt((np.asarray(e, float) ** 2).sum(-1).mean()))
PLANE_Z = CFG.get("plane_z_m", -0.0079)
act = json.loads((ROOT / CFG["active_calibration"]).read_text())
O_M = np.array(act["laser_ray_in_flange"]["origin_m"]); U_M = np.array(act["laser_ray_in_flange"]["direction_unit"])
T_BC = np.array(act["T_base_camera"]["matrix"])
K = np.array(CFG["rectified_K"]) if "rectified_K" in CFG else np.array(json.loads((ROOT / CFG["rectified_K_from"]).read_text())["rectified_K"])
E1 = np.cross(U_M, [1, 0, 0]); E1 /= np.linalg.norm(E1); E2 = np.cross(U_M, E1)
UNIT9 = np.r_[np.radians(1), np.radians(1), 1e-3, 1e-3]     # ray parameters: degrees, degrees, mm, mm


def hom(H, pts):
    q = np.c_[pts, np.ones(len(pts))] @ H.T
    return q[:, :2] / q[:, 2:]


def homography(src, dst):
    """Direct linear transform, src -> dst (both N x 2)."""
    s_c, d_c = src.mean(0), dst.mean(0); s_s = np.sqrt(2) / np.linalg.norm(src - s_c, axis=1).mean(); d_s = np.sqrt(2) / np.linalg.norm(dst - d_c, axis=1).mean()
    Ts = np.array([[s_s, 0, -s_s * s_c[0]], [0, s_s, -s_s * s_c[1]], [0, 0, 1]]); Td = np.array([[d_s, 0, -d_s * d_c[0]], [0, d_s, -d_s * d_c[1]], [0, 0, 1]])
    a = hom(Ts, src); b = hom(Td, dst); rows = []
    for (x, y), (u, v) in zip(a, b):
        rows.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u]); rows.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])
    H = np.linalg.svd(np.array(rows))[2][-1].reshape(3, 3)
    H = np.linalg.inv(Td) @ H @ Ts
    return H / H[2, 2]


def m_active(px):
    """Pixel -> base-plane point (mm) through the calibration file: the mapping the collection script evaluates with."""
    ray_c = np.linalg.solve(K, np.c_[px, np.ones(len(px))].T).T
    d = ray_c @ T_BC[:3, :3].T
    s = (PLANE_Z - T_BC[2, 3]) / d[:, 2]
    return (T_BC[:3, 3] + s[:, None] * d)[:, :2] * 1000


def inv_m_active(xy_mm):
    p = np.c_[xy_mm / 1000, np.full(len(xy_mm), PLANE_Z)]
    q = ((p - T_BC[:3, 3]) @ T_BC[:3, :3]) @ K.T
    return q[:, :2] / q[:, 2:]


def load(label, path, H_plan):
    rows = []
    for e in json.loads((ROOT / path).read_text())["events"]:
        if not e.get("evaluation"):
            continue
        t, ev = e["target"], e["evaluation"]
        if "actual_planar_base_mm" in ev:
            spot_px = inv_m_active(np.array([ev["actual_planar_base_mm"][:2]]))[0]
        else:                                                    # validation runs report H_plan(spot) directly
            spot_px = hom(np.linalg.inv(H_plan), np.array([ev["actual_xy_mm"]]))[0]
        rows.append(dict(set=label, source=t["source"], T=kin.pose_to_matrix(t["pose"]), nominal=np.array(t["nominal_target_base_mm"][:2]),
                         commanded=np.array(t.get("commanded_target_base_mm", t["nominal_target_base_mm"])[:2]), d_cmd=t["command_distance_mm"] / 1000,
                         d_meas=e["laser_precheck_mm"] / 1000, tilt=np.array(t["tilt_xy_deg"]), roll=float(t["roll_about_ray_deg"]), spot_px=spot_px,
                         target_px=np.array(t["pixel"]), err_reported=np.array(ev["error_xy_mm"])))
    return rows


def similarity(src, dst):
    best = None
    for refl in (False, True):
        a = src * (np.array([1, -1]) if refl else 1); ca, cb = a.mean(0), dst.mean(0)
        U, Sv, Vt = np.linalg.svd((a - ca).T @ (dst - cb)); Dm = np.diag([1, np.sign(np.linalg.det(Vt.T @ U.T))]); Rm = Vt.T @ Dm @ U.T
        s = (Sv * np.diag(Dm)).sum() / ((a - ca) ** 2).sum(); t = cb - s * Rm @ ca
        res = rms2(s * a @ Rm.T + t - dst)
        if best is None or res < best[0]:
            best = (res, s, Rm, t, refl)
    return best


def ray(x):
    u = U_M + x[0] * E1 + x[1] * E2; u /= np.linalg.norm(u)
    return O_M + x[2] * E1 + x[3] * E2, u


def plane(x):
    n = np.array([x[0], x[1], 1.0]); n /= np.linalg.norm(n)
    return n, PLANE_Z + x[2]


def camera_free_fit(T, dm):
    """Plate constraint: flange pose + laser reading must land on one plane. Ray direction (2), lateral origin (2), plane (3)."""
    def cost(x):
        o, u = ray(x[3:7]); n, c = plane(x[:3])
        P = T[:, :3, 3] + np.einsum("nij,nj->ni", T[:, :3, :3], o[None] + dm[:, None] * u[None])
        return (P @ n - c) * 1000
    P0 = T[:, :3, 3] + np.einsum("nij,nj->ni", T[:, :3, :3], O_M[None] + dm[:, None] * U_M[None]); c0 = P0.mean(0); n0 = np.linalg.svd(P0 - c0)[2][-1]
    sol = least_squares(cost, np.zeros(7), x_scale="jac", xtol=1e-13, ftol=1e-13)
    cov = np.linalg.inv(sol.jac.T @ sol.jac) * (sol.fun @ sol.fun) / max(len(dm) - 7, 1)
    return sol.x, np.sqrt(np.diag(cov)), float(np.sqrt((((P0 - c0) @ n0) ** 2).mean()) * 1000), float(np.sqrt(np.mean(sol.fun ** 2)))


def spots_on_plane(T, o, u, n, c):
    p0 = T[:, :3, 3] + T[:, :3, :3] @ o; w = T[:, :3, :3] @ u
    s = (c - p0 @ n) / (w @ n)
    return (p0 + s[:, None] * w)[:, :2] * 1000


def fit9(T, dm, board, idx):
    """Ray (4) + board<->base similarity (4) + surface-height offset (1), fitted to where the spots were seen on the printed board."""
    xcf = camera_free_fit(T[idx], dm[idx])[0]
    n, c = plane(xcf[:3]); o0, u0 = ray(xcf[3:7])
    m0 = similarity(board[idx], spots_on_plane(T[idx], o0, u0, n, c)); refl = np.array([1, -1]) if m0[4] else np.array([1, 1])

    def model(x, Tm, bm):
        o, u = ray(x[:4] * UNIT9)
        S = spots_on_plane(Tm, o, u, n, c + x[8] * 1e-3)
        sc, th = 1 + x[4] * 1e-2, x[5]; Rm = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        return sc * (bm * refl) @ Rm.T + x[6:8] - S
    x0 = np.r_[np.degrees(xcf[3:5]), xcf[5:7] * 1000, (m0[1] - 1) * 100, np.arctan2(m0[2][1, 0], m0[2][0, 0]), m0[3], 0.0]
    sol = least_squares(lambda z: model(z, T[idx], board[idx]).ravel(), x0, x_scale="jac", xtol=1e-12, ftol=1e-12)
    return sol.x, model, (n, c)


# ---- the colleague's polynomial (his feature set, normalisation and ridge selection) ----------------------------------
def features(N, d, tilt, roll):
    x = (N[:, 0] - 540) / 190; y = (N[:, 1] - 80) / 140; dd = (d * 1000 - 200) / 75; tx = tilt[:, 0] / 10; ty = tilt[:, 1] / 10; rr = roll / 15
    return np.stack([np.ones_like(x), x, y, x * x, x * y, y * y, dd, tx, ty, rr, dd * dd, tx * tx, ty * ty, rr * rr, dd * tx, dd * ty,
                     x * dd, y * dd, x * tx, y * tx, x * ty, y * ty], 1)


def ridge(X, Y, alpha):
    Rg = np.eye(X.shape[1]); Rg[0, 0] = 0
    return np.linalg.solve(X.T @ X + alpha * Rg, X.T @ Y)


def poly_fit(X, Y, groups):
    best = None
    for alpha in (1e-4, 1e-3, 1e-2, 3e-2, 0.1, 0.3, 1, 3, 10, 30, 100):
        pred = np.zeros_like(Y)
        for g in np.unique(groups):
            m = groups == g
            pred[m] = X[m] @ ridge(X[~m], Y[~m], alpha)
        r = rms2(Y - pred)
        if best is None or r < best[0]:
            best = (r, alpha)
    return ridge(X, Y, best[1]), best[1]


# ---- analysis ----------------------------------------------------------------------------------------------------------
reg = json.loads((ROOT / CFG["registration"]).read_text())
rp = np.array([p["pixel"] for p in reg["points"]]); rb = np.array([p["board_center_mm"] for p in reg["points"]]); rt = np.array([p["target_base_mm"][:2] for p in reg["points"]])
H_pb = homography(rp, rb)                          # pixel -> printed board (mm); camera only
H_plan = np.array(reg["left_to_base_plane"]["matrix"])   # pixel -> base plane; what generated the targets
rows = [r for label, path in CFG["reports"].items() for r in load(label, path, H_plan)]
n_pts = len(rows)
T = np.array([r["T"] for r in rows]); dm = np.array([r["d_meas"] for r in rows]); dcmd = np.array([r["d_cmd"] for r in rows])
N = np.array([r["nominal"] for r in rows]); tilt = np.array([r["tilt"] for r in rows]); roll = np.array([r["roll"] for r in rows])
spot_px = np.array([r["spot_px"] for r in rows]); tgt_px = np.array([r["target_px"] for r in rows])
sets = np.array([r["set"] for r in rows]); src = np.array([r["set"][0] + r["source"] for r in rows]); groups_pos = np.array([r["source"] for r in rows])
err_reported = np.array([r["err_reported"] for r in rows])
err = hom(H_plan, spot_px) - N                     # one mapping for target and landing (the colleague's 17:03 definition)
board = hom(H_pb, spot_px)                         # where the spot was seen on the printed board (mm)
X = features(N, dcmd, tilt, roll)
out = {"name": CFG.get("name", ""), "points": n_pts, "sets": {k: int((sets == k).sum()) for k in np.unique(sets)},
       "ranges": {"distance_mm": [float(dcmd.min() * 1000), float(dcmd.max() * 1000)], "tilt_deg": float(np.abs(tilt).max()), "roll_deg": float(np.abs(roll).max())}}

m_his = similarity(rb, rt)
out["board_scale"] = {"calibration_mapping_vs_printed_board_pct": float((m_his[1] - 1) * 100), "calibration_mapping_similarity_residual_mm": m_his[0],
                      "registration_template_rms_px": reg["template_to_left"].get("rms_px")}
out["raw_error_mm"] = {"as_reported_mixed_mappings": rms2(err_reported), "one_mapping": rms2(err), "mapping_difference_at_spots": rms2(m_active(spot_px) - hom(H_plan, spot_px)),
                       "pixel_error_px": float(np.sqrt(((spot_px - tgt_px) ** 2).sum(1).mean()))}

xcf, sd, before, after = camera_free_fit(T, dm)
o_cf, u_cf = ray(xcf[3:7])
ang = lambda a, b: float(np.degrees(np.arccos(np.clip(a @ b, -1, 1))))
out["camera_free_ray"] = {"plane_residual_file_ray_mm": before, "plane_residual_fitted_ray_mm": after, "direction_change_deg": ang(u_cf, U_M),
                          "direction_change_components_deg": np.degrees(xcf[3:5]).tolist(), "direction_std_deg": np.degrees(sd[3:5]).tolist(),
                          "lateral_origin_change_mm": (xcf[5:7] * 1000).tolist(), "lateral_origin_std_mm": (sd[5:7] * 1000).tolist(),
                          "file_direction": U_M.tolist(), "fitted_direction": u_cf.tolist()}

idx = np.arange(n_pts)
x9, model9, (n_pl, c_pl) = fit9(T, dm, board, idx)
res9 = model9(x9, T, board); o9, u9 = ray(x9[:4] * UNIT9)
V = np.c_[dcmd * 1000, tilt, roll]
out["layered_9"] = {"in_sample_rms_mm": rms2(res9), "ray_direction_change_deg": ang(u9, U_M), "fitted_direction": u9.tolist(), "lateral_origin_change_mm": x9[2:4].tolist(),
                    "surface_height_offset_mm": float(x9[8]), "board_vs_robot_scale_pct": float(x9[4]),
                    "residual_correlation": {nm: [float(np.corrcoef(V[:, j], res9[:, 0])[0, 1]), float(np.corrcoef(V[:, j], res9[:, 1])[0, 1])] for j, nm in enumerate(["distance", "tilt_x", "tilt_y", "roll"])}}

coef, alpha = poly_fit(X, err, groups_pos)
out["polynomial_44"] = {"in_sample_rms_mm": rms2(err - X @ coef), "ridge_alpha": alpha}


def evaluate(tr, te):
    res = {}
    c_, _ = poly_fit(X[tr], err[tr], groups_pos[tr]); res["polynomial_44"] = err[te] - X[te] @ c_
    c6, _ = poly_fit(X[tr][:, :6], err[tr], groups_pos[tr]); res["position_only_12"] = err[te] - X[te][:, :6] @ c6
    xx, mdl, _ = fit9(T, dm, board, tr); r_te, r_tr = mdl(xx, T[te], board[te]), mdl(xx, T[tr], board[tr])
    res["layered_9"] = r_te
    c2, _ = poly_fit(X[tr], r_tr, groups_pos[tr]); res["layered_9_plus_polynomial"] = r_te - X[te] @ c2
    return res


agg = {}
uniq = np.unique(src)
for seed in range(2):
    rng = np.random.default_rng(seed); u2 = uniq.copy(); rng.shuffle(u2)
    for f in np.array_split(u2, 5):
        te = idx[np.isin(src, f)]; tr = idx[~np.isin(src, f)]
        for k, v in evaluate(tr, te).items():
            agg.setdefault(k, []).append(v)
out["cv_5fold_by_position"] = {k: {"rms_mm": rms2(np.concatenate(v)), "median_mm": float(np.median(np.linalg.norm(np.concatenate(v), axis=1)))} for k, v in agg.items()}

curve = {}
for n_tr in (10, 20, 40, 80, 160):
    if n_tr >= n_pts - 20:
        continue
    acc = {}
    for seed in range(8):
        rng = np.random.default_rng(100 + seed); p = rng.permutation(n_pts); tr, te = p[:n_tr], p[n_tr:]
        for k, v in evaluate(tr, te).items():
            acc.setdefault(k, []).append(rms2(v))
    curve[str(n_tr)] = {k: float(np.median(v)) for k, v in acc.items()}
out["small_data_rms_mm"] = curve

if "cross_set_tests" in CFG:
    out["cross_set_rms_mm"] = []
    for tr_sets, te_sets in CFG["cross_set_tests"]:
        tr, te = idx[np.isin(sets, tr_sets)], idx[np.isin(sets, te_sets)]
        out["cross_set_rms_mm"].append({"train": tr_sets, "test": te_sets, "n_train": int(len(tr)), "n_test": int(len(te)), "raw": rms2(err[te]),
                                        **{k: rms2(v) for k, v in evaluate(tr, te).items()}})

if "validation_report" in CFG:          # shots executed with the colleague's 39-point compensation: predict where they landed
    val = load("val", CFG["validation_report"], H_plan)
    Tv = np.array([r["T"] for r in val]); bv = hom(H_pb, np.array([r["spot_px"] for r in val])); land = hom(H_plan, np.array([r["spot_px"] for r in val]))
    nom = np.array([r["nominal"] for r in val]); cmd = np.array([r["commanded"] for r in val])
    Xv = features(cmd, np.array([r["d_cmd"] for r in val]), np.array([r["tilt"] for r in val]), np.array([r["roll"] for r in val]))
    out["executed_validation_shots"] = {"count": len(val), "closed_loop_error_as_reported_mm": rms2(land - nom),
                                        "layered_9_prediction_error_mm": rms2(model9(x9, Tv, bv)), "polynomial_44_prediction_error_mm": rms2(land - (cmd + Xv @ coef))}

if "polynomial_model" in CFG:           # his published model: per-slider increments at one board point versus the 3 physical numbers
    his = json.loads((ROOT / CFG["polynomial_model"]).read_text()); C = np.array(his["coefficients_error_xy_mm"])
    def poly_at(P, D=200.0, tx=0.0, ty=0.0, r=0.0):
        return features(np.array([P]), np.array([D / 1000]), np.array([[tx, ty]]), np.array([r]))[0] @ C
    ref_R = Rotation.from_euler("xyz", CFG.get("reference_rpy_rad", [2.86175, 0.058165, 1.403348])).as_matrix(); ref_ray = ref_R @ U_M
    def make_rotation(tx, ty, rdeg):
        desired = Rotation.from_euler("xy", [tx, ty], degrees=True).apply(ref_ray); desired /= np.linalg.norm(desired)
        align, _ = Rotation.align_vectors(desired[None, :], ref_ray[None, :])
        return Rotation.from_rotvec(desired * math.radians(rdeg)).as_matrix() @ align.as_matrix() @ ref_R
    def landing(P, D, tx, ty, rdeg):
        Rm = make_rotation(tx, ty, rdeg); nominal = np.r_[P / 1000, PLANE_Z]
        flange = nominal - Rm @ (O_M + D / 1000 * U_M)
        p0 = flange + Rm @ o9; w = Rm @ u9; s = (c_pl + x9[8] * 1e-3 - p0 @ n_pl) / (w @ n_pl)
        return (p0 + s * w)[:2] * 1000 - P
    P = np.array(CFG.get("increment_point_mm", [532.1, 64.1])); base = landing(P, 200.0, 0, 0, 0); pbase = poly_at(P)
    inc = {}
    for label, kw in (("distance 200->130", dict(D=130.0)), ("distance 200->275", dict(D=275.0)), ("roll +30", dict(r=30.0)), ("roll -30", dict(r=-30.0)),
                      ("tilt_x +20", dict(tx=20.0)), ("tilt_x -20", dict(tx=-20.0)), ("tilt_y +20", dict(ty=20.0)), ("tilt_y -20", dict(ty=-20.0))):
        kk = {"D": 200.0, "tx": 0.0, "ty": 0.0, "r": 0.0, **kw}
        inc[label] = {"polynomial": (poly_at(P, **kk) - pbase).tolist(), "physical": (landing(P, kk["D"], kk["tx"], kk["ty"], kk["r"]) - base).tolist()}
    out["increments_at_point"] = {"point_mm": P.tolist(), "model": CFG["polynomial_model"], "rows": inc}

OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False))
print(json.dumps(out, indent=1, ensure_ascii=False))
