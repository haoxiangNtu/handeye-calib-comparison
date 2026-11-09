"""Layer 3 design study: can 'laser on a flat plate' calibrate the arm and the laser ray without any camera?

Offline simulation on the xArm7 URDF; recorded joint angles are used only as the 'task poses' that score a result.
Residual of one measurement:  r = n_k . FK(q; errors) (o + d u) - c_k     (plate k, laser range d)

Usage: layer3_plate_sim.py <urdf> <recorded_samples.json> <out.json>
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

import kin

URDF, SAMPLES, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
chain = kin.load_chain(URDF)
DEG, MM = np.pi / 180, 1e-3
O0 = np.array([-0.0152, -0.0011, 0.1019])            # laser origin in flange (current estimate)
U0 = np.array([0.0017, -0.0077, 1.0]); U0 /= np.linalg.norm(U0)
E1 = np.cross(U0, [1, 0, 0]); E1 /= np.linalg.norm(E1); E2 = np.cross(U0, E1)
SIGMA = 0.15 * MM                                     # per-measurement noise along the plate normal
RANGE = (0.130, 0.270)                                # usable laser range with margin [m]
MAX_INCIDENCE = 30 * DEG
LO = np.maximum(chain["lower"], -np.pi) + 3 * DEG
HI = np.minimum(chain["upper"], np.pi) - 3 * DEG
CAM_BOX = (np.array([0.38, -0.42, -0.2]), np.array([0.76, -0.23, 0.54]))  # camera head + post, conservative


def unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def plate(name, centre, normal, half=0.15):
    n = unit(normal)
    e1 = unit(np.cross([0, 1, 0], n)) if abs(n[1]) < 0.9 else unit(np.cross([1, 0, 0], n))
    return dict(name=name, centre=np.array(centre, float), n=n, e1=e1, e2=np.cross(n, e1), half=half)


def tilt(az_deg, tilt_deg):
    a, t = az_deg * DEG, tilt_deg * DEG
    return [np.sin(t) * np.cos(a), np.sin(t) * np.sin(a), np.cos(t)]


PLATES = {
    "H": plate("horizontal, on the table", [0.50, 0.08, 0.00], [0, 0, 1]),
    "T1": plate("tilted 40 deg, facing the robot", [0.52, 0.08, 0.13], tilt(180, 40)),
    "T2": plate("tilted 40 deg, facing sideways (+y)", [0.48, 0.02, 0.13], tilt(90, 40)),
    "T3": plate("tilted 40 deg, facing sideways (-y)", [0.48, 0.16, 0.13], tilt(-90, 40)),
    "V1": plate("vertical, facing the robot", [0.62, 0.08, 0.18], tilt(180, 90)),
    "V2": plate("vertical, facing sideways (+y)", [0.45, -0.05, 0.18], tilt(90, 90)),
}


def geometry(q):
    """Flange pose, all joint frames, laser origin/direction in base for a batch of configurations."""
    T, frames = kin.fk(chain, q, return_all=True)
    R, t = T[:, :3, :3], T[:, :3, 3]
    return T, frames, t + R @ O0, R @ U0


def feasible(q, pl):
    T, frames, p0, w = geometry(q)
    n = pl["n"]
    denom = w @ n
    with np.errstate(divide="ignore", invalid="ignore"):
        d = ((pl["centre"] - p0) @ n) / denom
    hit = p0 + d[:, None] * w
    rel = hit - pl["centre"]
    ok = (denom < -np.cos(MAX_INCIDENCE)) & (d > RANGE[0]) & (d < RANGE[1])
    ok &= (np.abs(rel @ pl["e1"]) < pl["half"] - 0.02) & (np.abs(rel @ pl["e2"]) < pl["half"] - 0.02)
    # crude body clearance: joint origins, link midpoints, flange and laser head
    pts = [f[:, :3, 3] for f in frames] + [p0]
    pts += [(a + b) / 2 for a, b in zip(pts[:-1], pts[1:])]
    for i, p in enumerate(pts):
        if i >= 2:
            ok &= p[:, 2] > 0.06                                   # above the table
        s = (p - pl["centre"]) @ n
        inside = (np.abs((p - pl["centre"]) @ pl["e1"]) < pl["half"] + 0.06) & (np.abs((p - pl["centre"]) @ pl["e2"]) < pl["half"] + 0.06)
        ok &= ~inside | (s > 0.07)                                 # stay in front of the plate
        ok &= ~np.all((p > CAM_BOX[0]) & (p < CAM_BOX[1]), axis=1)  # camera head / post
    flange = frames[6][:, :3, 3]
    ok &= (np.hypot(flange[:, 0], flange[:, 1]) > 0.20) | (flange[:, 2] > 0.45)  # not folded onto the base column
    return ok, d, hit


def build_pool(pl, rng, want=3000, chunk=400000, max_draw=12_000_000):
    qs, ds, drawn = [], [], 0
    while sum(len(a) for a in qs) < want and drawn < max_draw:
        q = rng.uniform(LO, HI, (chunk, 7))
        ok, d, _ = feasible(q, pl)
        qs.append(q[ok]); ds.append(d[ok]); drawn += chunk
    q, d = np.concatenate(qs)[:want], np.concatenate(ds)[:want]
    return q, d, drawn


# ---- parameter blocks -------------------------------------------------------------------------------------------
JOINT_NAMES = [f"J{i} zero" for i in range(1, 8)]
DH_NAMES = [f"d{i}" for i in range(1, 8)] + [f"a{i}" for i in range(1, 8)] + [f"alpha{i}" for i in range(1, 8)]
RAY_NAMES = ["ray origin x", "ray origin y", "ray origin z", "ray dir a", "ray dir b"]
UNIT = {"J": 0.01 * DEG, "d": 0.1 * MM, "a": 0.1 * MM, "al": 0.01 * DEG, "ro": 0.1 * MM, "rd": 0.01 * DEG, "pl_t": 0.01 * DEG, "pl_c": 0.1 * MM,
        "rz": 0.1 * MM, "rs": 1e-3}


def point_jacobian(q, d):
    """d(point in base)/d(parameter), shape (N, 3, 7+21+5+2), per unit parameter (rad or m).

    Order: 7 joint zeros, 7 d, 7 a, 7 alpha, ray origin (3), ray direction (2), range zero, range scale.
    """
    T, frames, p0, w = geometry(q)
    R = T[:, :3, :3]
    p = p0 + d[:, None] * w
    n = len(q)
    J = np.zeros((n, 3, 35))
    prev_o = np.zeros((n, 3)); prev_x = np.tile([1.0, 0, 0], (n, 1))
    for j in range(7):
        oj, zj = frames[j][:, :3, 3], frames[j][:, :3, 2]
        J[:, :, j] = np.cross(zj, p - oj)                # joint zero
        J[:, :, 7 + j] = zj                               # d_j
        J[:, :, 14 + j] = prev_x                          # a_j   (along parent x)
        J[:, :, 21 + j] = np.cross(prev_x, p - prev_o)    # alpha_j (about parent x)
        prev_o, prev_x = oj, frames[j][:, :3, 0]
    J[:, :, 28:31] = R
    J[:, :, 31] = d[:, None] * (R @ E1)
    J[:, :, 32] = d[:, None] * (R @ E2)
    J[:, :, 33] = w
    J[:, :, 34] = d[:, None] * w
    return J, p


POINT_UNITS = np.r_[[UNIT["J"]] * 7, [UNIT["d"]] * 7, [UNIT["a"]] * 7, [UNIT["al"]] * 7, [UNIT["ro"]] * 3, [UNIT["rd"]] * 2, UNIT["rz"], UNIT["rs"]]
POINT_NAMES = JOINT_NAMES + DH_NAMES + RAY_NAMES + ["range zero", "range scale"]


def design_matrix(sets, plates):
    """sets: list of (q, d) per plate. Returns whitened Jacobian in 'units', names and unit labels."""
    rows, names = [], list(POINT_NAMES)
    k = len(sets)
    for i, ((q, d), pl) in enumerate(zip(sets, plates)):
        Jp, p = point_jacobian(q, d)
        Jr = np.einsum("k,nkp->np", pl["n"], Jp) * POINT_UNITS
        blk = np.zeros((len(q), 3 * k))
        rel = p - pl["centre"]
        blk[:, 3 * i] = (rel @ pl["e1"]) * UNIT["pl_t"]      # plate tilt about e2
        blk[:, 3 * i + 1] = (rel @ pl["e2"]) * UNIT["pl_t"]  # plate tilt about e1
        blk[:, 3 * i + 2] = -UNIT["pl_c"]                    # plate offset
        rows.append(np.c_[Jr, blk])
    for i in range(k):
        names += [f"plate{i+1} tilt a", f"plate{i+1} tilt b", f"plate{i+1} offset"]
    return np.concatenate(rows) / SIGMA, names


def cols(names, want):
    return [i for i, n in enumerate(names) if n in want]


MAIN = ["J2 zero", "J3 zero", "J4 zero", "J5 zero", "J6 zero"] + RAY_NAMES   # what layer 3 estimates (+ plates)


def analyse(sets, plates, free=MAIN, label=""):
    Jw, names = design_matrix(sets, plates)
    idx = cols(names, set(free)) + [i for i, n in enumerate(names) if n.startswith("plate")]
    A = Jw[:, idx]
    sv = np.linalg.svd(A, compute_uv=False)
    cov = np.linalg.pinv(A.T @ A, rcond=1e-12)
    std = np.sqrt(np.diag(cov))
    return {"label": label, "n": int(len(A)), "cond": float(sv[0] / sv[-1]), "names": [names[i] for i in idx], "std_units": std.tolist(),
            "cov": cov, "idx": idx, "all_names": names}


def null_space(sets, plates, extra=()):
    Jw, names = design_matrix(sets, plates)
    free = JOINT_NAMES + RAY_NAMES + list(extra)
    idx = cols(names, set(free)) + [i for i, n in enumerate(names) if n.startswith("plate")]
    A = Jw[:, idx]
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    out = []
    for i, s in enumerate(S):
        if s < 1e-7 * S[0]:
            v = Vt[i]; order = np.argsort(-np.abs(v))[:5]
            out.append([(names[idx[j]], round(float(v[j]), 3)) for j in order if abs(v[j]) > 0.08])
    return {"params": len(idx), "rank": int((S > 1e-7 * S[0]).sum()), "null": out}


def greedy_d_optimal(pools, plates, per_plate, free=MAIN, rng=None):
    """Forward selection maximising log det of the information matrix of (free + plate) parameters."""
    k = len(pools)
    Jw, names = design_matrix(pools, plates)
    idx = cols(names, set(free)) + [i for i, n in enumerate(names) if n.startswith("plate")]
    A = Jw[:, idx]
    owner = np.concatenate([np.full(len(q), i) for i, (q, _) in enumerate(pools)])
    offs = np.cumsum([0] + [len(q) for q, _ in pools])
    Minv = np.eye(len(idx)) * 1e6                    # weak prior so the first picks are defined
    chosen, count = [], np.zeros(k, int)
    avail = np.ones(len(A), bool)
    for _ in range(per_plate * k):
        gain = np.einsum("ij,jk,ik->i", A, Minv, A)
        gain[~avail] = -1
        gain[count[owner] >= per_plate] = -1
        b = int(np.argmax(gain))
        a = A[b]
        Ma = Minv @ a
        Minv -= np.outer(Ma, Ma) / (1 + a @ Ma)
        chosen.append(b); avail[b] = False; count[owner[b]] += 1
    chosen = np.array(chosen)
    return [(pools[i][0][chosen[owner[chosen] == i] - offs[i]], pools[i][1][chosen[owner[chosen] == i] - offs[i]]) for i in range(k)]


def random_subset(pools, per_plate, rng):
    out = []
    for q, d in pools:
        i = rng.choice(len(q), per_plate, replace=False)
        out.append((q[i], d[i]))
    return out


def rigid_projector(P):
    """Projector removing the part of a point-displacement field that a rigid motion could explain."""
    n = len(P)
    G = np.zeros((3 * n, 6))
    for a in range(3):
        e = np.zeros(3); e[a] = 1
        G[a::3, a] = 1
        G[:, 3 + a] = np.cross(e, P - P.mean(0)).ravel()
    Q, _ = np.linalg.qr(G)
    return np.eye(3 * n) - Q @ Q.T


def tool_point_error(res, qv, dv):
    """RMS (mm) of the non-rigid error at the laser spot over task poses, given the layer-3 posterior."""
    Jp, P = point_jacobian(qv, dv)
    names = res["names"]
    sel = [POINT_NAMES.index(n) for n in names if n in POINT_NAMES]
    keep = [i for i, n in enumerate(names) if n in POINT_NAMES]
    Jv = (Jp[:, :, sel] * POINT_UNITS[sel]).reshape(-1, len(sel))
    C = res["cov"][np.ix_(keep, keep)]
    Pr = rigid_projector(P)
    Jn = Pr @ Jv
    return float(np.sqrt(np.trace(Jn @ C @ Jn.T) / len(qv)) * 1000), float(np.sqrt(np.trace(Jv @ C @ Jv.T) / len(qv)) * 1000)


def sensitivity(qv, dv):
    """Non-rigid tool-point RMS (mm) caused by one unit of each single error source over the task poses."""
    Jp, P = point_jacobian(qv, dv)
    Pr = rigid_projector(P)
    out = {}
    for i, n in enumerate(POINT_NAMES):
        v = (Jp[:, :, i] * POINT_UNITS[i]).reshape(-1)
        out[n] = (float(np.sqrt(((Pr @ v) ** 2).sum() / len(qv)) * 1000), float(np.sqrt((v ** 2).sum() / len(qv)) * 1000))
    return out


# ---- nonlinear Monte Carlo: simulate, solve, compare -------------------------------------------------------------
def true_points(q, d, th, dh, o, u):
    T = kin.fk(chain, q, dtheta=th, dd=dh[0], da=dh[1], dalpha=dh[2])
    return np.einsum("nij,nj->ni", T[:, :3, :3], o[None] + d[:, None] * u[None]) + T[:, :3, 3]


def solve_layer3(sets, plates, d_meas, with_joints=True):
    k = len(sets)
    def unpack(x):
        th = np.zeros(7)
        if with_joints:
            th[1:6] = x[:5] * UNIT["J"]
        b = 5 if with_joints else 0
        o = O0 + x[b:b + 3] * UNIT["ro"]
        u = unit(U0 + x[b + 3] * UNIT["rd"] * E1 + x[b + 4] * UNIT["rd"] * E2)
        return th, o, u, x[b + 5:].reshape(k, 3)
    def cost(x):
        th, o, u, pp = unpack(x)
        r = []
        for (q, _), pl, dm, (ta, tb, c) in zip(sets, plates, d_meas, pp):
            n = unit(pl["n"] + ta * UNIT["pl_t"] * pl["e1"] + tb * UNIT["pl_t"] * pl["e2"])
            P = true_points(q, dm, th, np.zeros((3, 7)), o, u)
            r.append(((P - pl["centre"]) @ n - c * UNIT["pl_c"]) / SIGMA)
        return np.concatenate(r)
    x0 = np.zeros((5 if with_joints else 0) + 5 + 3 * k)
    sol = least_squares(cost, x0, x_scale="jac", xtol=1e-13, ftol=1e-13, gtol=1e-13)
    th, o, u, _ = unpack(sol.x)
    return th, o, u, float(np.sqrt(np.mean(sol.fun ** 2)) * SIGMA * 1000)


def kabsch_rms(A, B):
    ca, cb = A.mean(0), B.mean(0)
    U, S, Vt = np.linalg.svd((A - ca).T @ (B - cb))
    Dm = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
    Rm = Vt.T @ Dm @ U.T
    return float(np.sqrt((((A - ca) @ Rm.T - (B - cb)) ** 2).sum(1).mean()) * 1000)


def monte_carlo(sets, plates, qv, dv, rng, trials, joint_sigma_deg, link_mm, twist_deg):
    rec = {"err_J_deg": [], "before": [], "ray_only": [], "full": [], "fit_rms": [], "fit_rms_ray_only": []}
    for _ in range(trials):
        th = rng.normal(0, joint_sigma_deg * DEG, 7)
        dh = np.array([rng.normal(0, link_mm * MM, 7), rng.normal(0, link_mm * MM, 7), rng.normal(0, twist_deg * DEG, 7)])
        o = O0 + rng.normal(0, 2 * MM, 3)
        u = unit(U0 + rng.normal(0, 1.0 * DEG) * E1 + rng.normal(0, 1.0 * DEG) * E2)
        d_meas, tp = [], []
        for (q, _), pl in zip(sets, plates):
            # the plate is not where we think: unknown tilt (2 deg) and offset (5 mm)
            n = unit(pl["n"] + rng.normal(0, 2 * DEG) * pl["e1"] + rng.normal(0, 2 * DEG) * pl["e2"])
            c = rng.normal(0, 5 * MM)
            T = kin.fk(chain, q, dtheta=th, dd=dh[0], da=dh[1], dalpha=dh[2])
            p0 = T[:, :3, 3] + T[:, :3, :3] @ o
            w = T[:, :3, :3] @ u
            d = (c - (p0 - pl["centre"]) @ n) / (w @ n)
            d_meas.append(d + rng.normal(0, SIGMA, len(d)) / np.abs(w @ n))   # noise along the plate normal
        Ptrue = true_points(qv, dv, th, dh, o, u)
        z = np.zeros((3, 7))
        rec["before"].append(kabsch_rms(true_points(qv, dv, np.zeros(7), z, O0, U0), Ptrue))
        th1, o1, u1, f1 = solve_layer3(sets, plates, d_meas, with_joints=False)
        rec["ray_only"].append(kabsch_rms(true_points(qv, dv, th1, z, o1, u1), Ptrue)); rec["fit_rms_ray_only"].append(f1)
        th2, o2, u2, f2 = solve_layer3(sets, plates, d_meas, with_joints=True)
        rec["full"].append(kabsch_rms(true_points(qv, dv, th2, z, o2, u2), Ptrue)); rec["fit_rms"].append(f2)
        rec["err_J_deg"].append((th2[1:6] - th[1:6]) / DEG)
    e = np.array(rec["err_J_deg"])
    return {"trials": trials, "truth": {"joint_sigma_deg": joint_sigma_deg, "link_sigma_mm": link_mm, "twist_sigma_deg": twist_deg},
            "joint_zero_error_rms_deg": np.sqrt((e ** 2).mean(0)).tolist(),
            "tool_point_nonrigid_rms_mm": {k: float(np.sqrt(np.mean(np.square(rec[k])))) for k in ("before", "ray_only", "full")},
            "plate_fit_rms_mm": {"ray_only": float(np.mean(rec["fit_rms_ray_only"])), "full": float(np.mean(rec["fit_rms"]))}}


def joint_std(res):
    std = dict(zip(res["names"], res["std_units"]))
    return [std[f"J{i} zero"] * 0.01 for i in range(2, 7)], std


def design_rows(pools, combos, totals, qv, dv, rng, keep=None):
    """Posterior of the joint-zero model for random and D-optimal pose sets drawn from the feasible pools."""
    rows = []
    for combo in combos:
        pl = [PLATES[k] for k in combo]; pp = [pools[k] for k in combo]
        for total in totals:
            per = total // len(combo)
            rs = [analyse(random_subset(pp, per, rng), pl) for _ in range(20)]
            sel = greedy_d_optimal(pp, pl, per)
            res = analyse(sel, pl); jd, std = joint_std(res)
            row = {"plates": "+".join(combo), "poses": per * len(combo),
                   "random_joint_zero_std_deg": np.median([joint_std(r)[0] for r in rs], axis=0).tolist(),
                   "random_tool_point_mm": float(np.median([tool_point_error(r, qv, dv)[0] for r in rs])),
                   "dopt_joint_zero_std_deg": jd, "dopt_ray_direction_std_deg": std["ray dir a"] * 0.01,
                   "dopt_ray_origin_std_mm": [std[f"ray origin {a}"] * 0.1 for a in "xyz"], "dopt_tool_point_mm": tool_point_error(res, qv, dv)[0]}
            rows.append(row)
            if keep is not None:
                keep[("+".join(combo), per * len(combo))] = (sel, pl)
            print(f" {row['plates']:12s} n={row['poses']:3d} random J2-J6 {np.round(row['random_joint_zero_std_deg'], 3)} tool {row['random_tool_point_mm']:.3f} mm"
                  f" | D-opt J2-J6 {np.round(jd, 3)} tool {row['dopt_tool_point_mm']:.3f} mm")
    return rows


if __name__ == "__main__":
    rng = np.random.default_rng(20261109)
    np.set_printoptions(precision=3, suppress=True, linewidth=170)
    seen, qv, dv = set(), [], []
    for s in json.loads(Path(SAMPLES).read_text()):
        r = s["robot"]["before"]; key = tuple(np.round(r["joints"], 4))
        if key in seen or not (120 <= s["laser"]["mean_mm"] <= 280):
            continue
        seen.add(key); qv.append(r["joints"]); dv.append(s["laser"]["mean_mm"] / 1000)
    qv, dv = np.array(qv), np.array(dv)
    result = {"noise_sigma_mm": SIGMA * 1000, "laser_range_m": RANGE, "max_incidence_deg": MAX_INCIDENCE / DEG, "task_poses": len(qv),
              "plates": {k: p["name"] for k, p in PLATES.items()}}

    wide = (LO.copy(), HI.copy())
    pools = {k: build_pool(PLATES[k], rng, want=2500)[:2] for k in ("H", "T1", "T2", "T3")}
    result["feasible_pool"] = {k: len(v[0]) for k, v in pools.items()}

    sens = sensitivity(qv, dv)
    result["task_pose_error_per_unit_mm"] = {n: {"non_rigid": sens[n][0], "total": sens[n][1]} for n in sens}
    result["units"] = {"joint zero": "0.01 deg", "d / a": "0.1 mm", "alpha": "0.01 deg", "ray origin": "0.1 mm", "ray dir": "0.01 deg"}

    result["unobservable"] = {"recorded poses, existing board": null_space([(qv, dv)], [PLATES["H"]], extra=["range zero"])}
    for combo in (["H"], ["H", "T1", "T2"]):
        result["unobservable"]["+".join(combo)] = null_space([pools[k] for k in combo], [PLATES[k] for k in combo], extra=["range zero"])
    print(json.dumps(result["unobservable"], ensure_ascii=False))

    res = analyse([(qv, dv)], [PLATES["H"]]); jd, std = joint_std(res)
    result["recorded_poses_as_layer3"] = {"poses": len(qv), "joint_zero_std_deg": jd, "ray_direction_std_deg": std["ray dir a"] * 0.01,
                                          "ray_origin_std_mm": [std[f"ray origin {a}"] * 0.1 for a in "xyz"], "tool_point_mm": tool_point_error(res, qv, dv)[0]}
    print("recorded poses:", json.dumps(result["recorded_poses_as_layer3"]))

    print("\njoint-zero model, full joint range")
    designs = {}
    result["joint_zero_model_full_range"] = design_rows(pools, (["H"], ["H", "T1", "T2"], ["H", "T1", "T2", "T3"]), (30, 60, 120), qv, dv, rng, keep=designs)

    print("\njoint-zero model, conservative envelope (joint ranges already used on 2026-11-09, widened by 15 deg)")
    LO[:] = np.maximum(qv.min(0) - 15 * DEG, wide[0]); HI[:] = np.minimum(qv.max(0) + 15 * DEG, wide[1])
    cpools = {k: build_pool(PLATES[k], rng, want=2500)[:2] for k in ("H", "T1", "T3")}
    cdesigns = {}
    result["joint_zero_model_conservative"] = {"envelope_deg": [np.degrees(LO).tolist(), np.degrees(HI).tolist()],
                                               "rows": design_rows(cpools, (["H"], ["H", "T1", "T3"]), (30, 60, 120), qv, dv, rng, keep=cdesigns)}
    LO[:], HI[:] = wide

    print("\nfull link model (28 DH errors; theta1, d1, a1, alpha1, theta7, d7 fixed as gauge)")
    FULL = [f"J{i} zero" for i in range(2, 7)] + [f"d{i}" for i in range(2, 7)] + [f"a{i}" for i in range(2, 8)] + [f"alpha{i}" for i in range(2, 8)] + RAY_NAMES
    result["full_link_model"] = []
    for combo in (["H"], ["H", "T1", "T2"], ["H", "T1", "T2", "T3"]):
        pl = [PLATES[k] for k in combo]; pp = [pools[k] for k in combo]
        Jw, names = design_matrix(pp, pl)
        idx = [i for i, n in enumerate(names) if n not in ("range zero", "range scale")]
        sv = np.linalg.svd(Jw[:, idx], compute_uv=False)
        unobs = len(idx) - int((sv > 1e-7 * sv[0]).sum())
        for total in (60, 120, 240):
            per = total // len(combo)
            res = analyse(greedy_d_optimal(pp, pl, per, free=FULL), pl, free=FULL)
            std = dict(zip(res["names"], res["std_units"]))
            row = {"plates": "+".join(combo), "poses": per * len(combo), "unobservable_combinations": unobs, "condition_number": res["cond"],
                   "well_posed": bool(res["cond"] < 1e6)}
            if row["well_posed"]:
                row.update({"joint_zero_std_deg_max": max(std[f"J{i} zero"] for i in range(2, 7)) * 0.01, "d_std_mm_max": max(std[f"d{i}"] for i in range(2, 7)) * 0.1,
                            "a_std_mm_max": max(std[f"a{i}"] for i in range(2, 8)) * 0.1, "alpha_std_deg_max": max(std[f"alpha{i}"] for i in range(2, 8)) * 0.01,
                            "tool_point_mm": tool_point_error(res, qv, dv)[0]})
            result["full_link_model"].append(row)
            print(" ", json.dumps(row))

    print("\nnonlinear Monte Carlo (simulate -> solve -> score on task poses, rigid part removed)")
    result["monte_carlo"] = {}
    for name, (sel, pl), kw in (("1 plate x60, joint zeros only", designs[("H", 60)], dict(joint_sigma_deg=0.05, link_mm=0.0, twist_deg=0.0)),
                                ("1 plate x60, with unmodelled link errors", designs[("H", 60)], dict(joint_sigma_deg=0.05, link_mm=0.1, twist_deg=0.01)),
                                ("1 plate x60 conservative envelope, with unmodelled link errors", cdesigns[("H", 60)], dict(joint_sigma_deg=0.05, link_mm=0.1, twist_deg=0.01)),
                                ("3 plates x20, with unmodelled link errors", designs[("H+T1+T2", 60)], dict(joint_sigma_deg=0.05, link_mm=0.1, twist_deg=0.01)),
                                ("3 plates x40, with unmodelled link errors", designs[("H+T1+T2", 120)], dict(joint_sigma_deg=0.05, link_mm=0.1, twist_deg=0.01))):
        r = monte_carlo(sel, pl, qv, dv, rng, 40, **kw)
        result["monte_carlo"][name] = r
        print(f" {name:64s} J2-J6 err {np.round(r['joint_zero_error_rms_deg'], 3)} | tool-point mm: {r['tool_point_nonrigid_rms_mm']} | plate fit {r['plate_fit_rms_mm']}")
    Path(OUT).write_text(json.dumps(result, ensure_ascii=False, indent=1))
