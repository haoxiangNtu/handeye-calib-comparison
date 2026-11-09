"""Layer 4 study: with camera, robot and laser models frozen, how well do N laser spots fix the 6 extrinsic numbers?
Also: what a camera scale error does when it is NOT removed in layer 1.  Offline; uses recorded spot positions only.
Usage: l4_sim.py <data_root> <out.json>
"""
import json, sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
import kin

np.set_printoptions(precision=3, suppress=True, linewidth=170)
D = Path(sys.argv[1]); R = D / "data/recalibration/20261109_camera_shift_recalibration"
active = json.loads((D / "calibration/active/final_robot_camera_laser_calibration.json").read_text())
train = json.loads((R / "train_complete_strict.json").read_text())
T_bc = np.array(active["T_base_camera"]["matrix"])
o = np.array(active["laser_ray_in_flange"]["origin_m"]); u = np.array(active["laser_ray_in_flange"]["direction_unit"])
F, B, CX, CY = 729.2711917567123, 0.12111325211874005, 933.8052520751953, 526.6510887145996
Tf = np.array([kin.pose_to_matrix(s["robot"]["before"]["pose"]) for s in train])
d = np.array([s["laser"]["mean_mm"] for s in train]) / 1000
P_board = np.einsum("nij,nj->ni", Tf[:, :3, :3], o[None] + d[:, None] * u[None]) + Tf[:, :3, 3]
centre = P_board.mean(0)
print("spots on board:", len(P_board), "centre", centre, "x", P_board[:, 0].min(), P_board[:, 0].max(), "y", P_board[:, 1].min(), P_board[:, 1].max())


def project(Pb, T, f=F, b=B):
    pc = (Pb - T[:3, 3]) @ T[:3, :3]
    xl = f * pc[:, 0] / pc[:, 2] + CX; yl = f * pc[:, 1] / pc[:, 2] + CY
    return np.c_[xl, yl, xl - f * b / pc[:, 2]], pc


def triangulate(obs, f=F, b=B):
    z = f * b / (obs[:, 0] - obs[:, 2])
    return np.c_[(obs[:, 0] - CX) * z / f, (obs[:, 1] - CY) * z / f, z]


def perturb(T, x):
    out = np.eye(4); Rm = Rotation.from_rotvec(x[:3]).as_matrix()
    out[:3, :3] = Rm @ T[:3, :3]; out[:3, 3] = T[:3, 3] + x[3:]
    return out


def extrinsic_cov(Pb, sigma_px):
    J = np.zeros((3 * len(Pb), 6)); h = 1e-6
    for k in range(6):
        e = np.zeros(6); e[k] = h
        J[:, k] = ((project(Pb, perturb(T_bc, e))[0] - project(Pb, perturb(T_bc, -e))[0]) / (2 * h)).ravel()
    return np.linalg.inv(J.T @ J) * sigma_px ** 2


def point_std(cov, p):
    """1-sigma position error (mm) at base point p caused by extrinsic uncertainty alone."""
    r = p - T_bc[:3, 3]
    A = np.zeros((3, 6))
    A[:, :3] = -np.array([[0, -r[2], r[1]], [r[2], 0, -r[0]], [-r[1], r[0], 0]])
    A[:, 3:] = np.eye(3)
    return float(np.sqrt(np.trace(A @ cov @ A.T)) * 1000)


tests = {"on board, centre": centre, "on board, corner": np.array([P_board[:, 0].max(), P_board[:, 1].max(), centre[2]]),
         "150 mm above board": centre + [0, 0, 0.15], "300 mm above board": centre + [0, 0, 0.30]}
rng = np.random.default_rng(3)
def raised(n, h):
    i = rng.choice(len(P_board), n, replace=False); P = P_board[i].copy(); P[:, 2] += h; return P
designs = {
    "28 spots, all on the board (today)": P_board,
    "12 spots on the board": P_board[rng.choice(len(P_board), 12, replace=False)],
    "28 on board + 10 at +100 mm + 10 at +200 mm": np.r_[P_board, raised(10, 0.10), raised(10, 0.20)],
    "16 on board + 16 at +100 mm + 16 at +200 mm": np.r_[P_board[rng.choice(len(P_board), 16, replace=False)], raised(16, 0.10), raised(16, 0.20)],
}
SIG = 0.6  # px: spot localisation ~0.5 px combined with ~0.2 mm robot+laser point uncertainty
out = {"sigma_px": SIG, "designs": {}}
print(f"\nextrinsic-only (6 numbers), sigma {SIG} px per image coordinate; error at a point = extrinsic uncertainty only")
for name, Pb in designs.items():
    cov = extrinsic_cov(Pb, SIG)
    rot = np.degrees(np.sqrt(np.trace(cov[:3, :3]))); tr = np.sqrt(np.trace(cov[3:, 3:])) * 1000
    res = {k: point_std(cov, p) for k, p in tests.items()}
    out["designs"][name] = {"n": len(Pb), "rot_std_deg": float(rot), "trans_std_mm": float(tr), "point_std_mm": res}
    print(f" {name:46s} rot {rot:.3f} deg, trans {tr:.2f} mm | " + " | ".join(f"{k}: {v:.2f}" for k, v in res.items()))

# single observation noise of the camera itself at the working distance (not a calibration error)
obs, pc = project(tests["on board, centre"][None], T_bc)
z = pc[0, 2]
out["camera_single_shot"] = {"depth_m": float(z), "lateral_mm_per_px": float(z / F * 1000), "depth_mm_per_px_disparity": float(z * z / (F * B) * 1000)}
print("\ncamera single-shot sensitivity at board centre:", out["camera_single_shot"])

print("\nif the camera scale is wrong and layer 1 is skipped (extrinsic re-fitted on the board spots, then used elsewhere):")
out["scale_error"] = {}
for eps_b, eps_f, label in ((0.01, 0.0, "baseline +1%"), (0.024, 0.0, "baseline +2.4%"), (0.0, 0.02, "focal length +2%")):
    obs, _ = project(P_board, T_bc)                               # what the true camera sees
    pc_wrong = triangulate(obs, f=F * (1 + eps_f), b=B * (1 + eps_b))   # reconstructed with the wrong model
    ca, cb = pc_wrong.mean(0), P_board.mean(0)
    U, S_, Vt = np.linalg.svd((pc_wrong - ca).T @ (P_board - cb)); Dm = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
    Rf = Vt.T @ Dm @ U.T; tf = cb - Rf @ ca
    fit = np.linalg.norm(pc_wrong @ Rf.T + tf - P_board, axis=1) * 1000
    row = {"fit_rms_mm": float(np.sqrt((fit ** 2).mean())), "camera_shift_mm": float(np.linalg.norm(tf - T_bc[:3, 3]) * 1000)}
    for k, p in tests.items():
        o1, _ = project(p[None], T_bc)
        est = triangulate(o1, f=F * (1 + eps_f), b=B * (1 + eps_b)) @ Rf.T + tf
        row[k] = float(np.linalg.norm(est[0] - p) * 1000)
    out["scale_error"][label] = row
    print(f" {label:18s} fit RMS on board {row['fit_rms_mm']:.2f} mm, fitted camera position moves {row['camera_shift_mm']:.1f} mm | " + " | ".join(f"{k}: {row[k]:.2f}" for k in tests))
Path(sys.argv[2]).write_text(json.dumps(out, indent=1))
