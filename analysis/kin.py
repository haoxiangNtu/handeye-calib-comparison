"""Minimal vectorised kinematics for a 7R arm described by a URDF (modified-DH compatible origins).

T_i(q) = Rx(dalpha_i) Tx(da_i) T_origin_i Rz(q_i + dtheta_i) Tz(dd_i)
so (dtheta, dd, da, dalpha) are exactly the modified-DH parameter errors of joint i.
"""
import xml.etree.ElementTree as ET

import numpy as np

JOINTS = tuple(f"joint{i}" for i in range(1, 8))


def rot_x(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rot_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def rpy_matrix(r, p, y):
    return rot_z(y) @ rot_y(p) @ rot_x(r)


def hom(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def load_chain(urdf_path):
    root = ET.parse(urdf_path).getroot()
    by_name = {j.get("name"): j for j in root.findall("joint")}
    origins, lower, upper = [], [], []
    for name in JOINTS:
        j = by_name[name]
        o = j.find("origin")
        xyz = [float(v) for v in o.get("xyz").split()]
        rpy = [float(v) for v in o.get("rpy").split()]
        axis = [float(v) for v in j.find("axis").get("xyz").split()]
        assert axis == [0.0, 0.0, 1.0], axis
        origins.append(hom(rpy_matrix(*rpy), xyz))
        lim = j.find("limit")
        lower.append(float(lim.get("lower")))
        upper.append(float(lim.get("upper")))
    return {"origins": np.array(origins), "lower": np.array(lower), "upper": np.array(upper)}


def fk(chain, q, dtheta=None, dd=None, da=None, dalpha=None, return_all=False):
    """q: (N, 7) joint angles [rad]. Returns flange poses (N, 4, 4); with return_all also every joint frame."""
    q = np.atleast_2d(np.asarray(q, dtype=float))
    n = q.shape[0]
    z = np.zeros(7)
    dtheta = z if dtheta is None else dtheta
    dd = z if dd is None else dd
    da = z if da is None else da
    dalpha = z if dalpha is None else dalpha
    T = np.tile(np.eye(4), (n, 1, 1))
    frames = []
    for i in range(7):
        A = hom(rot_x(dalpha[i]), [da[i], 0, 0]) @ chain["origins"][i]
        th = q[:, i] + dtheta[i]
        c, s = np.cos(th), np.sin(th)
        Rz = np.zeros((n, 4, 4))
        Rz[:, 0, 0], Rz[:, 0, 1], Rz[:, 1, 0], Rz[:, 1, 1] = c, -s, s, c
        Rz[:, 2, 2] = Rz[:, 3, 3] = 1.0
        Rz[:, 2, 3] = dd[i]
        T = T @ A @ Rz
        if return_all:
            frames.append(T.copy())
    return (T, frames) if return_all else T


def pose_to_matrix(pose):
    """xArm controller pose [x, y, z (mm), roll, pitch, yaw (rad)] -> 4x4 in metres."""
    pose = np.asarray(pose, dtype=float)
    return hom(rpy_matrix(*pose[3:6]), pose[:3] / 1000.0)


def rotvec_of(R):
    """Rotation matrix (..., 3, 3) -> rotation vector (..., 3)."""
    R = np.asarray(R)
    c = np.clip((np.trace(R, axis1=-2, axis2=-1) - 1) / 2, -1, 1)
    th = np.arccos(c)
    v = np.stack([R[..., 2, 1] - R[..., 1, 2], R[..., 0, 2] - R[..., 2, 0], R[..., 1, 0] - R[..., 0, 1]], axis=-1)
    k = np.where(th < 1e-9, 0.5, th / (2 * np.sin(np.where(th < 1e-9, 1.0, th))))
    return v * k[..., None]
