from typing import Optional, Tuple, Union

import mujoco
import numpy as np
from dm_robotics.transformations import transformations as tr

def pseudo_inverse(M, damped=True):
    lambda_ = 0.2 if damped else 0.0

    U, sing_vals, Vt = np.linalg.svd(M, full_matrices=False)
    S_inv = np.zeros_like(M, dtype=float)
    for i in range(len(sing_vals)):
        S_inv[i, i] = sing_vals[i] / (sing_vals[i] ** 2 + lambda_ ** 2)

    return np.dot(Vt.T, np.dot(S_inv, U.T))

def pd_control(
    x: np.ndarray,
    x_des: np.ndarray,
    dx: np.ndarray,
    kp_kv: np.ndarray,
    max_pos_error: float = 0.05,
) -> np.ndarray:
    # Compute error.
    x_err = x - x_des
    dx_err = dx

    # Clip pos error
    x_err = np.clip(x_err, -max_pos_error, max_pos_error)

    # Apply gains.
    x_err *= -kp_kv[:, 0]
    dx_err *= -kp_kv[:, 1]

    return x_err + dx_err


def pd_control_orientation(
    quat: np.ndarray,
    quat_des: np.ndarray,
    w: np.ndarray,
    kp_kv: np.ndarray,
    max_ori_error: float = 0.05,
) -> np.ndarray:
    # Compute error.
    quat_err = tr.quat_diff_active(source_quat=quat_des, target_quat=quat)
    ori_err = tr.quat_to_axisangle(quat_err)
    w_err = w

    # Clip ori error
    ori_err = np.clip(ori_err, -max_ori_error, max_ori_error)

    # Apply gains.
    ori_err *= -kp_kv[:, 0]
    w_err *= -kp_kv[:, 1]

    return ori_err + w_err

def saturate_torque_rate(tau_d_calculated, tau_J_d, delta_tau_max):
    tau_d_saturated = np.zeros_like(tau_d_calculated)
    for i in range(len(tau_d_calculated)):
        difference = tau_d_calculated[i] - tau_J_d[i]
        tau_d_saturated[i] = tau_J_d[i] + np.clip(difference, -delta_tau_max, delta_tau_max)
    return tau_d_saturated


def opspace_2(
    model,
    data,
    site_id,
    dof_ids: np.ndarray,
    pos: Optional[np.ndarray] = None,
    ori: Optional[np.ndarray] = None,
    joint: Optional[np.ndarray] = None,
    pos_gains: Union[Tuple[float, float, float], np.ndarray] = (400.0, 400.0, 400.0),
    ori_gains: Union[Tuple[float, float, float], np.ndarray] = (150.0, 150.0, 150.0),
    translational_damping: float = 40.0,
    rotational_damping: float = 7.16,
    nullspace_stiffness: float = 10.0,
    joint1_nullspace_stiffness: float = 10.0,
    max_pos_error: float = 0.05,
    max_ori_error: float = 0.05,
    delta_tau_max: float = 1.0,
    gravity_comp: bool = True,
) -> np.ndarray:
    if pos is None:
        x_des = data.site_xpos[site_id]
    else:
        x_des = np.asarray(pos)
    if ori is None:
        xmat = data.site_xmat[site_id].reshape((3, 3))
        quat_des = tr.mat_to_quat(xmat.reshape((3, 3)))
    else:
        ori = np.asarray(ori)
        if ori.shape == (3, 3):
            quat_des = tr.mat_to_quat(ori)
        else:
            quat_des = ori
    if joint is None:
        q_des = data.qpos[dof_ids]
    else:
        q_des = np.asarray(joint)

    kp = np.asarray(pos_gains)
    kd = translational_damping*np.ones_like(kp)
    kp_kv_pos = np.stack([kp, kd], axis=-1)

    kp = np.asarray(ori_gains)
    kd = rotational_damping*np.ones_like(kp)
    kp_kv_ori = np.stack([kp, kd], axis=-1)

    kp_joint = np.full((len(dof_ids),), nullspace_stiffness)
    kd_joint = 2 * np.sqrt(kp_joint)
    kp_kv_joint = np.stack([kp_joint, kd_joint], axis=-1)

    # Get current state.
    q = data.qpos[dof_ids]
    dq = data.qvel[dof_ids]

    # Compute Jacobian of the eef site in world frame.
    J_v = np.zeros((3, model.nv), dtype=np.float64)
    J_w = np.zeros((3, model.nv), dtype=np.float64)
    mujoco.mj_jacSite(
        model,
        data,
        J_v,
        J_w,
        site_id,
    )
    J_v = J_v[:, dof_ids]
    J_w = J_w[:, dof_ids]
    J = np.concatenate([J_v, J_w], axis=0)

    # Compute position PD control.
    x = data.site_xpos[site_id]
    dx = J_v @ dq
    ddx = pd_control(
        x=x,
        x_des=x_des,
        dx=dx,
        kp_kv=kp_kv_pos,
        max_pos_error=max_pos_error,
    )

    # Compute orientation PD control.
    quat = tr.mat_to_quat(data.site_xmat[site_id].reshape((3, 3)))
    if quat @ quat_des < 0.0:
        quat *= -1.0
    w = J_w @ dq
    dw = pd_control_orientation(
        quat=quat,
        quat_des=quat_des,
        w=w,
        kp_kv=kp_kv_ori,
        max_ori_error=max_ori_error,
    )

    # Compute Coriolis and centrifugal terms.
    C = data.qfrc_bias[dof_ids]

    # Compute task-space forces.
    F = np.concatenate([ddx, dw])

    # Compute joint torques using Jacobian transpose.
    tau_task = J.T @ F

    # Nullspace control.
    # jacobian_transpose_pinv = pseudo_inverse(J.T)
    q_error = q_des - q
    q_error[0] *= joint1_nullspace_stiffness
    dq_error = -dq
    dq_error[1] *= 2*np.sqrt(joint1_nullspace_stiffness)
    tau_nullspace = kp_kv_joint[:, 0] * q_error + kp_kv_joint[:, 1] * dq_error

    tau = tau_task + tau_nullspace
    if gravity_comp:
        tau += C

    tau_d = saturate_torque_rate(tau, data.qfrc_actuator[dof_ids], delta_tau_max)
    return tau_d