# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Robot-neutral event functions shared by the cube-stacking tasks.

These were previously defined in ``franka_stack_events.py``; they are robot-agnostic and are used
by every stack robot config, so they live here and ``franka_stack_events`` re-exports them for
backward compatibility.
"""

from __future__ import annotations

import math
import random
from typing import TYPE_CHECKING

import torch
import warp as wp

import isaaclab.utils.math as math_utils
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedEnv

def randomize_joint_by_gaussian_offset(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    mean: float,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Add Gaussian noise to the (non-gripper) joints of an asset on reset.

    The gripper joints are restored to their default pose rather than noised. They are resolved
    generically from ``env.cfg.gripper_joint_names`` (the same names the stack observations and
    terminations already use) instead of assuming the last two joints are the gripper -- the latter
    is Franka-specific and wrong for, e.g., the single-jaw SO-101 gripper or a UR10 surface gripper.

    Behavior of ``env.cfg.gripper_joint_names``:

    * a non-empty list -> those joints are held at their default pose (everything else is noised);
    * an empty list -> no gripper joints to hold, so every joint is noised (e.g. surface grippers);
    * unset / ``None`` -> backward-compatible fallback that holds the last two joints.
    """
    asset: Articulation = env.scene[asset_cfg.name]

    # Add gaussian noise to joint states
    joint_pos = asset.data.default_joint_pos.torch[env_ids].clone()
    joint_vel = asset.data.default_joint_vel.torch[env_ids].clone()
    joint_pos += math_utils.sample_gaussian(mean, std, joint_pos.shape, joint_pos.device)

    # Clamp joint pos to limits
    joint_pos_limits = asset.data.soft_joint_pos_limits.torch[env_ids]
    joint_pos = joint_pos.clamp_(joint_pos_limits[..., 0], joint_pos_limits[..., 1])

    # Don't noise the gripper joints (resolved generically; see the docstring).
    gripper_joint_names = getattr(env.cfg, "gripper_joint_names", None)
    if gripper_joint_names is None:
        # Backward-compatible fallback for callers that do not configure gripper joint names.
        joint_pos[:, -2:] = asset.data.default_joint_pos.torch[env_ids, -2:]
    elif gripper_joint_names:
        gripper_ids, _ = asset.find_joints(gripper_joint_names)
        joint_pos[:, gripper_ids] = asset.data.default_joint_pos.torch[env_ids][:, gripper_ids]

    # Set into the physics simulation
    asset.set_joint_position_target_index(target=joint_pos, env_ids=env_ids)
    asset.set_joint_velocity_target_index(target=joint_vel, env_ids=env_ids)
    asset.write_joint_position_to_sim_index(position=joint_pos, env_ids=env_ids)
    asset.write_joint_velocity_to_sim_index(velocity=joint_vel, env_ids=env_ids)

def sample_object_poses(
    num_objects: int,
    min_separation: float = 0.0,
    pose_range: dict[str, tuple[float, float]] = {},
    max_sample_tries: int = 5000,
):
    range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    pose_list = []

    for i in range(num_objects):
        for j in range(max_sample_tries):
            sample = [random.uniform(range[0], range[1]) for range in range_list]

            # Accept pose if it is the first one, or if reached max num tries
            if len(pose_list) == 0 or j == max_sample_tries - 1:
                pose_list.append(sample)
                break

            # Check if pose of object is sufficiently far away from all other objects
            separation_check = [math.dist(sample[:3], pose[:3]) > min_separation for pose in pose_list]
            if False not in separation_check:
                pose_list.append(sample)
                break

    return pose_list

def randomize_object_pose(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfgs: list[SceneEntityCfg],
    min_separation: float = 0.0,
    pose_range: dict[str, tuple[float, float]] = {},
    max_sample_tries: int = 5000,
):
    if env_ids is None:
        return

    # Randomize poses in each environment independently
    for cur_env in env_ids.tolist():
        pose_list = sample_object_poses(
            num_objects=len(asset_cfgs),
            min_separation=min_separation,
            pose_range=pose_range,
            max_sample_tries=max_sample_tries,
        )

        # Randomize pose for each object
        for i in range(len(asset_cfgs)):
            asset_cfg = asset_cfgs[i]
            asset = env.scene[asset_cfg.name]

            # Write pose to simulation
            pose_tensor = torch.tensor([pose_list[i]], device=env.device)
            positions = pose_tensor[:, 0:3] + env.scene.env_origins[cur_env, 0:3]
            orientations = math_utils.quat_from_euler_xyz(pose_tensor[:, 3], pose_tensor[:, 4], pose_tensor[:, 5])
            asset.write_root_pose_to_sim_index(
                root_pose=torch.cat([positions, orientations], dim=-1),
                env_ids=torch.tensor([cur_env], device=env.device),
            )
            asset.write_root_velocity_to_sim_index(
                root_velocity=torch.zeros(1, 6, device=env.device), env_ids=torch.tensor([cur_env], device=env.device)
            )


def rand_tower_pos_and_cube_3(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    cube_1_cfg: SceneEntityCfg = SceneEntityCfg("cube_1"),
    cube_2_cfg: SceneEntityCfg = SceneEntityCfg("cube_2"),
    cube_3_cfg: SceneEntityCfg = SceneEntityCfg("cube_3"),
    min_separation: float = 0.14,
    pose_range: dict[str, tuple[float, float]] = {},
    max_sample_tries: int = 5000,
    cube_size : float  = 0.05,
    stack_gap : float = 0.001,
    xy_rand: float = 0.01,
    yaw_rand: float = 0.5,

): 
    if env_ids is None:
        return
    
    for cur_env in env_ids.tolist():
        pose_list = sample_object_poses(
            num_objects=2,
            min_separation=min_separation,
            pose_range=pose_range,
            max_sample_tries=max_sample_tries,
        )
    
        cube_1_pose = pose_list[0]
        cube_2_pose = [
            cube_1_pose[0] + random.uniform(-xy_rand,xy_rand),
            cube_1_pose[1] + random.uniform(-xy_rand,xy_rand),
            cube_1_pose[2] + cube_size + stack_gap, 
            cube_1_pose[3],
            cube_1_pose[4],
            cube_1_pose[5] + random.uniform(-yaw_rand,yaw_rand),
            ]
        cube_3_pose = pose_list[1]

        for cfg, pose in ((cube_1_cfg,cube_1_pose,),(cube_2_cfg,cube_2_pose),(cube_3_cfg,cube_3_pose)):
            asset = env.scene[cfg.name]
            pose_tensor = torch.tensor([pose], device=env.device)
            positions = pose_tensor[:, 0:3] + env.scene.env_origins[cur_env, 0:3]
            orientations = math_utils.quat_from_euler_xyz(pose_tensor[:, 3], pose_tensor[:, 4], pose_tensor[:, 5])
            asset.write_root_pose_to_sim_index(
                root_pose=torch.cat([positions, orientations], dim=-1),
                env_ids=torch.tensor([cur_env], device=env.device),
            )
            asset.write_root_velocity_to_sim_index(
                root_velocity=torch.zeros(1, 6, device=env.device), env_ids=torch.tensor([cur_env], device=env.device)
            )