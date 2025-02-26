from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import wrap_to_pi
from isaaclab.assets import Articulation, RigidObject


if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def joint_pos_target_l2(env: ManagerBasedRLEnv, target: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize joint position deviation from a target value."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # wrap the joint positions to (-pi, pi)
    joint_pos = wrap_to_pi(asset.data.joint_pos[:, asset_cfg.joint_ids])
    # compute the reward
    return torch.sum(torch.square(joint_pos - target), dim=1)

def proximity_to_point_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    
    
    target =  torch.tensor(env.scene.env_origins + torch.tensor([0.0, -1.0, 0.0], device=env.device))
    # TODO: Remove print
    # print('origins', env.scene.env_origins)
    # print('goal', target)

    current_pos = asset.data.root_pos_w  # (x, y, z) world position
    # print('current_pos', current_pos)
    distance = torch.sum(torch.square(current_pos - target), dim=1)
    # print('distance', distance)
    # Compute L2 squared distance to the target point
    return 2 - distance

def orientation(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Terminate when the asset's orientation is too far from the desired orientation limits.

    This is computed by checking the angle between the projected gravity vector and the z-axis.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    orientation_rew = torch.acos(-asset.data.projected_gravity_b[:, 2]).abs()
    # print(orientation_rew)
    return orientation_rew

