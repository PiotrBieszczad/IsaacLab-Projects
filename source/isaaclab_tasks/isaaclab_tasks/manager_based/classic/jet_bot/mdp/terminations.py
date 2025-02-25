# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to activate certain terminations.

The functions can be passed to the :class:`isaaclab.managers.TerminationTermCfg` object to enable
the termination introduced by the function.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers.command_manager import CommandTerm

"""
MDP terminations.
"""


def joint_pos_out_of_triangle_limit(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), distance: float = 3.0
) -> torch.Tensor:
    """Terminate when (x, y) joint positions are outside the triangular region.

    Constraints:
    - x <= 3
    - 0 <= y <= min(x + 5, -x + 5)
    """
    asset: Articulation = env.scene[asset_cfg.name]
    if asset_cfg.joint_ids is None:
        raise ValueError("Joint not defined.")
    x, y = asset.data.root_pos_w[0, 0], asset.data.root_pos_w[0, 1]

    # Compute triangle limits
    upper_y_limit = torch.minimum(x + distance, -x + distance)
    # print("x:", x, "upper_limit_x:" distance, "y:", y, "upper_y_limit:", upper_y_limit)
    # Check violations
    out_of_x_limit = x > distance
    out_of_y_limit = (y < -0.5) | (y > upper_y_limit + 0.2)
    return -1 if torch.logical_or(out_of_x_limit, out_of_y_limit) else 0


def time_out_penalised(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Terminate the episode when the episode length exceeds the maximum episode length."""
    return -1 if env.episode_length_buf >= env.max_episode_length else 0

def bad_orientation_penalised(
    env: ManagerBasedRLEnv, limit_angle: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Terminate when the asset's orientation is too far from the desired orientation limits.

    This is computed by checking the angle between the projected gravity vector and the z-axis.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return -1 if torch.acos(-asset.data.projected_gravity_b[:, 2]).abs() > limit_angle else 0
