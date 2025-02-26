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
    x = asset.data.root_pos_w[:, 0] - env.scene.env_origins[:, 0]
    y = asset.data.root_pos_w[:, 1] - env.scene.env_origins[:, 1]
    # Compute triangle limits
    upper_y_limit = torch.minimum(x + distance, -x + distance)
    # print("x:", x, "upper_limit_x:" distance, "y:", y, "upper_y_limit:", upper_y_limit)
    # Check violations
    out_of_x_limit = y > distance
    out_of_y_limit = (x < -0.2) | (x > upper_y_limit + 0.2)
    terminate = torch.logical_or(out_of_x_limit, out_of_y_limit)
    # if terminate.any():
    #     print("x:", x, "upper_limit_x:", distance, "y:", y, "upper_y_limit:", upper_y_limit)
    return terminate

def bad_orientation(
    env: ManagerBasedRLEnv, limit_angle: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Terminate when the asset's orientation is too far from the desired orientation limits.

    This is computed by checking the angle between the projected gravity vector and the z-axis.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    heading = asset.data.heading_w_rev.abs()
    heading_adjusted = (heading).abs()
    terminate = heading > limit_angle
    # if terminate.any():
    #     print('fail')
    #     print("angle:", heading_adjusted, "limit_angle:", limit_angle)
    return terminate
