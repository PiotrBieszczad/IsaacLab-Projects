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

    joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    x, y = joint_pos[:, 0], joint_pos[:, 1]

    # Compute triangle limits
    upper_y_limit = torch.minimum(x + distance, -x + distance)

    # Check violations
    out_of_x_limit = x > distance
    out_of_y_limit = (y < 0) | (y > upper_y_limit)

    return torch.logical_or(out_of_x_limit, out_of_y_limit)

