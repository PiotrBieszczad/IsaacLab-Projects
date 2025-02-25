import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on creating a cartpole base environment.")
parser.add_argument("--num_envs", type=int, default=16, help="Number of environments to spawn.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app



# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg, ManagerBasedRLEnv, ManagerBasedEnvCfg, ManagerBasedEnv
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
import torch

import isaaclab_tasks.manager_based.classic.jet_bot.mdp as mdp

##
# Pre-defined configs
##
from isaaclab_assets.robots.jet_bot import JETBOT_CFG


##
# Scene definition
##


@configclass
class CartpoleSceneCfg(InteractiveSceneCfg):
    """Configuration for a cart-pole scene."""

    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)),
    )

    # cartpole
    robot: ArticulationCfg = JETBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )


##
# MDP settings
##


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""
    # TODO: Robot wheels
    wheel_velocities = mdp.JointVelocityActionCfg(asset_name="robot", joint_names=["left_wheel_joint", "right_wheel_joint"], scale=100.0)

@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        base_lin_vel = ObsTerm(mdp.base_lin_vel)
        base_ang_vel = ObsTerm(mdp.base_ang_vel)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""
    # TODO: Reset robot properly
    # reset
    reset_jet_bot_position = EventTerm(
        func=mdp.reset_scene_to_default,
        mode="reset"
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""
    # reward scales
    # # TODO: Reward scales
    # rew_scale_alive = 1.0
    # rew_scale_terminated = 1.0
    # rew_scale_robot_velocity = 1.0
    # rew_scale_robot_angle = 1.0 # -0.01
    # rew_scale_robot_proximity = 1.0 # -0.005
    # rew_scale_action_rate = 1.0


    # (1) Constant running reward
    alive = RewTerm(func=mdp.is_alive, weight=1.0)
    # # (2) Failure penalty
    # terminating = RewTerm(func=mdp.is_terminated, weight=rew_scale_terminated)
    # # (3) Primary task: proximity to goal
    # # TODO: Robot centre
    # pole_pos = RewTerm(
    #     func=mdp.proximity_to_point_l2,
    #     weight=rew_scale_robot_proximity,
    #     params={"asset_cfg": SceneEntityCfg("robot", joint_names=[""]), "target": torch.tensor([0.0, 3.0, 0.0])},
    # )
    # # (4) Shaping tasks: change in velocity
    # cart_vel = RewTerm(
    #     func=mdp.action_rate_l2,
    #     weight=rew_scale_action_rate,
    # )
    # # (5) Shaping tasks: lower pole angular velocity
    # pole_vel = RewTerm(
    #     func=mdp.joint_vel_l1,
    #     weight=-0.005,
    #     params={"asset_cfg": SceneEntityCfg("robot", joint_names=["cart_to_pole"])},
    # )

# NOTE: This can be used as an observation to the agent. I.e telling the agent what to do.
# @configclass
# class CommandsCfg:
#     """Command terms for the MDP."""

#     pose_command = mdp.UniformPose2dCommandCfg(
#         asset_name="robot",
#         simple_heading=False,
#         resampling_time_range=(8.0, 8.0),
#         debug_vis=True,
#         ranges=mdp.UniformPose2dCommandCfg.Ranges(pos_x=(2.0, ), pos_y=(-3.0, 3.0), heading=(-math.pi, math.pi)),
#     )

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # (1) Time out
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # (2) Cart out of bounds
    # cart_out_of_bounds = DoneTerm(
    #     # TODO:
    #     func=mdp.joint_pos_out_of_triangle_limit,
    #     params={"asset_cfg": SceneEntityCfg("robot", joint_names=["slider_to_cart"]), "distance": 3.0},
    # )


##
# Environment configuration
##


@configclass
class JetBotEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the cartpole environment."""
    # Scene settings
    scene: CartpoleSceneCfg = CartpoleSceneCfg(num_envs=4096, env_spacing=4.0)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    # # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.episode_length_s = 5
        # viewer settings
        self.viewer.eye = (8.0, 0.0, 5.0)
        # simulation settings
        self.sim.dt = 1 / 120
        self.sim.render_interval = self.decimation


def main():
    """Main function."""
    # create environment configuration
    env_cfg = JetBotEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    # setup RL environment
    env = ManagerBasedRLEnv(cfg=env_cfg)

    # simulate physics
    count = 0
    while simulation_app.is_running():
        with torch.inference_mode():
            # reset
            if count % 300 == 0:
                count = 0
                env.reset()
                print("-" * 80)
                print("[INFO]: Resetting environment...")
            # sample random actions
            joint_efforts = torch.randn_like(env.action_manager.action)
            # step the environment
            obs, rew, terminated, truncated, info = env.step(joint_efforts)
            # print current orientation of pole
            # update counter
            count += 1

    # close the environment
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()