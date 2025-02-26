import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
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
import gym
import isaaclab_tasks 

import isaaclab_tasks.manager_based.classic.jet_bot.mdp as mdp
from isaaclab_tasks.utils import parse_env_cfg


##
# Pre-defined configs
##
from isaaclab_assets.robots.jet_bot import JETBOT_CFG


##
# Scene definition
##


@configclass
class JetBotSceneCfg(InteractiveSceneCfg):
    """Configuration for a cart-pole scene."""

    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)),
    )

    # cartpole
    robot: ArticulationCfg = JETBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # Spawn mesh cuboid
    goal_cone = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/GoalCone",
        spawn=sim_utils.MeshConeCfg(radius=0.1, height=0.2),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(1.0, 0.0, 0.0)),
    )

    # Spawn mesh cuboid
    bounds = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/RoobotBounds",
        spawn=sim_utils.MeshCuboidCfg(size=(1, 1, 0.001)),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.5, 0, 0.001), rot=( 0.3826834, 0, 0, 0.9238795 )),
    )

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
        base_pos = ObsTerm(mdp.root_pos_w)
        base_orient = ObsTerm(mdp.root_quat_w)
        goal_pos = ObsTerm(mdp.goal_pos)

        # TODO: Adjust the velocities
        # base_lin_vel = ObsTerm(mdp.base_lin_vel)
        # base_ang_vel = ObsTerm(mdp.base_ang_vel)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""
    # reset
    reset_jet_bot_position = EventTerm(
        func=mdp.reset_scene_to_default,
        mode="reset",
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""
    # (1) Constant running reward
    # alive = RewTerm(func=mdp.is_alive, weight=0.1)
    # (2) Failure penalty
    terminating = RewTerm(func=mdp.is_terminated, weight=-2.0)
    # (3) Primary task: proximity to goal
    goal_proximity = RewTerm(
        func=mdp.proximity_to_point_l2,
        weight=1.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    # TODO: Add this for faster learning
    # (4) Shaping tasks: angle to goal
    # angle_to_goal = RewTerm(
    #     func=mdp.orientation,
    #     weight=-0.1
    # )
    # TODO: Add this for smoothness
    # (5) Shaping tasks: change in velocity
    # cart_vel = RewTerm(
    #     func=mdp.action_rate_l2,
    #     weight=1.0,
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
    # time_out = DoneTerm(func=mdp.time_out_penalised, time_out=True)
    # (2) JetBot out of bounds
    jet_bot_out_of_bounds = DoneTerm(
        func=mdp.joint_pos_out_of_triangle_limit,
        params={"asset_cfg": SceneEntityCfg("robot"), "distance": 1.0},
    )
    # (3) JetBot orientation out of bounds
    angle_out_of_bounds = DoneTerm(
        func=mdp.bad_orientation,
        params={"limit_angle": math.radians(90), 
                "asset_cfg": SceneEntityCfg("robot")
        },
    )


##
# Environment configuration
##


@configclass
class JetBotEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the cartpole environment."""
    # Scene settings
    scene: JetBotSceneCfg = JetBotSceneCfg(num_envs=4096, env_spacing=1.5)
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
        self.episode_length_s = 10
        # viewer settings
        self.viewer.eye = (8.0, 0.0, 5.0)
        # simulation settings
        self.sim.dt = 1 / 120
        self.sim.render_interval = self.decimation