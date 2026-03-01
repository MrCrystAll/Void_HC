import os

# needed to prevent numpy from using a ton of memory in env processes and causing them to throttle each other
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["PYANY_SERDE_UNPICKLE_WITHOUT_PROMPT"] = "1"

os.environ["WANDB_API_KEY"] = os.environ.get("WANDB_DEV_API_KEY", "")
os.environ["WANDB_BASE_URL"] = "https://api.wandb.ai"

RENDER_MODE_KEY = "RENDER_MODE"
VERSION = "v1.0.1-Hybrid"
CONTROLLER_NAME = "Void_Hybrid"


def get_render_mode() -> bool:
    return os.getenv(RENDER_MODE_KEY, "False").lower() in ("true", "1", "t")


def pnw(value: float, positive_weight: float = 1, negative_weight: float = 1):
    """
    Multiplies a positive value by the positive weight and a negative value by the negative weight
    :param value: The value to multiply
    :param positive_weight: The weight for positive values
    :param negative_weight: The weight for negative values
    :return: The multiplied value
    """
    if value > 0:
        return value * positive_weight

    if value < 0:
        return value * negative_weight

    return value


def build_rlgym_v2_env():
    import numpy as np
    from rlgym.api import RLGym
    from rlgym.rocket_league import void_hc_values
    from rlgym.rocket_league.action_parsers import LookupTableAction, RepeatAction
    from rlgym.rocket_league.done_conditions import (
        AnyCondition,
        GoalCondition,
        NoTouchTimeoutCondition,
        TimeoutCondition,
    )
    from rlgym.rocket_league.obs_builders import DefaultObs
    from rlgym.rocket_league.reward_functions import (
        CombinedReward,
        GoalReward,
        TouchReward,
    )
    from rlgym.rocket_league.sim import RocketSimEngine
    from rlgym.rocket_league.state_mutators import (
        FixedTeamSizeMutator,
        KickoffMutator,
        MutatorSequence,
    )

    from void_logging.api.rewards import RewardLogger, LoggedCombinedReward
    from void_logging.api.wrappers import ChainWrapper
    from rlgym_tools.rocket_league.reward_functions.velocity_player_to_ball_reward import (
        VelocityPlayerToBallReward,
    )
    from rlgym_tools.rocket_league.shared_info_providers.multi_provider import (
        MultiProvider,
    )
    from rlgym_tools.rocket_league.reward_functions.goal_prob_reward import (
        GoalViewReward,
    )
    from rlgym_tools.rocket_league.reward_functions.advanced_touch_reward import (
        AdvancedTouchReward,
    )

    from void_hc.action_parser import HCBotActionParser

    spawn_opponents = True
    team_size = 1
    blue_team_size = team_size
    orange_team_size = team_size if spawn_opponents else 0
    action_repeat = 8
    no_touch_timeout_seconds = 15
    game_timeout_seconds = 150

    action_parser = RepeatAction(HCBotActionParser(), repeats=action_repeat)
    termination_condition = GoalCondition()
    truncation_condition = AnyCondition(
        NoTouchTimeoutCondition(timeout_seconds=no_touch_timeout_seconds),
        TimeoutCondition(timeout_seconds=game_timeout_seconds),
    )

    reward_fn = RewardLogger(
        LoggedCombinedReward(
            ChainWrapper(VelocityPlayerToBallReward())
            .to_logged()
            .apply_operation(lambda val: pnw(val, 1, 0.2))
            .weight(5.0),
            ChainWrapper(AdvancedTouchReward(acceleration_reward=1))
            .to_logged()
            .weight(10.0),
            ChainWrapper(GoalReward()).to_logged().weight(100.0),
            ChainWrapper(GoalViewReward()).to_logged().weight(5.0),
        )
    )

    from void_logging.rlgym_learn.reward_shared_info_provider import (
        RewardSharedInfoProvider,
    )
    from void_logging.rocket_league.player_metric_providers import (
        PlayerOnGroundRatioMetricSharedInfoProvider,
        PlayerTouchMetricSharedInfoProvider,
        PlayerVelocityMetricSharedInfoProvider,
        PlayerBallHitForceMetricSharedInfoProvider,
        PlayerHeightMetricSharedInfoProvider,
    )
    from void_logging.rocket_league.state_metric_providers import (
        GoalMetricSharedInfoProvider,
        GoalScoreSpeedSharedInfoProvider,
    )
    from void_logging.rocket_league.ball_metric_providers import (
        BallHeightMetricSharedInfoProvider,
        BallAccelerationMetricSharedInfoProvider,
        BallVelocityMetricSharedInfoProvider,
    )
    from rlgym_tools.rocket_league.renderers.rocketsimvis_renderer import (
        RocketSimVisRenderer,
    )

    obs_builder = DefaultObs(
        zero_padding=team_size,
        pos_coef=np.asarray(
            [
                1 / void_hc_values.SIDE_WALL_X,
                1 / void_hc_values.BACK_NET_Y,
                1 / void_hc_values.CEILING_Z,
            ]
        ),
        ang_coef=1 / np.pi,
        lin_vel_coef=1 / void_hc_values.CAR_MAX_SPEED,
        ang_vel_coef=1 / void_hc_values.CAR_MAX_ANG_VEL,
        boost_coef=1 / 100.0,
    )

    state_mutator = MutatorSequence(
        FixedTeamSizeMutator(blue_size=blue_team_size, orange_size=orange_team_size),
        KickoffMutator(),
    )
    return RLGym(
        state_mutator=state_mutator,
        obs_builder=obs_builder,
        action_parser=action_parser,
        reward_fn=reward_fn,
        termination_cond=termination_condition,
        truncation_cond=truncation_condition,
        transition_engine=RocketSimEngine(),
        shared_info_provider=MultiProvider(
            RewardSharedInfoProvider(),
            BallHeightMetricSharedInfoProvider(),
            BallVelocityMetricSharedInfoProvider(),
            BallAccelerationMetricSharedInfoProvider(),
            PlayerBallHitForceMetricSharedInfoProvider(),
            PlayerHeightMetricSharedInfoProvider(),
            PlayerOnGroundRatioMetricSharedInfoProvider(),
            PlayerTouchMetricSharedInfoProvider(),
            PlayerVelocityMetricSharedInfoProvider(),
            GoalMetricSharedInfoProvider(),
            GoalScoreSpeedSharedInfoProvider(),
        ),
        renderer=RocketSimVisRenderer(),
    )


def get_latest_checkpoint_to_load(controller_name: str, version: str) -> str | None:
    import os.path

    SAVE_FOLDER = "agent_controllers_checkpoints/Void"

    import glob

    load_folder = os.path.join(SAVE_FOLDER, controller_name, version)

    list_of_files = glob.glob(load_folder + "/*")

    checkpoint_to_load = (
        max(list_of_files, key=os.path.getctime) if len(list_of_files) > 0 else None
    )

    return checkpoint_to_load


if __name__ == "__main__":
    from argparse import ArgumentParser

    arg_parser = ArgumentParser(prog="Void", description="Void's training script")

    arg_parser.add_argument(
        "--render", action="store_true", help="Activate render mode"
    )

    args = arg_parser.parse_args()

    os.environ[RENDER_MODE_KEY] = str(args.render)

    from typing import Tuple

    import numpy as np
    from rlgym_learn_algos.logging import (
        WandbMetricsLogger,
        WandbMetricsLoggerConfigModel,
    )
    from rlgym_learn_algos.ppo import (
        BasicCritic,
        DiscreteFF,
        ExperienceBufferConfigModel,
        GAETrajectoryProcessor,
        GAETrajectoryProcessorConfigModel,
        NumpyExperienceBuffer,
        PPOAgentController,
        PPOAgentControllerConfigModel,
        PPOLearnerConfigModel,
        PPOMetricsLogger,
    )

    from rlgym_learn import (
        BaseConfigModel,
        LearningCoordinator,
        LearningCoordinatorConfigModel,
        NumpySerdeConfig,
        ProcessConfigModel,
        PyAnySerdeType,
        SerdeTypesModel,
        generate_config,
    )
    from rlgym_learn.rocket_league import GameStatePythonSerde
    from void_logging.logging_utils import REWARDS_HEADER, METRICS_HEADER
    from void_logging.rlgym_learn.multi_logger import MultiLogger
    from void_logging.rlgym_learn.metric_logger import (
        custom_metrics_serde,
        CustomMetricLogger,
    )
    from void_logging.rlgym_learn.reward_metrics_logger import (
        reward_metric_logger_serde,
        RewardMetricsLogger,
    )

    # The obs_space_type and action_space_type are determined by your choice of ObsBuilder and ActionParser respectively.
    # The logic used here assumes you are using the types defined by the DefaultObs and LookupTableAction above.
    DefaultObsSpaceType = Tuple[str, int]
    DefaultActionSpaceType = Tuple[str, int]

    def actor_factory(
        obs_space: DefaultObsSpaceType,
        action_space: DefaultActionSpaceType,
        device: str,
    ):
        return DiscreteFF(obs_space[1], action_space[1], (256, 256, 256), device)

    def critic_factory(obs_space: DefaultObsSpaceType, device: str):
        return BasicCritic(obs_space[1], (256, 256, 256), device)

    if get_render_mode():
        process_config = ProcessConfigModel(
            n_proc=1,
            render=True,
            render_delay=8.0 / 120.0,
            # Number of processes to spawn to run environments. Increasing will use more RAM but should increase steps per second, up to a point
        )
    else:
        process_config = ProcessConfigModel(
            n_proc=64,
            # Number of processes to spawn to run environments. Increasing will use more RAM but should increase steps per second, up to a point
        )

    # Create the config that will be used for the run
    config = LearningCoordinatorConfigModel(
        base_config=BaseConfigModel(
            serde_types=SerdeTypesModel(
                agent_id_serde_type=PyAnySerdeType.STRING(),
                action_serde_type=PyAnySerdeType.NUMPY(np.int64),
                obs_serde_type=PyAnySerdeType.NUMPY(np.float64),
                reward_serde_type=PyAnySerdeType.FLOAT(),
                obs_space_serde_type=PyAnySerdeType.TUPLE(
                    (PyAnySerdeType.STRING(), PyAnySerdeType.INT())
                ),
                action_space_serde_type=PyAnySerdeType.TUPLE(
                    (PyAnySerdeType.STRING(), PyAnySerdeType.INT())
                ),
                shared_info_serde_type=PyAnySerdeType.TYPEDDICT(
                    {
                        REWARDS_HEADER: reward_metric_logger_serde,
                        METRICS_HEADER: custom_metrics_serde,
                    }
                ),
            ),
            timestep_limit=1_000_000_000,  # Train for 1B steps
        ),
        process_config=process_config,
        agent_controllers_config={
            CONTROLLER_NAME: PPOAgentControllerConfigModel(
                timesteps_per_iteration=2**16,
                run_name=VERSION,
                learner_config=PPOLearnerConfigModel(
                    ent_coef=0.01,  # Sets the entropy coefficient used in the PPO algorithm
                    actor_lr=2e-3,  # Sets the learning rate of the actor model
                    critic_lr=2e-3,  # Sets the learning rate of the critic model
                    batch_size=2**16,
                    n_minibatches=10,
                    n_epochs=10,
                ),
                experience_buffer_config=ExperienceBufferConfigModel(
                    max_size=2
                    ** 16,  # Sets the number of timesteps to store in the experience buffer. Old timesteps will be pruned to only store the most recently obtained timesteps.
                    trajectory_processor_config=GAETrajectoryProcessorConfigModel().model_dump(),
                    save_experience_buffer_in_checkpoint=False,
                ),
                metrics_logger_config=WandbMetricsLoggerConfigModel(
                    group="cryy_salt",
                    project="Void",
                    enable=not get_render_mode(),
                    run=VERSION,
                ).model_dump(),
                checkpoint_load_folder=get_latest_checkpoint_to_load(
                    CONTROLLER_NAME, VERSION
                ),
                add_unix_timestamp=False,
                n_checkpoints_to_keep=30,
                save_every_ts=2**18,
            ).model_dump()
        },
        agent_controllers_save_folder="agent_controllers_checkpoints/Void",  # (default value) WARNING: THIS PROCESS MAY DELETE ANYTHING INSIDE THIS FOLDER. This determines the parent folder for the runs for each agent controller. The runs folder for the agent controller will be this folder and then the agent controller config key as a subfolder.
    )

    # Generate the config file for reference (this file location can be
    # passed to the learning coordinator via config_location instead of defining
    # the config object in code and passing that)
    generate_config(
        learning_coordinator_config=config,
        config_location="config.json",
        force_overwrite=True,
    )

    learning_coordinator = LearningCoordinator(
        build_rlgym_v2_env,
        agent_controllers={
            CONTROLLER_NAME: PPOAgentController(
                actor_factory=actor_factory,
                critic_factory=critic_factory,
                experience_buffer=NumpyExperienceBuffer(GAETrajectoryProcessor()),
                metrics_logger=WandbMetricsLogger(
                    MultiLogger(
                        CustomMetricLogger(), RewardMetricsLogger(), PPOMetricsLogger()
                    )
                ),
                obs_standardizer=None,
            )
        },
        config=config,
    )
    learning_coordinator.start()
