import time

from rlgym.rocket_league.obs_builders import DefaultObs
from rlgym.rocket_league.action_parsers import RepeatAction
from rlgym.rocket_league.state_mutators import FixedTeamSizeMutator, MutatorSequence
from rlgym.rocket_league.reward_functions import TouchReward
from rlgym.rocket_league.done_conditions import GoalCondition, TimeoutCondition
from rlgym.rocket_league.sim import RocketSimEngine

from rlgym.rocket_league.common_values import JUMP, BOOST, YAW, PITCH, STEER

from rlgym_tools.rocket_league.renderers.rocketsimvis_renderer import (
    RocketSimVisRenderer,
)
from rlgym_tools.rocket_league.state_mutators.random_physics_mutator import (
    RandomPhysicsMutator,
)

from rlgym.api import RLGym

from common.action_parser import HCBotAction, HCBotActionParser


if __name__ == "__main__":
    act_parser = RepeatAction(HCBotActionParser(), 8)
    
    env = RLGym(
        state_mutator=MutatorSequence(
            FixedTeamSizeMutator(1, 1), RandomPhysicsMutator()
        ),
        action_parser=act_parser,
        obs_builder=DefaultObs(),
        renderer=RocketSimVisRenderer(),
        reward_fn=TouchReward(),
        termination_cond=GoalCondition(),
        truncation_cond=TimeoutCondition(10),
        transition_engine=RocketSimEngine(),
    )

    running = True

    print("Running env")
    while running:
        try:
            env.reset()

            truncated = {agent: False for agent in env.agents}
            terminated = {agent: False for agent in env.agents}

            while not (any(truncated.values()) or any(terminated.values())):
                env.render()
                time.sleep(8.0 / 120.0)

                # Change to multi discrete ?
                # bins = [2, 2, 2]
                # 2 for go to ball and go away from ball
                # 2 for jump (0/1)
                # 2 for boost (0/1)

                # agent_actions = act_parser.parser.pick(env.agents, HCBotAction.FLIP_TO_BALL_IDX)
                agent_actions = act_parser.parser.pick(env.agents, HCBotAction.GO_TO_BALL_SPEED_SCALED)

                _, _, terminated, truncated = env.step(agent_actions)
        except KeyboardInterrupt:
            print("Stopping")
            running = False
            break

    env.close()
