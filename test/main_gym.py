import time

from rlgym.rocket_league.obs_builders import DefaultObs
from rlgym.rocket_league.state_mutators import FixedTeamSizeMutator, MutatorSequence, KickoffMutator
from rlgym.rocket_league.reward_functions import TouchReward
from rlgym.rocket_league.done_conditions import GoalCondition, TimeoutCondition
from rlgym.rocket_league.sim import RocketSimEngine

from rlgym_tools.rocket_league.renderers.rocketsimvis_renderer import (
    RocketSimVisRenderer,
)
from rlgym.api import RLGym

from common.action_parser import HCBotEnhancedActionParser
from common.target_shared_info_provider import TargetSharedInfoProvider


if __name__ == "__main__":
    tick_skip = 16
    act_parser = HCBotEnhancedActionParser(tick_skip)
    
    env = RLGym(
        state_mutator=MutatorSequence(
            FixedTeamSizeMutator(1, 1), KickoffMutator()
        ),
        action_parser=act_parser,
        obs_builder=DefaultObs(),
        renderer=RocketSimVisRenderer(),
        reward_fn=TouchReward(),
        termination_cond=GoalCondition(),
        truncation_cond=TimeoutCondition(10),
        transition_engine=RocketSimEngine(),
        shared_info_provider=TargetSharedInfoProvider()
    )

    running = True
    
    actions = iter([2, 2, 2, 2, 2, 1] * 100)

    print("Running env")
    while running:
        try:
            env.reset()

            truncated = {agent: False for agent in env.agents}
            terminated = {agent: False for agent in env.agents}

            while not (any(truncated.values()) or any(terminated.values())):
                env.render()
                time.sleep(tick_skip/120.0)
                
                # agent_actions = act_parser.sample(env.agents)                
                agent_actions = act_parser.pick(env.agents, next(actions))

                _, _, terminated, truncated = env.step(agent_actions)
        except KeyboardInterrupt:
            print("Stopping")
            running = False
            break

    env.close()
