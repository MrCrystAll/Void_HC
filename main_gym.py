import time
from typing import Any, Dict, Hashable, List

import numpy as np
from rlgym.api import RLGym, ActionParser

from rlgym.rocket_league.api import GameState
from rlgym.rocket_league.obs_builders import DefaultObs
from rlgym.rocket_league.action_parsers import RepeatAction
from rlgym.rocket_league.state_mutators import FixedTeamSizeMutator, MutatorSequence, KickoffMutator
from rlgym.rocket_league.reward_functions import TouchReward
from rlgym.rocket_league.done_conditions import GoalCondition
from rlgym.rocket_league.sim import RocketSimEngine

from rlgym_tools.rocket_league.renderers.rocketsimvis_renderer import RocketSimVisRenderer

from hc_gym_bot import GymActionReturn, HCGymBot

class NoopActionParser(ActionParser[Hashable, GymActionReturn, np.ndarray, GameState, tuple[str, int]]):
    def parse_actions(self, actions: Dict[Hashable, GymActionReturn], state: GameState, shared_info: Dict[str, Any]) -> Dict[Hashable, np.ndarray]:
        acts = {agent: np.asarray([action.to_list()]) for agent, action in actions.items()}
        # print(acts)
        return acts
    
    def reset(self, agents: List[Hashable], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        pass

if __name__ == "__main__":
    env = RLGym(
        state_mutator=MutatorSequence(
            FixedTeamSizeMutator(4, 4),
            KickoffMutator()
        ),
        action_parser=RepeatAction(NoopActionParser(), 8),
        obs_builder=DefaultObs(),
        renderer=RocketSimVisRenderer(),
        reward_fn=TouchReward(),
        termination_cond=GoalCondition(),
        truncation_cond=GoalCondition(),
        transition_engine=RocketSimEngine()
    )

    running = True
    
    bot = HCGymBot()
    
    print("Running env")
    while running:
        try:
            env.reset()
            bot.reset(env.agents, env.state, env.shared_info)
            
            truncated = {agent: False for agent in env.agents}
            terminated = {agent: False for agent in env.agents}
            
            while not (any(truncated.values()) or any(terminated.values())):
                env.render()
                time.sleep(8.0/120.0)
                
                actions = bot.get_output(env.agents, env.state, env.shared_info)
                
                _, _, terminated, truncated = env.step(actions)
        except KeyboardInterrupt:
            print("Stopping")
            env.close()
            running = False
            break