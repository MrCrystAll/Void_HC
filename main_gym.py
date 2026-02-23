from collections.abc import Hashable
import time
from typing import Any, Dict, List

import numpy as np
from rlgym.api import RLGym, ActionParser

from rlgym.rocket_league.api import GameState
from rlgym.rocket_league.obs_builders import DefaultObs
from rlgym.rocket_league.action_parsers import RepeatAction
from rlgym.rocket_league.state_mutators import FixedTeamSizeMutator, MutatorSequence
from rlgym.rocket_league.reward_functions import TouchReward
from rlgym.rocket_league.done_conditions import GoalCondition, TimeoutCondition
from rlgym.rocket_league.sim import RocketSimEngine

from rlgym_tools.rocket_league.renderers.rocketsimvis_renderer import (
    RocketSimVisRenderer,
)
from rlgym_tools.rocket_league.state_mutators.random_physics_mutator import (
    RandomPhysicsMutator,
)

from hc_gym_bot import GymActionReturn, HCGymBot


class NoopActionParser(
    ActionParser[Hashable, GymActionReturn, np.ndarray, GameState, tuple[str, int]]
):
    def parse_actions(
        self,
        actions: Dict[Hashable, GymActionReturn],
        state: GameState,
        shared_info: Dict[str, Any],
    ) -> Dict[Hashable, np.ndarray]:
        acts = {
            agent: np.asarray([action.to_list()]) for agent, action in actions.items()
        }
        # print(acts)
        return acts

    def reset(
        self,
        agents: List[Hashable],
        initial_state: GameState,
        shared_info: Dict[str, Any],
    ) -> None:
        pass


class HCBotActionParser(
    ActionParser[Hashable, np.ndarray, np.ndarray, GameState, tuple[str, int]]
):
    GO_TO_BALL_IDX: int = 90

    def __init__(self) -> None:
        self.hc_bot = HCGymBot()
        self._lookup_tables = {}

    def get_action_space(self, agent: Hashable) -> tuple[str, int]:
        return "discrete", 91

    def _make_lookup_tables(self, agents: list[Hashable]) -> dict[Hashable, np.ndarray]:
        returns = {}

        for agent in agents:
            actions = []
            # Ground
            for throttle in (-1, 0, 1):
                for steer in (-1, 0, 1):
                    for boost in (0, 1):
                        for handbrake in (0, 1):
                            if boost == 1 and throttle != 1:
                                continue
                            actions.append(
                                [
                                    throttle or boost,
                                    steer,
                                    0,
                                    steer,
                                    0,
                                    0,
                                    boost,
                                    handbrake,
                                ]
                            )
            # Aerial
            for pitch in (-1, 0, 1):
                for yaw in (-1, 0, 1):
                    for roll in (-1, 0, 1):
                        for jump in (0, 1):
                            for boost in (0, 1):
                                if (
                                    jump == 1 and yaw != 0
                                ):  # Only need roll for sideflip
                                    continue
                                if pitch == roll == jump == 0:  # Duplicate with ground
                                    continue
                                # Enable handbrake for potential wavedashes
                                handbrake = jump == 1 and (
                                    pitch != 0 or yaw != 0 or roll != 0
                                )
                                actions.append(
                                    [
                                        boost,
                                        yaw,
                                        pitch,
                                        yaw,
                                        roll,
                                        jump,
                                        boost,
                                        handbrake,
                                    ]
                                )

            # Placeholder for go to ball
            actions.append([0] * 8)

            returns[agent] = np.asarray(actions, dtype=float)

        return returns

    def parse_actions(
        self,
        actions: Dict[Hashable, np.ndarray],
        state: GameState,
        shared_info: Dict[str, Any],
    ) -> Dict[Hashable, np.ndarray]:
        results = self.hc_bot.get_output(list(actions.keys()), state, shared_info)

        returns = {}

        for agent, action in actions.items():
            # Action can have shape (Ticks, 1) or (Ticks)
            assert len(action.shape) == 1 or (
                len(action.shape) == 2 and action.shape[1] == 1
            )

            if len(action.shape) == 2:
                action = action.squeeze(1)

            self._lookup_tables[agent][self.GO_TO_BALL_IDX] = results[agent].to_list()
            returns[agent] = self._lookup_tables[agent][action]

        return returns

    def reset(
        self,
        agents: List[Hashable],
        initial_state: GameState,
        shared_info: Dict[str, Any],
    ) -> None:
        self.hc_bot.reset(agents, initial_state, shared_info)
        self._lookup_tables = self._make_lookup_tables(agents)


if __name__ == "__main__":
    env = RLGym(
        state_mutator=MutatorSequence(
            FixedTeamSizeMutator(1, 0), RandomPhysicsMutator()
        ),
        action_parser=RepeatAction(HCBotActionParser(), 8),
        obs_builder=DefaultObs(),
        renderer=RocketSimVisRenderer(),
        reward_fn=TouchReward(),
        termination_cond=GoalCondition(),
        truncation_cond=TimeoutCondition(2),
        transition_engine=RocketSimEngine(),
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
                time.sleep(8.0 / 120.0)

                actions = {agent: np.asarray([90]) for agent in env.agents}

                _, _, terminated, truncated = env.step(actions)
        except KeyboardInterrupt:
            print("Stopping")
            running = False
            break

    env.close()
