from collections.abc import Hashable
from enum import IntEnum, auto
import random
from typing import Any, Dict, List

import numpy as np
from rlgym.api import ActionParser

from rlgym.rocket_league.api import GameState
from rlgym.rocket_league.common_values import BOOST, JUMP, THROTTLE

from common.hc_gym_bot import HCGymBot

class HCBotAction(IntEnum):
    GO_TO_BALL = 0
    GO_AWAY_FROM_BALL = auto()
    
    GO_TO_BALL_BOOST = auto()
    GO_AWAY_FROM_BALL_BOOST = auto()
    
    GO_TO_BALL_SPEED_SCALED = auto()

    FLIP_TO_BALL = auto()
    
    DO_NOTHING = auto()
    BOOST = auto()
    JUMP = auto()
    SLOW_DOWN = auto()
    ACCELERATE = auto()
    
    N_ACTIONS = auto()

class HCBotActionParser(
    ActionParser[Hashable, np.ndarray, np.ndarray, GameState, tuple[str, int]]
):

    def __init__(self) -> None:
        self.hc_bot = HCGymBot()
        self._lookup_tables = {}
        self._kickoff_sequences = {}
        self._states = {}
        self._last_actions: dict[Hashable, np.ndarray] = {}

    def get_action_space(self, agent: Hashable) -> tuple[str, int]:
        return "discrete", HCBotAction.N_ACTIONS
    
    def _make_lookup_tables(self, agents: list[Hashable]) -> dict[Hashable, np.ndarray]:
        returns = {}

        for agent in agents:
            actions = []

            for _ in range(HCBotAction.N_ACTIONS):
                actions.append([0] * 8)

            returns[agent] = np.asarray(actions, dtype=float)

        return returns

    def parse_actions(
        self,
        actions: Dict[Hashable, np.ndarray],
        state: GameState,
        shared_info: Dict[str, Any],
    ) -> Dict[Hashable, np.ndarray]:
        returns: dict[Hashable, np.ndarray] = {}
        
        self.hc_bot.update_errors(list(actions.keys()), state, shared_info)
        
        _go_to_ball = self.hc_bot.go_to_ball_output(list(actions.keys()), state, shared_info, self._last_actions)
        _go_away_from_ball = self.hc_bot.go_away_from_ball_output(list(actions.keys()), state, shared_info, self._last_actions)
        _go_to_ball_scaled = self.hc_bot.go_to_ball_speed_scaled(list(actions.keys()), state, shared_info, self._last_actions)

        for agent, action in actions.items():
            # Action can have shape (Ticks, 1) or (Ticks)
            assert len(action.shape) == 1 or (
                len(action.shape) == 2 and action.shape[1] == 1
            )

            if len(action.shape) == 2:
                action = action.squeeze(1)
                
            if action != HCBotAction.DO_NOTHING:
                self._states[agent] = action

            self._lookup_tables[agent][HCBotAction.GO_TO_BALL] = _go_to_ball[agent].to_ndarray()
            self._lookup_tables[agent][HCBotAction.GO_AWAY_FROM_BALL] = _go_away_from_ball[agent].to_ndarray()
            
            self._lookup_tables[agent][HCBotAction.GO_TO_BALL_BOOST] = _go_to_ball[agent].to_ndarray()
            self._lookup_tables[agent][HCBotAction.GO_TO_BALL_BOOST][BOOST] = 1
            
            self._lookup_tables[agent][HCBotAction.GO_AWAY_FROM_BALL_BOOST] = _go_away_from_ball[agent].to_ndarray()
            self._lookup_tables[agent][HCBotAction.GO_AWAY_FROM_BALL_BOOST][BOOST] = 1
            
            self._lookup_tables[agent][HCBotAction.GO_TO_BALL_SPEED_SCALED] = _go_to_ball_scaled[agent].to_ndarray()
            
            if action == HCBotAction.FLIP_TO_BALL:
                returns[agent] = self.hc_bot.flip_to_ball([agent], state, shared_info, self._last_actions)[agent].to_ndarray()
            elif action == HCBotAction.BOOST:
                returns[agent] = self._last_actions[agent]
                returns[agent][BOOST] = True
            elif action == HCBotAction.JUMP:
                returns[agent] = self._last_actions[agent]
                returns[agent][JUMP] = True
            elif action == HCBotAction.SLOW_DOWN:
                returns[agent] = self._last_actions[agent]
                returns[agent][THROTTLE] = max(0, returns[agent][THROTTLE] - 0.05)
            elif action == HCBotAction.ACCELERATE:
                returns[agent] = self._last_actions[agent]
                returns[agent][THROTTLE] = min(1, returns[agent][THROTTLE] + 0.05)
            else:
                returns[agent] = self._lookup_tables[agent][self._states[agent]]
                
            if returns[agent].shape[0] == 1:
                returns[agent] = returns[agent].squeeze(0)
                
            self._last_actions[agent] = returns[agent]

        return returns
    
    def sample(self, agents: List[Hashable]):
        return {agent: np.asarray([random.randint(0, HCBotAction.N_ACTIONS - 1)]) for agent in agents}

    def pick(self, agents: List[Hashable], action: int):
        return {agent: np.asarray([action]) for agent in agents}

    def reset(
        self,
        agents: List[Hashable],
        initial_state: GameState,
        shared_info: Dict[str, Any],
    ) -> None:
        self.hc_bot.reset(agents, initial_state, shared_info)
        self._lookup_tables = self._make_lookup_tables(agents)
        for agent in agents:
            self._states[agent] = HCBotAction.GO_TO_BALL
            self._last_actions[agent] = np.zeros((8, ))