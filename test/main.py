import math

import numpy as np
from rlbot.managers import Bot
from rlbot_flatbuffers import ControllerState, GamePacket

from rotation_utils import euler_to_rotation
from vec_op import OpVector3


class VoidHC(Bot):
    def get_output(self, packet: GamePacket) -> ControllerState:
        _agent = packet.players[self.index]
        _ball = packet.balls[0]

        _agent_to_ball = OpVector3(_ball.physics.location) - _agent.physics.location

        _agent_rot_mat = euler_to_rotation(
            _agent.physics.rotation.pitch,
            _agent.physics.rotation.yaw,
            _agent.physics.rotation.roll,
        )

        _agent_forward = OpVector3.from_numpy(_agent_rot_mat[:, 0])
        _project_to_ball = _agent_forward - _agent_to_ball.normalized
        _angle = _agent_forward.angle(_agent_to_ball.normalized) / math.pi

        return ControllerState(
            throttle=1, steer=float(_angle * np.sign(_project_to_ball[1]))
        )


if __name__ == "__main__":
    VoidHC().run(wants_ball_predictions=False, wants_match_communications=False)
