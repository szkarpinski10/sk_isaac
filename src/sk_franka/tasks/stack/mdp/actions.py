from __future__ import annotations

import torch

from isaaclab.envs.mdp.actions.actions_cfg import JointActionCfg, RelativeJointPositionActionCfg
from isaaclab.envs.mdp.actions.joint_actions import JointAction, RelativeJointPositionAction
from isaaclab.utils import configclass


class SmoothedRelativeJointPositionAction(RelativeJointPositionAction):
    """Cel = q + skala * akcja (jak RelativeJointPositionAction), ale wygładzony jak w sterowniku robota:
    co krok fizyki cel przesuwa się o część `smoothing` pozostałej drogi (panda_py JointPosition.set_filter(0.05)
    co 1 ms = 0.40 co 10 ms), więc nie skacze co krok policy."""

    def __init__(self, cfg: SmoothedRelativeJointPositionActionCfg, env):
        super().__init__(cfg, env)
        self._target = self._asset.data.joint_pos.torch[:, self._joint_ids].clone()
        self._needs_init = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)

    def apply_actions(self):
        q = self._asset.data.joint_pos.torch[:, self._joint_ids]
        self._target[self._needs_init] = q[self._needs_init]
        self._needs_init[:] = False
        self._target += self.cfg.smoothing * (q + self.processed_actions - self._target)
        self._asset.set_joint_position_target_index(target=self._target, joint_ids=self._joint_ids)

    def reset(self, env_ids=None):
        super().reset(env_ids)
        if env_ids is None:
            env_ids = slice(None)
        self._needs_init[env_ids] = True


@configclass
class SmoothedRelativeJointPositionActionCfg(RelativeJointPositionActionCfg):
    class_type: type = SmoothedRelativeJointPositionAction
    smoothing: float = 0.40


class RobotJointPositionAction(JointAction):
    """Sterowanie stawami tak samo jak robot_franka.movement() na prawdziwym robocie:
    komenda liczona od poprzedniej komendy, z limitem prędkości, plus prędkość docelowa step / DT."""

    def __init__(self, cfg: RobotJointPositionActionCfg, env):
        super().__init__(cfg, env)
        self._q_command = self._asset.data.joint_pos.torch[:, self._joint_ids].clone()
        self._velocity = torch.zeros_like(self._q_command)
        self._needs_init = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)

    def process_actions(self, actions: torch.Tensor):
        super().process_actions(actions)
        q = self._asset.data.joint_pos.torch[:, self._joint_ids]
        dt = self._env.step_dt

        self._q_command[self._needs_init] = q[self._needs_init]
        self._needs_init[:] = False

        q_goal = q + self.processed_actions
        step = torch.clamp(q_goal - self._q_command, -self.cfg.max_velocity * dt, self.cfg.max_velocity * dt)
        self._q_command += step
        self._velocity = step / dt

    def apply_actions(self):
        self._asset.set_joint_position_target_index(target=self._q_command, joint_ids=self._joint_ids)
        self._asset.set_joint_velocity_target_index(target=self._velocity, joint_ids=self._joint_ids)

    def reset(self, env_ids=None):
        super().reset(env_ids)
        if env_ids is None:
            env_ids = slice(None)
        self._needs_init[env_ids] = True
        self._velocity[env_ids] = 0.0


@configclass
class RobotJointPositionActionCfg(JointActionCfg):
    class_type: type = RobotJointPositionAction
    max_velocity: float = 0.8
