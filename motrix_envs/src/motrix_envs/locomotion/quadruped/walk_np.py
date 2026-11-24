# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Generic quadruped walk task with joystick velocity tracking."""

import gymnasium as gym
import numpy as np
from motrix_robots.quadruped import QuadrupedRobotCfg

from motrix_env_core.array.env import ArrayEnvState, NpObs
from motrix_env_core.base import ObsSpace
from motrix_env_core.direct.env import DirectEnv
from motrix_env_core.mdp.action_space import asymmetric_residual_action_space
from motrix_env_core.sim import (
    ActuatorCtrlQuery,
    ActuatorKdQuery,
    ActuatorKpQuery,
    BodyAngularVelocityWrite,
    BodyJointPositionQuery,
    BodyJointPositionWrite,
    BodyJointVelocityQuery,
    BodyLinearVelocityWrite,
    BodyPositionWrite,
    BodyRotationWrite,
    DofVelocityQuery,
    GeomPositionQuery,
    GeomFrictionQuery,
    LinkCenterOfMassQuery,
    LinkMassQuery,
    LinkPositionQuery,
    SensorValuesQuery,
)
from motrix_env_core.sim.model import ActuatorType
from motrix_env_core.sim.write import (
    ActuatorDampingWrite,
    ActuatorKpWrite,
    BodyJointVelocityWrite,
    CtrlTargetsWrite,
    GeomFrictionWrite,
    LinkComWrite,
    LinkMassWrite,
)
from motrix_envs.locomotion.quadruped.cfg import QuadrupedWalkEnvCfg
from motrix_envs.locomotion.quadruped.velocity_command import RandomPlanarVelocityBinding


def _sim_data_queries(cfg: QuadrupedWalkEnvCfg):
    robot = cfg.scene.objs.robot
    base_link_name = robot.resolved_base_link_name
    scene_sensor_names = {name for name, _ in cfg.scene.iter_sensors()}

    def resolve_sensor_name(name: str) -> str:
        return name if name in scene_sensor_names else robot.resolve_name(name)

    queries = {
        "joint_dof_pos": BodyJointPositionQuery(body=base_link_name),
        "joint_dof_vel": BodyJointVelocityQuery(body=base_link_name),
        "dof_vel": DofVelocityQuery(),
        "actuator_ctrls": ActuatorCtrlQuery(),
        "local_linvel": SensorValuesQuery(sensors=(resolve_sensor_name(cfg.sensor.local_linvel),)),
        "gyro": SensorValuesQuery(sensors=(resolve_sensor_name(cfg.sensor.gyro),)),
        "upvector": SensorValuesQuery(sensors=(resolve_sensor_name(cfg.sensor.upvector),)),
        "base_pos": LinkPositionQuery(link=base_link_name),
        "front_left_contact": SensorValuesQuery(sensors=("front_left_contact",)),
        "front_right_contact": SensorValuesQuery(sensors=("front_right_contact",)),
        "rear_left_contact": SensorValuesQuery(sensors=("rear_left_contact",)),
        "rear_right_contact": SensorValuesQuery(sensors=("rear_right_contact",)),
        "FL_pos": SensorValuesQuery(sensors=(resolve_sensor_name(cfg.sensor.foot_positions[0]),)),
        "FR_pos": SensorValuesQuery(sensors=(resolve_sensor_name(cfg.sensor.foot_positions[1]),)),
        "RL_pos": SensorValuesQuery(sensors=(resolve_sensor_name(cfg.sensor.foot_positions[2]),)),
        "RR_pos": SensorValuesQuery(sensors=(resolve_sensor_name(cfg.sensor.foot_positions[3]),)),
    }
    if isinstance(robot, QuadrupedRobotCfg):
        foot_geom_names = robot.foot_contact_geom_names
        for key, geom_name in zip(("FL_world_pos", "FR_world_pos", "RL_world_pos", "RR_world_pos"), foot_geom_names):
            if geom_name is None:
                raise ValueError("Quadruped walk requires a resolved foot contact geom name")
            queries[key] = GeomPositionQuery(geom=geom_name)
    return queries


def _sim_model_queries(cfg: QuadrupedWalkEnvCfg):
    base_link_name = cfg.scene.objs.robot.resolved_base_link_name
    return {
        "actuator_kp": ActuatorKpQuery(),
        "actuator_kd": ActuatorKdQuery(),
        "base_mass": LinkMassQuery(name=base_link_name),
        "base_com": LinkCenterOfMassQuery(name=base_link_name),
        "ground_friction": GeomFrictionQuery(name=cfg.ground_geom_name),
    }


_FOOTPRINT = np.array(
    [[dx, dy] for dx in (-0.25, 0.0, 0.25) for dy in (-0.15, 0.0, 0.15)],
    dtype=np.float32,
)
def _resolve_key_pose(
    robot: QuadrupedRobotCfg,
    actuators,
    pose_name: str,
) -> np.ndarray:
    actuator_joint_names: list[str] = []
    for spec in actuators:
        actuator_joint_names.append(spec.target_name)

    if pose_name not in robot.key_pose.poses:
        raise ValueError(f"Quadruped robot does not define key pose {pose_name!r}")
    positions = {
        robot.resolve_name(name): value
        for name, value in zip(robot.key_pose.joint_names, robot.key_pose.poses[pose_name])
    }
    missing = sorted(set(actuator_joint_names).difference(positions))
    extra = sorted(set(positions).difference(actuator_joint_names))
    if missing or extra:
        raise ValueError(
            f"Quadruped robot key pose {pose_name!r} must match actuator joint targets exactly: "
            f"missing={missing}, extra={extra}"
        )
    return np.asarray([positions[name] for name in actuator_joint_names], dtype=np.float32)


class QuadrupedWalkTask(DirectEnv[QuadrupedWalkEnvCfg]):
    """Base quadruped walk task driven by a :class:`QuadrupedRobotCfg`."""

    # Keep the nominal posture constraints intact for standing and sagittal
    # walking, while making them a soft constraint for lateral/yaw commands.
    # These are implementation constants rather than user-facing reward
    # hyperparameters: the command envelope is already part of the task
    # contract and we do not want another tuning surface for this correction.
    _POSTURE_GATE_START = 0.1
    _POSTURE_GATE_END = 0.5
    _POSTURE_GATE_FLOOR = 0.40

    # The nominal joint pose is the handoff pose used by deployment.  Keep
    # the existing weak regularizer during motion, but make the same term a
    # strong standing-only reference so a zero-command policy cannot settle
    # into a different balance posture after takeover.
    _STANDING_SIMILAR_TO_DEFAULT_GAIN = 5.0

    # Static-stance foot anchoring is deliberately kept as implementation
    # policy rather than another public reward/configuration knob.  The
    # deadband filters simulator/contact noise; beyond it, 5 mm corresponds
    # to a small but meaningful penalty.  The rational barrier approaches the
    # existing saturation value 4.0 without becoming flat at a finite error,
    # so accumulated creep continues to receive a learning signal.
    _STATIC_ANCHOR_DEADBAND = 0.001
    _STATIC_ANCHOR_SATURATION_SCALE = 0.02

    # Dynamic COM envelope used by the existing base_com_position term.  A
    # moving quadruped needs a small forward COM shift to accelerate, but the
    # allowed shift must remain bounded so the body cannot progressively
    # pitch/translate ahead of the stance feet.
    # Nominal dynamic-COM envelope used by the earlier regular recipe. Keep
    # the default handoff COM as the target while allowing bounded motion.
    _DYNAMIC_COM_FORWARD_ALLOWANCE = 0.02
    _DYNAMIC_COM_BACKWARD_ALLOWANCE = 0.025

    def __init__(self, cfg: QuadrupedWalkEnvCfg, num_envs=1, backend: str | None = None):
        robot = cfg.scene.objs.robot
        if not isinstance(robot, QuadrupedRobotCfg):
            raise TypeError(f"Quadruped walk scene robot must be QuadrupedRobotCfg, got {type(robot).__name__}")
        legs = (
            robot.legs.front_left,
            robot.legs.front_right,
            robot.legs.rear_left,
            robot.legs.rear_right,
        )
        if any(leg.contact_geom_name is None for leg in legs):
            raise ValueError("Quadruped walk requires a contact geom name for every leg")
        foot_position_sensors = cfg.sensor.foot_positions
        if len(foot_position_sensors) != len(legs) or any(not name for name in foot_position_sensors):
            raise ValueError("Quadruped walk requires one non-empty foot position sensor name per leg")
        if len(set(foot_position_sensors)) != len(foot_position_sensors):
            raise ValueError("Quadruped walk foot position sensor names must be unique")

        super().__init__(cfg, num_envs, backend=backend)
        self.model = self.sim.compile_model(_sim_model_queries(cfg))
        self.sim_data = self.sim.compile_reads(_sim_data_queries(cfg))
        self._ctrl_writes = self.sim.write_compiler.compile({"ctrl": CtrlTargetsWrite()})
        self._base_link_name = robot.resolved_base_link_name
        self._reset_program = self.sim.write_compiler.compile(
            {
                "base_position": BodyPositionWrite((self._base_link_name,)),
                "base_rotation": BodyRotationWrite((self._base_link_name,)),
                "base_linear_velocity": BodyLinearVelocityWrite((self._base_link_name,)),
                "base_angular_velocity": BodyAngularVelocityWrite((self._base_link_name,)),
                "joints_position": BodyJointPositionWrite(self._base_link_name),
                "joints_velocity": BodyJointVelocityWrite(self._base_link_name),
            },
            reset=True,
        )
        self._reset_position = self._reset_program.buffer("base_position")[:, 0]
        self._reset_rotation = self._reset_program.buffer("base_rotation")[:, 0]
        self._reset_linear_velocity = self._reset_program.buffer("base_linear_velocity")[:, 0]
        self._reset_angular_velocity = self._reset_program.buffer("base_angular_velocity")[:, 0]
        self._reset_joint_position = self._reset_program.buffer("joints_position")
        self._reset_joint_velocity = self._reset_program.buffer("joints_velocity")
        self._num_action = self.num_actuators
        self._num_feet = len(legs)
        self._feet_contact_sensors = (
            "front_left_contact",
            "front_right_contact",
            "rear_left_contact",
            "rear_right_contact",
        )
        self._feet_position_sensors = foot_position_sensors
        velocity_cfg = cfg.commands.velocity
        root_seed = int(np.random.randint(0, np.iinfo(np.uint32).max, dtype=np.uint32))
        command_seed, randomization_seed = np.random.SeedSequence(root_seed).spawn(2)
        self._command_rng = np.random.default_rng(command_seed)
        self._randomization_rng = np.random.default_rng(randomization_seed)
        self._velocity_command_binding = RandomPlanarVelocityBinding(
            velocity_cfg.lower,
            velocity_cfg.upper,
            rng=self._command_rng,
            standing_probability=velocity_cfg.standing_probability,
            hard_case_probability=velocity_cfg.hard_case_probability,
            hard_cases=velocity_cfg.hard_cases,
        )
        if not np.isfinite(velocity_cfg.standing_threshold) or velocity_cfg.standing_threshold < 0.0:
            raise ValueError(
                f"standing_threshold must be finite and non-negative, got {velocity_cfg.standing_threshold}"
            )

        self.default_angles = _resolve_key_pose(robot, self.model.actuators, cfg.key_pose_name)
        if self.default_angles.shape != (self._num_action,):
            raise ValueError(
                "Quadruped default joint positions must match actuator count: "
                f"got {self.default_angles.shape}, expected {(self._num_action,)}"
            )
        ctrl_ranges = np.asarray([spec.ctrl_range for spec in self.model.actuators], dtype=np.float32)
        self._joint_position_lower = ctrl_ranges[:, 0]
        self._joint_position_upper = ctrl_ranges[:, 1]
        self._init_action_space()
        self._init_obs_space()

        self._init_base_pose = self.model.init_dof_pos[:7].copy()
        self._init_base_pose[:3] = np.asarray(cfg.initial_base_position, dtype=np.float32)
        self._init_joint_position = self.default_angles.copy()
        self._hip_indices = np.asarray(
            [index for index, actuator in enumerate(self.model.actuators) if "hip" in actuator.target_name],
            dtype=np.int64,
        )

        has_non_position_actuator = any(
            spec.actuator_type is not ActuatorType.POSITION for spec in self.model.actuators
        )
        if cfg.randomization.enabled and has_non_position_actuator:
            raise TypeError("Quadruped PD randomization requires position actuators")
        self._randomize_action_delay = cfg.randomization.enabled and any(cfg.randomization.action_delay_steps)
        # Declare the enabled override writes once and batch every reset's
        # randomization into a single program execution.
        randomize_writes = {}
        self._randomize_kp = cfg.randomization.enabled and cfg.randomization.kp_scale_range != (1.0, 1.0)
        self._randomize_damping = cfg.randomization.enabled and cfg.randomization.damping_scale_range != (1.0, 1.0)
        self._randomize_friction = cfg.randomization.enabled and cfg.randomization.sliding_friction_range is not None
        self._randomize_base_mass = cfg.randomization.enabled and cfg.randomization.base_mass_scale_range != (1.0, 1.0)
        self._randomize_base_com = cfg.randomization.enabled and any(cfg.randomization.base_com_offset_noise)
        actuator_names = tuple(spec.name for spec in self.model.actuators)
        if self._randomize_kp:
            randomize_writes["kp"] = ActuatorKpWrite(actuator_names)
        if self._randomize_damping:
            randomize_writes["damping"] = ActuatorDampingWrite(actuator_names)
        if self._randomize_friction:
            randomize_writes["friction"] = GeomFrictionWrite((cfg.ground_geom_name,))
        if self._randomize_base_mass:
            randomize_writes["mass"] = LinkMassWrite((self._base_link_name,))
        if self._randomize_base_com:
            randomize_writes["com"] = LinkComWrite((self._base_link_name,))
        self._randomize_writes = self.sim.write_compiler.compile(randomize_writes) if randomize_writes else None

        self.feet_contact = np.zeros((num_envs, self._num_feet), dtype=bool)
        self._last_feet_contact = np.zeros((num_envs, self._num_feet), dtype=bool)
        self.feet_pos = np.zeros((num_envs, self._num_feet, 3), dtype=np.float32)
        self._last_feet_pos = np.zeros_like(self.feet_pos)
        self.feet_world_pos = np.zeros_like(self.feet_pos)
        self._last_feet_world_pos = np.zeros_like(self.feet_world_pos)
        self._static_anchor_pos = np.zeros((num_envs, self._num_feet, 2), dtype=np.float32)
        self._static_anchor_valid = np.zeros((num_envs, self._num_feet), dtype=bool)
        self._last_joint_vel = np.zeros((num_envs, self._num_action), dtype=np.float32)
        self._default_feet_width: np.ndarray | None = None
        self._default_base_com_forward_offset: float | None = None

        # Episode-scoped task state: full-batch buffers, reset writes the done
        # rows in place.
        self._commands = np.zeros((num_envs, 3), dtype=np.float32)
        self._command_resampling_time = np.full(num_envs, np.inf, dtype=np.float32)
        self._command_control_steps = 0
        self._phase = np.zeros(num_envs, dtype=np.float32)
        self._feet_phase = np.zeros((num_envs, self._num_feet), dtype=np.float32)
        self._gait_schedule_alpha = np.zeros(num_envs, dtype=np.float32)
        self._gait_frequency = np.full(num_envs, float(cfg.gait_frequency), dtype=np.float32)
        self._swing_fraction = np.full(
            num_envs, float(cfg.reward_config.swing_fraction), dtype=np.float32
        )
        self._current_actions = np.zeros((num_envs, self._num_action), dtype=np.float32)
        self._last_actions = np.zeros((num_envs, self._num_action), dtype=np.float32)
        self._action_delay_steps = np.zeros(num_envs, dtype=np.int32)
        self._set_hard_case_probability(0)

    # ---- obs/action space plumbing ----------------------------------------

    def _policy_obs_dim(self) -> int:
        # gyro(3) + gravity(3) + dof_diff(A) + dof_vel(A) + last_actions(A)
        # + command(3) + feet_phase(num_feet)
        return 3 + 3 + 3 * self._num_action + 3 + self._num_feet

    def _value_obs_dim(self) -> int:
        return self._policy_obs_dim() + 3  # + local linvel(3)

    def _init_action_space(self):
        self._action_space = asymmetric_residual_action_space(
            np.asarray([self._joint_position_lower, self._joint_position_upper], dtype=np.float32),
            self.default_angles,
            self.cfg.control_config.action_scale,
        )

    def _init_obs_space(self):
        self._observation_space = ObsSpace(
            policy=gym.spaces.Box(-np.inf, np.inf, (self._policy_obs_dim(),), dtype=np.float32),
            value=gym.spaces.Box(-np.inf, np.inf, (self._value_obs_dim(),), dtype=np.float32),
        )

    @property
    def action_space(self) -> gym.spaces.Box:
        return self._action_space

    @property
    def observation_space(self) -> ObsSpace:
        return self._observation_space

    # ---- state accessors --------------------------------------------------

    def get_dof_pos(self) -> np.ndarray:
        return self.sim_data["joint_dof_pos"]

    def get_dof_vel(self) -> np.ndarray:
        return self.sim_data["joint_dof_vel"]

    def get_local_linvel(self) -> np.ndarray:
        return self.sim_data["local_linvel"]

    def get_gyro(self) -> np.ndarray:
        return self.sim_data["gyro"]

    def get_gravity(self) -> np.ndarray:
        return self.sim_data["upvector"]

    # ---- contact / phase helpers -----------------------------------------

    def _obs_noise(self, data: np.ndarray, scale: float) -> np.ndarray:
        level = float(self.cfg.noise_config.level)
        if level <= 0.0:
            return data
        noise = np.random.uniform(-1.0, 1.0, data.shape).astype(data.dtype)
        return data + level * scale * noise

    def _read_feet_contact(self, env_ids: np.ndarray | None = None) -> np.ndarray:
        rows = slice(None) if env_ids is None else env_ids
        inputs = self.sim_data
        contacts = np.zeros((self._num_envs, self._num_feet), dtype=bool)
        for i, name in enumerate(self._feet_contact_sensors):
            value = inputs[name][rows]
            contacts[rows, i] = value[:, 0] > 0.0
        return contacts

    def _update_feet_buffers(self, env_ids: np.ndarray | None = None) -> None:
        contacts = self._read_feet_contact(env_ids)
        rows = slice(None) if env_ids is None else env_ids
        self._last_feet_contact[rows, :] = self.feet_contact[rows, :]
        self.feet_contact[rows, :] = contacts[rows, :]
        for i, name in enumerate(self._feet_position_sensors):
            self.feet_pos[rows, i, :] = self.sim_data[name][rows]
        for i, name in enumerate(("FL_world_pos", "FR_world_pos", "RL_world_pos", "RR_world_pos")):
            self.feet_world_pos[rows, i, :] = self.sim_data[name][rows]

    def _advance_phase(self):
        commands = self._commands
        if self.cfg.gait_schedule_enabled:
            # Anchor the schedule to the measured forward gaits.  Lateral and
            # yaw commands map to equivalent sagittal speeds so they do not
            # remain trapped in the measured low-speed anchors.
            speed = np.maximum(
                np.maximum(
                    np.abs(commands[:, 0]),
                    float(self.cfg.gait_schedule_lateral_weight) * np.abs(commands[:, 1]),
                ),
                float(self.cfg.gait_schedule_yaw_weight) * np.abs(commands[:, 2]),
            )
            speed_low = float(self.cfg.gait_schedule_speed_low)
            speed_high = float(self.cfg.gait_schedule_speed_high)
            target_alpha = np.clip((speed - speed_low) / (speed_high - speed_low), 0.0, 1.0)
            alpha_rate = self.cfg.ctrl_dt / float(self.cfg.gait_schedule_smoothing_time_s)
            delta = target_alpha - self._gait_schedule_alpha
            self._gait_schedule_alpha += np.sign(delta) * np.minimum(np.abs(delta), alpha_rate)
            self._gait_frequency[:] = (
                float(self.cfg.gait_schedule_frequency_low)
                + self._gait_schedule_alpha
                * (float(self.cfg.gait_schedule_frequency_high) - float(self.cfg.gait_schedule_frequency_low))
            )
            self._swing_fraction[:] = (
                float(self.cfg.gait_schedule_swing_fraction_low)
                + self._gait_schedule_alpha
                * (
                    float(self.cfg.gait_schedule_swing_fraction_high)
                    - float(self.cfg.gait_schedule_swing_fraction_low)
                )
            )
        else:
            self._gait_schedule_alpha.fill(0.0)
            self._gait_frequency.fill(float(self.cfg.gait_frequency))
            self._swing_fraction.fill(float(self.cfg.reward_config.swing_fraction))
        if np.any((self._swing_fraction <= 0.0) | (self._swing_fraction >= 1.0)):
            raise ValueError("scheduled swing fraction must lie strictly inside (0, 1)")

        standing = np.linalg.norm(commands, axis=1) < self.cfg.commands.velocity.standing_threshold
        phase = self._phase
        np.fmod(phase + self.cfg.ctrl_dt * self._gait_frequency, 1.0, out=phase)
        # Stand and walk share phase zero.  Advancing a hidden clock while the
        # observed phase is masked makes the first moving observation jump to an
        # arbitrary leg phase; freezing both keeps starts in full support.
        phase[standing] = 0.0
        feet_phase = self._feet_phase
        # Each trot pair swings together; pair ``n`` is offset by half a cycle
        # relative to pair ``n-1``.
        for pair_idx, (i, j) in enumerate(self.cfg.trot_pairs):
            value = (phase + 0.5 * pair_idx) % 1.0
            feet_phase[:, i] = value
            feet_phase[:, j] = value
        feet_phase[standing] = 0.0

    # ---- env loop ---------------------------------------------------------

    def apply_action(self, actions: np.ndarray, state: ArrayEnvState) -> ArrayEnvState:
        actions = np.asarray(actions, dtype=np.float32)
        self._last_actions[:] = self._current_actions
        self._current_actions[:] = actions
        if self._randomize_action_delay:
            use_last = self._action_delay_steps == 1
            exec_actions = np.where(use_last[:, None], self._last_actions, actions)
        else:
            exec_actions = self._last_actions if self.cfg.control_config.simulate_action_latency else actions
        targets = exec_actions * self.cfg.control_config.action_scale + self.default_angles
        ctrl = self._ctrl_writes.buffer("ctrl")
        ctrl[:] = targets.astype(np.float32, copy=False)
        self._ctrl_writes.execute()
        return state

    def compute_observation(self, state: ArrayEnvState) -> ArrayEnvState:
        noise_cfg = self.cfg.noise_config
        diff = self.get_dof_pos() - self.default_angles
        noisy_gyro = self._obs_noise(self.get_gyro(), noise_cfg.scale_gyro)
        noisy_gravity = self._obs_noise(self.get_gravity(), noise_cfg.scale_gravity)
        noisy_diff = self._obs_noise(diff, noise_cfg.scale_joint_angle)
        noisy_dof_vel = self._obs_noise(self.get_dof_vel(), noise_cfg.scale_joint_vel)
        noisy_linvel = self._obs_noise(self.get_local_linvel(), noise_cfg.scale_linvel)
        command = self._commands
        last_actions = self._current_actions
        feet_phase = self._feet_phase

        policy = np.concatenate(
            [
                noisy_gyro,
                -noisy_gravity,
                noisy_diff,
                noisy_dof_vel,
                last_actions,
                command,
                feet_phase,
            ],
            axis=1,
        ).astype(np.float32, copy=False)
        value = np.concatenate([policy, noisy_linvel], axis=1).astype(np.float32, copy=False)
        expected_policy = self._policy_obs_dim()
        expected_value = self._value_obs_dim()
        if policy.shape[1] != expected_policy or value.shape[1] != expected_value:
            raise ValueError(
                f"Unexpected quadruped obs shapes: policy={policy.shape}, "
                f"expected policy[{expected_policy}]; value={value.shape}, "
                f"expected value[{expected_value}]"
            )
        return state.replace(obs=NpObs(policy=policy, value=value))

    def compute_transition(self, state: ArrayEnvState) -> ArrayEnvState:
        self.sim_data.execute()
        self._command_control_steps += 1
        self._set_hard_case_probability(
            self._command_control_steps
            // self.cfg.commands.velocity.hard_case_curriculum_steps_per_iteration
        )
        self._update_commands()
        self._advance_phase()
        self._update_feet_buffers()

        terminated = self.get_gravity()[:, 2] <= self.cfg.termination_min_up_z
        reward = self._compute_reward(state, self.get_local_linvel(), self.get_gyro(), self.get_dof_pos())
        self._last_feet_pos[:] = self.feet_pos
        self._last_feet_world_pos[:] = self.feet_world_pos
        self._last_joint_vel[:] = self.get_dof_vel()
        return state.replace(reward=reward, terminated=terminated)

    def _compute_reward(
        self, state: ArrayEnvState, linvel: np.ndarray, gyro: np.ndarray, dof_pos: np.ndarray
    ) -> np.ndarray:
        reward = np.zeros((self._num_envs,), dtype=np.float32)
        reward_cfg = self.cfg.reward_config
        reward_fns = {
            "tracking_lin_vel": self._reward_tracking_lin_vel,
            "tracking_ang_vel": self._reward_tracking_ang_vel,
            "lin_vel_z": self._reward_lin_vel_z,
            "ang_vel_xy": self._reward_ang_vel_xy,
            "base_height": self._reward_base_height,
            "action_rate": self._reward_action_rate,
            "similar_to_default": self._reward_similar_to_default,
            "feet_spacing": self._reward_feet_spacing,
            "contact": self._reward_contact,
            "swing_feet_z": self._reward_swing_feet_z,
            "swing_contact": self._reward_swing_contact,
            "touchdown_velocity": self._reward_touchdown_velocity,
            "flat_orientation": self._reward_flat_orientation,
            "dof_acc": self._reward_dof_acc,
            "hip_position": self._reward_hip_position,
            "base_com_position": self._reward_base_com_position,
        }
        reward_items = {}
        for name, reward_fn in reward_fns.items():
            if name in reward_cfg.excluded_rewards:
                continue
            scale = getattr(reward_cfg.scales, name)
            if scale == 0:
                continue
            rew = reward_fn(linvel, gyro, dof_pos)
            weighted = (rew * scale).astype(np.float32, copy=False)
            reward += weighted
            reward_items[name] = weighted

        state.reward_terms = reward_items
        return reward * self.cfg.ctrl_dt

    def reset(self, env_ids: np.ndarray) -> None:
        num_reset = len(env_ids)
        if self._default_feet_width is None and (
            self.cfg.reward_config.scales.feet_spacing != 0.0
            or self.cfg.reward_config.scales.base_com_position != 0.0
        ):
            self._capture_default_feet_width(env_ids)

        base_pose = np.tile(self._init_base_pose, (num_reset, 1))
        base_linear_velocity = np.zeros((num_reset, 3), dtype=np.float32)
        base_angular_velocity = np.zeros((num_reset, 3), dtype=np.float32)
        joint_position = np.tile(self._init_joint_position, (num_reset, 1))
        joint_velocity = np.zeros((num_reset, self._reset_joint_velocity.shape[1]), dtype=np.float32)
        self._randomize_dof_noise(env_ids, base_linear_velocity, base_angular_velocity, joint_position, joint_velocity)

        spawn_range = self.cfg.spawn_xy_range
        if spawn_range > 0.0:
            xy = np.random.uniform(-spawn_range, spawn_range, size=(num_reset, 2)).astype(np.float32)
            ground_height = self.sim.sample_terrain_height(
                self.cfg.ground_geom_name, env_ids, xy[:, None, :] + _FOOTPRINT[None, :, :]
            ).max(axis=1)
            base_pose[:, :2] = xy
            base_pose[:, 2] = self._init_base_pose[2] + ground_height

        _reset_pose = base_pose
        self._reset_position[env_ids] = _reset_pose[:, :3]
        self._reset_rotation[env_ids] = _reset_pose[:, 3:7]
        self._reset_linear_velocity[env_ids] = base_linear_velocity
        self._reset_angular_velocity[env_ids] = base_angular_velocity
        self._reset_joint_position[env_ids] = joint_position
        self._reset_joint_velocity[env_ids] = joint_velocity
        self._reset_program.execute(env_ids)
        self._randomize_params(env_ids)

        self.sim_data.execute(np.asarray(env_ids, dtype=np.int64))
        self._last_feet_contact[env_ids] = False
        self.feet_contact[env_ids] = False
        self._static_anchor_valid[env_ids] = False
        self._update_feet_buffers(env_ids)
        self._last_feet_pos[env_ids] = self.feet_pos[env_ids]
        self._last_feet_world_pos[env_ids] = self.feet_world_pos[env_ids]
        self._last_joint_vel[env_ids] = self.get_dof_vel()[env_ids]

        self._commands[env_ids] = self.resample_commands(num_reset)
        self._command_resampling_time[env_ids] = self._sample_command_resampling_time(num_reset)
        self._phase[env_ids] = 0.0
        self._feet_phase[env_ids] = 0.0
        self._gait_schedule_alpha[env_ids] = 0.0
        if self.cfg.gait_schedule_enabled:
            self._gait_frequency[env_ids] = float(self.cfg.gait_schedule_frequency_low)
            self._swing_fraction[env_ids] = float(self.cfg.gait_schedule_swing_fraction_low)
        else:
            self._gait_frequency[env_ids] = float(self.cfg.gait_frequency)
            self._swing_fraction[env_ids] = float(self.cfg.reward_config.swing_fraction)
        self._current_actions[env_ids] = 0.0
        self._last_actions[env_ids] = 0.0

    def _set_hard_case_probability(self, iteration: int) -> None:
        """Apply the configured command curriculum for the current train step."""
        velocity_cfg = self.cfg.commands.velocity
        curriculum = velocity_cfg.hard_case_curriculum
        if not curriculum:
            probability = velocity_cfg.hard_case_probability
        elif iteration <= curriculum[0][0]:
            probability = curriculum[0][1]
        elif iteration >= curriculum[-1][0]:
            probability = curriculum[-1][1]
        else:
            probability = curriculum[-1][1]
            for (left_iteration, left_probability), (right_iteration, right_probability) in zip(
                curriculum, curriculum[1:]
            ):
                if left_iteration <= iteration <= right_iteration:
                    fraction = (iteration - left_iteration) / (right_iteration - left_iteration)
                    probability = left_probability + fraction * (right_probability - left_probability)
                    break
        self._velocity_command_binding.set_hard_case_probability(probability)

    def _capture_default_feet_width(self, env_ids: np.ndarray) -> None:
        """Read the nominal front and rear foot widths from the default pose."""
        num_reset = len(env_ids)
        base_pose = np.tile(self._init_base_pose, (num_reset, 1))
        joint_position = np.tile(self._init_joint_position, (num_reset, 1))
        joint_velocity = np.zeros((num_reset, self._reset_joint_velocity.shape[1]), dtype=np.float32)

        self._reset_position[env_ids] = base_pose[:, :3]
        self._reset_rotation[env_ids] = base_pose[:, 3:7]
        self._reset_linear_velocity[env_ids] = 0.0
        self._reset_angular_velocity[env_ids] = 0.0
        self._reset_joint_position[env_ids] = joint_position
        self._reset_joint_velocity[env_ids] = joint_velocity
        self._reset_program.execute(env_ids)
        self.sim_data.execute(np.asarray(env_ids, dtype=np.int64))
        self._last_feet_contact[env_ids] = False
        self.feet_contact[env_ids] = False
        self._static_anchor_valid[env_ids] = False
        self._update_feet_buffers(env_ids)
        self._last_feet_world_pos[env_ids] = self.feet_world_pos[env_ids]

        front_width = self.feet_pos[env_ids, 0, 1] - self.feet_pos[env_ids, 1, 1]
        rear_width = self.feet_pos[env_ids, 2, 1] - self.feet_pos[env_ids, 3, 1]
        self._default_feet_width = np.stack([front_width, rear_width], axis=1).mean(axis=0)
        base_com_x = float(self.model.others["base_com"][0])
        self._default_base_com_forward_offset = float(
            np.mean(base_com_x - np.mean(self.feet_pos[env_ids, :, 0], axis=1))
        )

    def resample_commands(self, num_envs: int) -> np.ndarray:
        return self._velocity_command_binding.read_command(batch_size=num_envs).values.copy()

    def _sample_command_resampling_time(self, num_envs: int) -> np.ndarray:
        interval = self.cfg.commands.velocity.resampling_seconds_range
        if interval is None:
            return np.full((num_envs,), np.inf, dtype=np.float32)
        return self._command_rng.uniform(*interval, size=num_envs).astype(np.float32)

    def _update_commands(self) -> None:
        if self.cfg.commands.velocity.resampling_seconds_range is None:
            return
        remaining = self._command_resampling_time
        remaining -= self.cfg.ctrl_dt
        due = remaining <= 0.0
        num_due = int(np.count_nonzero(due))
        if num_due:
            self._commands[due] = self.resample_commands(num_due)
            remaining[due] = self._sample_command_resampling_time(num_due)

    def _randomize_dof_noise(
        self,
        env_ids: np.ndarray,
        base_linear_velocity: np.ndarray,
        base_angular_velocity: np.ndarray,
        joint_position: np.ndarray,
        joint_velocity: np.ndarray,
    ) -> None:
        num_reset = len(env_ids)
        randomization = self.cfg.randomization
        if not randomization.enabled:
            return
        rng = self._randomization_rng
        num_action = self._num_action

        if randomization.joint_pos_noise > 0.0:
            joint_position[:, -num_action:] += rng.uniform(
                -randomization.joint_pos_noise,
                randomization.joint_pos_noise,
                size=(num_reset, num_action),
            ).astype(np.float32)
            np.clip(
                joint_position[:, -num_action:],
                self._joint_position_lower,
                self._joint_position_upper,
                out=joint_position[:, -num_action:],
            )
        if randomization.joint_vel_noise > 0.0:
            joint_velocity[:, -num_action:] += rng.uniform(
                -randomization.joint_vel_noise,
                randomization.joint_vel_noise,
                size=(num_reset, num_action),
            ).astype(np.float32)

        if any(randomization.base_lin_vel_noise):
            base_lin_vel_noise = np.asarray(randomization.base_lin_vel_noise, dtype=np.float32)
            base_linear_velocity[:] += rng.uniform(-base_lin_vel_noise, base_lin_vel_noise, size=(num_reset, 3)).astype(
                np.float32
            )
        if any(randomization.base_ang_vel_noise):
            base_ang_vel_noise = np.asarray(randomization.base_ang_vel_noise, dtype=np.float32)
            base_angular_velocity[:] += rng.uniform(
                -base_ang_vel_noise, base_ang_vel_noise, size=(num_reset, 3)
            ).astype(np.float32)

    def _randomize_params(self, env_ids: np.ndarray) -> None:
        num_reset = len(env_ids)
        randomization = self.cfg.randomization
        if not randomization.enabled:
            return
        rng = self._randomization_rng
        num_action = self._num_action

        if self._randomize_action_delay:
            delay_low, delay_high = randomization.action_delay_steps
            if delay_low == delay_high:
                self._action_delay_steps[env_ids] = delay_low
            else:
                self._action_delay_steps[env_ids] = rng.integers(delay_low, delay_high + 1, size=num_reset)

        # All override writes for this reset batch into one program execution.
        program = self._randomize_writes
        if program is None:
            return

        if self._randomize_kp:
            kp_scale = rng.uniform(*self.cfg.randomization.kp_scale_range, size=(num_reset, num_action)).astype(
                np.float32
            )
            program.buffer("kp")[env_ids] = self.model.others["actuator_kp"][None, :] * kp_scale
        if self._randomize_damping:
            damping_scale = rng.uniform(
                *self.cfg.randomization.damping_scale_range, size=(num_reset, num_action)
            ).astype(np.float32)
            program.buffer("damping")[env_ids] = self.model.others["actuator_kd"][None, :] * damping_scale
        if self._randomize_friction:
            friction = np.tile(self.model.others["ground_friction"], (num_reset, 1))
            friction_low, friction_high = self.cfg.randomization.sliding_friction_range
            # Keep the configured range unchanged while exposing the policy to
            # the high-traction end more often.  The square-root transform
            # raises the probability of mu >= 1.2 from about 25% to about 45%,
            # targeting the forward/reverse failures seen in the audit without
            # adding a new reward or randomization field.
            friction_u = np.sqrt(rng.uniform(0.0, 1.0, size=num_reset))
            friction[:, 0] = (friction_low + (friction_high - friction_low) * friction_u).astype(np.float32)
            program.buffer("friction")[env_ids, 0] = friction
        if self._randomize_base_mass:
            base_mass_scale = rng.uniform(*self.cfg.randomization.base_mass_scale_range, size=num_reset).astype(
                np.float32
            )
            program.buffer("mass")[env_ids, 0] = np.full((num_reset,), self.model.others["base_mass"]) * base_mass_scale
        if self._randomize_base_com:
            base_com_noise = np.asarray(self.cfg.randomization.base_com_offset_noise, dtype=np.float32)
            base_com_offset = rng.uniform(-base_com_noise, base_com_noise, size=(num_reset, 3)).astype(np.float32)
            program.buffer("com")[env_ids, 0] = np.tile(self.model.others["base_com"], (num_reset, 1)) + base_com_offset
        program.execute(env_ids)

    # ---- reward functions -------------------------------------------------

    def _reward_tracking_lin_vel(self, linvel, gyro, dof_pos) -> np.ndarray:
        del gyro, dof_pos
        reward_cfg = self.cfg.reward_config
        commands = self._commands
        error = np.square(commands[:, :2] - linvel[:, :2])
        sigma = np.full(
            (self._num_envs, 2),
            float(reward_cfg.tracking_lin_vel_sigma),
            dtype=np.float32,
        )
        forward_sigma = reward_cfg.tracking_lin_vel_forward_sigma
        if forward_sigma is not None:
            sigma[:, 0] = np.where(
                commands[:, 0] > 0.0,
                float(forward_sigma),
                float(reward_cfg.tracking_lin_vel_sigma),
            )
        lateral_sigma = reward_cfg.tracking_lin_vel_lateral_sigma
        if lateral_sigma is not None:
            sigma[:, 1] = float(lateral_sigma)
        return np.exp(-np.sum(error / sigma, axis=1))

    def _reward_tracking_ang_vel(self, linvel, gyro, dof_pos) -> np.ndarray:
        del linvel, dof_pos
        commands = self._commands
        error = np.square(commands[:, 2] - gyro[:, 2])
        return np.exp(-error / self.cfg.reward_config.tracking_ang_vel_sigma)

    def _reward_lin_vel_z(self, linvel, gyro, dof_pos) -> np.ndarray:
        del gyro, dof_pos
        return np.square(linvel[:, 2])

    def _reward_ang_vel_xy(self, linvel, gyro, dof_pos) -> np.ndarray:
        del linvel, dof_pos
        return np.sum(np.square(gyro[:, :2]), axis=1)

    def _reward_base_height(self, linvel, gyro, dof_pos) -> np.ndarray:
        del linvel, gyro, dof_pos
        base_pos = self.sim_data["base_pos"]
        env_ids = np.arange(self._num_envs, dtype=np.int64)
        ground_height = self.sim.sample_terrain_height(self.cfg.ground_geom_name, env_ids, base_pos[:, None, :2])[:, 0]
        base_height = base_pos[:, 2].astype(np.float32) - ground_height
        return np.square(base_height - self.cfg.reward_config.base_height_target)

    def _reward_action_rate(self, linvel, gyro, dof_pos) -> np.ndarray:
        del linvel, gyro, dof_pos
        current = self._current_actions
        last = self._last_actions
        return np.sum(np.square(current - last), axis=1)

    def _reward_similar_to_default(self, linvel, gyro, dof_pos) -> np.ndarray:
        del linvel, gyro
        error = np.sum(np.abs(dof_pos - self.default_angles), axis=1)
        standing = np.linalg.norm(self._commands, axis=1) < self.cfg.commands.velocity.standing_threshold
        gain = 1.0 + (self._STANDING_SIMILAR_TO_DEFAULT_GAIN - 1.0) * standing
        return error * gain

    def _posture_constraint_scale(self) -> np.ndarray:
        """Scale posture constraints down only for lateral/yaw commands.

        Standing and pure forward/backward commands retain full strength.  A
        smooth minimum of the lateral and yaw command magnitudes transitions
        the scale from 1.0 to a non-zero floor, so the constraints remain
        stabilizing during sidestepping and turning without dictating their
        nominal leg geometry.
        """
        activity = np.maximum(np.abs(self._commands[:, 1]), np.abs(self._commands[:, 2]))
        progress = np.clip(
            (activity - self._POSTURE_GATE_START)
            / (self._POSTURE_GATE_END - self._POSTURE_GATE_START),
            0.0,
            1.0,
        )
        smooth = np.square(progress) * (3.0 - 2.0 * progress)
        return 1.0 - (1.0 - self._POSTURE_GATE_FLOOR) * smooth

    def _reward_feet_spacing(self, linvel, gyro, dof_pos) -> np.ndarray:
        del linvel, gyro, dof_pos
        if self._default_feet_width is None:
            return np.zeros((self._num_envs,), dtype=np.float32)

        front_width = self.feet_pos[:, 0, 1] - self.feet_pos[:, 1, 1]
        rear_width = self.feet_pos[:, 2, 1] - self.feet_pos[:, 3, 1]
        target_width = np.asarray(self._default_feet_width, dtype=np.float32)
        reward_cfg = self.cfg.reward_config
        pair_widths = np.stack([front_width, rear_width], axis=1)
        errors = np.abs(pair_widths - target_width)
        excess = np.maximum(errors - reward_cfg.feet_spacing_tolerance, 0.0)
        pair_rewards = np.exp(-np.square(excess / reward_cfg.feet_spacing_sigma))
        return np.mean(pair_rewards, axis=1) * self._posture_constraint_scale()

    def _reward_swing_feet_z(self, linvel, gyro, dof_pos) -> np.ndarray:
        del linvel, gyro, dof_pos
        feet_phase = self._feet_phase
        contacts = self.feet_contact
        reward_cfg = self.cfg.reward_config
        # feet_pos is body-relative (framepos with ref=imu). The foot lifts
        # `target_height` above its stance position. Convert that world-frame
        # lift into the sensor frame using the configured base-to-sensor offset.
        swing_fraction = self._swing_fraction[:, None]
        swing_start = 1.0 - swing_fraction
        valid_swing = (feet_phase >= swing_start) & ~contacts
        sensor_ground_z = -reward_cfg.base_height_target - reward_cfg.foot_height_reference_offset
        target_z = sensor_ground_z + reward_cfg.target_foot_height
        if reward_cfg.swing_feet_z_sinusoidal:
            # Shape the vertical trajectory instead of holding the peak for the
            # entire swing.  The final part of the gait cycle is swing, so map
            # it to a smooth lift-and-place arc.  Rise and plateau fractions are
            # normalized by swing duration and make velocity zero at endpoints.
            swing_progress = np.clip((feet_phase - swing_start) / swing_fraction, 0.0, 1.0)
            rise_fraction = float(reward_cfg.swing_rise_fraction)
            plateau_fraction = float(reward_cfg.swing_plateau_fraction)
            if not 0.0 < rise_fraction < 1.0 or plateau_fraction < 0.0 or rise_fraction + plateau_fraction >= 1.0:
                raise ValueError("swing rise and plateau fractions must lie strictly inside the swing phase")
            plateau_start = rise_fraction
            descent_start = rise_fraction + plateau_fraction
            target_profile = np.empty_like(swing_progress)
            rising = swing_progress < plateau_start
            plateauing = (swing_progress >= plateau_start) & (swing_progress < descent_start)
            descending = swing_progress >= descent_start
            target_profile[rising] = 0.5 * (1.0 - np.cos(np.pi * swing_progress[rising] / rise_fraction))
            target_profile[plateauing] = 1.0
            descent_progress = (swing_progress[descending] - descent_start) / (1.0 - descent_start)
            target_profile[descending] = 0.5 * (1.0 + np.cos(np.pi * descent_progress))
            target_z = sensor_ground_z + reward_cfg.target_foot_height * target_profile
        height_error = np.square(self.feet_pos[:, :, 2] - target_z)
        sigma_sq = reward_cfg.swing_feet_height_sigma**2
        swing_rew = np.exp(-height_error / sigma_sq) * valid_swing
        return np.sum(swing_rew, axis=1) / self._num_feet

    def _reward_touchdown_velocity(self, linvel, gyro, dof_pos) -> np.ndarray:
        del linvel, gyro, dof_pos
        reward_cfg = self.cfg.reward_config
        current_height = self.feet_pos[:, :, 2]
        previous_height = self._last_feet_pos[:, :, 2]
        vertical_speed = (current_height - previous_height) / max(float(self.cfg.ctrl_dt), 1e-6)
        sensor_ground_z = (
            reward_cfg.touchdown_height_threshold
            - reward_cfg.base_height_target
            - reward_cfg.foot_height_reference_offset
        )
        near_ground = (current_height < sensor_ground_z) & (vertical_speed < 0.0)
        normalized_speed = np.minimum(
            np.abs(vertical_speed) / reward_cfg.touchdown_speed_sigma,
            reward_cfg.touchdown_speed_clip,
        )
        return np.mean(np.square(normalized_speed) * near_ground, axis=1)

    def _reward_contact(self, linvel, gyro, dof_pos) -> np.ndarray:
        del linvel, gyro, dof_pos
        res = np.zeros((self._num_envs,), dtype=np.float32)
        feet_phase = self._feet_phase
        contacts = self.feet_contact
        swing_start = 1.0 - self._swing_fraction
        for i in range(self._num_feet):
            target_contact = feet_phase[:, i] < swing_start
            res += (contacts[:, i] == target_contact).astype(np.float32)
        return res / self._num_feet

    def _reward_swing_contact(self, linvel, gyro, dof_pos) -> np.ndarray:
        """Penalize contact most strongly during the middle of swing.

        Brief contact at lift-off and touchdown is a normal transition in a
        discrete-time simulator.  Weighting the interior of swing more
        strongly targets sustained dragging without adding another tuning
        parameter or changing the average penalty for a foot that remains in
        contact throughout swing.
        """
        del linvel, gyro, dof_pos
        swing_fraction = self._swing_fraction[:, None]
        swing_start = 1.0 - swing_fraction
        is_swing = self._feet_phase >= swing_start
        swing_progress = np.clip(
            (self._feet_phase - swing_start) / swing_fraction,
            0.0,
            1.0,
        )
        phase_weight = 2.0 * np.square(np.sin(np.pi * swing_progress))
        swing_contacts = self.feet_contact & is_swing
        weighted_contacts = swing_contacts.astype(np.float32) * phase_weight

        # Once a foot has remained in contact for two consecutive control
        # ticks, its horizontal displacement is ground slip rather than a
        # touchdown transition.  Keep the existing linear velocity penalty
        # for moving commands; static commands use the position anchor below
        # so slow creep cannot accumulate for free.
        consecutive_stance = self.feet_contact & self._last_feet_contact & ~is_swing
        horizontal_displacement = np.linalg.norm(
            self.feet_world_pos[:, :, :2] - self._last_feet_world_pos[:, :, :2], axis=2
        )
        horizontal_speed = horizontal_displacement / max(float(self.cfg.ctrl_dt), 1e-6)
        standing = np.linalg.norm(self._commands, axis=1) < self.cfg.commands.velocity.standing_threshold
        stance_motion = np.minimum(6.0 * horizontal_speed, 4.0) * consecutive_stance

        # In static stance, latch the world-frame contact point once contact
        # is confirmed for two consecutive ticks.  The latch is released on
        # command motion or liftoff, so it cannot constrain walking, turning,
        # or a normal swing/landing transition.
        static_stance = consecutive_stance & standing[:, None]
        self._static_anchor_valid &= standing[:, None] & self.feet_contact
        new_anchors = static_stance & ~self._static_anchor_valid
        self._static_anchor_pos[new_anchors] = self.feet_world_pos[:, :, :2][new_anchors]
        self._static_anchor_valid[new_anchors] = True
        anchor_error = np.linalg.norm(
            self.feet_world_pos[:, :, :2] - self._static_anchor_pos,
            axis=2,
        )
        anchor_excess = np.maximum(anchor_error - self._STATIC_ANCHOR_DEADBAND, 0.0)
        anchor_penalty = 4.0 * anchor_excess / (
            anchor_excess + self._STATIC_ANCHOR_SATURATION_SCALE
        )
        anchor_motion = (
            anchor_penalty
            * self._static_anchor_valid
            * standing[:, None]
            * self.feet_contact
        )
        stance_motion = np.maximum(stance_motion, anchor_motion)
        raw_penalty = np.sum(weighted_contacts + stance_motion, axis=1) / self._num_feet
        forward_scale = self.cfg.reward_config.swing_contact_forward_scale
        if forward_scale is None:
            return raw_penalty
        # The low-damping policy can evade speed tracking by keeping feet on
        # the ground and dragging.  Low-speed forward gait is especially prone
        # to that shortcut, so ramp this existing penalty with forward command
        # magnitude only.  Reverse, lateral, yaw, and standing commands retain
        # the nominal scale until they receive dedicated diagnostic evidence.
        direction_scale = 1.0 + (float(forward_scale) - 1.0) * np.clip(
            self._commands[:, 0], 0.0, 1.0
        )
        return raw_penalty * direction_scale

    def _reward_flat_orientation(self, linvel, gyro, dof_pos) -> np.ndarray:
        """Penalize roll/pitch tilt while leaving yaw unconstrained."""
        del linvel, gyro, dof_pos
        upvector = self.get_gravity()
        return np.sum(np.square(upvector[:, :2]), axis=1)

    def _reward_dof_acc(self, linvel, gyro, dof_pos) -> np.ndarray:
        """Penalize the largest joint acceleration estimated from velocity changes."""
        del linvel, gyro, dof_pos
        joint_vel = self.get_dof_vel()
        acceleration = (joint_vel - self._last_joint_vel) / max(float(self.cfg.ctrl_dt), 1e-6)
        return np.max(np.square(acceleration), axis=1)

    def _reward_hip_position(self, linvel, gyro, dof_pos) -> np.ndarray:
        del linvel, gyro
        if self._hip_indices.size == 0:
            return np.zeros((self._num_envs,), dtype=np.float32)

        reward_cfg = self.cfg.reward_config
        hip_error = np.abs(dof_pos[:, self._hip_indices] - self.default_angles[self._hip_indices])
        excess = np.maximum(hip_error - reward_cfg.hip_position_tolerance, 0.0)
        normalized = np.minimum(excess / reward_cfg.hip_position_sigma, 3.0)
        return np.mean(np.square(normalized), axis=1) * self._posture_constraint_scale()

    def _reward_base_com_position(self, linvel, gyro, dof_pos) -> np.ndarray:
        """Keep COM inside the standing or motion-specific envelope.

        Zero command retains the nominal handoff posture.  During motion the
        target remains the nominal handoff COM relative to the mean of all
        four feet, with a bounded command-dependent envelope.
        """
        del linvel, gyro, dof_pos
        reward_cfg = self.cfg.reward_config
        if self._default_base_com_forward_offset is None:
            return np.zeros((self._num_envs,), dtype=np.float32)

        base_com_x = float(self.model.others["base_com"][0])
        # Keep the training reference identical to the audit metric:
        # COM x relative to the mean x of all four feet.  Using only the
        # currently contacting feet makes the reference jump every half gait
        # cycle and lets the policy evade the dynamic COM constraint while
        # still showing a large measured forward extension.
        all_feet_center = np.mean(self.feet_pos[:, :, 0], axis=1)
        current_offset = base_com_x - all_feet_center

        commands = self._commands
        standing = np.linalg.norm(commands, axis=1) < self.cfg.commands.velocity.standing_threshold
        forward_speed = np.clip(commands[:, 0], 0.0, 1.0)
        backward_speed = np.clip(-commands[:, 0], 0.0, 1.0)
        upper_allowance = self._DYNAMIC_COM_FORWARD_ALLOWANCE * forward_speed
        lower_allowance = self._DYNAMIC_COM_BACKWARD_ALLOWANCE * backward_speed
        target = self._default_base_com_forward_offset
        upper = target + upper_allowance
        lower = target - lower_allowance
        excess = np.maximum(np.maximum(current_offset - upper, lower - current_offset), 0.0)
        # Standing remains a symmetric nominal-position constraint; movement
        # uses the one-sided dynamic envelope above.
        standing_error = np.abs(current_offset - self._default_base_com_forward_offset)
        excess = np.where(standing, standing_error, excess)
        excess = np.maximum(excess - reward_cfg.base_com_position_tolerance, 0.0)
        normalized = excess / reward_cfg.base_com_position_sigma
        dynamic_scale = self._posture_constraint_scale()
        return np.square(normalized) * np.where(standing, 1.0, dynamic_scale)

__all__ = ["QuadrupedWalkTask"]
