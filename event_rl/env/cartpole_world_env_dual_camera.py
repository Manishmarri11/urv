import os

import numpy as np
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.envs.registration import register, registry
from gymnasium.spaces import Box

XML_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cartpole_world_dual_camera.xml")

DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 0,
    "distance": 8.5,
    "elevation": -10,
}


class CartpoleWorldDualCameraEnv(MujocoEnv, utils.EzPickle):
    """Same task/dynamics as CartpoleWorldEnv, but the XML defines two cameras
    instead of one: "world" (fixed to the cart, oblique third-person view, for
    convenience renders/GIFs) and "pole" (fixed to the pole body itself, mounted
    just past its tip and tilted back toward the cart, so it physically rotates
    with the pole -- this is the true egocentric observation feed).

    The pole can tip in two independent planes (hinge_x, hinge_y) and the cart
    can slide in both to catch it (slider_x, slider_y), same gimbal setup as
    the inspo codebase. qpos/qvel order is [slider_x, slider_y, hinge_x,
    hinge_y], so observation = [cart_x, cart_y, pole_angle_x, pole_angle_y,
    cart_vx, cart_vy, pole_angvel_x, pole_angvel_y] (8-element), action is
    2-element continuous force (slide_motor_x, slide_motor_y). +1 reward per
    step both pole angles stay within 0.2 rad of upright.
    """

    metadata = {
        "render_modes": ["human", "rgb_array", "depth_array", "rgbd_tuple"],
    }

    def __init__(
        self,
        frame_skip: int = 2,
        reset_noise_scale: float = 0.01,
        default_camera_config: dict = DEFAULT_CAMERA_CONFIG,
        **kwargs,
    ):
        utils.EzPickle.__init__(self, frame_skip, reset_noise_scale, default_camera_config, **kwargs)

        observation_space = Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float64)
        self._reset_noise_scale = reset_noise_scale

        MujocoEnv.__init__(
            self,
            XML_PATH,
            frame_skip,
            observation_space=observation_space,
            default_camera_config=default_camera_config,
            **kwargs,
        )

        self.observation_structure = {
            "qpos": self.data.qpos.size,
            "qvel": self.data.qvel.size,
        }

        self.metadata["render_fps"] = int(np.round(1.0 / self.dt))

    def step(self, action):  # TODO: reward is angle-based for now; swap to event-frame-based once v2e is wired in
        self.do_simulation(action, self.frame_skip)
        observation = self._get_obs()

        # qpos = [slider_x, slider_y, hinge_x, hinge_y] -> observation[2:4] are the pole tilt angles
        pole_angle_x, pole_angle_y = observation[2], observation[3]
        terminated = bool(
            not np.isfinite(observation).all() or abs(pole_angle_x) > 0.2 or abs(pole_angle_y) > 0.2
        )
        reward = float(not terminated)
        info = {"reward_survive": reward}

        if self.render_mode == "human":
            self.render()

        return observation, reward, terminated, False, info

    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=noise_low, high=noise_high)
        qvel = self.init_qvel + self.np_random.uniform(size=self.model.nv, low=noise_low, high=noise_high)
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        return np.concatenate([self.data.qpos, self.data.qvel]).ravel()


if "CartpoleWorldDualCamera-v0" not in registry:
    register(
        id="CartpoleWorldDualCamera-v0",
        entry_point=f"{__name__}:CartpoleWorldDualCameraEnv",
        max_episode_steps=1000,
    )
