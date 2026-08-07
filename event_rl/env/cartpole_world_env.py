import os
import numpy as np
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.envs.registration import register, registry
from gymnasium.spaces import Box
XML_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cartpole_world.xml")
DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 0,
    "distance": 8.5,
    "elevation": -10,
}
class CartpoleWorldEnv(MujocoEnv, utils.EzPickle):
    """Cartpole balancing task set in a textured world (checkered ground, gradient
    sky, distant skyline silhouette) instead of the bare-XML InvertedPendulum scene.
    Follows the same MujocoEnv subclassing pattern as Gymnasium's built-in
    InvertedPendulumEnv: 4-element observation (cart pos, pole angle, cart vel,
    pole angular vel), 1-element continuous force action, +1 reward per step the
    pole stays within 0.2 rad of upright.
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
        observation_space = Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float64)
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
    def step(self, action):
        self.do_simulation(action, self.frame_skip)
        observation = self._get_obs()
        terminated = bool(not np.isfinite(observation).all() or abs(observation[1]) > 0.2)
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
if "CartpoleWorld-v0" not in registry:
    register(
        id="CartpoleWorld-v0",
        entry_point=f"{__name__}:CartpoleWorldEnv",
        max_episode_steps=1000,
    )
