import gymnasium as gym
import imageio.v2 as imageio
import cartpole_world_env  # noqa: F401  (registers "CartpoleWorld-v0")
FPS = 30
DURATION_SECONDS = 5
NUM_FRAMES = FPS * DURATION_SECONDS
def main():
    env = gym.make(
        "CartpoleWorld-v0",
        render_mode="rgb_array",
        width=480,
        height=320,
        camera_name="side",
    )
    obs, info = env.reset(seed=0)
    frames = []
    for _ in range(NUM_FRAMES):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        frames.append(env.render())
        if terminated or truncated:
            obs, info = env.reset()
    env.close()
    imageio.mimsave("cartpole_world.mp4", frames, fps=FPS)
    print(f"Saved cartpole_world.mp4 ({NUM_FRAMES} frames, {DURATION_SECONDS}s @ {FPS}fps)")
if __name__ == "__main__":
    main()
