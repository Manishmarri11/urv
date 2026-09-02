import gymnasium as gym
import imageio.v2 as imageio

import cartpole_world_env_dual_camera  # noqa: F401  (registers "CartpoleWorldDualCamera-v0")


FPS = 30
DURATION_SECONDS = 15 #u can raise or lower it as you like it, depending on if u want a longer or shorter training vid.
NUM_FRAMES = FPS * DURATION_SECONDS


def main():
    env = gym.make(
        "CartpoleWorldDualCamera-v0",
        render_mode="rgb_array",
        width=480,
        height=320,
        camera_name="pole",  # convenience/demo view; use "pole" for the true egocentric observation feed to input into event cam; use "world" for convenient 3rd-person view
    )

    obs, info = env.reset(seed=0)
    frames = []

    for _ in range(NUM_FRAMES): #MAIN TRAINING LOOP
        action = env.action_space.sample() #need to feed our RL model with the Representation Learning output here somewhere.
        obs, reward, terminated, truncated, info = env.step(action)
        frames.append(env.render())

        if terminated or truncated:
            obs, info = env.reset()

    env.close()

    imageio.mimsave("cartpole_world_dual_camera.mp4", frames, fps=FPS)
    print(f"Saved cartpole_world_dual_camera.mp4 ({NUM_FRAMES} frames, {DURATION_SECONDS}s @ {FPS}fps)")


if __name__ == "__main__":
    main()
