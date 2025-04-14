import os
import time
import gym
import cv2
import numpy as np
import imageio
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecTransposeImage
from stable_baselines3.common.monitor import Monitor

class Gym26Wrapper(gym.Wrapper):
    def reset(self, **kwargs):
        obs, _ = self.env.reset(**kwargs)
        return obs

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs, reward, terminated or truncated, info

timestamp = int(time.time())
models_dir = f"models/{timestamp}/"
logdir = f"logs/{timestamp}/"
video_path = f"{models_dir}/boxing_playback.mp4"
os.makedirs(models_dir, exist_ok=True)
os.makedirs(logdir, exist_ok=True)

raw_env = gym.make("ALE/Boxing-v5", render_mode="rgb_array")
raw_env = Gym26Wrapper(raw_env)
raw_env = Monitor(raw_env)
env = DummyVecEnv([lambda: raw_env])
env = VecFrameStack(env, n_stack=4)
env = VecTransposeImage(env)

model = PPO("CnnPolicy", env, verbose=1, tensorboard_log=logdir)

TIMESTEPS = 100_000
ITERS = 5
for i in range(ITERS):
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="PPO")
    model.save(f"{models_dir}/{TIMESTEPS * (i + 1)}")

model_path = f"{models_dir}/{TIMESTEPS * ITERS}.zip"
model = PPO.load(model_path, env=env)

obs = env.reset()
done = False
frames = []

while not done:
    action, _ = model.predict(obs)
    obs, _, done, _ = env.step(action)
    frame = raw_env.render() 
    frames.append(frame)
    cv2.imshow("Boxing Agent", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
raw_env.close()

print(f"Saving gameplay to {video_path}...")
imageio.mimsave(video_path, frames, fps=30)
print("Video saved.")
