# rl_atari_boxing

🥊 Atari Boxing PPO Agent

This project trains a reinforcement learning agent using Proximal Policy Optimization (PPO) to play the classic Atari game Boxing via the OpenAI Gym ALE environment.

Setup

Ensure you're using Python 3.10 and install dependencies via:

pip install -r requirements.txt

Once installed, run:

AutoROM --accept-license

This downloads the Atari ROMs required to train the agent.

Training the Agent

Train the PPO agent by running:

python boxing_rl_agent.py

By default, the agent will:

Train for 10,000 timesteps

Use CNN-based policy (CnnPolicy)

Save the final model and gameplay video in the models/ directory

Log training metrics to logs/ (TensorBoard-compatible)

Agent Behavior

The agent controls the white boxer (Player 1).

Opponent (black boxer) is the built-in Atari AI.

The environment returns visual observations, which are processed and stacked to form the agent's input.

Output

After training, a sample gameplay is saved as an MP4 video (requires imageio-ffmpeg backend).

Dependencies

All dependencies (including exact versions) are listed in requirements.txt.

Monitoring

To monitor training progress:

tensorboard --logdir logs/

Notes

Environment used: ALE/Boxing-v5

Frameworks: Stable-Baselines3, Gym, ALE-Py

Compatible with Gym v0.26.x API via a wrapper (Gym26Wrapper)

Future Work

Reward shaping for smarter aggression

Hyperparameter tuning

Frame-skipping or RAM-based observation
