# compile cython modules
import os
os.system('python experience_replay_setup.py build_ext --inplace')

# load dependencies
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
tf.config.experimental.set_memory_growth(physical_devices[0], True)

from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras import initializers

import gym

import numpy as np

from deep_q_agents import EpsAnnDQNAgent
from deep_q_networks import DeepQNetwork
from experience_replay import PrioritizedExperienceReplay
from atari_preprocessing import atari_montezuma_processor, ProcessedAtariEnv
from openai_baseline_wrappers import make_atari, wrap_deepmind
from load_data import LoadAtariHeadData

def get_env(expert_reset_prob=0.5, nrpi=False, shaky_hands=0.0):

    if nrpi:
        ale_states_path = "mz_demos_ale.npz"
        loaded_zip = np.load(ale_states_path, allow_pickle=True)
        ale_states = loaded_zip["ale"].reshape([-1,])

        # expert_reset_prob = args.expert_reset_prob if args.expert_reset_prob is not None else 0.0
    else:
        ale_states = None
        expert_reset_prob = 0.0

    #create environment
    frame_processor = atari_montezuma_processor
    game_id = 'MontezumaRevengeNoFrameskip-v4'
    game_name = 'montezuma_revenge'
    env = make_atari(game_id)
    env = wrap_deepmind(env)
    env = ProcessedAtariEnv(env, frame_processor, reward_processor = lambda x: np.sign(x) * np.log(1 + np.abs(x)), ale_states=ale_states, expert_reset_prob=expert_reset_prob, shaky_hands=shaky_hands)
    return env