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
from load_data import LoadAtariHeadData, LoadAtariRNDData

import random



import argparse

parser = argparse.ArgumentParser(
                    prog = 'MontezumaDemoExperiment',
                    description = 'Trains a DQfD agent for the Montezuma Revenge Atari game',
                    epilog = 'apple')

# parser.add_argument('filename')           # positional argument
parser.add_argument('-p', '--expert-reset-prob', type=float)      # option that takes a value
parser.add_argument('-e', '--nrpi',
                    action='store_true')  # on/off flag
parser.add_argument('-f', '--experiment-name', type=str)
parser.add_argument('-s', '--use-sqil-rewards', action='store_true')
parser.add_argument('-d', '--shaky-hands', type=float, default=0.0)
parser.add_argument("--data", type=str, default="head")
parser.add_argument('--seed', default=12, type=int) 
parser.add_argument('--save_path', default="experiments/montezuma_standard_experiment/", type=str)
args = parser.parse_args()

use_sqil_rewards = args.use_sqil_rewards

if args.nrpi:
    print('NRPI: Loading ALE States')
    ale_states_path = "/home/gokul/dev/efficient_irl/HyQ/MRevenge/offline_data/easy/ales.npy"
    # loaded_zip = np.load(ale_states_path, allow_pickle=True)
    ale_states = np.load(ale_states_path, allow_pickle=True).reshape([-1,])

    expert_reset_prob = args.expert_reset_prob if args.expert_reset_prob is not None else 1.0
else:
    ale_states = None
    expert_reset_prob = 0.0



np.random.seed(args.seed)
random.seed(args.seed)
tf.random.set_seed(args.seed)


#create environment
frame_processor = atari_montezuma_processor
game_id = 'MontezumaRevengeNoFrameskip-v4'
game_name = 'montezuma_revenge'
env = make_atari(game_id)
env = wrap_deepmind(env)
env = ProcessedAtariEnv(env, frame_processor, reward_processor = lambda x: np.sign(x) * np.log(1 + np.abs(x)), ale_states=ale_states, expert_reset_prob=expert_reset_prob, shaky_hands=args.shaky_hands, use_sqil_rewards=use_sqil_rewards)

env.seed(args.seed)

# additional env specific parameters
frame_shape = env.reset().shape
frame_skip = 4
num_stacked_frames = 4
num_actions = env.action_space.n

# replay parameters
batch_size = 32
max_frame_num = 2**20
prioritized_replay = True
prio_coeff = 0.4
is_schedule = [0.6, 1.0, 1500000]
replay_epsilon = 0.001
expert_epsilon = 1.0
memory_restore_path = None

# network training parameters
dueling = True
double_q = True
lr_schedule = [[0.0000625, 0.0000625, 10000000]]
optimizer = Adam
discount_factor = 0.99
n_step = 10
one_step_weight = 1.0/3.0
n_step_weight = 1.0/3.0
expert_weight = 1.0/3.0
l2_weight = 0.00001
large_margin_coeff = 0.8
model_restore_path = None #'/home/gokul/dev/efficient_irl/gks/DQfD/experiments/best_validation_model_710.h5'

# network architecture
conv_layers = {'filters': [32, 64, 64, 1024],
               'kernel_sizes': [8, 4, 3, 7],
               'strides': [4, 2, 1, 1],
               'paddings': ['valid' for _ in range(4)],
               'activations': ['relu' for _ in range(4)],
               'initializers': [initializers.VarianceScaling(scale = 2.0) for _ in range(4)],
               'names': ['conv_%i'%(i) for i in range(1,5)]}
dense_layers = None

# exploration parameters
eps_schedule = [[0.25, 0.1, 250000],
                [0.1, 0.01, 5000000],
                [0.01, 0.001, 5000000]]

# training session parameters
target_interval = 10000
warmup_steps = 50000
pretrain_steps = 500000
learning_interval = 4
num_steps = 15000000
num_episodes = 10000
max_steps_per_episode = 18000
output_freq = 1000
save_freq = 500
store_memory = True
save_path = "experiments/montezuma_demo_experiment"


# create replay memory
memory = PrioritizedExperienceReplay(frame_shape = frame_shape,
                                     max_frame_num = max_frame_num,
                                     num_stacked_frames = num_stacked_frames,
                                     batch_size = batch_size,
                                     prio_coeff = prio_coeff,
                                     is_schedule = is_schedule,
                                     epsilon = replay_epsilon,
                                     restore_path = memory_restore_path)

# expert memory
# data_loader = LoadAtariHeadData(game_name = game_name, frame_processor = frame_processor)
# expert_memory = data_loader.demonstrations_to_per(max_frame_num = max_frame_num,
#                                                   num_stacked_frames = num_stacked_frames,
#                                                   frame_shape = frame_shape,
#                                                   batch_size = batch_size,
#                                                   prio_coeff = prio_coeff,
#                                                   is_schedule = is_schedule,
#                                                   epsilon = expert_epsilon,
#                                                   recompute_demonstrations = True,
#                                                   only_highscore = False,
#                                                   frame_skip = frame_skip)
# expert_memory = PrioritizedExperienceReplay(frame_shape = frame_shape,
#                                      max_frame_num = max_frame_num,
#                                      num_stacked_frames = num_stacked_frames,
#                                      batch_size = batch_size,
#                                      prio_coeff = prio_coeff,
#                                      is_schedule = is_schedule,
#                                      epsilon = replay_epsilon,
#                                      restore_path = "AtariHEADArchives")


# buffer_path = args.offline_buffer_path

# states_data = np.load("{}/states.npy".format(buffer_path))
# actions_data = np.load("{}/actions.npy".format(buffer_path))
# rewards_data = np.load("{}/rewards.npy".format(buffer_path))
# dones_data =  np.load("{}/dones.npy".format(buffer_path)).astype(np.uint8)

# priorities = np.ones(actions_data.shape[0], dtype = np.single)

# expert_memory = PrioritizedExperienceReplay(
#                                     max_frame_num = max_frame_num,
#                                     num_stacked_frames = 4,
#                                     batch_size = batch_size,
#                                     frames = states_data,
#                                     actions = actions_data,
#                                     rewards = rewards_data,
#                                     priorities = priorities, 
#                                     episode_endings = dones_data,
#                                     prio_coeff = prio_coeff,
#                                     is_schedule = is_schedule,
#                                     epsilon = expert_epsilon)

if args.data == "head":
    data_loader = LoadAtariHeadData(game_name = game_name, frame_processor = frame_processor, use_sqil_rewards=use_sqil_rewards)
    expert_memory = data_loader.demonstrations_to_per(max_frame_num = max_frame_num,
                                                    num_stacked_frames = num_stacked_frames,
                                                    frame_shape = frame_shape,
                                                    batch_size = batch_size,
                                                    prio_coeff = prio_coeff,
                                                    is_schedule = is_schedule,
                                                    epsilon = expert_epsilon,
                                                    recompute_demonstrations = False,
                                                    only_highscore = False,
                                                    frame_skip = frame_skip)
elif args.data == "rnd":
    data_loader = LoadAtariRNDData(game_name = game_name, frame_processor = frame_processor, use_sqil_rewards=use_sqil_rewards)
    expert_memory = data_loader.demonstrations_to_per(max_frame_num = max_frame_num,
                                                  num_stacked_frames = num_stacked_frames,
                                                  frame_shape = frame_shape,
                                                  batch_size = batch_size,
                                                  prio_coeff = prio_coeff,
                                                  is_schedule = is_schedule,
                                                  epsilon = expert_epsilon,
                                                  recompute_demonstrations = False,
                                                  only_highscore = False,
                                                  frame_skip = frame_skip)
elif args.data == "yuda":
    buffer_path = '/home/gokul/dev/efficient_irl/HyQ/MRevenge/offline_data/easy'
    states_data = np.load("{}/states.npy".format(buffer_path))
    actions_data = np.load("{}/actions.npy".format(buffer_path))
    rewards_data = np.load("{}/rewards.npy".format(buffer_path))

    if use_sqil_rewards:
        print("SQIL: setting expert rewards to 1")
        rewards_data = np.ones_like(rewards_data)

    dones_data =  np.load("{}/dones.npy".format(buffer_path)).astype(np.uint8)
    print(dones_data.shape)

    priorities = np.ones(actions_data.shape[0], dtype = np.single)

    expert_memory = PrioritizedExperienceReplay(
                                        max_frame_num = max_frame_num,
                                        num_stacked_frames = 4,
                                        batch_size = batch_size,
                                        frames = states_data,
                                        actions = actions_data,
                                        rewards = rewards_data,
                                        priorities = priorities, 
                                        episode_endings = dones_data,
                                        prio_coeff = prio_coeff,
                                        is_schedule = is_schedule,
                                        epsilon = expert_epsilon)
else:
    print(f"Specified Data {args.data} is not supported.")


# create policy network
policy_network = DeepQNetwork(in_shape = (num_stacked_frames, *frame_shape),
                              conv_layers = conv_layers,
                              dense_layers = dense_layers,
                              num_actions = num_actions,
                              optimizer = optimizer,
                              lr_schedule = lr_schedule,
                              dueling = dueling,
                              one_step_weight = one_step_weight,
                              n_step_weight = n_step_weight,
                              expert_weight = expert_weight)


if model_restore_path is not None:
    policy_network.model = load_model(model_restore_path)

# create target network
target_network = DeepQNetwork(in_shape = (num_stacked_frames, *frame_shape),
                              conv_layers = conv_layers,
                              dense_layers = dense_layers,
                              num_actions = num_actions,
                              optimizer = optimizer,
                              lr_schedule = lr_schedule,
                              dueling = dueling,
                              one_step_weight = one_step_weight,
                              n_step_weight = n_step_weight,
                              expert_weight = expert_weight)

# if model_restore_path is not None:
#     target_network.model.load_weights(model_restore_path, by_name = True)

# create agent
agent = EpsAnnDQNAgent(env = env,
                       memory = memory,
                       policy_network = policy_network, 
                       target_network = target_network,
                       num_actions = num_actions,
                       frame_shape = frame_shape,
                       discount_factor = discount_factor,
                       save_path = save_path,
                       eps_schedule = eps_schedule,
                       double_q = double_q,
                       n_step = n_step,
                       expert_memory = expert_memory,
                       prioritized_replay = prioritized_replay)


agent.policy_network.model.save(save_path + "/trained_models/initial_model.h5")

# train the agent
agent.train(num_episodes = num_episodes,
            num_steps = num_steps,
            max_steps_per_episode = max_steps_per_episode,
            warmup_steps = warmup_steps,
            pretrain_steps = pretrain_steps,
            target_interval = target_interval,  
            learning_interval = learning_interval,
            output_freq = output_freq,
            save_freq = save_freq,
            store_memory = store_memory)
