from cv2 import cvtColor, COLOR_BGR2GRAY, resize, INTER_AREA
import numpy as np
from numpy import absolute
import gym

from PIL import Image
import random


def atari_enduro_processor(raw_frame):
    # convert input frame to gray scale
    gray_frame = cvtColor(raw_frame, COLOR_BGR2GRAY)
    # resize frame
    resized_frame = resize(gray_frame, (84, 105), interpolation=INTER_AREA)
    # return cropped frame
    return(resized_frame[0:84])

def atari_montezuma_processor(raw_frame):
    # # convert input frame to gray scale
    # gray_frame = cvtColor(raw_frame, COLOR_BGR2GRAY)
    # # resize frame
    # resized_frame = resize(gray_frame, (84, 84), interpolation=INTER_AREA)
    # # return cropped frame
    # return(resized_frame[15:99])

    X = np.array(Image.fromarray(raw_frame).convert('L')).astype('uint8')
    x = resize(X, (84,84))
    return x

def atari_pong_processor(raw_frame):
    # convert input frame to gray scale
    gray_frame = cvtColor(raw_frame, COLOR_BGR2GRAY)
    # resize frame
    resized_frame = resize(gray_frame, (84, 111), interpolation=INTER_AREA)
    # return cropped frame
    return(resized_frame[18:102])

def atari_spaceinvaders_processor(raw_frame):
    # convert input frame to gray scale
    gray_frame = cvtColor(raw_frame, COLOR_BGR2GRAY)
    # resize frame
    resized_frame = resize(gray_frame, (84, 97), interpolation=INTER_AREA)
    # return cropped frame
    return(resized_frame[6:90])

def atari_breakout_processor(raw_frame):
    # convert input frame to gray scale
    gray_frame = cvtColor(raw_frame, COLOR_BGR2GRAY)
    # resize frame
    resized_frame = resize(gray_frame, (84, 105), interpolation=INTER_AREA)
    # return cropped frame
    return(resized_frame[15:99])

def atari_mspacman_processor(raw_frame):
    # convert input frame to gray scale
    gray_frame = cvtColor(raw_frame, COLOR_BGR2GRAY)
    # resize frame
    resized_frame = resize(gray_frame, (84, 102), interpolation=INTER_AREA)
    # return cropped frame
    return(resized_frame[0:84])




class ProcessedAtariEnv(gym.Wrapper):
    """
    ***********************
    ** ProcessedAtariEnv **
    ***********************
        Class for handling some preprocessing techniques 
        (such as processing frames, actions or rewards) 
        for the openai gym environment wrappers

        -----------
        Parameters:
        -----------
            env:                      object;
                                      the basic (possibly already wrapped) OpenAI gym environment
            
            frame_processor:          callable;
                                      function for processing (eg. grayscale conversion, resizing, cropping) the raw frames

            action_processor:         callable;
                                      function for processing the raw actions

            reward_processor:         callable;
                                      function for processing the raw rewards

            neg_reward_terminal:      bool;
                                      variable indicating that a negative reward is considered as the end of an episode
            neg_reward_for_life_loss: bool;
                                      variable indicating that losing a life will yield a negative reward equal to the current episode score
    """
    
    def __init__(self, 
                 env = gym.make('PongDeterministic-v4'), 
                 frame_processor = atari_pong_processor,
                 action_processor = lambda x: x, 
                 reward_processor = lambda x: x,
                 neg_reward_terminal = False,
                 neg_reward_for_life_loss = False,
                 ale_states = None,
                 expert_reset_prob = 0.0,
                 shaky_hands=0.0,
                 use_sqil_rewards=False):
        gym.Wrapper.__init__(self, env)
        
        # custom environment processors
        self.frame_processor = frame_processor
        self.action_processor = action_processor
        self.reward_processor = reward_processor
        
        # reward options
        self.neg_reward_terminal = neg_reward_terminal
        self.neg_reward_for_life_loss = neg_reward_for_life_loss

        self.ale_states = ale_states
        self.expert_reset_prob = expert_reset_prob
        if self.expert_reset_prob > 0:
            assert self.ale_states is not None, "Must provide ale_states if expert_reset_prob > 0"
        self.shaky_hands = shaky_hands
        print("Expert Reset Prob", self.expert_reset_prob, "Shaky Hands", self.shaky_hands)
        
        # internal variables
        self._unprocessed_reward = 0.
        self._unprocessed_score = 0.
        self._unprocessed_frame = self.env.reset()

        self.use_sqil_rewards = use_sqil_rewards
        
        if self.use_sqil_rewards:
            print("SQIL: setting learner rewards to 0")
            self.reward_processor = lambda x: 0.0

        self.previous_action = 0
    
  
    def true_reset(self, validation=False):
        """Perform a true reset on OpenAI's EpisodicLifeEnv"""
        if not validation and random.random() < self.expert_reset_prob:
            random_expert_state = self.ale_states[np.random.randint(0, len(self.ale_states))]
            self.unwrapped.restore_state(random_expert_state)
            s, _, _, _ = self.unwrapped.step(self.unwrapped.action_space.sample())
            return (s)
        return(self.unwrapped.reset())
    
    def reset(self):
        """Reset the environment and return the processed frame"""
        self.previous_action = 0
        return(self.frame_processor(self.env.reset()))
    
    def step(self, action):
        """Perform one step in the processed environment"""
        if np.random.rand() <= 0.25:
            action = self.previous_action
        self.previous_action = action
        if random.random() < self.shaky_hands:
            action = self.unwrapped.action_space.sample()
        
        action = self.action_processor(action)
        frame, reward, done, info = self.env.step(action)
        
        # record unprocessed observations
        self._unprocessed_reward = reward
        self._unprocessed_frame = frame
        self._unprocessed_score += self._unprocessed_reward
        
        # give negative reward equal to the current total score for the end of an episode (if requested)
        if done:
            if self.neg_reward_for_life_loss:
                reward = -1 * absolute(self._unprocessed_score)
            self._unprocessed_score = 0.
        
        # end the episode when observing negative reward (if requested)
        if self.neg_reward_terminal:
            done = done or reward < 0
            
        # return the processed observations
        return(self.frame_processor(frame), self.reward_processor(reward), done, info)


        
