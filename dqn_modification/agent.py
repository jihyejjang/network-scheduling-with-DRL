#!/usr/bin/env python

import numpy as np
import random
from random import randrange
from collections import deque
from dqn import DeepQNetwork
import warnings

warnings.filterwarnings('ignore')
# from memory import Memory

BATCH = 64
PRIORITY_QUEUE = 2
STATE = 3
STATE_SIZE = STATE * PRIORITY_QUEUE
GCL_LENGTH = 3
ACTION_SIZE = 2 ** (PRIORITY_QUEUE * GCL_LENGTH)
EPSILON_MAX = 1
EPSILON_DECAY = 0.9993
EPSILON_MIN = 0.01
DISCOUNT_FACTOR = 0.3  # 할인율. 1에 가까울 수록 미래에 받는 보상도 중요, 0에 가까울수록 즉각적인 보상이 중요

def number_to_action(action_id): #number -> binary gcl code
    b_id = format(action_id, '06b')
    action_ = np.array(list(map(int, b_id)))
    return action_.reshape((PRIORITY_QUEUE, GCL_LENGTH))

def action_to_number(action):
    action_ = action.flatten()
    bin = ''
    for a in action_:
        bin += str(a)
    return int(bin, 2)

class Agent(): #one node agent
    def __init__(self):
        self.model = DeepQNetwork()
        # self.episode = 0
        self.epsilon = EPSILON_MAX
        self.memory = deque(maxlen=999999999999999)
        # self.action_list =
        #     np.array(['1111111111', '1111100000'
        #                                 , '0000011111', '0000000000'])
        # self.action_list = np.array(['1111111111', '1111100000'
        #                             , '1010101010', '0000011111', '1100110011'
        #                             , '0000000001', '0000100001', '0000000000'])

    def reset(self):
        # self.episode +=1
        self._epsilon_decay_()
        return self.epsilon

    def choose_action(self, state):
        if np.random.random() < self.epsilon:
            id = randrange(ACTION_SIZE)
            action = number_to_action(id)
            return action
        else:
            n = self.model.predict_one(state)
            id = np.argmax(n)
            action = number_to_action(id)
            return action

    def _epsilon_decay_(self):
        if self.epsilon > EPSILON_MIN:
            self.epsilon = self.epsilon * EPSILON_DECAY
        else:
            self.epsilon = EPSILON_MIN

    # agent memory
    def sample(self):  # 학습용 샘플 생성
        sample_batch = random.sample(self.memory, BATCH)
        return sample_batch

    def observation(self, state, action_id, reward, next_state, done):  # sample = [state,action,reward,next_state,done] 저장
        #action = "".join(action)
        sample = [state, action_id, reward, next_state, done]
        self.memory.append(sample)

    def state_target(self, batch):  # sample을 받아서 dqn의 input(state)과 target(predicted q-value)로 데이터셋을 나눠주는작업

        states = np.array([o[0] for o in batch])
        states_ = np.array([o[3] for o in batch])  # next state

        p = self.model.predict(states)  # model predict with state
        pTarget_ = self.model.predict(states_, target=True)  # target_model predict with next state

        x = np.zeros((BATCH, STATE_SIZE))
        y = np.zeros((BATCH, ACTION_SIZE))
        # errors = np.zeros(batch_len)

        for i in range(BATCH):
            o = batch[i]  # batch=[state, action, reward, next_state, done]
            s = o[0]
            a = o[1]
            #a = action_to_number(o[1])
            r = o[2]
            # ns = o[3]
            done = o[4]

            t = p[i]
            # old_value = t[a]
            if done:
                t[a] = r
            else:
                t[a] = r + DISCOUNT_FACTOR * np.amax(pTarget_[i])

            x[i] = s
            y[i] = t

        return [x, y]

    # build the replay buffer 
    def replay(self):

        if len(self.memory) < BATCH:
            return 99999

        batch = self.sample()
        x, y = self.state_target(batch)
        min_loss = self.model.train(x, y)

        return min_loss

    def update_target_model(self):
        self.model.update_target_model()
