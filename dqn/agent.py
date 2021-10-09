#!/usr/bin/env python

import numpy as np
import random
from collections import deque
from dqn import DeepQNetwork
import warnings
warnings.filterwarnings('ignore')
#from memory import Memory

STATE_SIZE = 1
ACTION_SIZE = 8
EPSILON_MAX = 1
EPSILON_DECAY = 0.999
EPSILON_MIN = 0.01
DISCOUNT_FACTOR = 0.9  #할인율. 1에 가까울 수록 미래에 받는 보상도 중요, 0에 가까울수록 즉각적인 보상이 중요

class Agent():
    def __init__(self):
        self.model = DeepQNetwork()
        # self.episode = 0
        self.epsilon = EPSILON_MAX
        self.memory = deque(maxlen=99999)
        self.action_list = np.array(['1111111111','1111100000'
                        ,'1010101010','0000011111','1100110011'
                        ,'0000000001','0000100001','0000000000'])

    def reset(self):
        # self.episode +=1
        self._epsilon_decay_()
        return self.epsilon

    def choose_action(self,state):
        if np.random.random() < self.epsilon:
            return random.choice(self.action_list)
        else:
            n = self.model.predict_one(state)
            return self.action_list[np.argmax(n)]

    def _epsilon_decay_(self):
        if (self.epsilon > EPSILON_MIN):
            self.epsilon = self.epsilon * EPSILON_DECAY
        else :
            self.epsilon = EPSILON_MIN
        
    #agent memory
    def sample(self, n): #학습용 샘플 생성
        sample_batch = random.sample(self.memory, n)
        return sample_batch
    
    def observation(self, state, action, reward, next_state, done): #sample = [state,action,reward,next_state,done] 저장
        sample = [state, action, reward, next_state, done]
        self.memory.append(sample)

    def state_target(self, batch): #sample을 받아서 dqn의 input(state)과 target(predicted q-value)로 데이터셋을 나눠주는작업
        batch_len = len(batch)

        states = np.array([o[0] for o in batch])
        states_ = np.array([o[3] for o in batch]) #next state

        p = self.model.predict(states) #model predict with state
        pTarget_ = self.model.predict(states_, target=True)#target_model predict with next state

        x = np.zeros((batch_len, STATE_SIZE))
        y = np.zeros((batch_len, ACTION_SIZE))
        #errors = np.zeros(batch_len)

        for i in range(batch_len):
            o = batch[i]
            s = o[0]
            a = np.where(self.action_list == o[1])
            r = o[2]
            #ns = o[3]
            done = o[4]

            t = p[i]
            #old_value = t[a]
            if done:
                t[a] = r
            else:
                t[a] = r + DISCOUNT_FACTOR * np.amax(pTarget_[i])

            x[i] = s
            y[i] = t

        return [x, y]

    # build the replay buffer 
    def replay(self):
        
        batch_size = 32
        if len(self.memory) < batch_size: #buffer에 저장된 memory가 buffer의 총 batch_size보다 작다면 return
            return 
        
        batch = self.sample(batch_size)
        x, y = self.state_target(batch)
        min_loss = self.model.train(x, y)
        
        return min_loss

    def update_target_model(self):
        self.model.update_target_model()


# In[ ]:




