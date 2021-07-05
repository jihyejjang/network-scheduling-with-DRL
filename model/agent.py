#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import random
import math
from collections import deque 
from itertools import product
from dqn import DQN
#from memory import Memory

class Agent(object):
    def __init__(self,priority_queue,tdm_cycle):
        self.tdm_cycle = tdm_cycle
        self.priority_queue = priority_queue
        self.model = DQN(priority_queue,tdm_cycle)
        self.state_size = priority_queue
        self.action_size = 2**tdm_cycle
        self.epsilon = 0.999
        self.epsilon_min = 0.01
        self.step = 0
        self.test = False
        self.step_decrease = 2500 #step_decrease마다 epsilon 감소, 전체 episode를 약 10번으로 나눠서 epsilon을 감소
        self.discount_factor = 0.8 #할인율. 1에 가까울 수록 미래에 받는 보상도 중요, 0에 가까울수록 즉각적인 보상이 중요
        self.memory = deque(maxlen=9999999999)
        #self.mse_loss = []
        
    def choose_action(self,state):
        if np.random.random() < self.epsilon: 
            return format(random.randrange(self.action_size),'0'+str(self.tdm_cycle)+'b')
        else:
            return format(np.argmax(self.model.predict_one(state)),'0'+str(self.tdm_cycle)+'b') #tdm cycle자리수의 이진수 변환 '111111'
        
    def epsilon_decay(self):
        
        if self.test:
            self.epsilon = self.epsilon_min
        
        else:
            if (self.step%self.step_decrease ==0):
                self.epsilon = max(self.epsilon_min,pow(self.epsilon,int(self.step/self.step_decrease+1)))
       
        self.step +=1 
        
        return self.epsilon
        
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

        x = np.zeros((batch_len, self.state_size)) 
        y = np.zeros((batch_len, self.action_size))
        errors = np.zeros(batch_len)

        for i in range(batch_len):
            o = batch[i]
            s = o[0]
            a = int(o[1],2) #binary action을 action id로 matching
            r = o[2]
            s_ = o[3]
            done = o[4]

            t = p[i]
            old_value = t[a]
            if done:
                t[a] = r
            else:
                t[a] = r + self.discount_factor * np.amax(pTarget_[i])

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




