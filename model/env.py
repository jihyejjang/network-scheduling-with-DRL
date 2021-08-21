#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import simpy
import random
from agent import Agent
import time
import warnings
warnings.filterwarnings('ignore')

start = time.time()

from dataclasses import dataclass 

#TODO : class 를 반대로
@dataclass 
class Flow: #type(class1:besteffort,2:c&c,3:video,4:audio),Num,deadline,generate_time,depart_time,bits
    type_ : int = None
    num_ : int = None
    deadline_ : float = None #millisecond 단위, depart_time - arrival time < deadline 이어야 함
    generate_time_ : float = None #millisecond 단위
    depart_time_ : float =  None 
    bit_ : int = None
    met_ : bool = None
           

class GateControllEnv(object):
    
    def __init__(self):
        self.env = simpy.Environment()
        self.PRIORITY_QUEUE = 4 #How many priority queues = number of agents
        self.agents = []
        self.TDM_CYCLE = 10 #millisecond, LCM of periods
        #self.agents.append(Agent(self.PRIORITY_QUEUE,self.TDM_CYCLE)
        self.agents = [Agent(self.TDM_CYCLE) for _ in range(self.PRIORITY_QUEUE)]#4 of multiagents which is number of priority queue
        self.class_based_queues=[simpy.Store(self.env) for _ in range(self.PRIORITY_QUEUE)]#simpy store, queue 하나당 1Gbit
#         self.strict_priority_scheduler = [] #simpy resource
        
        #self.best_effort = 100 #best effort traffic (Even)
        self.best_effort = 200
        self.cnt4 = 0 #전송된 flow 개수 카운트
        #self.command_control = 40 #c&c flow number (Even)
        self.command_control = 80
        self.cnt1 = 0
        #self.video = 2 #video flow number (Even)
        self.video = 10 #video flow number (Even)
        self.cnt3 = 0
        #self.audio = 8 #audio flow number (Even)
        self.audio = 32
        self.cnt2 = 0

        # self.be_period = 0.002
        # self.cc_period = 0.005 # to 80
        # self.vd_period = 0.033
        # self.ad_period = 0.001 # milliseconds

        self.be_period = 0.002
        self.cc_period = 0.001  # to 80
        self.vd_period = 0.003
        self.ad_period = 0.002  # milliseconds
        
        
        self.total_episode = 0
        self.max_episode = 1000
        self.timeslot_size = 0.0005 #millisecond 단위, 0.5ms 마다 timeout
        self.simulation_duration = 0 
        self.timestep = 0

        self.action_size = 2**self.TDM_CYCLE

        self.state = np.zeros((4,2))
        self.actions = []
        self.reward_ = 0
#         self.flows = [] #전송한 flow들을 기록, deadline 맞췄는 지 확인용
        self.done = False
        self.next_state = np.zeros((4,2))
        self.log = pd.DataFrame(columns = ['Episode','Time','Final step','Score','Epsilon','Min_loss']  )

        self.start_time = 0 #episode 시작
        self.end_time = 0 #episode 끝났을 때

        self.minloss = []
        self.received_packet = np.zeros(3, dtype=np.int32)

        self.total_timestep = 0
        
        
    def reset(self): #initial state, new episode start
        self.class_based_queues = [simpy.Store(self.env) for _ in range(self.PRIORITY_QUEUE)]#simpy store, queue 하나당 1Gbit
#         self.flows = []
        self.state = np.zeros((4,2))
        self.actions = np.array([list(map(int,format(random.randrange(self.action_size),'0'+str(self.TDM_CYCLE)+'b'))) for _ in range(self.PRIORITY_QUEUE)])
        self.reward_ = 0
        self.done = False
        self.next_state = np.zeros((4,2))
        self.time = 0
        self.cnt1 = 0 #전송된 flow 개수 카운트
        self.cnt2 = 0
        self.cnt3 = 0
        self.cnt4 = 0
        self.received_packet = np.zeros(3, dtype=np.int32)
        self.timestep = 0

#type에 맞게 flow scheme을 설정하는 모듈

    def flow_generator(self,type_num,fnum,now,period=None): #flow structure에 맞게 flow생성, timestamp등 남길 수 있음
        
        f=Flow()
        
        #flow type, num으로 특정 가능
        
        if type_num == 4: #best effort
            f.type_ = type_num
            f.num_ = fnum
            f.deadline_ = 0.010
            f.generate_time_ = now
            f.depart_time_ = None
            f.bit_ = 32*8
            f.met_ = False
            
        elif type_num == 1: #c&c
            f.type_ = type_num
            f.num_ = fnum
            f.deadline_= period
            f.generate_time_ = now
            f.depart_time_= None
            f.bit_ = random.randrange(53,300)*8
            f.met_ = False


        elif type_num == 3: #video
            f.type_ = type_num
            f.num_ = fnum
            f.deadline_= 0.030
            f.generate_time_ = now
            f.depart_time_= None
            f.bit_ = 30*1500*8
            f.met_ = False


        else : #audio
            f.type_ = type_num
            f.num_ = fnum
            f.deadline_= random.choice([4,10])*0.001
            f.generate_time_ = now
            f.depart_time_= None
            f.bit_ = random.choice([128,256])*8 
            f.met_ = False

                        
        return f


#주기마다 flow를 생성하는 process

    def flow_generate_BE(self,env, store): #flow 생성 process(producer) 1, store = class_based_queue[0]
        for i in range(self.best_effort):
            yield env.timeout(self.be_period) #be 주기
            flow = self.flow_generator(4,i,self.env.now-self.end_time) #type,f_num,env.now,period=none(c&c)
            yield store.put(flow) 
            # flowname="best effort flow {:2d}".format(i)
            # #print("{} : {} 추가,{} flows left in queue 1".format(env.now, flowname , len(store.items)))
            self.cnt4+=1

    def flow_generate_CC(self,env,store): #cq[1] 
        for i in range(self.command_control):
            yield env.timeout(self.cc_period) #cc 주기
            flow = self.flow_generator(1,i,self.env.now-self.end_time,self.cc_period) #type,f_num,env.now,period=none(c&c)
            yield store.put(flow)
            # flowname="command&control flow {:2d}".format(i)
            # #print("{} : {} 추가,{} flows left in queue 2".format(env.now, flowname , len(store.items)))
            self.cnt1+=1
            
    def flow_generate_VD(self,env,store): #cq[2]
        for i in range(self.video):
            yield env.timeout(self.vd_period) #video 주기
            flow = self.flow_generator(3,i,self.env.now-self.end_time) #type,f_num,env.now,period=none(c&c)
            yield store.put(flow)
            # flowname="video flow {:2d}".format(i)
            # #print("{} : {} 추가,{} flows left in queue 3".format(env.now, flowname , len(store.items)))
            self.cnt3 +=1
            
    def flow_generate_AD(self,env,store): #cq[3]
        for i in range(self.audio):
            yield env.timeout(self.ad_period) #audio 주기
            flow = self.flow_generator(2,i,self.env.now-self.end_time) #type,f_num,env.now,period=none(c&c)
            yield store.put(flow)
            # flowname="audio flow {:2d}".format(i)
            # #print("{} : {} 추가,{} flows left in queue 4".format(env.now, flowname , len(store.items)))
            self.cnt2 +=1

    def action_choose(self):
        gcl = [list(self.agents[a].choose_action(self.state[a])) for a in range(4)] #GCL update
        #print("gcl",gcl)
        return np.array(gcl)

    def episode(self,env):
        flow = [] #reward 용 전송된 flow
        
        #actions (GCL)을 받아서 전송하면 flow(전송한 flow)에 넣고, 처리된 시간을 기록 -> 모든 flow가 전송완료(done)일 때까지
        
        for episode_num in range(self.max_episode): #max episode만큼 하나의 episode를 실행
            rewards_all = []

            timestep = 0
            epsilon = 0
            self.start_time = self.env.now
            self.total_episode +=1
            self.reset()
            
            #episode 시작 시 마다 flow generator process를 실행

            self.env.process(self.flow_generate_BE(self.env, self.class_based_queues[3]))
            self.env.process(self.flow_generate_CC(self.env, self.class_based_queues[0]))
            self.env.process(self.flow_generate_VD(self.env, self.class_based_queues[2]))
            self.env.process(self.flow_generate_AD(self.env, self.class_based_queues[1]))


            print ( "***에피소드" + str(self.total_episode) + "시작***" )

            while (self.done == False): #모든 flow가 전송 완료 (episode종료 조건)
                loss = []
                flow = []
                self.reward_ = 0
                #print ("--Tdm cycle" + str(timestep) + "시작--")

                self.state = self.next_state
                self.actions = self.action_choose()
                self.total_timestep += 1
                self.timestep += 1


                for t in range(self.TDM_CYCLE): #cycle
                    gcl = self.actions[:,t] #GCL에서 각 queue별 gate open 정보를 불러옴
                    #print ("Time : {}".format(env.now))

                    for n in range(len(gcl)): #queue
                        if (gcl[n] == '1') and (len(self.class_based_queues[n].items)): #gcl이 열려있고, flow가 존재하면
                            f=yield self.class_based_queues[n].get()
                            flow.append(f) #전송된 flow 추가
                            if (n != 3):
                                self.received_packet[n] += 1
                            f.depart_time_ = env.now-self.end_time
                            if ((f.depart_time_ - f.generate_time_) <= f.deadline_): 
                                f.met_ = True

                    yield env.timeout(self.timeslot_size)

                self.next_state, self.reward_ , self.done = self.step(flow)
                rewards_all.append(self.reward_)

                for a in range(len(self.agents)):
                    #print ("observe" ,a)
                    #print (self.state)
                    #print (self.actions[a])
                    #print("state,act,rewared,nextstate",self.state[a],self.actions[a],self.reward_,self.next_state[a])
                    self.agents[a].observation(self.state[a], self.actions[a], self.reward_, self.next_state[a], self.done)
                    epsilon = self.agents[a].epsilon_decay()
                    loss.append(self.agents[a].replay())
                    if (self.total_timestep%100 == 0):
                        self.agents[a].update_target_model()


            self.end_time = self.env.now
            print(loss)
            log_ = pd.DataFrame([(episode_num, self.end_time-self.start_time ,self.timestep ,np.sum(rewards_all),  epsilon, np.mean(loss))], columns = ['Episode','Time','Final step','Score','Epsilon','avg_loss'])
            self.minloss.append(min(loss))
            if (self.total_episode >= 100) and (min(self.minloss) >= np.mean(loss)):
                #self.minloss = min(loss)
                i=0
                for agent in self.agents:
                    i+=1
                    agent.model.save_model("./result/0819_1_train/"+ "agent[" + str(i) +"]"+ str(np.mean(loss)) + ".h5")
                self.log.to_csv("./result/0819_1_train/log_0819_train_1.csv")

            self.log = self.log.append(log_,ignore_index=True)
            
            print("Episode {p}, Score: {s}, Final Step: {t}, now: {n},epsilon: {e} , avg_loss: {m}".format(p=episode_num, s=np.sum(rewards_all) ,t=self.timestep , n= self.end_time - self.start_time , e=epsilon ,m=np.mean(loss)))

    def reward(self,state,flows):
        #state = [전송된 패킷, 생성해야 할 전체 패킷의 개수, 생성한 패킷의 개수]

        w1 = [6,3,2]
        w2 = 4
        w3 = 10

        # reward 1
        # 생성해야할 패킷 대비 전송된 패킷 : 전송된 패킷이 많아질 수록 점수를 많이 부여하기 때문에 빨리 전송할 수록 보상이 많이 주어짐
        #오류 발견 : class별로 차등 점수를 부여하지 않았음;
        reward1 = 0
        for i in range(3):
            reward1 += state[i][0] * w1[i]
        #print("reward1",reward1)
        #
        #reward 2
        # 생성한 패킷 대비 전송된 패킷 : 생성한 패킷이 모두 전송완료될때까지 panelty
        reward2 = 0
        for i in range(3):
            if state[i][1] != 0:
                #print("state", state[i][0],state[i][2])
                r = w2 * (1 - state[i][1])
                reward2 -= r*(3-i)*(3-i)
        #print("reward2", reward2)
        #
        # #reward 3
        # #높은 우선순위 Flow를 빨리 전송했을수록 가산점
        reward3 = 0
        for f in range(len(flows)):
            if (flows[f].met_ != True):
                # rn = flows[f].type_
                # reward3 += 4-rn #기간내에 전송 완료했을 때
            # else:
                rn = flows[f].type_
                reward3 -= w3*(4-rn) #기간내에 전송 못했을 때 : 큰 panelty
        # #print("reward3",reward3)
        return round(reward1+reward2+reward3)
            
    def step(self,flows):
        
        #state 관측

        state = np.zeros((4,2))

        try:
            state[0] = round(self.received_packet[0]/self.command_control,2) , round(self.received_packet[0]/self.cnt1,2)
            state[1] = round(self.received_packet[1]/self.command_control,2) , round(self.received_packet[0]/self.cnt1,2)
            state[2] = round(self.received_packet[2]/self.command_control,2) , round(self.received_packet[0]/self.cnt1,2)
            state[3] = state[0][0]*0.5+state[1][0]*0.3+state[2][0]*0.2, state[0][1]*0.5+state[1][1]*0.3+state[2][1]*0.2

        except: #devided by zero
            state[0] = round(self.received_packet[0] / self.command_control, 2), 0
            state[1] = round(self.received_packet[1] / self.command_control, 2), 0
            state[2] = round(self.received_packet[2] / self.command_control, 2), 0
            state[3] = state[0][0] * 0.5 + state[1][0] * 0.3 + state[2][0] * 0.2, 0

        #reward 측정
        rewards = self.reward(state,flows)
        
        #done 검사
        if (self.cnt1 == self.command_control) and (self.cnt2 == self.audio) and (self.cnt3 == self.video) and (self.cnt4 == self.best_effort) :
            done = True
        
        else :
            done = False
        
        return [state, rewards, done]
           
                
    def run(self):
        self.env.process(self.episode(self.env))
        self.env.run(until=100000)
        i = 0
        for agent in self.agents:
            i += 1
            agent.model.save_model("./result/0819_1_train/" + "agent[" + str(i) + "]" + str(min(self.minloss)) + ".h5")
        self.log.to_csv("./result/0819_1_train/log_0819_train_1.csv")
        
if __name__ =="__main__":
    env_ = GateControllEnv()
    env_.run()
                                         


# ### Bug
# 
# 1. none is not a generator
#     
#     env process에서 yield가 없었을 때 생기는 error
#     
#    
# ## task
# 
# 21.4.20 : 
# 
# -agent별로 dqn 생성해서 그에 맞는 GCL을 tdm cycle마다 예측
# 
# -예측한 데이터를 바탕으로 state, reward를 관측해서 agent에 넘겨줌
# 
# -발표 전까지 한번 train 돌려볼것
# 
# 
# 21.4.21 :
# 
# -simulation은 완성
# 
# -episode 별 csv에 로그 작성
# 
# -save model
# 
# 이후 :
# 
# -model tuning
# 
# -successive node 구현

# In[ ]:

#
# if __name__ =="__main__":
#     all_agents=[]
#
#
# # In[18]:
#
#
# import numpy as np
# import simpy
# from dataclasses import dataclass
#
#
# a=1
# b=1
#
# @dataclass
# class Flow: #type(class1:besteffort,2:c&c,3:video,4:audio),Num,deadline,generate_time,depart_time,byte
#     type_ : int = None
#     num_ : int = None
#     deadline_ : float = None #millisecond 단위, depart_time - arrival time < deadline 이어야 함
#     generate_time_ : float = None #millisecond 단위
#     depart_time_ : float =  None
#     bit_ : int = None
#
#
# def producer(env, store): #flow 생성, store = class_based_queue[0]
#
#     global a
#
#     ## 무한으로 진행되는 generator
#     ## 1초에 제품을 하나씩 생성해냄.
#     while (a<=30): #for문이 끝나면 process가 끝나버림
#         yield env.timeout(2) #be 주기
#         f= Flow()
#         f.type_ = 1
#         f.num_ = a
#         f.deadline_ = 5
#         f.generate_time_ = env.now
#         f.depart_time_ = None
#         f.bit_ = 300
#
#         flow_name = "best_effort_flow{:2d}".format(a)
#         ## store method: items, put, get
#         yield store.put(f)
#         ## 현재 창고에 있는 제품들 출력
#         print("{:6.2f}: best effort 플로우 추가 {}, items: {}".format(env.now, flow_name,len(store.items)))
#         a+=1
#
#
# def consumer(env,stores):
#
#     global a
#     global b
#     for i in range(10): #max episode
#
#         env.process(producer(env, stores[0]))
#         env.process(producer2(env, stores[1]))
#         done=False
#         while True: #특정 종료 조건이 발동되면 episode 종료
#
#             item1 = Flow()
#             item2 = Flow()
#             yield env.timeout(3) #time slot
#             print ("시간:{}".format(env.now))
#             #print("{:6.2f}: requesting flow".format(env.now))
#             ## 아래도 request와 마찬가지로 획득할때까지 기다렸다가 생기면 가져감
#             if (len(stores[0].items)>0): #store 0 에 flow가 있으면
#                 item1 = yield stores[0].get()
#                 print("{:6.2f}:{} best effort flow 전송".format(env.now, item1.num_))
#             if (len(stores[1].items)>0):
#                 item2 = yield stores[1].get()
#                 print("{:6.2f}:{} command control flow 전송".format(env.now, item2.num_))
#
#             #item = yield store.get() #store에 item이 더이상 들어오지 않으면 들어올때 까지 기다리기 때문에 진행이 안됨..
#
#             item1.depart_time_ = env.now
#             item2.depart_time_ = env.now
#             #item2 = yield stores[1].get() #이부분에서 GCL에 따른 if문 작성
#             #print("{} was waiting during {:6.2f}".format(name, waiting_time))
#
#             print("a,b",a,b)
#             if (a==31) and (b==41) :
#                 done=True
#
#             if (done == True):
#
#                 print ("done")
#
#                 a=1
#                 b=1
#
#                 break
#
#
#         print ("끝")
#
#
# def producer2(env,store):
#     global b
#
#     while (b<=40): #for문이 끝나면 process가 끝나버림
#         yield env.timeout(4) #c,c 주기
#         f= Flow()
#         f.type_ = 2
#         f.num_ = b
#         f.deadline_ = 10
#         f.generate_time_ = env.now
#         f.depart_time_ = None
#         f.bit_ = 330
#
#         flow_name = "command_control_flow{:2d}".format(b)
#         yield store.put(f)
#         print("{:6.2f}: command control 플로우 추가 {}, items: {}".format(env.now, flow_name, len(store.items)))
#         b+=1

#
# env = simpy.Environment()
# stores = [simpy.Store(env, capacity=20000) for _ in range(2)]
#
# #cons1 = env.process(consumer(env,stores[0]))
# #cons2 = env.process(consumer(env,stores[1]))
# env.process(consumer(env,stores))
# env.run(until=1000)
#
