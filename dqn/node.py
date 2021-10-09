#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import simpy
import random
from agent import Agent
from dataclasses import dataclass
import time
import warnings
warnings.filterwarnings('ignore')

PRIORITY_QUEUES = 4
ACTION_SIZE = 8
GCL_LENGTH = 10
COMMAND_CONTROL = 30
AUDIO = 8
VIDEO = 2
BEST_EFFORT = 100
CC_PERIOD = 5  # milliseconds
AD_PERIOD = 1
VD_PERIOD = 33
BE_PERIOD = 2
TIMESLOT_SIZE = 0.4  # milliseconds
BITS_PER_MS = 1000000  # 1ms당 전송하는 비트 수 (1백만 = 1Mbits)
MAX_BURST = 364448  # maximum burst는 약 0.364448밀리초만에 전송됨
LEARNING_NODE = 1


@dataclass
class Flow:  # type(class1:cc,2:ad,3:vd,4:be),Num,deadline,generate_time,depart_time,bits
    type_: int = None
    num_: int = None
    deadline_: float = None  # millisecond 단위, arrival time - generated time < deadline 이어야 함
    generated_time_: float = None  # millisecond 단위
    node_arrival_time_: list = None
    node_departure_time_: list = None
    arrival_time_: float = None
    bits_: int = None
    met_: bool = None


class Node():

    def __init__(self, datapath_id, env):
        # TODO: 학습 노드 구현
        self.datapath_id = datapath_id
        self.agents = [Agent() for _ in range(PRIORITY_QUEUES)]  # 4 of multiagents which is number of priority queue
        self.class_based_queues = [simpy.Store(env) for _ in range(PRIORITY_QUEUES)]
        self.learning = False
        if (self.datapath_id == LEARNING_NODE):
            self.learning = True
        self.timeslots = 0
        self.state = np.zeros((4, 2))
        self.actions = np.array([list(map(int,'1111111111')) for _ in range(PRIORITY_QUEUES)])
        self.reward_ = 0
        self.done = False
        self.next_state = np.zeros((4, 2))
        self.start_time = env.now  # episode 시작
        self.end_time = 0  # episode 끝났을 때

        self.avgloss = []
        self.received_packet = np.zeros(3, dtype=np.int32)

        self.total_timestep = 0

    def reset(self, env):  # initial state, new episode start
        self.class_based_queues = [simpy.Store(env) for _ in range(PRIORITY_QUEUES)]
        self.state = np.zeros((4, 2))
        self.actions = np.array([list(map(int,'1111111111')) for _ in range(PRIORITY_QUEUES)])
        self.reward_ = 0
        self.done = False
        self.next_state = np.zeros((4, 2))
        self.timeslots = 0

        self.received_packet = np.zeros(3, dtype=np.int32)

        # epsilon_decay
        self.agents[0].reset()
        self.agents[1].reset()
        self.agents[2].reset()
        self.agents[3].reset()

    def packet_in(self, time, flow):
        self.class_based_queues[flow.type_-1].put(flow)
        flow.node_arrival_time_.append(time)

    def gcl_update(self):  # GCL update (cycle : 0.4*10)
        gcl_ = [list(self.agents[a].choose_action(self.state[a])) for a in range(PRIORITY_QUEUES)]
        gcl = [list(map(int,g)) for g in gcl_]
        self.actions = np.array(gcl)

    def packet_out(self, env, trans_dict, t):  # 매 timeslot마다 발생하는 일
        gcl = self.actions[:, t]
        bits_sum = 0

        for q in range(PRIORITY_QUEUES):
            if (gcl[q] == '1') and len(self.class_based_queues[q].items):
                bits_sum += f.bits_
                if bits_sum > MAX_BURST: break
                f = yield self.class_based_queues[q].get()
                f.node_departure_time_.append(env.now)
                yield trans_dict.put(f)

                # def packet_out(self, env, trans_dict):  # 매 timeslot마다 발생하는 일
                #     t = 0
                #     while True:
                #         yield env.timeout(TIMESLOT_SIZE)
                #         gcl = self.actions[:, t % GCL_LENGTH]
                #         t += 1
                #         q = 0
                #         bits_sum = 0
                #
                #         while (bits_sum <= MAX_BURST) and (q < 4):
                #             if (gcl[1] == '1') and len(self.class_based_queues[q].items):
                #                 f = self.class_based_queues[q].get()
                #                 f.node_departure_time_ = time.time()
                #                 yield trans_dict.put(f)
                #                 bits_sum += f.bits_
                #             else:
                #                 q += 1

                #     if (gcl[1] == '1') and (len(self.class_based_queues[q].items) != 0):
                #         flow = yield self.class_based_queues[q].get()
                #         self.strict_priority_queue.put(flow)
                # self.strict_priority_queue.sorted(self.strict_priority_queue, key=lambda flow: flow.type_)  # class에 따라 정렬

                # 0.4ms동안 내보낼 수 있는 bits 만큼 패킷을 내보냄
                # for p in range(len(self.strict_priority_queue.items)):

                # while (len(self.class_based_queues[q].items != 0)):
                #     if bits_sum <= MAX_BURST:
                #
                #         f = self.strict_priority_queue.get()
                #         f.node_departure_time_ = time.time()
                #         yield f
                #         bits_sum += f.bits_
                #     else: #maximum bits만큼 내보냈을 때 packet out을 종료한다
                #         return
                # queue의 모든 packet을 내보냈을 때도 packet out을 종료한다
            #
            #     for t in range(self.TDM_CYCLE):  # cycle
            #         gcl = self.actions[:, t]  # GCL에서 각 queue별 gate open 정보를 불러옴
            #         # print ("Time : {}".format(env.now))
            #
            #         for n in range(len(gcl)):  # queue
            #             if (gcl[n] == '1') and (len(self.class_based_queues[n].items)):  # gcl이 열려있고, flow가 존재하면
            #                 f = yield self.class_based_queues[n].get()
            #                 flow.append(f)  # 전송된 flow 추가
            #                 if (n != 3):
            #                     self.received_packet[n] += 1
            #                 f.depart_time_ = env.now - self.end_time
            #                 if ((f.depart_time_ - f.generate_time_) <= f.deadline_):
            #                     f.met_ = True
            #
            #         yield env.timeout(self.timeslot_size)
            #
            #     self.next_state, self.reward_, self.done = self.step(flow)
            #     rewards_all.append(self.reward_)
            #
            #     for a in range(len(self.agents)):
            #         # print ("observe" ,a)
            #         # print (self.state)
            #         # print (self.actions[a])
            #         # print("state,act,rewared,nextstate",self.state[a],self.actions[a],self.reward_,self.next_state[a])
            #         self.agents[a].observation(self.state[a], self.actions[a], self.reward_, self.next_state[a],
            #                                    self.done)
            #         epsilon = self.agents[a].epsilon_decay()
            #         loss.append(self.agents[a].replay())
            #         if (self.total_timestep % 100 == 0):
            #             self.agents[a].update_target_model()
            #
            # self.end_time = self.env.now
            # print(loss)
            # log_ = pd.DataFrame([(episode_num, self.end_time - self.start_time, self.timestep, np.sum(rewards_all),
            #                       epsilon, np.mean(loss))],
            #                     columns=['Episode', 'Time', 'Final step', 'Score', 'Epsilon', 'avg_loss'])
            # self.minloss.append(min(loss))
            # if (self.total_episode >= 100) and (min(self.minloss) >= np.mean(loss)):
            #     # self.minloss = min(loss)
            #     i = 0
            #     for agent in self.agents:
            #         i += 1
            #         agent.model.save_model(
            #             "./result/0819_1_train/" + "agent[" + str(i) + "]" + str(np.mean(loss)) + ".h5")
            #     self.log.to_csv("./result/0819_1_train/log_0819_train_1.csv")
            #
            # self.log = self.log.append(log_, ignore_index=True)
            #
            # print(
            #     "Episode {p}, Score: {s}, Final Step: {t}, now: {n},epsilon: {e} , avg_loss: {m}".format(p=episode_num,
            #                                                                                              s=np.sum(
            #                                                                                                  rewards_all),
            #                                                                                              t=self.timestep,
            #                                                                                              n=self.end_time - self.start_time,
            #                                                                                              e=epsilon,
            #                                                                                              m=np.mean(
            #                                                                                                  loss)))
    #
    # def reward(self, state, flows):
    #     # state = [전송된 패킷, 생성해야 할 전체 패킷의 개수, 생성한 패킷의 개수]
    #
    #     w1 = [6, 3, 2]
    #     w2 = 4
    #     w3 = 10
    #
    #     # reward 1
    #     # 생성해야할 패킷 대비 전송된 패킷 : 전송된 패킷이 많아질 수록 점수를 많이 부여하기 때문에 빨리 전송할 수록 보상이 많이 주어짐
    #     # 오류 발견 : class별로 차등 점수를 부여하지 않았음;
    #     reward1 = 0
    #     for i in range(3):
    #         reward1 += state[i][0] * w1[i]
    #     # print("reward1",reward1)
    #     #
    #     # reward 2
    #     # 생성한 패킷 대비 전송된 패킷 : 생성한 패킷이 모두 전송완료될때까지 panelty
    #     reward2 = 0
    #     # for i in range(3):
    #     #     if state[i][1] != 0:
    #     #         #print("state", state[i][0],state[i][2])
    #     #         r = w2 * (1 - state[i][1])
    #     #         reward2 -= r*(3-i)*(3-i)
    #     for i in range(3):
    #         # if state[i][1] != 0:
    #         # print("state", state[i][0],state[i][2])
    #         # r = w2 * (1 - state[i][1])
    #         reward2 += state[i][1] * w1[i]
    #     # print("reward2", reward2)
    #     #
    #     # #reward 3
    #     # #높은 우선순위 Flow를 빨리 전송했을수록 가산점
    #     reward3 = 0
    #     for f in range(len(flows)):
    #         if (flows[f].met_ != True):
    #             # rn = flows[f].type_
    #             # reward3 += 4-rn #기간내에 전송 완료했을 때
    #             # else:
    #             rn = flows[f].type_
    #             reward3 -= w3 * (4 - rn)  # 기간내에 전송 못했을 때 : 큰 panelty
    #     # #print("reward3",reward3)
    #     return round(reward1 + reward2 + reward3)
    #
    # def step(self, flows):
    #
    #     # state 관측
    #
    #     state = np.zeros((4, 2))
    #
    #     try:
    #         state[0] = round(self.received_packet[0] / self.command_control, 2), round(
    #             self.received_packet[0] / self.cnt1, 2)
    #         state[1] = round(self.received_packet[1] / self.command_control, 2), round(
    #             self.received_packet[0] / self.cnt1, 2)
    #         state[2] = round(self.received_packet[2] / self.command_control, 2), round(
    #             self.received_packet[0] / self.cnt1, 2)
    #         state[3] = state[0][0] * 0.5 + state[1][0] * 0.3 + state[2][0] * 0.2, state[0][1] * 0.5 + state[1][
    #             1] * 0.3 + state[2][1] * 0.2
    #
    #     except:  # devided by zero
    #         state[0] = round(self.received_packet[0] / self.command_control, 2), 0
    #         state[1] = round(self.received_packet[1] / self.command_control, 2), 0
    #         state[2] = round(self.received_packet[2] / self.command_control, 2), 0
    #         state[3] = state[0][0] * 0.5 + state[1][0] * 0.3 + state[2][0] * 0.2, 0
    #
    #     # reward 측정
    #     rewards = self.reward(state, flows)
    #
    #     # done 검사
    #     if (self.cnt1 == self.command_control) and (self.cnt2 == self.audio) and (self.cnt3 == self.video) and (
    #             self.cnt4 == self.best_effort):
    #         done = True
    #
    #     else:
    #         done = False
    #
    #     return [state, rewards, done]
    #
    # def run(self):
    #     self.env.process(self.episode(self.env))
    #     self.env.run(until=100000)
    #     i = 0
    #     for agent in self.agents:
    #         i += 1
    #         agent.model.save_model("./result/0819_1_train/" + "agent[" + str(i) + "]" + str(min(self.minloss)) + ".h5")
    #     self.log.to_csv("./result/0819_1_train/log_0819_train_1.csv")

# if __name__ == "__main__":
#     env_ = GateControllSimulation()
#     env_.run()
