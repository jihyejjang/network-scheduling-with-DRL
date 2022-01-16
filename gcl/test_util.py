#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import simpy
import random
from node import Node
from dataclasses import dataclass
import warnings
import time
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import *

warnings.filterwarnings('ignore')

DATE = '1118'
FILENAME = '[1999]0.011379198171198368.h5'  # weight file name
WEIGHT_FILE = './result/' + DATE + '/' + FILENAME
# if not os.path.exists("./result/" + DATE):
#     os.makedirs("./result/" + DATE)
PRIORITY_QUEUE = 2
STATE = 3  # 3
STATE_SIZE = STATE * PRIORITY_QUEUE
GCL_LENGTH = 3
ACTION_SIZE = 2 ** (GCL_LENGTH * PRIORITY_QUEUE)
MAX_EPISODE = 2000
COMMAND_CONTROL = 40
AUDIO = 8
VIDEO_FRAME = 30
VIDEO = 2 * VIDEO_FRAME
BEST_EFFORT = 100
CC_PERIOD = 20
AD_PERIOD = 10
VD_PERIOD = 11
BE_PERIOD = 0.3
CC_BYTE = 300
AD_BYTE = 256
VD_BYTE = 1500
BE_BYTE = 1024
action_list = [63, 49, 35, 42, 28, 21, 14, 7]


TIMESLOT_SIZE = 0.6
NODES = 6  # switch의 수
UPDATE = 3000
W = [6, 0.05]

BANDWIDTH = 20000  # bits per msec (20Mbps)


def max_burst():
    burst = BANDWIDTH * TIMESLOT_SIZE
    print("burst", burst)


max_burst()


def utilization():  # without Best-effort traffic
    f1 = (CC_BYTE * 8) / CC_PERIOD  # bits per ms
    f2 = (AD_BYTE * 8) / AD_PERIOD
    f3 = (VD_BYTE * 8) / VD_PERIOD
    f4 = (BE_BYTE * 8) / BE_PERIOD
    utilization = (f1 + f2 + f3) / BANDWIDTH
    utilization_ = f4 / BANDWIDTH
    print("utilzation without BE ", round(utilization, 2))
    print("BE ", round(utilization_, 2))
    # f4 = (CC_BYTE * 8) / (CC_PERIOD * 0.001)


utilization()

def number_to_action(action_id): #number -> binary gcl code
    b_id = format(action_list[action_id], '06b')
    action_ = np.array(list(map(int, b_id)))
    return action_.reshape((PRIORITY_QUEUE, GCL_LENGTH))

def action_to_number(action):
    action_ = action.flatten()
    bin = ''
    for a in action_:
        bin += str(a)
    return action_list.index(int(bin, 2))


@dataclass
class Flow:  # type(class1:cc,2:ad,3:vd,4:be),Num,deadline,generate_time,depart_time,bits
    type_: int = None
    num_: int = None
    deadline_: float = None  # millisecond 단위, arrival time - generated time < deadline 이어야 함
    generated_time_: float = None  # millisecond 단위
    queueing_delay_: list = None  # node departure time - node arrival time
    node_arrival_time_: float = None
    # node_departure_time_: list = None
    arrival_time_: float = None
    bits_: int = None
    met_: bool = None
    hops_: int = None
    priority_: int = None


class GateControlTestSimulation:

    def __init__(self):
        # self.success_at_episode = [0, 0, 0, 0]  # deadline met
        #self.model = tf.keras.models.load_model(WEIGHT_FILE)
        #print(self.model.get_weights())
        self.env = simpy.Environment()
        self.nodes = [Node(n + 1, self.env) for n in range(NODES)]
        # self.agent = Agent()
        self.trans_dict = {1: simpy.Store(self.env),
                           2: simpy.Store(self.env),
                           3: simpy.Store(self.env),
                           4: simpy.Store(self.env),
                           5: simpy.Store(self.env),
                           6: simpy.Store(self.env)}  # node에서 받아온 전송할 packet들
        self.cnt1 = 0  # 전송된 flow 개수 카운트
        self.cnt2 = 0
        self.cnt3 = 0
        self.cnt4 = 0
        # self.total_episode = 0
        # self.total_timeslots = 0
        self.timeslots = 0
        self.start_time = self.env.now  # episode 시작
        state_shape = np.zeros((PRIORITY_QUEUE * STATE))
        self.state = {1: state_shape,
                      2: state_shape,
                      3: state_shape,
                      4: state_shape,
                      5: state_shape,
                      6: state_shape}
        self.done = False
        self.next_state = {1: state_shape,
                           2: state_shape,
                           3: state_shape,
                           4: state_shape,
                           5: state_shape,
                           6: state_shape}
        self.end_time = 0  # episode 끝났을 때
        self.received_packet = 0  # 전송완료 패킷수
        self.loss_min = 999999
        self.gate_control_list = [0 for _ in range(NODES)] # action

        # save logs
        self.log = pd.DataFrame(
            columns=['Episode', 'Duration', 'Slots', 'Score', 'Epsilon', 'min_loss', 'c&c', 'audio', 'video'])
        self.delay = pd.DataFrame(columns=['Episode', 'c&c', 'audio', 'video'])

        self.success = [0, 0, 0, 0]
        self.s = [[0, 0, 0, 0] for _ in range(PRIORITY_QUEUE)]  # arrived within deadline at the episode
        self.avg_delay = [[], [], [], []]
        self.state_and_action = []

    def gcl_predict(self, state):
        n = self.model.predict(state)
        id_ = np.argmax(n)
        action = number_to_action(id_)
        return action

    def reset(self):  # initial state, new episode start
        #self.model = tf.keras.models.load_model(WEIGHT_FILE)
        self.start_time = self.env.now  # episode 시작
        for n in range(NODES):
            self.nodes[n].reset(self.env, self.start_time)
        # self.total_episode += 1
        self.timeslots = 0
        self.cnt1 = 0  # 전송된 flow 개수 카운트
        self.cnt2 = 0
        self.cnt3 = 0
        self.cnt4 = 0
        state_shape = np.zeros((PRIORITY_QUEUE * STATE))
        self.state = {1: state_shape,
                      2: state_shape,
                      3: state_shape,
                      4: state_shape,
                      5: state_shape,
                      6: state_shape}
        self.next_state = {1: state_shape,
                           2: state_shape,
                           3: state_shape,
                           4: state_shape,
                           5: state_shape,
                           6: state_shape}
        self.done = False

        self.end_time = 0
        self.received_packet = 0
        self.success = [0, 0, 0, 0]  # deadline met
        self.avg_delay = [[], [], [], []]
        self.s = [[0, 0, 0, 0] for _ in range(PRIORITY_QUEUE)]
        # epsilon_decay
        # e = self.agent.reset()

    def flow_generator(self, time, type_num, fnum):  # flow structure에 맞게 flow생성, timestamp등 남길 수 있음

        f = Flow()

        if type_num == 1:  # c&c
            f.type_ = 1
            f.priority_ = 1
            f.num_ = fnum
            f.deadline_ = 7 * 0.001
            f.generated_time_ = time - self.start_time
            f.queueing_delay_ = 0
            f.node_arrival_time_ = 0
            f.arrival_time_ = 0
            f.bits_ = CC_BYTE * 8  # originally random.randrange(53, 300)
            f.met_ = -1
            f.hops_ = 4  # it means how many packet-in occur

        elif type_num == 2:  # audio
            f.type_ = 2
            f.priority_ = 1
            f.num_ = fnum
            f.deadline_ = 8 * 0.001  # originally random.choice([4, 10]) * 0.001
            f.generated_time_ = time - self.start_time
            f.queueing_delay_ = 0
            f.node_arrival_time_ = 0
            f.arrival_time_ = 0
            f.bits_ = AD_BYTE * 8  # originally random.choice([128, 256])
            f.met_ = -1
            f.hops_ = 4

        elif type_num == 3:  # video
            f.type_ = 3
            f.priority_ = 1
            f.num_ = fnum
            f.deadline_ = 0.030  # originally 30ms
            f.generated_time_ = time - self.start_time
            f.queueing_delay_ = 0
            f.node_arrival_time_ = 0
            f.arrival_time_ = 0
            f.bits_ = VD_BYTE * 8
            f.met_ = -1
            f.hops_ = 4

        else:  # best effort
            f.type_ = 4
            f.priority_ = 2
            f.num_ = fnum
            f.deadline_ = 0.05
            f.generated_time_ = time - self.start_time
            f.queueing_delay_ = 0
            f.node_arrival_time_ = 0
            f.arrival_time_ = 0
            f.bits_ = BE_BYTE * 8
            f.met_ = -1
            f.hops_ = 4

        return f

    def generate_cc(self, env):
        r = True
        for i in range(COMMAND_CONTROL):
            yield env.timeout(CC_PERIOD / 1000)
            if r:
                flow1 = self.flow_generator(env.now, 1, i)
                yield env.process(self.nodes[0].packet_in(env.now, flow1))
            else:
                flow2 = self.flow_generator(env.now, 1, i)
                yield env.process(self.nodes[1].packet_in(env.now, flow2))
            self.cnt1 += 1
            r = not r

    def generate_ad(self, env):
        r = True
        for i in range(AUDIO):
            yield env.timeout(AD_PERIOD / 1000)
            if r:
                flow1 = self.flow_generator(env.now, 2, i)
                yield env.process(self.nodes[0].packet_in(env.now, flow1))
            else:
                flow2 = self.flow_generator(env.now, 2, i)
                yield env.process(self.nodes[1].packet_in(env.now, flow2))
            self.cnt2 += 1
            r = not r

    def generate_vd(self, env):
        r = True
        for i in range(VIDEO):
            yield env.timeout(VD_PERIOD / 1000)
            if r:
                flow1 = self.flow_generator(env.now, 3, i)
                yield env.process(self.nodes[0].packet_in(env.now, flow1))
            else:
                flow2 = self.flow_generator(env.now, 3, i)
                yield env.process(self.nodes[1].packet_in(env.now, flow2))
            self.cnt3 += 1
            r = not r

    def generate_be(self, env):
        r = True
        for i in range(BEST_EFFORT):
            yield env.timeout(BE_PERIOD / 1000)
            if r:
                flow1 = self.flow_generator(env.now, 4, i)
                yield env.process(self.nodes[0].packet_in(env.now, flow1))
            else:
                flow2 = self.flow_generator(env.now, 4, i)
                yield env.process(self.nodes[1].packet_in(env.now, flow2))
            self.cnt4 += 1
            r = not r

    def sendTo_next_node(self, env, dpid):
        route = [2, 2, 3]
        flows = self.trans_dict[dpid]
        if not (len(flows.items)):
            return
        if dpid < 4:
            for _ in range(len(flows.items)):
                f = yield flows.get()
                yield env.process(self.nodes[route[dpid - 1]].packet_in(env.now, f))
            # yield env.timeout(TIMESLOT_SIZE / 1000)

        elif dpid > 4:  # transmission completed
            for _ in range(len(flows.items)):
                f = yield flows.get()
                t = f.type_ - 1
                self.received_packet += 1
                f.arrival_time_ = env.now - self.start_time
                delay = f.bits_ / 20000000.0 + f.queueing_delay_ + f.node_arrival_time_ - f.generated_time_
                #delay = f.arrival_time_ - f.generated_time_
                # print (f.queueing_delay_)
                self.avg_delay[t].append(delay)
                if delay <= f.deadline_:
                    f.met_ = 1
                    self.success[t] += 1
                    if dpid == 5:
                        self.s[0][t] += 1
                    else:
                        self.s[1][t] += 1
                else:
                    f.met_ = 0

        else:  # The Node4 sends packets to node5 or node6 according to the packet number

            for _ in range(len(flows.items)):
                f = yield flows.get()
                n = f.num_
                if not n % 2:
                    yield env.process(self.nodes[4].packet_in(env.now, f))
                else:
                    yield env.process(self.nodes[5].packet_in(env.now, f))
            # yield env.timeout(TIMESLOT_SIZE / 1000)

    def gcl_extract(self, env):  # mainprocess
        s = time.time()
        rewards_all = []

        # episode 시작 시 마다 flow generator process를 실행
        env.process(self.generate_cc(env))
        env.process(self.generate_ad(env))
        env.process(self.generate_vd(env))
        env.process(self.generate_be(env))

        gcl = {1: number_to_action(ACTION_SIZE - 1),
               2: number_to_action(ACTION_SIZE - 1),
               3: number_to_action(ACTION_SIZE - 1),
               4: number_to_action(ACTION_SIZE - 1),
               5: number_to_action(ACTION_SIZE - 1),
               6: number_to_action(ACTION_SIZE - 1)}

        while not self.done:  # 1회의 episode가 종료될 때 까지 cycle을 반복하는 MAIN process
            self.timeslots += GCL_LENGTH
            self.s = [[0, 0, 0, 0] for _ in range(PRIORITY_QUEUE)]

            for t in range(GCL_LENGTH):
                for n in range(NODES):
                    env.process(self.nodes[n].packet_out(env, self.trans_dict[n + 1], t))
                    env.process(self.sendTo_next_node(env, n + 1))
                yield env.timeout(TIMESLOT_SIZE / 1000)

            # training starts when a timeslot cycle has finished
            qlen = np.zeros((NODES, PRIORITY_QUEUE))  # flow type
            qdata = np.zeros((NODES, PRIORITY_QUEUE))
            t = self.env.now
            for i in range(NODES):
                qdata[i], _, _, qlen[i] = self.nodes[i].queue_info(t)

            # GCL predict & update
            for n in range(NODES):  # convey the predicted gcl and get states of queue
                self.next_state[n + 1], reward, self.done = self.step(n + 1, qlen, qdata)
                rewards_all.append(reward)
                self.state[n + 1] = self.next_state[n + 1]
                gcl[n + 1] = self.gcl_predict(self.state[n + 1].reshape((1, STATE_SIZE)))  # new state로 gcl 업데이트
                self.nodes[n].gcl_update(gcl[n + 1])
                self.gate_control_list.append(action_to_number(gcl[n + 1]))
                # print(self.gate_control_list)

        # Episode ends
        self.end_time = env.now
        e = time.time() - s
        print("avg delay", list(map(np.mean, self.avg_delay)))
        print("gcl distribution", np.unique(self.gate_control_list, return_counts=True))
        print("gcl extract success rate", self.success)
        print("reward", sum(rewards_all))

    def gcl_apply(self, env):
        print("Test 시작")
        env.process(self.generate_cc(env))
        env.process(self.generate_ad(env))
        env.process(self.generate_vd(env))
        env.process(self.generate_be(env))
        i = 0
        # print(np.unique(self.gate_control_list, return_counts=True))
        while not self.done:  # 1회의 episode가 종료될 때 까지 cycle을 반복하는 MAIN process
            for t in range(GCL_LENGTH):
                for n in range(NODES):
                    env.process(self.nodes[n].packet_out(env, self.trans_dict[n + 1], t))
                    env.process(self.sendTo_next_node(env, n + 1))
                yield env.timeout(TIMESLOT_SIZE / 1000)

            # training starts when a timeslot cycle has finished
            # qlen = np.zeros((NODES, PRIORITY_QUEUE))  # flow type
            # qdata = np.zeros((NODES, PRIORITY_QUEUE))

            #gcl_ = self.gate_control_list[i * 6:i * 6 + 6]
            #gcl = list(map(number_to_action, gcl_))

            # if not gcl:
            #     #print("not gcl")
            #     gcl = list(map(number_to_action, [0 for _ in range(NODES)]))
            t = env.now
            for n in range(NODES):
                gcl = list(map(number_to_action, [0 for _ in range(NODES)])) #action
                self.nodes[n].gcl_update(gcl[n])
                self.gate_control_list.append(action_to_number(gcl[n]))
                # qdata[n],_,_, qlen[n]= self.nodes[n].queue_info(t)
                #_, _, self.done = self.step(n + 1, qlen, qdata)
                #print(self.received_packet)
                #print(self.cnt1, self.cnt2, self.cnt3, self.cnt4)
                if self.received_packet == BEST_EFFORT:  # originally (CC + A + V + BE) COMMAND_CONTROL + AUDIO + VIDEO +
                    self.done = True
                    if n == 5:
                        self.success[2] //= VIDEO_FRAME

            i += 1

        print("test결과 success rate", self.success)
        print("avg delay", list(map(np.mean, self.avg_delay)))
        print("gcl distribution", np.unique(self.gate_control_list, return_counts=True))
        #print("gcl extract success rate", self.success)
        #print("reward", sum(rewards_all))

    def simulation(self):
        # gcl extract
        # self.env.process(self.gcl_extract(self.env))
        # self.env.run()
        # print("GCL Extract 완료")

        # test
        self.reset()
        self.env.process(self.gcl_apply(self.env))
        self.env.run()
        print("Test simulation 완료")

    def step(self, node, qlen, qdata):
        reward = 0
        if node >= 5:
            reward = self.reward2(node)
        hops = [3, 3, 2, 1, 0, 0]
        qt = qdata.transpose()
        previous_node = [0 for _ in range(PRIORITY_QUEUE)]

        if 2 < node < 5:  # 3,4 node
            # previous_node = list(map(sum, qt[:node - 1]))
            previous_node = [sum(qt[c][:node - 1]) for c in range(PRIORITY_QUEUE)]
        elif node > 4:  # 5,6 node
            previous_node = [0.5 * sum(qt[c][:node - 1]) for c in range(PRIORITY_QUEUE)]

        state = np.zeros((PRIORITY_QUEUE, STATE))
        state[:, 0] = qdata[node - 1]
        state[:, 1] = previous_node
        state[:, 2] = hops[node - 1]
        state = state.flatten()

        done = False
        if self.received_packet == COMMAND_CONTROL + AUDIO + VIDEO+ BEST_EFFORT:  # originally (CC + A + V + BE)
            done = True
            if node == 6:
                self.success[2] //= VIDEO_FRAME
        return [state, reward, done]

    def reward2(self, node):
        w = np.array([2, 3, 0.1, 0.01])
        # print("s:",self.s)
        r = round(sum(w * np.array(self.s[node - 5])), 1)
        # print ("r2:", r)
        return r


if __name__ == "__main__":
    test = GateControlTestSimulation()
    test.simulation()
