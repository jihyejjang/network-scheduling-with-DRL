#!/usr/bin/env python
# coding: utf-8
#TODO: 비교용 FIFO 시뮬레이션

import numpy as np
import pandas as pd
import simpy
from node import Node
from dataclasses import dataclass
import time
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import *
from parameter import *

max_burst()


# utilization()


class GateControlTestSimulation:

    def __init__(self):
        self.model = tf.keras.models.load_model(WEIGHT_FILE)
        # print(self.model.get_weights())
        self.env = simpy.Environment()
        self.node = Node(1, self.env)
        self.trans_list = simpy.Store(self.env)
        # self.agent = Agent()
        self.cnt1 = 0  # 전송된 flow 개수 카운트
        # self.cnt2 = 0
        # self.cnt3 = 0
        self.cnt4 = 0
        self.timeslots = 0  # progressed timeslots in a episode
        self.start_time = self.env.now  # episode 시작
        state_shape = np.zeros((PRIORITY_QUEUE * STATE))
        self.state = state_shape
        self.next_state = state_shape
        self.reward = 0
        self.done = False
        self.received_packet = 0  # 전송완료 패킷수
        self.loss_min = 999999
        self.gate_control_list = [INITIAL_ACTION for _ in range(NODES)]  # action

        # save logs
        self.log = pd.DataFrame(
            columns=['Episode', 'Duration', 'Slots', 'Score', 'Epsilon', 'min_loss', 'c&c', 'audio', 'video'])
        self.delay = pd.DataFrame(columns=['Episode', 'c&c', 'audio', 'video'])

        self.success = [0, 0]  # deadline met
        self.avg_delay = [[], []]
        self.estimated_e2e = [[],[]]
        self.s = [[0, 0] for _ in range(PRIORITY_QUEUE)]
        self.state_and_action = []
        self.sequence_p1, self.sequence_p2 = random_sequence()

    def gcl_predict(self, state):
        n = self.model.predict(state)
        id_ = np.argmax(n)
        # action = number_to_action(id_)
        return id_

    def reset(self):  # initial state, new episode start
        # self.model = tf.keras.models.load_model(WEIGHT_FILE)
        self.start_time = self.env.now  # episode 시작
        self.node = Node(1, self.env)

        # self.total_episode += 1
        self.timeslots = 0
        self.cnt1 = 0  # 전송된 flow 개수 카운트
        # self.cnt2 = 0
        # self.cnt3 = 0
        self.cnt4 = 0
        state_shape = np.zeros((PRIORITY_QUEUE * STATE))
        self.state = state_shape
        self.next_state = state_shape
        self.reward = 0
        self.done = False

        self.end_time = 0
        self.received_packet = 0
        self.success = [0, 0]  # deadline met
        self.avg_delay = [[], []]
        self.estimated_e2e = [[],[]]
        # epsilon_decay
        # e = self.agent.reset()

    def flow_generator(self, type_num, fnum):  # flow structure에 맞게 flow생성, timestamp등 남길 수 있음

        f = Flow()

        if type_num == 1:  # c&c
            f.type_ = 1
            f.priority_ = 1
            f.num_ = fnum
            f.deadline_ = CC_DEADLINE * 0.001
            # f.generated_time_ = time - self.start_time
            f.current_delay_ = self.sequence_p1[0][fnum]
            f.queueing_delay_ = 0
            f.node_arrival_time_ = 0
            f.bits_ = CC_BYTE * 8
            f.met_ = -1
            f.hops_ = self.sequence_p1[1][fnum]


        else:  # best effort
            f.type_ = 4
            f.priority_ = 2
            f.num_ = fnum
            f.deadline_ = BE_DEADLINE * 0.001
            f.current_delay_ = self.sequence_p2[0][fnum]
            # f.generated_time_ = time - self.start_time
            f.queueing_delay_ = 0
            f.node_arrival_time_ = 0
            f.arrival_time_ = 0
            f.bits_ = BE_BYTE * 8
            f.met_ = -1
            f.hops_ = self.sequence_p2[1][fnum]

        return f

    def generate_cc(self, env):
        for i in range(COMMAND_CONTROL):
            flow = self.flow_generator(1, i)
            # print("c&c generate time slot", self.timeslots)
            # if i < 10:
            #     print("p1 generated in timeslot", self.timeslots)
            yield env.process(self.node.packet_in(env.now, flow))
            self.cnt1 += 1
            yield env.timeout(TIMESLOT_SIZE * RANDOM_PERIOD_CC / 1000)

    def generate_be(self, env):
        for i in range(BEST_EFFORT):
            flow = self.flow_generator(4, i)
            # print("be generate time slot", self.timeslots)
            # if i < 10:
            #     print("p2 generated in timeslot", self.timeslots)
            yield env.process(self.node.packet_in(self.timeslots, flow))
            self.cnt4 += 1
            yield env.timeout(TIMESLOT_SIZE * RANDOM_PERIOD_BE / 1000)

    def sendTo_next_node(self, env):
        flows = self.trans_list

        if not (len(flows.items)):
            return

        # transmission completed
        for _ in range(len(flows.items)):
            self.reward += A  # 패킷을 전송했을 때 reward
            f = yield flows.get()
            t = f.priority_ - 1
            n = f.num_
            self.received_packet += 1
            f.arrival_time_ = env.now - self.start_time
            ET = (f.queueing_delay_ + f.current_delay_ + f.hops_) * TIMESLOT_SIZE / 1000
            delay = f.queueing_delay_ * TIMESLOT_SIZE / 1000
            self.avg_delay[t].append(delay)
            self.estimated_e2e[t].append(ET)

            if ET <= f.deadline_:
                f.met_ = 1
                self.success[t] += 1
                self.reward += W[t]

            else:
                f.met_ = 0

    def gcl_extract(self, env):  # mainprocess
        s = time.time()
        rewards_all = []
        i=1
        while sum(rewards_all) < 20:
            self.sequence_p1, self.sequence_p2 = random_sequence()
            print("{i}번째 시도".format(i = i))
            i+=1
            rewards_all = []
            self.reset()
            # episode 시작 시 마다 flow generator process를 실행
            env.process(self.generate_cc(env))
            # env.process(self.generate_ad(env))
            # env.process(self.generate_vd(env))
            env.process(self.generate_be(env))

            while not self.done:  # 1회의 episode가 종료될 때 까지 cycle을 반복하는 MAIN process


                yield env.process(self.node.packet_out(self.trans_list))
                env.process(self.sendTo_next_node(env))
                yield env.timeout(TIMESLOT_SIZE / 1000)
                t = self.env.now

                # training starts when a timeslot cycle has finished
                qlen, max_et = self.node.queue_info()  # state에 필요한 정보 (Q_p, maxET, index)
                self.next_state, self.done = self.step(qlen, max_et)
                rewards_all.append(self.reward)
                self.reward = 0
                self.state = self.next_state
                # print(self.state)
                gcl = self.gcl_predict(np.array(self.state).reshape(1, INPUT_SIZE))  # new state로 gcl 업데이트
                self.node.gcl_update(gcl)
                self.gate_control_list.append(gcl)
                self.timeslots += 1



        # Episode ends
        self.end_time = env.now
        e = time.time() - s
        print("extract 소요시간", e)
        print("avg delay", list(map(np.mean, self.avg_delay)))
        print("ET", list(map(np.mean, self.estimated_e2e)))
        print("gcl distribution", np.unique(self.gate_control_list, return_counts=True))
        print("gcl extract success rate", self.success)
        print("reward", sum(rewards_all))

    def gcl_apply(self, env):
        print("Test 시작")
        rewards_all = []
        env.process(self.generate_cc(env))
        env.process(self.generate_be(env))
        while not self.done:  # 1회의 episode가 종료될 때 까지 cycle을 반복하는 MAIN process
            gcl = self.gate_control_list[self.timeslots]

            yield env.process(self.node.packet_out(self.trans_list))
            env.process(self.sendTo_next_node(env))
            yield env.timeout(TIMESLOT_SIZE / 1000)

            qlen, max_et = self.node.queue_info()
            self.next_state, self.done = self.step(qlen, max_et)
            rewards_all.append(self.reward)
            self.reward = 0
            self.state = self.next_state
            self.node.gcl_update(gcl)
            self.timeslots += 1

        print("test결과 success rate", self.success)
        print("avg delay", list(map(np.mean, self.avg_delay)))
        print("ET", list(map(np.mean, self.estimated_e2e)))
        print("gcl distribution", np.unique(self.gate_control_list, return_counts=True))

    def FIFO(self, env):
        print("FIFO test")
        rewards_all = []
        env.process(self.generate_cc(env))
        env.process(self.generate_be(env))
        while not self.done:  # 1회의 episode가 종료될 때 까지 cycle을 반복하는 MAIN process
            #gcl = self.gate_control_list[self.timeslots]

            yield env.process(self.node.packet_FIFO_out(self.trans_list))
            env.process(self.sendTo_next_node(env))
            yield env.timeout(TIMESLOT_SIZE / 1000)

            qlen, max_et = self.node.queue_info()
            self.next_state, self.done = self.step(qlen, max_et)
            rewards_all.append(self.reward)
            self.reward = 0
            self.state = self.next_state
            #self.node.gcl_update(gcl)
            self.timeslots += 1

        print("test결과 success rate", self.success)
        print("avg delay", list(map(np.mean, self.avg_delay)))
        print("ET", list(map(np.mean, self.estimated_e2e)))
        print("reward", sum(rewards_all))

    def simulation(self):
        # gcl extract
        self.env.process(self.gcl_extract(self.env))
        self.env.run()
        print("@@@@@@@@@GCL Extract 완료@@@@@@@@@")

        #FIFO
        self.reset()
        self.env.process(self.FIFO(self.env))
        self.env.run()
        print("@@@@@@@@@FIFO 완료@@@@@@@@@")

        # test
        self.reset()
        self.env.process(self.gcl_apply(self.env))
        self.env.run()
        print("@@@@@@@@@Test simulation 완료@@@@@@@@@")

    def step(self, qlen, max_et):
        state = np.zeros((PRIORITY_QUEUE, STATE))
        state[:, 0] = qlen
        state[:, 1] = max_et
        # state[:, 2] = max_qp
        state = state.flatten()

        done = False

        if MAXSLOT_MODE:
            if (self.received_packet == COMMAND_CONTROL + BEST_EFFORT) or (self.timeslots == MAXSLOTS):
                done = True
        else:
            if self.received_packet == COMMAND_CONTROL + BEST_EFFORT:  # originally (CC + A + V + BE)
                done = True

        return [state, done]

    # def reward2(self, node):
    #     w = np.array([2, 3, 0.1, 0.01])
    #     # print("s:",self.s)
    #     r = round(sum(w * np.array(self.s[node - 5])), 1)
    #     # print ("r2:", r)
    #     return r


if __name__ == "__main__":
    test = GateControlTestSimulation()
    test.simulation()
