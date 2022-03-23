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


# max_burst()

# utilization()


class GateControlTestSimulation:

    def __init__(self):
        self.model = tf.keras.models.load_model(WEIGHT_FILE)
        self.env = simpy.Environment()
        self.node = Node(1, self.env)
        self.trans_list = simpy.Store(self.env)
        self.cnt1 = 0
        self.cnt4 = 0
        self.timeslots = 0  # progressed timeslots in a episode
        self.start_time = self.env.now  # episode 시작
        state_shape = np.zeros((PRIORITY_QUEUE * STATE))
        self.state = state_shape
        self.next_state = state_shape
        self.reward = 0
        self.done = False
        self.received_packet = 0  # 전송완료 패킷수
        # self.loss_min = 999999
        self.gate_control_list = [0]  # action
        self.state_list = []

        # save logs
        self.ex = pd.DataFrame(columns=['timeslot', 'state', 'gcl'])
        self.ap = pd.DataFrame(columns=['timeslot', 'state', 'gcl'])

        self.log1 = pd.DataFrame(
            columns=['Slots', 'Score', 'p1', 'p2'])  # extract
        self.log2 = pd.DataFrame(
            columns=['Slots', 'Score', 'p1', 'p2'])  # fifo(spq)
        self.log3 = pd.DataFrame(
            columns=['Slots', 'Score', 'p1', 'p2'])  # apply
        # self.delay = pd.DataFrame(columns=['Episode', 'c&c', 'audio', 'video'])

        self.success = [0, 0]  # deadline met
        self.avg_delay = [[], []]
        self.estimated_e2e = [[], []]
        self.s = [[0, 0] for _ in range(PRIORITY_QUEUE)]
        self.state_and_action = []
        self.sequence_p1, self.sequence_p2 = random_sequence()

    def gcl_predict(self, state):
        n = self.model.predict(state)
        id_ = np.argmax(n)
        # action = number_to_action(id_)
        return id_

    def reset(self):  # initial state, new episode start
        self.start_time = self.env.now  # episode 시작
        self.node = Node(1, self.env)

        self.timeslots = 0
        self.cnt1 = 0
        self.cnt4 = 0
        state_shape = np.zeros((PRIORITY_QUEUE * STATE))
        self.state = state_shape
        self.next_state = state_shape
        self.reward = 0
        self.done = False

        self.end_time = 0
        self.received_packet = 0
        self.success = [0, 0]  # deadline met
        self.state_list = []
        self.ex = pd.DataFrame(columns=['timeslot', 'state', 'gcl'])
        self.ap = pd.DataFrame(columns=['timeslot', 'state', 'gcl'])

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
            f.remain_hops_ = self.sequence_p1[1][fnum]


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
            f.remain_hops_ = self.sequence_p2[1][fnum]

        return f

    def generate_cc(self, env):
        for i in range(COMMAND_CONTROL):
            flow = self.flow_generator(1, i)
            # print("c&c generate time slot", self.timeslots)
            # if i < 10:
            #     print("p1 generated in timeslot", self.timeslots)
            yield env.process(self.node.packet_in(env.now, flow))
            self.cnt1 += 1
            yield env.timeout(TIMESLOT_SIZE * PERIOD_CC / 1000)

    def generate_be(self, env):
        for i in range(BEST_EFFORT):
            flow = self.flow_generator(4, i)
            # print("be generate time slot", self.timeslots)
            # if i < 10:
            #     print("p2 generated in timeslot", self.timeslots)
            yield env.process(self.node.packet_in(self.timeslots, flow))
            self.cnt4 += 1
            yield env.timeout(TIMESLOT_SIZE * PERIOD_BE / 1000)

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
            ET = (f.queueing_delay_ + f.current_delay_ + f.remain_hops_) * TIMESLOT_SIZE / 1000
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
        self.sequence_p1, self.sequence_p2 = random_sequence()
        self.reset()
        gcl = 0

        env.process(self.generate_cc(env))
        env.process(self.generate_be(env))

        while not self.done:
            self.timeslots += 1
            #print("timeslot",self.timeslots)
            log = pd.DataFrame([(self.timeslots, self.state, gcl)], columns=['timeslot', 'state', 'gcl'])
            yield env.process(self.node.packet_out(self.trans_list))
            env.process(self.sendTo_next_node(env))
            yield env.timeout(TIMESLOT_SIZE / 1000)

            qlen, max_et = self.node.queue_info()  # state에 필요한 정보 (Q_p, maxET, index)
            self.next_state, self.done = self.step(qlen, max_et)
            rewards_all.append(self.reward)
            self.reward = 0
            self.state = self.next_state
            # print(self.state)
            gcl = self.gcl_predict(np.array(self.state).reshape(1, INPUT_SIZE))  # new state로 gcl 업데이트
            #print("next slot gcl",gcl)
            self.node.gcl_update(gcl)
            self.gate_control_list.append(gcl)

            self.ex = self.ex.append(log, ignore_index=True)

        # Episode ends
        self.end_time = env.now
        e = time.time() - s
        print("extract 소요시간", e)
        log_ = pd.DataFrame([(self.timeslots, np.sum(rewards_all), self.success[0], self.success[1])],
                            columns=['Slots', 'Score', 'p1', 'p2'])

        self.log1 = self.log1.append(log_, ignore_index=True)

        if sum(rewards_all) < 30 :
            self.ex.to_csv('result/test/ex_analysis.csv')

        # print("avg delay", list(map(np.mean, self.avg_delay)))
        # print("ET", list(map(np.mean, self.estimated_e2e)))
        # print("gcl distribution", np.unique(self.gate_control_list, return_counts=True))
        # print("gcl extract success rate", self.success)
        # print("reward", sum(rewards_all))

        # x = range(int(len(self.avg_delay[0])))
        # # y1 = self.avg_delay[0]
        # y2 = self.estimated_e2e[0]
        # # plt.scatter(x, y1, s=3, label='p1_q')
        # plt.scatter(x, y2, s=3, label='p1_e')
        # plt.plot(x, [5*0.001 for _ in range(len(x))], c='orange', label='deadline')
        # plt.savefig("test_p1.png", dpi=300)
        # plt.clf()
        # X = range(int(len(self.avg_delay[1])))
        # # Y1 = self.avg_delay[1]
        # Y2 = self.estimated_e2e[1]
        # # plt.scatter(X, Y1, s=3, label='p2_q')
        # plt.scatter(X, Y2, s=3, label='p2_e')
        # plt.plot(X, [50*0.001 for _ in range(len(X))], c='orange', label='deadline')
        # plt.savefig("test_p2.png", dpi=300)
        # plt.clf()

    def gcl_apply(self, env):
        print("applied gcl")
        rewards_all = []
        env.process(self.generate_cc(env))
        env.process(self.generate_be(env))
        gcl=0
        while not self.done:  # 1회의 episode가 종료될 때 까지 cycle을 반복하는 MAIN process
            self.timeslots += 1
            log = pd.DataFrame([(self.timeslots, self.state, gcl)], columns=['timeslot', 'state', 'gcl'])

            yield env.process(self.node.packet_out(self.trans_list))
            env.process(self.sendTo_next_node(env))
            yield env.timeout(TIMESLOT_SIZE / 1000)

            qlen, max_et = self.node.queue_info()
            self.next_state, self.done = self.step(qlen, max_et)
            rewards_all.append(self.reward)
            self.reward = 0
            self.state_list.append(self.state)
            self.state = self.next_state
            gcl = self.gate_control_list[self.timeslots]
            self.node.gcl_update(gcl)

            self.ap = self.ap.append(log, ignore_index=True)

        # if sum(rewards_all) < 30:
        #     self.ap.to_csv('ap_analysis.csv')

        self.gate_control_list = [0]


        log_ = pd.DataFrame([(self.timeslots, np.sum(rewards_all), self.success[0], self.success[1])],
                            columns=['Slots', 'Score', 'p1', 'p2'])

        self.log2 = self.log2.append(log_, ignore_index=True)

    def FIFO(self, env):
        print("FIFO test")
        rewards_all = []
        env.process(self.generate_cc(env))
        env.process(self.generate_be(env))
        while not self.done:  # 1회의 episode가 종료될 때 까지 cycle을 반복하는 MAIN process
            # gcl = self.gate_control_list[self.timeslots]

            yield env.process(self.node.packet_FIFO_out(self.trans_list))
            env.process(self.sendTo_next_node(env))
            yield env.timeout(TIMESLOT_SIZE / 1000)

            qlen, max_et = self.node.queue_info()
            self.next_state, self.done = self.step(qlen, max_et)
            rewards_all.append(self.reward)
            self.reward = 0
            self.state = self.next_state
            # self.node.gcl_update(gcl)
            self.timeslots += 1

        # print("test결과 success rate", self.success)
        # print("avg delay", list(map(np.mean, self.avg_delay)))
        # print("ET", list(map(np.mean, self.estimated_e2e)))
        # print("reward", sum(rewards_all))

        # x = range(int(len(self.avg_delay[0])))
        # y1 = self.avg_delay[0]
        # y2 = self.estimated_e2e[0]
        # plt.scatter(x, y1, s=3, label='p1_q')
        # plt.plot(x, y2,  label='p1_e')
        # plt.plot(x, [5*0.001 for _ in range(len(x))], c='orange', label='deadline')
        # plt.savefig("fifo_p1.png", dpi=300)
        # plt.clf()

        # X = range(int(len(self.avg_delay[1])))
        # Y1 = self.avg_delay[1]
        # Y2 = self.estimated_e2e[1]
        # plt.scatter(X, Y1, s=3, label='p2_q')
        # plt.plot(X, Y2,  label='p2_e')
        # plt.plot(X, [50*0.001 for _ in range(len(X))], c='orange', label='deadline')
        # plt.savefig("fifo_p2.png", dpi=300)
        # plt.clf()

        log_ = pd.DataFrame([(self.timeslots, np.sum(rewards_all), self.success[0], self.success[1])],
                            columns=['Slots', 'Score', 'p1', 'p2'])

        self.log3 = self.log3.append(log_, ignore_index=True)

    def simulation(self):
        iter = 10
        for _ in range(iter):
            self.env.process(self.gcl_extract(self.env))
            self.env.run()
            # print("@@@@@@@@@GCL Extract 완료@@@@@@@@@")

            # FIFO
            self.reset()
            self.env.process(self.FIFO(self.env))
            self.env.run()
            # print("@@@@@@@@@FIFO 완료@@@@@@@@@")

            # test
            # self.reset()
            # self.env.process(self.gcl_apply(self.env))
            # self.env.run()
            # # print("@@@@@@@@@Test simulation 완료@@@@@@@@@")
            self.sequence_p1, self.sequence_p2 = random_sequence()

        self.log1.to_csv("result/test/extract.csv")
        # self.log2.to_csv("result/test/apply.csv")
        self.log3.to_csv("result/test/fifo.csv")

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
