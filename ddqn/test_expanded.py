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

#TODO: extract성능이 path 길이에 따라 현저히 떨어지는 이유 분석 (state-action을 보고 분석해야 할듯)
#TODO: route 변경해보기 (link를 공유하는 방식으로)
#TODO: 논문 읽고 스터디하기
#TODO: 해결 방안 고안하기

# max_burst()

# utilization()


class GateControlTestSimulation:

    def __init__(self):
        self.model = tf.keras.models.load_model(WEIGHT_FILE)
        self.env = simpy.Environment()
        self.start_time = self.env.now
        self.nodes = [Node(n + 1, self.env) for n in range(NODES)]
        self.output = {1: simpy.Store(self.env),
                       2: simpy.Store(self.env),
                       3: simpy.Store(self.env),
                       4: simpy.Store(self.env),
                       5: simpy.Store(self.env),
                       6: simpy.Store(self.env),
                       7: simpy.Store(self.env),
                       8: simpy.Store(self.env),
                       9: simpy.Store(self.env)}
        self.timeslots = 0
        state_shape = np.zeros(INPUT_SIZE)
        self.state = [state_shape for _ in range(NODES)]
        self.reward = [0 for _ in range(NODES)]
        self.next_state = [state_shape for _ in range(NODES)]
        self.done = [0 for _ in range(NODES)]
        self.end_time = 0
        self.received_packet = [0 for _ in range(NODES)]
        self.success = [[0, 0, 0], [0, 0, 0]]

        # save logs
        # self.ex = pd.DataFrame(columns=['timeslot', 'state', 'gcl'])
        # self.ap = pd.DataFrame(columns=['timeslot', 'state', 'gcl'])
        self.node0 = pd.DataFrame(columns=['node','timeslot', 'state', 'gcl'])
        self.node1 = pd.DataFrame(columns=['node','timeslot', 'state', 'gcl'])
        self.node2 = pd.DataFrame(columns=['node','timeslot', 'state', 'gcl'])
        self.node3 = pd.DataFrame(columns=['node','timeslot', 'state', 'gcl'])
        self.node5 = pd.DataFrame(columns=['node','timeslot', 'state', 'gcl'])
        self.node6 = pd.DataFrame(columns=['node','timeslot', 'state', 'gcl'])
        self.node7 = pd.DataFrame(columns=['node', 'timeslot', 'state', 'gcl'])
        self.node8 = pd.DataFrame(columns=['node', 'timeslot', 'state', 'gcl'])


        self.log1 = pd.DataFrame(
            columns=['Slots', 'Score', 'p1', 'p2'])  # extract
        self.log2 = pd.DataFrame(
            columns=['Slots', 'Score', 'p1', 'p2'])  # fifo(spq)
        self.log3 = pd.DataFrame(
            columns=['Slots', 'Score', 'p1', 'p2'])  # apply
        # self.delay = pd.DataFrame(columns=['Episode', 'c&c', 'audio', 'video'])

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
        self.nodes = [Node(n + 1, self.env) for n in range(NODES)]
        self.timeslots = 0
        self.output = {1: simpy.Store(self.env),
                       2: simpy.Store(self.env),
                       3: simpy.Store(self.env),
                       4: simpy.Store(self.env),
                       5: simpy.Store(self.env),
                       6: simpy.Store(self.env),
                       7: simpy.Store(self.env),
                       8: simpy.Store(self.env),
                       9: simpy.Store(self.env)}
        state_shape = np.zeros((PRIORITY_QUEUE * STATE))
        self.state = [state_shape for _ in range(NODES)]
        self.next_state = [state_shape for _ in range(NODES)]
        self.reward = [0 for _ in range(NODES)]
        self.done = [0 for _ in range(NODES)]
        self.end_time = 0
        self.received_packet = [0 for _ in range(NODES)]
        self.success = [[0, 0, 0], [0, 0, 0]]
        self.node0 = pd.DataFrame(columns=['node', 'timeslot', 'state', 'gcl'])
        self.node1 = pd.DataFrame(columns=['node', 'timeslot', 'state', 'gcl'])
        self.node2 = pd.DataFrame(columns=['node', 'timeslot', 'state', 'gcl'])
        self.node3 = pd.DataFrame(columns=['node', 'timeslot', 'state', 'gcl'])
        self.node5 = pd.DataFrame(columns=['node', 'timeslot', 'state', 'gcl'])
        self.node6 = pd.DataFrame(columns=['node', 'timeslot', 'state', 'gcl'])
        self.node7 = pd.DataFrame(columns=['node', 'timeslot', 'state', 'gcl'])
        self.node8 = pd.DataFrame(columns=['node', 'timeslot', 'state', 'gcl'])

        # self.ex = pd.DataFrame(columns=['timeslot', 'state', 'gcl'])
        # self.ap = pd.DataFrame(columns=['timeslot', 'state', 'gcl'])

    def flow_generator(self, src, fnum):
        flow = Flow()

        p = 1
        if src > 3:
            p = 2

        flow.src_ = src
        flow.dst_ = src
        # print(src)
        flow.route_ = route[src - 1]
        flow.priority_ = p
        flow.num_ = fnum
        flow.generated_time_ = self.env.now - self.start_time
        flow.current_delay_ = 0
        flow.queueing_delay_ = 0
        flow.remain_hops_ = len(route[src - 1]) - 1
        flow.met_ = -1
        flow.random_delay_ = 0

        if flow.priority_ == 1:
            flow.deadline_ = CC_DEADLINE * 0.001
            flow.random_delay_ = self.sequence_p1[0][fnum]
            flow.bits_ = CC_BYTE * 8

        else:
            flow.deadline_ = BE_DEADLINE * 0.001
            flow.random_delay_ = self.sequence_p2[0][fnum]
            flow.bits_ = BE_BYTE * 8

        return flow

    def send(self, env, src):
        if src < 4:
            for i in range(COMMAND_CONTROL):
                flow = self.flow_generator(src, i)
                r = flow.route_[0]
                yield env.process(self.nodes[r - 1].packet_in(flow))
                yield env.timeout(TIMESLOT_SIZE * PERIOD_CC / 1000)

        else:
            for i in range(BEST_EFFORT):
                flow = self.flow_generator(src, i)
                r = flow.route_[0]
                yield env.process(self.nodes[r - 1].packet_in(flow))
                yield env.timeout(TIMESLOT_SIZE * PERIOD_BE / 1000)

    def sendTo_next_node(self, env, output):

        for i, pkts in output.items():
            if not len(pkts.items):
                continue
            for _ in range(len(pkts.items)):
                node = i - 1
                self.reward[node] += A
                packet = yield pkts.get()
                packet.current_delay_ += packet.queueing_delay_
                packet.queueing_delay_ = 0
                p = packet.priority_ - 1
                src = packet.src_
                ET = (packet.queueing_delay_ + packet.current_delay_ + packet.remain_hops_ + packet.random_delay_) * TIMESLOT_SIZE / 1000
                # print (env.now, packet.generated_time_)
                delay = env.now - self.start_time - packet.generated_time_ + packet.random_delay_ * TIMESLOT_SIZE / 1000

                if ET <= packet.deadline_:
                    self.reward[node] += W[p]

                # transmission completed
                if not packet.route_:
                    self.received_packet[src - 1] += 1
                    self.avg_delay[p].append(delay)
                    self.estimated_e2e[p].append(ET)
                    if delay <= packet.deadline_:
                        packet.met_ = 1
                        self.success[p][src % 3 - 1] += 1
                    else:
                        packet.met_ = 0
                else:
                    r = packet.route_[0]
                    yield env.process(self.nodes[r - 1].packet_in(packet))

    def gcl_extract(self, env):  # mainprocess
        print("EX")
        s = time.time()
        rewards_all = []
        self.sequence_p1, self.sequence_p2 = random_sequence()
        self.reset()
        gcl = [INITIAL_ACTION for _ in range(NODES)]

        # env.process(self.send(env, 1))
        # env.process(self.send(env, 2))
        env.process(self.send(env, 3))
        env.process(self.send(env, 4))
        # env.process(self.send(env, 5))
        # env.process(self.send(env, 6))

        while not sum(self.done) == NODES:
            # s1=time.time()
            self.timeslots += 1
            log0 = pd.DataFrame([('1',self.timeslots, self.state[0], gcl[0])], columns=['node', 'timeslot', 'state', 'gcl'])
            log1 = pd.DataFrame([('2',self.timeslots, self.state[1], gcl[1])], columns=['node', 'timeslot', 'state', 'gcl'])
            log2 = pd.DataFrame([('3',self.timeslots, self.state[2], gcl[2])], columns=['node', 'timeslot', 'state', 'gcl'])
            log3 = pd.DataFrame([('4',self.timeslots, self.state[3], gcl[3])], columns=['node', 'timeslot', 'state', 'gcl'])
            log5 = pd.DataFrame([('6',self.timeslots, self.state[5], gcl[5])], columns=['node', 'timeslot', 'state', 'gcl'])
            log6 = pd.DataFrame([('7',self.timeslots, self.state[6], gcl[6])], columns=['node', 'timeslot', 'state', 'gcl'])
            log7 = pd.DataFrame([('8',self.timeslots, self.state[7], gcl[7])], columns=['node', 'timeslot', 'state', 'gcl'])
            log8 = pd.DataFrame([('9',self.timeslots, self.state[8], gcl[8])], columns=['node', 'timeslot', 'state', 'gcl'])

            for n in range(NODES):
                # s2 = time.time() - s1
                yield env.process(self.nodes[n].packet_out(self.output[n + 1]))
            env.process(self.sendTo_next_node(env, self.output))
            yield env.timeout(TIMESLOT_SIZE / 1000)

            for n in range(NODES):
                # s3 = time.time() - s1
                state = self.nodes[n].queue_info()
                self.next_state[n], self.done[n] = self.step(n, state)

            rewards_all.append(np.sum(self.reward))
            self.reward = [0 for _ in range(NODES)]
            self.state = self.next_state

            for n in range(NODES):
                gcl[n] = self.gcl_predict(np.array(self.state[n]).reshape(1, INPUT_SIZE))  # new state로 gcl 업데이트
                self.nodes[n].gcl_update(gcl[n])

            # self.gate_control_list.append(gcl)
            # self.ex = self.ex.append(log, ignore_index=True)
            self.node0 = self.node0.append(log0, ignore_index=True)
            self.node1 = self.node1.append(log1, ignore_index=True)
            self.node2 = self.node2.append(log2, ignore_index=True)
            self.node3 = self.node3.append(log3, ignore_index=True)
            self.node5 = self.node5.append(log5, ignore_index=True)
            self.node6 = self.node6.append(log6, ignore_index=True)
            self.node7 = self.node7.append(log7, ignore_index=True)
            self.node8 = self.node8.append(log8, ignore_index=True)


        # Episode ends
        self.end_time = env.now
        e = time.time() - s
        print("extract 소요시간", e)
        print("Score: {s}, Final Step: {t}, success: {l}, avg_qdelay: {d}".format(
            s=np.sum(rewards_all),
            t=self.timeslots,
            l=self.success,
            d=list(map(np.mean, self.avg_delay))))
        log_ = pd.DataFrame([(self.timeslots, np.sum(rewards_all), self.success[0], self.success[1])],
                            columns=['Slots', 'Score', 'p1', 'p2'])

        self.log1 = self.log1.append(log_, ignore_index=True)
        self.node0.to_csv('result/test/node0.csv')
        self.node1.to_csv('result/test/node1.csv')
        self.node2.to_csv('result/test/node2.csv')
        self.node3.to_csv('result/test/node3.csv')
        self.node5.to_csv('result/test/node5.csv')
        self.node6.to_csv('result/test/node6.csv')
        self.node7.to_csv('result/test/node7.csv')
        self.node8.to_csv('result/test/node8.csv')



        # if sum(rewards_all) < 30:
        # self.ex.to_csv('result/test/ex_analysis.csv')

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

    '''
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
    '''

    def SP(self, env):
        print("SP")
        s = time.time()
        rewards_all = []
        # env.process(self.send(env, 1))
        # env.process(self.send(env, 2))
        env.process(self.send(env, 3))
        env.process(self.send(env, 4))
        # env.process(self.send(env, 5))
        # env.process(self.send(env, 6))

        while not sum(self.done) == NODES:  # 1회의 episode가 종료될 때 까지 cycle을 반복하는 MAIN process
            self.timeslots += 1
            log0 = pd.DataFrame([('1', self.timeslots, self.state[0])],
                                columns=['node', 'timeslot', 'state'])
            log1 = pd.DataFrame([('2', self.timeslots, self.state[1])],
                                columns=['node', 'timeslot', 'state'])
            log2 = pd.DataFrame([('3', self.timeslots, self.state[2])],
                                columns=['node', 'timeslot', 'state'])
            log3 = pd.DataFrame([('4', self.timeslots, self.state[3])],
                                columns=['node', 'timeslot', 'state']),
            log5 = pd.DataFrame([('6', self.timeslots, self.state[5])],
                                columns=['node', 'timeslot', 'state'])
            log6 = pd.DataFrame([('7', self.timeslots, self.state[6])],
                                columns=['node', 'timeslot', 'state'])
            log7 = pd.DataFrame([('8', self.timeslots, self.state[7])],
                                columns=['node', 'timeslot', 'state'])
            log8 = pd.DataFrame([('9', self.timeslots, self.state[8])],
                                columns=['node', 'timeslot', 'state'])

            for n in range(NODES):
                yield env.process(self.nodes[n].strict_priority(self.output[n + 1]))
            env.process(self.sendTo_next_node(env, self.output))
            yield env.timeout(TIMESLOT_SIZE / 1000)

            for n in range(NODES):
                state = self.nodes[n].queue_info()
                self.next_state[n], self.done[n] = self.step(n, state)
            rewards_all.append(np.sum(self.reward))
            self.reward = [0 for _ in range(NODES)]
            self.state = self.next_state

            self.node0 = self.node0.append(log0, ignore_index=True)
            self.node1 = self.node1.append(log1, ignore_index=True)
            self.node2 = self.node2.append(log2, ignore_index=True)
            self.node3 = self.node3.append(log3, ignore_index=True)
            self.node5 = self.node5.append(log5, ignore_index=True)
            self.node6 = self.node6.append(log6, ignore_index=True)
            self.node7 = self.node7.append(log7, ignore_index=True)
            self.node8 = self.node8.append(log8, ignore_index=True)

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

        e = time.time() - s
        print("SP 소요시간", e)
        print("Score: {s}, Final Step: {t}, success: {l}, avg_qdelay: {d}".format(
            s=np.sum(rewards_all),
            t=self.timeslots,
            l=self.success,
            d=list(map(np.mean, self.avg_delay))))
        log_ = pd.DataFrame([(self.timeslots, np.sum(rewards_all), self.success[0], self.success[1])],
                            columns=['Slots', 'Score', 'p1', 'p2'])

        self.log3 = self.log3.append(log_, ignore_index=True)
        self.node0.to_csv('result/test/SP/node0.csv')
        self.node1.to_csv('result/test/SP/node1.csv')
        self.node2.to_csv('result/test/SP/node2.csv')
        self.node3.to_csv('result/test/SP/node3.csv')
        self.node5.to_csv('result/test/SP/node5.csv')
        self.node6.to_csv('result/test/SP/node6.csv')
        self.node7.to_csv('result/test/SP/node7.csv')
        self.node8.to_csv('result/test/SP/node8.csv')

    def simulation(self):
        iter = 1
        for _ in range(iter):
            self.env.process(self.gcl_extract(self.env))
            self.env.run()
            print("@@@@@@@@@GCL Extract 완료@@@@@@@@@")

            # FIFO
            self.reset()
            self.env.process(self.SP(self.env))
            self.env.run()
            # print("@@@@@@@@@FIFO 완료@@@@@@@@@")

            # test
            # self.reset()
            # self.env.process(self.gcl_apply(self.env))
            # self.env.run()
            # print("@@@@@@@@@Test simulation 완료@@@@@@@@@")
            self.sequence_p1, self.sequence_p2 = random_sequence()

        # self.log1.to_csv("result/test/extract.csv")
        # self.log2.to_csv("result/test/apply.csv")
        # self.log3.to_csv("result/test/fifo.csv")

    def step(self, node, states=None):
        qlen = states[0]
        max_et = states[1]
        state = np.zeros((PRIORITY_QUEUE, STATE))
        state[:, 0] = qlen
        state[:, 1] = max_et
        state = state.flatten()

        done = 0

        if MAXSLOT_MODE:
            if (self.received_packet[node] == COMMAND_CONTROL + BEST_EFFORT) or (self.timeslots >= MAXSLOTS):
                done = 1
        else:
            if self.received_packet[node] == COMMAND_CONTROL + BEST_EFFORT:  # originally (CC + A + V + BE)
                done = 1

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
