#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import simpy
import random
from node import Node
from agent import Agent
from dataclasses import dataclass
import warnings
import time

warnings.filterwarnings('ignore')

DATE = str(1030)
PRIORITY_QUEUES = 4
INPUT_SIZE = 2
GCL_LENGTH = 10
MAX_EPISODE = 2000
COMMAND_CONTROL = 40
AUDIO = 8
VIDEO = 4
BEST_EFFORT = 40
CC_PERIOD = 5  # milliseconds #TODO: originally 5 (simulation duration을 위해 잠시 줄임)
AD_PERIOD = 1
VD_PERIOD = 33
BE_PERIOD = 0.5
TIMESLOT_SIZE = 0.4
NODES = 6  # switch의 수
UPDATE = 10000

@dataclass
class Flow:  # type(class1:cc,2:ad,3:vd,4:be),Num,deadline,generate_time,depart_time,bits
    type_: int = None
    num_: int = None
    deadline_: float = None  # millisecond 단위, arrival time - generated time < deadline 이어야 함
    generated_time_: float = None  # millisecond 단위
    queueing_delay_: float = None  # node departure time - node arrival time
    node_arrival_time_: float = None
    # node_departure_time_: list = None
    arrival_time_: float = None
    bits_: int = None
    met_: bool = None
    hops_: int = None


class GateControlSimulation:

    def __init__(self):
        self.success_at_episode = [0, 0, 0, 0]  # deadline met
        self.env = simpy.Environment()
        self.nodes = [Node(n + 1, self.env) for n in range(NODES)]
        self.agents = [Agent() for _ in range(PRIORITY_QUEUES)]  # 4 of multiagents which is number of priority queue
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
        self.total_episode = 0
        self.total_timeslots = 0
        self.timeslots = 0
        self.start_time = self.env.now  # episode 시작
        self.state = {1: np.zeros((PRIORITY_QUEUES, INPUT_SIZE)),
                      2: np.zeros((PRIORITY_QUEUES, INPUT_SIZE)),
                      3: np.zeros((PRIORITY_QUEUES, INPUT_SIZE)),
                      4: np.zeros((PRIORITY_QUEUES, INPUT_SIZE)),
                      5: np.zeros((PRIORITY_QUEUES, INPUT_SIZE)),
                      6: np.zeros((PRIORITY_QUEUES, INPUT_SIZE))}
        self.done = False
        self.next_state = {1: np.zeros((PRIORITY_QUEUES, INPUT_SIZE)),
                           2: np.zeros((PRIORITY_QUEUES, INPUT_SIZE)),
                           3: np.zeros((PRIORITY_QUEUES, INPUT_SIZE)),
                           4: np.zeros((PRIORITY_QUEUES, INPUT_SIZE)),
                           5: np.zeros((PRIORITY_QUEUES, INPUT_SIZE)),
                           6: np.zeros((PRIORITY_QUEUES, INPUT_SIZE))}
        self.end_time = 0  # episode 끝났을 때

        self.received_packet = 0  # 전송완료 패킷수
        self.loss_min = 9999

        # save logs
        self.log = pd.DataFrame(
            columns=['Episode', 'Duration', 'Slots', 'Score', 'Epsilon', 'min_loss', 'success rate'])
        self.success = [0, 0, 0, 0]
        # npz
        self.e2e_delay = [[], [], [], []]
        self.state_and_action = []

    def reset(self):  # initial state, new episode start
        self.start_time = self.env.now  # episode 시작
        for n in range(NODES):
            self.nodes[n].reset(self.env, self.start_time)
        self.total_episode += 1
        self.timeslots = 0
        self.cnt1 = 0  # 전송된 flow 개수 카운트
        self.cnt2 = 0
        self.cnt3 = 0
        self.cnt4 = 0
        self.state = {1: np.zeros((PRIORITY_QUEUES, INPUT_SIZE)),
                      2: np.zeros((PRIORITY_QUEUES, INPUT_SIZE)),
                      3: np.zeros((PRIORITY_QUEUES, INPUT_SIZE)),
                      4: np.zeros((PRIORITY_QUEUES, INPUT_SIZE)),
                      5: np.zeros((PRIORITY_QUEUES, INPUT_SIZE)),
                      6: np.zeros((PRIORITY_QUEUES, INPUT_SIZE))}
        self.next_state = {1: np.zeros((PRIORITY_QUEUES, INPUT_SIZE)),
                           2: np.zeros((PRIORITY_QUEUES, INPUT_SIZE)),
                           3: np.zeros((PRIORITY_QUEUES, INPUT_SIZE)),
                           4: np.zeros((PRIORITY_QUEUES, INPUT_SIZE)),
                           5: np.zeros((PRIORITY_QUEUES, INPUT_SIZE)),
                           6: np.zeros((PRIORITY_QUEUES, INPUT_SIZE))}
        self.done = False

        self.end_time = 0
        self.received_packet = 0
        self.success = [0, 0, 0, 0]  # deadline met
        self.fail = [[0, 0, 0, 0] for _ in range(2)]

        # epsilon_decay
        e = self.agents[0].reset()
        self.agents[1].reset()
        self.agents[2].reset()
        self.agents[3].reset()

        return e

    def flow_generator(self, time, type_num, fnum):  # flow structure에 맞게 flow생성, timestamp등 남길 수 있음

        f = Flow()

        if type_num == 1:  # c&c
            f.type_ = type_num
            f.num_ = fnum
            f.deadline_ = CC_PERIOD / 1000
            f.generated_time_ = time - self.start_time
            f.queueing_delay_ = 0
            f.node_arrival_time_ = 0
            f.arrival_time_ = 0
            f.bits_ = random.randrange(53, 300) * 8
            f.met_ = -1
            f.hops_ = 4  # it means how many packet-in occur

        elif type_num == 2:  # audio
            f.type_ = type_num
            f.num_ = fnum
            f.deadline_ = random.choice([4, 10]) * 0.001
            f.generated_time_ = time - self.start_time
            f.queueing_delay_ = 0
            f.node_arrival_time_ = 0
            f.arrival_time_ = 0
            f.bits_ = random.choice([128, 256]) * 8
            f.met_ = -1
            f.hops_ = 4

        elif type_num == 3:  # video
            f.type_ = type_num
            f.num_ = fnum
            f.deadline_ = 0.030
            f.generated_time_ = time - self.start_time
            f.queueing_delay_ = 0
            f.node_arrival_time_ = 0
            f.arrival_time_ = 0
            f.bits_ = 30 * 1500 * 8
            f.met_ = -1
            f.hops_ = 4

        else:  # best effort
            f.type_ = type_num
            f.num_ = fnum
            f.deadline_ = 0.010
            f.generated_time_ = time - self.start_time
            f.queueing_delay_ = 0
            f.node_arrival_time_ = 0
            f.arrival_time_ = 0
            f.bits_ = 32 * 8
            f.met_ = -1
            f.hops_ = 4

        return f

    def generate_cc(self, env):
        half = int(COMMAND_CONTROL / 2)
        for i in range(half):
            # print("c&c flow를 생성합니다.", env.now - self.start_time)
            yield env.timeout(CC_PERIOD / 1000)
            flow1 = self.flow_generator(env.now, 1, i)
            flow2 = self.flow_generator(env.now, 1, i + half)
            yield env.process(self.nodes[0].packet_in(env.now, flow1))
            yield env.process(self.nodes[1].packet_in(env.now, flow2))
            self.cnt1 += 2

    def generate_ad(self, env):
        half = int(AUDIO / 2)
        for i in range(half):
            # print("audio flow를 생성합니다.", env.now - self.start_time)
            yield env.timeout(AD_PERIOD / 1000)
            flow1 = self.flow_generator(env.now, 2, i)
            flow2 = self.flow_generator(env.now, 2, i + half)
            yield env.process(self.nodes[0].packet_in(env.now, flow1))
            yield env.process(self.nodes[1].packet_in(env.now, flow2))
            self.cnt2 += 2

    def generate_vd(self, env):
        half = int(VIDEO / 2)
        for i in range(half):
            # print("video flow를 생성합니다.", env.now - self.start_time)
            yield env.timeout(VD_PERIOD / 1000)
            flow1 = self.flow_generator(env.now, 3, i)
            flow2 = self.flow_generator(env.now, 3, i + half)
            yield env.process(self.nodes[0].packet_in(env.now, flow1))
            yield env.process(self.nodes[1].packet_in(env.now, flow2))
            self.cnt3 += 2

    def generate_be(self, env):
        half = int(BEST_EFFORT / 2)
        for i in range(half):
            # print("be flow를 생성합니다.", env.now - self.start_time)
            yield env.timeout(BE_PERIOD / 1000)
            flow1 = self.flow_generator(env.now, 4, i)
            flow2 = self.flow_generator(env.now, 4, i + half)
            yield env.process(self.nodes[0].packet_in(env.now, flow1))
            yield env.process(self.nodes[1].packet_in(env.now, flow2))
            self.cnt4 += 2

    def sendTo_next_node(self, env, dpid):
        route = [2, 2, 3]
        flows = self.trans_dict[dpid]
        if not (len(flows.items)):
            return
        classes = [COMMAND_CONTROL, AUDIO, VIDEO, BEST_EFFORT]
        if dpid < 4:
            # print('{} 노드에서 {}노드로 전송(sendTo)'.format(dpid, route[dpid - 1] + 1))
            # print('    전송할 trans_dict:', flows.items)
            for _ in range(len(flows.items)):
                f = yield flows.get()
                yield env.process(self.nodes[route[dpid - 1]].packet_in(env.now, f))
            yield env.timeout(TIMESLOT_SIZE / 1000)

        elif dpid > 4:  # transmission completed
            yield env.timeout(TIMESLOT_SIZE / 1000)
            for _ in range(len(flows.items)):
                # print("@@@@@TRANSMISSION COMPLETED@@@@@")
                f = yield flows.get()
                c = f.type_ - 1
                self.received_packet += 1
                f.arrival_time_ = env.now - self.start_time
                delay = f.arrival_time_ - f.generated_time_
                self.e2e_delay[c].append(delay)
                if delay <= f.deadline_:
                    f.met_ = 1
                    self.success[c] += 1
                else:
                    f.met_ = 0
                    #self.fail[dpid-5][c] += 1
                # print("전송 완료 패킷:", f)
                # print("delay :", delay)

        else:  # The Node4 sends packets to node5 or node6 according to the packet number
            # print('{} 노드에서 전송(sendTo)'.format(dpid))
            # print('    전송할 trans_dict:', flows.items)
            for _ in range(len(flows.items)):
                f = yield flows.get()
                c = f.type_ - 1
                n = f.num_
                if n < classes[c] / 2:
                    yield env.process(self.nodes[4].packet_in(env.now, f))
                else:
                    yield env.process(self.nodes[5].packet_in(env.now, f))
            yield env.timeout(TIMESLOT_SIZE / 1000)

    def episode(self, env):  # mainprocess
        for episode_num in range(MAX_EPISODE):
            # print("****** 에피소드" + str(self.total_episode) + "시작 ******")
            s = time.time()
            rewards_all = []
            gcl = ['1111111111' for _ in range(PRIORITY_QUEUES)]
            loss = []
            epsilon = self.reset()
            # episode 시작 시 마다 flow generator process를 실행
            env.process(self.generate_cc(env))
            env.process(self.generate_ad(env))
            env.process(self.generate_vd(env))
            env.process(self.generate_be(env))

            while not self.done:  # 1회의 episode가 종료될 때 까지 cycle을 반복하는 MAIN process

                # self.fail = [[0, 0, 0, 0] for _ in range(2)]
                self.timeslots += 10
                self.total_timeslots += 10

                for t in range(GCL_LENGTH):
                    for n in range(NODES):
                        env.process(self.nodes[n].packet_out(env, self.trans_dict[n + 1], t))
                        env.process(self.sendTo_next_node(env, n + 1))
                    yield env.timeout(TIMESLOT_SIZE / 1000)

                    if self.total_timeslots % 100:  # logging states and actions
                        self.state_and_action.append([self.state, gcl])

                # training starts when a timeslot cycle has finished
                qlen = np.zeros((NODES, PRIORITY_QUEUES))

                for i in range(NODES):
                    qlen[i] = self.nodes[i].queue_length()

                for i in range(NODES):  # convey the predicted gcl and get states of queue
                    self.next_state[i + 1], reward, self.done = self.step(i + 1, qlen)
                    rewards_all.append(reward)
                    for a in range(PRIORITY_QUEUES):  # model training
                        self.agents[a].observation(self.state[i + 1][a], gcl[a], reward, self.next_state[i + 1][a],
                                                   self.done)
                        loss.append(self.agents[a].replay())  # training

                    self.state = self.next_state
                    gcl = [list(self.agents[a].choose_action(self.state[i + 1][a])) for a in
                           range(PRIORITY_QUEUES)]  # new state로 gcl 업데이트
                    self.nodes[i].gcl_update(gcl)
                    #print(gcl)

                if self.total_timeslots % UPDATE == 0:
                    print("Target models update")
                    for a in range(len(self.agents)):
                        self.agents[a].update_target_model()
            # Episode ends
            self.end_time = env.now
            log_ = pd.DataFrame([(episode_num, self.end_time - self.start_time, self.timeslots, np.sum(rewards_all),
                                  epsilon, min(loss), self.success)],
                                columns=['Episode', 'Duration', 'Slots', 'Score', 'Epsilon', 'min_loss',
                                         'success_rate'])
            self.log = self.log.append(log_, ignore_index=True)

            if (self.total_episode >= 100) and (self.loss_min >= np.min(loss)):
                self.loss_min = min(loss)
                i = 0
                for agent in self.agents:
                    i += 1
                    agent.model.save_model(
                        "./result/"+DATE+"_1_train/" + "agent[" + str(i) + "]" + ".h5")
                self.log.to_csv("./result/"+DATE+"_1_train/log_"+DATE+"_train_1.csv")
                np.savez(DATE+"_1_npz", delay=self.e2e_delay, stateaction=self.state_and_action)

            e = time.time() - s
            print("실제소요시간 : %s 분 %s 초, 예상소요시간 : %s 시간" % (
                int(e / 60), int(e % 60), int(e * (MAX_EPISODE - self.total_episode) / 3600)))
            print(
                "Episode {p}, Score: {s}, Final Step: {t}, Duration: {n}, Epsilon: {e} , Min loss: {m}, success: {l}".format(
                    p=episode_num,
                    s=np.sum(rewards_all),
                    t=self.timeslots,
                    n=self.end_time - self.start_time,
                    e=epsilon,
                    m=np.min(loss),
                    l=self.success))

    # TODO:학습파라미터 세팅
    def step(self, node, qlen):
        # qlen 은 전체노드를 참조하고있음
        rewards = self.reward1(qlen[node-1])
        # if node > 4 :
        #     rewards = self.reward2(self.fail[node-5])
        #self.reward1(node, qlen[node - 1]) +
        # print (rewards)
        hops = [3, 3, 2, 1, 0, 0]
        qt = qlen.transpose()
        previous_node = [0 for _ in range(PRIORITY_QUEUES)]

        if 2 < node < 5:  # 3,4 node
            # previous_node = list(map(sum, qt[:node - 1]))
            previous_node = [sum(qt[c][:node - 1]) for c in range(PRIORITY_QUEUES)]
        elif node > 4:  # 5,6 node
            previous_node = [0.5 * sum(qt[c][:node - 1]) for c in range(PRIORITY_QUEUES)]
        # state
        state = np.zeros((PRIORITY_QUEUES, INPUT_SIZE))
        state[:, 0] = qlen[node - 1]  # 해당노드 큐에 남아서 대기중인 패킷 개수 (queue legnth)
        state[:, 1] = previous_node  # 이전 노드들 큐의 합
        #state[:, 2] = hops[node - 1]  # 전송까지 남은 홉수
        # state[:, 3] = self.success  # deadline 맞춘 패킷
        # state[:, 4] = self.received_packet #전송 완료된 패킷
        # print(state)
        # reward
        # done
        done = False
        if self.received_packet == COMMAND_CONTROL + AUDIO + VIDEO + BEST_EFFORT:  # originally (CC + A + V + BE)
            done = True
        return [state, rewards, done]

    # def reward2(self,fail):
    #     w = np.array([2, 2, 1, 0.1])
    #     r = -round(sum(w * np.array(fail)), 1)
    #     # print ("r2", r)
    #     return r

    def reward1(self, q_len):
        # print(q_len)
        w = [1, 1, 1, 0.05]
        r = 0
        # reward 1
        for i in range(PRIORITY_QUEUES):
            r -= round(q_len[i] * w[i], 1)
        return r

    def run(self):
        self.env.process(self.episode(self.env))
        self.env.run(until=1000000)
        # i = 0
        # for agent in self.agents:
        #     i += 1
        #     agent.model.save_model("./result/1017_1_train/" + "agent[" + str(i) + "]" + str(min(self.loss_min)) + ".h5")
        # self.log.to_csv("./result/1019_1_train/log_1019_train_1.csv")


if __name__ == "__main__":
    environment = GateControlSimulation()
    environment.run()
