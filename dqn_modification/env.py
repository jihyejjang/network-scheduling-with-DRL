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
import os

warnings.filterwarnings('ignore')

DATE = '1118'
if not os.path.exists("./result/" + DATE):
    os.makedirs("./result/" + DATE)
PRIORITY_QUEUE = 2
STATE = 2
STATE_SIZE = STATE * PRIORITY_QUEUE
GCL_LENGTH = 3
ACTION_SIZE = 2 ** (GCL_LENGTH * PRIORITY_QUEUE)
MAX_EPISODE = 4500
COMMAND_CONTROL = 40
AUDIO = 8
VIDEO_FRAME = 30
VIDEO = 2 * VIDEO_FRAME
BEST_EFFORT = 20
CC_PERIOD = 5  # milliseconds #TODO: originally 5 (simulation duration을 위해 잠시 줄임)
AD_PERIOD = 1
VD_PERIOD = 1.1
BE_PERIOD = 10
TIMESLOT_SIZE = 0.6
NODES = 6  # switch의 수
UPDATE = 3000
W = [6, 0.05]


def action_to_number(action):
    action_ = action.flatten()
    bin = ''
    for a in action_:
        bin += str(a)
    return int(bin, 2)


def number_to_action(action_id):  # number -> binary gcl code
    b_id = format(action_id, '06b')
    action_ = np.array(list(map(int, b_id)))
    return action_.reshape((PRIORITY_QUEUE, GCL_LENGTH))


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
    priority_: int = None


class GateControlSimulation:

    def __init__(self):
        self.success_at_episode = [0, 0, 0, 0]  # deadline met
        self.env = simpy.Environment()
        self.nodes = [Node(n + 1, self.env) for n in range(NODES)]
        self.agent = Agent()
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

        # save logs
        self.log = pd.DataFrame(
            columns=['Episode', 'Duration', 'Slots', 'Score', 'Epsilon', 'min_loss', 'c&c', 'audio', 'video'])
        self.delay = pd.DataFrame(columns=['Episode', 'c&c', 'audio', 'video'])

        self.success = [0, 0, 0, 0]
        self.avg_delay = [[], [], [], []]
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

        # epsilon_decay
        e = self.agent.reset()
        return e

    def flow_generator(self, time, type_num, fnum):  # flow structure에 맞게 flow생성, timestamp등 남길 수 있음

        f = Flow()

        if type_num == 1:  # c&c
            f.type_ = 1
            f.priority_ = 1
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
            f.type_ = 2
            f.priority_ = 2
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
            f.type_ = 3
            f.priority_ = 2
            f.num_ = fnum
            f.deadline_ = 0.030
            f.generated_time_ = time - self.start_time
            f.queueing_delay_ = 0
            f.node_arrival_time_ = 0
            f.arrival_time_ = 0
            f.bits_ = 1500 * 8
            f.met_ = -1
            f.hops_ = 4

        else:  # best effort
            f.type_ = 4
            f.priority_ = 2
            f.num_ = fnum
            f.deadline_ = 0.3
            f.generated_time_ = time - self.start_time
            f.queueing_delay_ = 0
            f.node_arrival_time_ = 0
            f.arrival_time_ = 0
            f.bits_ = 1024 * 8
            f.met_ = -1
            f.hops_ = 4

        return f

    def generate_cc(self, env):
        r = True
        for i in range(COMMAND_CONTROL):
            # print("c&c flow를 생성합니다.", env.now - self.start_time)
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
            # print("audio flow를 생성합니다.", env.now - self.start_time)
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
            # print("video flow를 생성합니다.", env.now - self.start_time)
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
            # print("be flow를 생성합니다.", env.now - self.start_time)
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
            # print('{} 노드에서 {}노드로 전송(sendTo)'.format(dpid, route[dpid - 1] + 1))
            # print('    전송할 trans_dict:', flows.items)
            for _ in range(len(flows.items)):
                f = yield flows.get()
                yield env.process(self.nodes[route[dpid - 1]].packet_in(env.now, f))
            # yield env.timeout(TIMESLOT_SIZE / 1000)

        elif dpid > 4:  # transmission completed
            # yield env.timeout(TIMESLOT_S애IZE / 1000)
            for _ in range(len(flows.items)):
                # print("@@@@@TRANSMISSION COMPLETED@@@@@")
                f = yield flows.get()
                t = f.type_ - 1
                self.received_packet += 1
                f.arrival_time_ = env.now - self.start_time
                delay = f.arrival_time_ - f.generated_time_
                self.avg_delay[t].append(delay)
                if delay <= f.deadline_:
                    f.met_ = 1
                    self.success[t] += 1
                else:
                    f.met_ = 0
            # yield env.timeout(TIMESLOT_SIZE / 1000)
            # self.fail[dpid-5][c] += 1
            # print("전송 완료 패킷:", f)
            # print("delay :", delay)

        else:  # The Node4 sends packets to node5 or node6 according to the packet number
            # print('{} 노드에서 전송(sendTo)'.format(dpid))
            # print('    전송할 trans_dict:', flows.items)
            for _ in range(len(flows.items)):
                f = yield flows.get()
                n = f.num_
                if not n % 2:
                    yield env.process(self.nodes[4].packet_in(env.now, f))
                else:
                    yield env.process(self.nodes[5].packet_in(env.now, f))
            # yield env.timeout(TIMESLOT_SIZE / 1000)

    def episode(self, env):  # mainprocess
        # cnt = 1
        for episode_num in range(MAX_EPISODE):
            # print("****** 에피소드" + str(self.total_episode) + "시작 ******")
            s = time.time()
            rewards_all = []
            gcl = number_to_action(63)
            loss = []
            epsilon = self.reset()
            # episode 시작 시 마다 flow generator process를 실행
            env.process(self.generate_cc(env))
            env.process(self.generate_ad(env))
            env.process(self.generate_vd(env))
            env.process(self.generate_be(env))

            while not self.done:  # 1회의 episode가 종료될 때 까지 cycle을 반복하는 MAIN process
                # s = [[0, 0, 0, 0] for _ in range(2)]
                # self.fail = [[0, 0, 0, 0] for _ in range(2)]
                self.timeslots += GCL_LENGTH
                self.total_timeslots += GCL_LENGTH

                for t in range(GCL_LENGTH):
                    for n in range(NODES):
                        env.process(self.nodes[n].packet_out(env, self.trans_dict[n + 1], t))
                        env.process(self.sendTo_next_node(env, n + 1))
                    yield env.timeout(TIMESLOT_SIZE / 1000)

                # training starts when a timeslot cycle has finished
                qlen = np.zeros((NODES, PRIORITY_QUEUE))  # flow type
                qdata = np.zeros((NODES, PRIORITY_QUEUE))

                for i in range(NODES):
                    qlen[i], qdata[i] = self.nodes[i].queue_length()

                for n in range(NODES):  # convey the predicted gcl and get states of queue
                    self.next_state[n + 1], reward, self.done = self.step(n + 1, qlen, qdata)
                    rewards_all.append(reward)
                    self.agent.observation(self.state[n + 1], action_to_number(gcl), reward,
                                           self.next_state[n + 1], self.done)
                    self.state[n + 1] = self.next_state[n + 1]
                    gcl = self.agent.choose_action(self.state[n + 1])  # new state로 gcl 업데이트
                    self.nodes[n].gcl_update(gcl)
                    if (episode_num % 100 == 0) or (episode_num == MAX_EPISODE - 1):  # logging states and actions
                        self.state_and_action.append([episode_num, self.state[n + 1], gcl])

                loss.append(self.agent.replay())  # train

                if self.total_timeslots % UPDATE == 0:
                    # print("Target models update")
                    # print(self.success)
                    self.agent.update_target_model()

            # Episode ends
            self.end_time = env.now
            log_ = pd.DataFrame([(episode_num, self.end_time - self.start_time, self.timeslots, np.sum(rewards_all),
                                  epsilon, min(loss), self.success[0], self.success[1], self.success[2])],
                                columns=['Episode', 'Duration', 'Slots', 'Score', 'Epsilon', 'min_loss',
                                         'c&c', 'audio', 'video'])

            delay_ = pd.DataFrame([(episode_num, np.mean(self.avg_delay[0]),
                                    np.mean(self.avg_delay[1]), np.mean(self.avg_delay[2]))],
                                  columns=['Episode', 'c&c', 'audio', 'video'])

            self.log = self.log.append(log_, ignore_index=True)
            self.delay = self.delay.append(delay_, ignore_index=True)

            if ((self.total_episode >= 1000) and (self.loss_min >= np.min(loss))) or episode_num == MAX_EPISODE - 1:
                self.loss_min = min(loss)
                self.agent.model.save_model(
                    "./result/" + DATE + "/" + "[" + str(episode_num) + "]" + str(min(loss)) + ".h5")
                self.log.to_csv("./result/" + DATE + "/log_" + DATE + ".csv")
                self.delay.to_csv("./result/" + DATE + "/avg_delay_" + DATE + ".csv")
                np.save("./result/" + DATE + "_S&A.npy", self.state_and_action)
            e = time.time() - s
            print("실제소요시간 : %s 분 %s 초, 예상소요시간 : %s 시간" % (
                int(e / 60), int(e % 60), int(e * (MAX_EPISODE - self.total_episode) / 3600)))
            print(
                "Episode {p}, Score: {s}, Final Step: {t}, Duration: {n}, Epsilon: {e} , Min loss: {m}, success: {l}, "
                "avg_delay: {d}".format(
                    p=episode_num,
                    s=np.sum(rewards_all),
                    t=self.timeslots,
                    n=round(self.end_time - self.start_time, 4),
                    e=round(epsilon, 4),
                    m=round(np.min(loss), 4),
                    l=self.success[:3],
                    d=[round(np.mean(self.avg_delay[0]),4),
                       round(np.mean(self.avg_delay[1]),4), round(np.mean(self.avg_delay[2]),4)]))

    # TODO:학습파라미터 세팅
    def step(self, node, qlen, qdata):
        # qlen 은 전체노드를 참조하고있음
        rewards = self.reward1(qlen[node - 1])
        # if node > 4 :
        #     rewards = self.reward2(self.fail[node-5])
        # self.reward1(node, qlen[node - 1]) +
        # print (rewards)
        # hops = [3, 3, 2, 1, 0, 0]
        qt = qdata.transpose()
        previous_node = [0 for _ in range(PRIORITY_QUEUE)]

        if 2 < node < 5:  # 3,4 node
            # previous_node = list(map(sum, qt[:node - 1]))
            previous_node = [sum(qt[c][:node - 1]) for c in range(PRIORITY_QUEUE)]
        elif node > 4:  # 5,6 node
            previous_node = [0.5 * sum(qt[c][:node - 1]) for c in range(PRIORITY_QUEUE)]

        # state
        # available_data = [d if d <= 1500 else 1500 for d in qdata[node - 1]]

        state = np.zeros((PRIORITY_QUEUE, STATE))
        # state[:, 0] = [1, 2]  # priority
        state[:, 0] = qdata[node - 1]  # 해당노드 큐에 남아서 대기중인 패킷 개수 (queue legnth)
        state[:, 1] = previous_node  # 이전 노드들 큐의 합
        # state[:, 3] = available_data
        state = state.flatten()

        done = False
        if self.received_packet == COMMAND_CONTROL + AUDIO + VIDEO + BEST_EFFORT:  # originally (CC + A + V + BE)
            done = True
            if node == 6:
                self.success[2] //= VIDEO_FRAME
        return [state, rewards, done]

    # def reward2(self,fail):
    #     w = np.array([2, 2, 1, 0.1])
    #     r = -round(sum(w * np.array(fail)), 1)
    #     # print ("r2", r)
    #     return r

    def reward1(self, q_len):
        # print(q_len)
        r = 0
        # reward 1
        for i in range(PRIORITY_QUEUE):  # flow type
            r -= round(q_len[i] * W[i], 1)
        return r

    def run(self):
        self.env.process(self.episode(self.env))
        self.env.run(until=1000000)


if __name__ == "__main__":
    environment = GateControlSimulation()
    environment.run()
