#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import simpy
from node import Node
from agent import Agent
import warnings
import time
import tensorflow as tf
from parameter import *

tf.compat.v1.disable_eager_execution()
warnings.filterwarnings('ignore')


class GateControlSimulation:

    def __init__(self):
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
        self.time_step = 0
        self.total_episode = 0
        self.total_timeslots = 0  # episode
        self.timeslots = 0  # timestep
        self.start_time = self.env.now
        state_shape = np.zeros(INPUT_SIZE)
        self.state = state_shape
        # self.state = {1: state_shape,
        #               2: state_shape,
        #               3: state_shape,
        #               4: state_shape,
        #               5: state_shape,
        #               6: state_shape}
        self.done = False
        self.done_timestep = 0
        # self.reward = [0 for _ in range(NODES)]
        self.next_state = state_shape
        # self.next_state = {1: state_shape,
        #                    2: state_shape,
        #                    3: state_shape,
        #                    4: state_shape,
        #                    5: state_shape,
        #                    6: state_shape}
        self.end_time = 0  # episode 끝났을 때
        self.received_packet = 0  # 전송완료 패킷수
        self.remain_packet = [0, 0]
        self.loss_min = 999999

        # save logs
        self.log = pd.DataFrame(
            columns=['Episode', 'Duration', 'Slots', 'Score', 'Epsilon', 'min_loss', 'c&c', 'audio', 'video', 'be'])
        self.delay = pd.DataFrame(columns=['Episode', 'c&c', 'audio', 'video', 'be'])

        self.success = [0, 0, 0, 0]
        self.s = [0, 0, 0, 0]  # arrived within deadline at the timestep
        self.delay_at_timestep = [[], [], [], []]
        self.avg_delay = [[], [], [], []]
        self.gate_control_list = []

    def timestep_reset(self):
        # self.start_time = self.env.now
        # for n in range(NODES):
        #     self.nodes[n].timestep_reset(self.start_time)
        self.time_step += 1
        self.timeslots = 0
        self.s = [0, 0, 0, 0]  # arrived within deadline at the timestep
        # self.avg_delay = [[], [], [], []]
        self.delay_at_timestep = [[], [], [], []]
        self.done = False
        self.received_packet = 0

    def episode_reset(self):  # initial state, new episode start
        self.start_time = self.env.now  # episode 시작

        for n in range(NODES):
            self.nodes[n].reset(self.env, self.start_time)

        self.total_episode += 1
        self.timeslots = 0
        # self.total_timeslots = 0
        self.cnt1 = 0  # 전송된 flow 개수 카운트
        self.cnt2 = 0
        self.cnt3 = 0
        self.cnt4 = 0
        state_shape = np.zeros(INPUT_SIZE)
        self.state = state_shape
        # self.state = {1: state_shape,
        #               2: state_shape,
        #               3: state_shape,
        #               4: state_shape,
        #               5: state_shape,
        #               6: state_shape}
        self.next_state = state_shape
        # self.next_state = {1: state_shape,
        #                    2: state_shape,
        #                    3: state_shape,
        #                    4: state_shape,
        #                    5: state_shape,
        #                    6: state_shape}
        self.done = False
        self.done_timestep = 0
        # self.reward = [0 for _ in range(NODES)]
        self.end_time = 0
        self.received_packet = 0
        self.remain_packet = [0, 0]
        self.success = [0, 0, 0, 0]  # deadline met
        self.avg_delay = [[], [], [], []]
        self.delay_at_timestep = [[], [], [], []]
        self.s = [0, 0, 0, 0]
        self.gate_control_list = []
        # epsilon_decay
        e = self.agent.reset()
        return e

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
            f.deadline_ = 0.5
            f.generated_time_ = time - self.start_time
            f.queueing_delay_ = 0
            f.node_arrival_time_ = 0
            f.arrival_time_ = 0
            f.bits_ = BE_BYTE * 8
            f.met_ = -1
            f.hops_ = 4

        return f

    def generate_cc(self, env, p1):
        r = True
        for i in range(COMMAND_CONTROL):
            # print("c&c flow를 생성합니다.", env.now - self.start_time)
            yield env.timeout(p1 / 1000)
            if r:
                flow1 = self.flow_generator(env.now, 1, i)
                yield env.process(self.nodes[0].packet_in(env.now, flow1))
            else:
                flow2 = self.flow_generator(env.now, 1, i)
                yield env.process(self.nodes[1].packet_in(env.now, flow2))
            self.cnt1 += 1
            r = not r

    def generate_ad(self, env, p2):
        r = True
        for i in range(AUDIO):
            # print("audio flow를 생성합니다.", env.now - self.start_time)
            yield env.timeout(p2 / 1000)
            if r:
                flow1 = self.flow_generator(env.now, 2, i)
                yield env.process(self.nodes[0].packet_in(env.now, flow1))
            else:
                flow2 = self.flow_generator(env.now, 2, i)
                yield env.process(self.nodes[1].packet_in(env.now, flow2))
            self.cnt2 += 1
            r = not r

    def generate_vd(self, env, p3):
        r = True
        for i in range(VIDEO):
            # print("video flow를 생성합니다.", env.now - self.start_time)
            yield env.timeout(p3 / 1000)
            if r:
                flow1 = self.flow_generator(env.now, 3, i)
                yield env.process(self.nodes[0].packet_in(env.now, flow1))
            else:
                flow2 = self.flow_generator(env.now, 3, i)
                yield env.process(self.nodes[1].packet_in(env.now, flow2))
            self.cnt3 += 1
            r = not r

    def generate_be(self, env, p4):
        r = True
        for i in range(BEST_EFFORT):
            # print("be flow를 생성합니다.", env.now - self.start_time)
            yield env.timeout(p4 / 1000)
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
                # self.reward[dpid - 1] += f.reward_
                yield env.process(self.nodes[route[dpid - 1]].packet_in(env.now, f))

        elif dpid > 4:  # transmission completed
            for _ in range(len(flows.items)):
                # print("@@@@@TRANSMISSION COMPLETED@@@@@")
                f = yield flows.get()
                # self.reward[dpid - 1] += f.reward_
                t = f.type_ - 1
                # print("전송완료")
                self.received_packet += 1
                f.arrival_time_ = env.now - self.start_time
                delay = f.bits_ / 20000000.0 + f.queueing_delay_ + f.node_arrival_time_ - f.generated_time_  # f.arrival_time_ - f.generated_time_
                self.avg_delay[t].append(delay)
                self.delay_at_timestep[t].append(delay)
                if delay <= f.deadline_:
                    f.met_ = 1
                    self.success[t] += 1
                    self.s[t] += 1
                else:
                    f.met_ = 0

        else:  # The Node4 sends packets to node5 or node6 according to the packet number
            # print('{} 노드에서 전송(sendTo)'.format(dpid))
            # print('    전송할 trans_dict:', flows.items)
            for _ in range(len(flows.items)):
                f = yield flows.get()
                # self.reward[dpid - 1] += f.reward_
                n = f.num_
                if not n % 2:
                    yield env.process(self.nodes[4].packet_in(env.now, f))
                else:
                    yield env.process(self.nodes[5].packet_in(env.now, f))

    def episode(self, env):
        epsilon = EPSILON_MAX
        # gcl = {1: number_to_action(0),
        #        2: number_to_action(0),
        #        3: number_to_action(0),
        #        4: number_to_action(0),
        #        5: number_to_action(0),
        #        6: number_to_action(0)}
        gcl = number_to_action(INITIAL_ACTION)

        for episode_num in range(MAX_EPISODE):
            # initial step
            epsilon = self.episode_reset()

            s = time.time()
            rewards_all = []
            loss = []
            u1 = random.randint(1, 11)
            u2 = random.randint(1, 11)
            # print("u1, u2", u1, u2)
            p1, p2, p3, p4 = util_calculation(u1, u2)
            self.state = [u1, u2, 0, 0]
            gcl = self.agent.choose_action(self.state)
            self.gate_control_list.append(action_to_number(gcl))
            for n in range(NODES):
                self.nodes[n].gcl_update(gcl)
            # for n in range(NODES):
            #     gcl[n + 1] = self.agent.choose_action(self.state[n + 1])

            for _ in range(6):  # time step
                self.timestep_reset()
                env.process(self.generate_cc(env, p1))
                env.process(self.generate_ad(env, p2))
                env.process(self.generate_vd(env, p3))
                env.process(self.generate_be(env, p4))

                qlen = np.zeros((NODES, PRIORITY_QUEUE))  # flow type
                qdelay = np.zeros((NODES, PRIORITY_QUEUE))

                while not self.done:  # timeslots

                    self.timeslots += GCL_LENGTH
                    self.total_timeslots += GCL_LENGTH

                    for t in range(GCL_LENGTH):
                        for n in range(NODES):
                            env.process(self.nodes[n].packet_out(env, self.trans_dict[n + 1], t))
                            env.process(self.sendTo_next_node(env, n + 1))
                        yield env.timeout(TIMESLOT_SIZE / 1000)

                    now = self.env.now
                    for i in range(NODES):
                        qdelay[i], qlen[i] = self.nodes[i].queue_info(now)

                    self.observe_done(gcl, qlen)

                # step ends
                self.s[2] //= VIDEO_FRAME
                u1 = random.randint(1, 11)
                u2 = random.randint(1, 11)
                # print('u1, u2:', u1, u2 )
                p1, p2, p3, p4 = util_calculation(u1, u2)

                self.next_state, reward = self.step(qdelay, u1, u2, gcl)
                rewards_all.append(reward)
                if self.done_timestep == 6:
                    d = True
                else:
                    d = False
                self.agent.observation(self.state, action_to_number(gcl), reward,
                                       self.next_state, d)
                # print(self.state, action_to_number(gcl), reward,
                #       self.next_state, d)
                self.state = self.next_state
                gcl = self.agent.choose_action(self.state)  # new state로 gcl 업데이트
                self.gate_control_list.append(action_to_number(gcl))
                for n in range(NODES):
                    self.nodes[n].gcl_update(gcl)

                loss.append(self.agent.replay())  # train

            if self.total_episode % UPDATE == 0:
                print("Target models update")
                print("action distribution : ", np.unique(self.gate_control_list, return_counts=True))
                # print(self.success)
                self.agent.update_target_model()

            # Episode ends
            self.success[2] //= VIDEO_FRAME
            # print(self.success)
            # print(self.avg_delay)
            # print(sum(rewards_all))
            self.end_time = env.now
            log_ = pd.DataFrame([(episode_num, self.end_time - self.start_time, self.timeslots, np.sum(rewards_all),
                                  epsilon, min(loss), self.success[0], self.success[1], self.success[2],
                                  self.success[3])],
                                columns=['Episode', 'Duration', 'Slots', 'Score', 'Epsilon', 'min_loss',
                                         'c&c', 'audio', 'video', 'be'])

            delay_ = pd.DataFrame([(episode_num, np.mean(self.avg_delay[0]),
                                    np.mean(self.avg_delay[1]), np.mean(self.avg_delay[2]),
                                    np.mean(self.avg_delay[3]))],
                                  columns=['Episode', 'c&c', 'audio', 'video', 'be'])

            self.log = self.log.append(log_, ignore_index=True)
            self.delay = self.delay.append(delay_, ignore_index=True)

            # if ((self.total_episode >= 1500) and (self.loss_min >= np.min(loss))) or episode_num == MAX_EPISODE - 1:
            if (self.total_episode >= 8000) and (self.loss_min > min(loss)):
                self.loss_min = min(loss)
                self.agent.model.save_model(
                    "./result/" + DATE + "/" + "[" + str(episode_num) + "]" + str(min(loss)) + ".h5")
                self.log.to_csv("./result/" + DATE + "/log_" + DATE + ".csv")
                self.delay.to_csv("./result/" + DATE + "/avg_delay_" + DATE + ".csv")
                # np.save("./result/" + DATE + "_S&A.npy", self.state_and_action)
            e = time.time() - s

            print("소요시간 : %s 초, 예상소요시간 : %s 시간" % (
                round(e % 60, 2), round(e * (MAX_EPISODE - self.total_episode) / 3600, 2)))

            print("Episode {p}, Score: {s}, Final Step: {t}, Epsilon: {e} , Min loss: {m}, success: {l}, "
                  "avg_delay: {d}".format(
                p=episode_num,
                s=np.sum(rewards_all),
                t=self.timeslots,
                e=round(epsilon, 4),
                m=round(np.min(loss), 4),
                l=self.success,
                d=[round(np.mean(self.avg_delay[0]), 4),
                   round(np.mean(self.avg_delay[1]), 4), round(np.mean(self.avg_delay[2]), 4)]))

        print("simulation ends")
        self.agent.model.save_model(
            "./result/" + DATE + "/" + "[" + str(episode_num) + "]" + str(min(loss)) + ".h5")
        self.log.to_csv("./result/" + DATE + "/log_" + DATE + ".csv")
        self.delay.to_csv("./result/" + DATE + "/avg_delay_" + DATE + ".csv")

    def observe_done(self, gcl, qlen):
        gcl_ = action_to_number(gcl)
        q1 = qlen[0][0] + qlen[1][0]
        q2 = qlen[0][1] + qlen[1][1]
        total = COMMAND_CONTROL + AUDIO + VIDEO + BEST_EFFORT
        # print("total, remain", total, self.remain_packet)
        # print(gcl_, q1, q2, self.received_packet)
        if gcl_ == 0:  # p2 suspend|
            if self.received_packet + q2 == total + self.remain_packet[0] + self.remain_packet[1]:
                self.done = True
                self.done_timestep += 1
                self.remain_packet[0] = 0
                self.remain_packet[1] = q2
                # print("d")
        elif gcl_ == 1:  # p1 suspend
            if self.received_packet + q1 == total + self.remain_packet[0] + self.remain_packet[1]:
                self.done = True
                self.done_timestep += 1
                self.remain_packet[0] = q1
                self.remain_packet[1] = 0
                # print("d")
        else:
            if self.received_packet == total + self.remain_packet[0] + self.remain_packet[1]:
                self.done = True
                self.done_timestep += 1
                self.remain_packet = [0, 0]
                # print("d")

    # def initial_state(self, u1, u2):
    # hops = [3, 3, 2, 1, 0, 0]

    # s = [u1, u2, 0, 0]
    # self.state = {1: s + [hops[0]],
    #               2: s + [hops[1]],
    #               3: s + [hops[2]],
    #               4: s + [hops[3]],
    #               5: s + [hops[4]],
    #               6: s + [hops[5]]}

    def step(self, delay, u1, u2, gcl):
        # reward = self.reward1(qlen[node - 1])
        pd1 = []
        pd2 = []
        for i in range(NODES):
            pd1.append(delay[i][0])
            pd2.append(delay[i][1])
        # hops = [3, 3, 2, 1, 0, 0]
        #print("pd1", pd1)
        s = [u1, u2, np.mean(pd1), np.mean(pd2)]
        state = s
        reward = self.reward_function(action_to_number(gcl))
        # print (reward)

        # TODO: REWARD avg delay 기반으로 만들고, state에 timestep별 avg_delay로 변경
        return [state, reward]

    def reward_function(self, gcl):
        r = 0
        deadline = [CC_DEADLINE / 1000, AD_DEADLINE / 1000, VD_DEADLINE / 1000, BE_DEADLINE / 1000]
        a = [(deadline[i] - np.mean(self.delay_at_timestep[i]))*W[i] for i in range(3)]
        b = W[3] * (deadline[3] - np.mean( self.delay_at_timestep[3]))

        if (gcl != 1) and (gcl != 0):
            s = [sum(self.s[:3]), self.s[3]]
            r = round(s[0]+s[1]*0.2,1)
            #r = round(sum(np.array([sum(a), b]))*10, 1)
        return r

    # def reward1(self, q_len):
    #     # print(q_len)
    #     r = 0
    #     # reward 1
    #     for i in range(PRIORITY_QUEUE):  # flow type
    #         r -= q_len[i] * W[i]
    #     return r

    def run(self):
        self.env.process(self.episode(self.env))
        self.env.run(until=1000000)


if __name__ == "__main__":
    environment = GateControlSimulation()
    environment.run()
