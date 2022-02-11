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
        self.node = Node(1, self.env)
        self.agent = Agent()
        self.trans_list = simpy.Store(self.env)
        self.cnt1 = 0  # 전송된 flow 개수 카운트
        # self.cnt2 = 0
        # self.cnt3 = 0
        self.cnt4 = 0
        self.timeslots = 0
        self.start_time = self.env.now
        state_shape = np.zeros(INPUT_SIZE)
        self.state = state_shape
        self.reward = 0
        self.next_state = state_shape
        self.done = False
        self.end_time = 0
        self.received_packet = 0
        self.loss_min = 999999

        # save logs
        self.log = pd.DataFrame(
            columns=['Episode', 'Duration', 'Slots', 'Score', 'Epsilon', 'min_loss', 'p1', 'p2'])
        self.delay = pd.DataFrame(columns=['Episode', 'p1_q', 'p2_q', 'p1_e', 'p2_e'])
        self.success = [0, 0]
        self.avg_delay = [[], []]
        self.estimated_e2e = [[], []]
        self.gate_control_list = []
        self.state_and_action = []

        self.total_episode = 0
        self.sequence_p1, self.sequence_p2 = random_sequence()

    def reset(self):
        self.start_time = self.env.now
        self.node = Node(1, self.env)
        self.trans_list = simpy.Store(self.env)
        self.timeslots = 0
        self.cnt1 = 0
        # self.cnt2 = 0
        # self.cnt3 = 0
        self.cnt4 = 0
        self.timeslots = 0
        state_shape = np.zeros((PRIORITY_QUEUE * STATE))
        self.state = state_shape
        self.next_state = state_shape
        self.reward = 0
        self.done = False
        self.received_packet = 0
        self.end_time = 0
        self.success = [0, 0]
        self.avg_delay = [[], []]
        self.estimated_e2e = [[], []]

        e = self.agent.reset()
        return e

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

    def episode(self, env):

        epsilon = EPSILON_MAX

        for episode_num in range(MAX_EPISODE):
            self.total_episode += 1
            env.process(self.generate_cc(env))
            env.process(self.generate_be(env))
            s = time.time()
            rewards_all = []
            loss = []
            gcl = INITIAL_ACTION

            while not self.done:
                self.timeslots += 1

                yield env.process(self.node.packet_out(self.trans_list))
                #print("gcl, trans_list", gcl, self.trans_list.items)
                #print("node", time.time())
                env.process(self.sendTo_next_node(env))
                yield env.timeout(TIMESLOT_SIZE / 1000)
                t = self.env.now

                # training starts when a timeslot cycle has finished
                qlen, max_et = self.node.queue_info()  # state에 필요한 정보 (Q_p, maxET, index)
                self.next_state, self.done = self.step(qlen, max_et)
                # print(self.state)
                if any(self.state):
                    self.agent.observation(self.state, gcl, self.reward, self.next_state, self.done)
                    #if gcl==0:
                        #print("State,  Reward",self.state, self.reward)
                rewards_all.append(self.reward)
                self.reward = 0
                self.state = self.next_state
                gcl = self.agent.choose_action(self.state)  # new state로 gcl 업데이트
                self.node.gcl_update(gcl)
                self.gate_control_list.append(gcl)
                loss.append(self.agent.replay())  # train

            if self.total_episode % UPDATE == 0:
                print("Target models update")
                print("action distribution : ", np.unique(self.gate_control_list, return_counts=True))
                f.write("action distribution : {a} \n".format(
                    a = np.unique(self.gate_control_list, return_counts=True)))
                self.gate_control_list = []
                # print(self.success)
                self.agent.update_target_model()

            # Episode ends
            self.end_time = env.now
            log_ = pd.DataFrame([(episode_num, self.end_time - self.start_time, self.timeslots, np.sum(rewards_all),
                                  epsilon, min(loss), self.success[0], self.success[1])],
                                columns=['Episode', 'Duration', 'Slots', 'Score', 'Epsilon', 'min_loss',
                                         'p1', 'p2'])

            delay_ = pd.DataFrame([(episode_num, np.mean(self.avg_delay[0]),
                                    np.mean(self.avg_delay[1]), np.mean(self.estimated_e2e[0]),
                                    np.mean(self.estimated_e2e[1]))],
                                  columns=['Episode', 'p1_q', 'p2_q', 'p1_e', 'p2_e'])

            self.log = self.log.append(log_, ignore_index=True)
            self.delay = self.delay.append(delay_, ignore_index=True)

            if (self.total_episode >= MAX_EPISODE/2) and (self.loss_min > min(loss)):
                self.loss_min = min(loss)
                self.agent.model.save_model(
                    "./result/" + DATE + "/" + "[" + str(episode_num) + "]" + str(min(loss)) + ".h5")
                #self.log.to_csv("./result/" + DATE + "/log_" + DATE + ".csv")
                #self.delay.to_csv("./result/" + DATE + "/avg_delay_" + DATE + ".csv")
                # np.save("./result/" + DATE + "_S&A.npy", self.state_and_action)
            e = time.time() - s

            print("소요시간 : %s 초, 예상소요시간 : %s 시간" % (
                round(e % 60, 2), round(e * (MAX_EPISODE - self.total_episode) / 3600, 2)))

            print("Episode {p}, Score: {s}, Final Step: {t}, Epsilon: {e} , Min loss: {m}, success: {l}, "
                  "avg_qdelay: {d}".format(
                p=episode_num,
                s=np.sum(rewards_all),
                t=self.timeslots,
                e=round(epsilon, 4),
                m=round(np.min(loss), 4),
                l=self.success,
                d=list(map(np.mean, self.avg_delay))))

            epsilon = self.reset()

        print("simulation ends")
        self.agent.model.save_model(
            "./result/" + DATE + "/" + "[" + str(episode_num) + "]" + str(min(loss)) + "_last.h5")
        self.log.to_csv("./result/" + DATE + "/log_last" + DATE + ".csv")
        self.delay.to_csv("./result/" + DATE + "/avg_delay_last" + DATE + ".csv")
        draw_result(self.log)
        f.close()

    def step(self, qlen, max_et):
        state = np.zeros((PRIORITY_QUEUE, STATE))
        state[:, 0] = qlen
        state[:, 1] = max_et
        #state[:, 2] = max_qp
        state = state.flatten()

        done = False

        if MAXSLOT_MODE :
            if (self.received_packet == COMMAND_CONTROL + BEST_EFFORT) or (self.timeslots == MAXSLOTS):
                done = True
        else:
            if self.received_packet == COMMAND_CONTROL + BEST_EFFORT:  # originally (CC + A + V + BE)
                done = True

        return [state, done]

    # def reward_function(self, gcl):
    #     r = 0
    #     deadline = [CC_DEADLINE / 1000, AD_DEADLINE / 1000, VD_DEADLINE / 1000, BE_DEADLINE / 1000]
    #     a = [(deadline[i] - np.mean(self.delay_at_timestep[i]))*W[i] for i in range(3)]
    #     b = W[3] * (deadline[3] - np.mean( self.delay_at_timestep[3]))
    #
    #     if (gcl != 1) and (gcl != 0):
    #         s = [sum(self.s[:3]), self.s[3]]
    #         r = round(s[0]+s[1]*0.2,1)
    #         #r = round(sum(np.array([sum(a), b]))*10, 1)
    #     return r

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
