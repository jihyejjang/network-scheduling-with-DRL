#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import simpy
from node import Node
from src import Src
from ddqn_agent import Agent
import warnings
import time
import tensorflow as tf
from parameter import *

tf.compat.v1.disable_eager_execution()
warnings.filterwarnings('ignore')


class GateControlSimulation:

    def __init__(self):
        self.env = simpy.Environment()
        self.start_time = self.env.now
        self.sources = [Src(n + 1, self.start_time) for n in range(SRCES)]
        self.nodes = [Node(n + 1, self.start_time) for n in range(NODES)]
        self.agent = Agent()
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
        self.state = {1: state_shape,
                      2: state_shape,
                      3: state_shape,
                      4: state_shape,
                      5: state_shape,
                      6: state_shape,
                      7: state_shape,
                      8: state_shape,
                      9: state_shape}
        self.reward = [0 for _ in range(NODES)]
        self.next_state = {1: state_shape,
                           2: state_shape,
                           3: state_shape,
                           4: state_shape,
                           5: state_shape,
                           6: state_shape,
                           7: state_shape,
                           8: state_shape,
                           9: state_shape}
        self.done = [0 for _ in range(NODES)]
        self.end_time = 0
        self.received_packet = [0 for _ in range(NODES)]
        self.success = [[0, 0, 0], [0, 0, 0]]

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
        self.reward_max = 0

    def reset(self):
        self.start_time = self.env.now
        self.sources = [Src(n + 1, self.start_time) for n in range(SRCES)]
        self.nodes = [Node(n + 1, self.start_time) for n in range(NODES)]
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
        self.state = {1: state_shape,
                      2: state_shape,
                      3: state_shape,
                      4: state_shape,
                      5: state_shape,
                      6: state_shape,
                      7: state_shape,
                      8: state_shape,
                      9: state_shape}
        self.next_state = {1: state_shape,
                           2: state_shape,
                           3: state_shape,
                           4: state_shape,
                           5: state_shape,
                           6: state_shape,
                           7: state_shape,
                           8: state_shape,
                           9: state_shape}
        self.reward = [0 for _ in range(NODES)]
        self.done = [0 for _ in range(NODES)]
        self.end_time = 0
        self.success = [0, 0]
        self.avg_delay = [[], []]
        self.estimated_e2e = [[], []]
        self.received_packet = [0 for _ in range(NODES)]
        self.success = [[0, 0, 0], [0, 0, 0]]

        e = self.agent.reset()
        return e

    def sendTo_next_node(self, env, output):

        for i, pkts in enumerate(output):
            if not len(pkts.items):
                return
            for _ in range(len(pkts.items)):
                node = i
                self.reward[node-1] += A
                packet = yield pkts.get()
                p = packet.priority_ - 1
                src = packet.src_
                ET = (packet.queueing_delay_ + packet.current_delay_ + packet.remain_hops_) * TIMESLOT_SIZE / 1000

                if ET <= packet.deadline_:
                    self.reward[node] += W[p]

                    if not packet.route_:
                        # received
                        self.received_packet[src - 1] += 1
                        packet.met_ = 1
                        self.success[p][src - 1] += 1
                        delay = env.now() - packet.generated_time_
                        self.avg_delay[p].append(delay)
                        self.estimated_e2e[p].append(ET)
                else:
                    if not packet.route_:
                        # received
                        self.received_packet[src - 1] += 1
                        packet.met_ = 0
                        delay = env.now() - packet.generated_time_
                        self.avg_delay[p].append(delay)
                        self.estimated_e2e[p].append(ET)

    def episode(self, env):
        start_time = time.time()
        epsilon = EPSILON_MAX

        for episode_num in range(MAX_EPISODE):
            self.total_episode += 1

            for i in range(SRCES):
                env.process(self.sources[i].send(env, self.nodes))

            s = time.time()
            rewards_all = []
            loss = []
            gcl = [INITIAL_ACTION for _ in range(NODES)]

            while not sum(self.done) == NODES:
                self.timeslots += 1
                for n in range(NODES):
                    yield env.process(self.nodes[n].packet_out(self.output[n+1]))
                env.process(self.sendTo_next_node(env, self.output))
                yield env.timeout(TIMESLOT_SIZE / 1000)

                for n in range(NODES):
                    state = self.nodes[n].queue_info()
                    self.next_state[n], self.done[n] = self.step(state)
                    self.agent.observation(self.state[n], gcl[n], self.reward[n], self.next_state[n], self.done[n])

                rewards_all.append(self.reward)
                self.reward = [0 for _ in range(NODES)]
                self.state = self.next_state

                for n in range(NODES):
                    gcl[n] = self.agent.choose_action(self.state[n])  # new state로 gcl 업데이트
                    self.nodes[n].gcl_update(gcl)

                self.gate_control_list.append(gcl)
                loss.append(self.agent.replay())  # train

            if self.total_episode % UPDATE == 0:
                print("Target models update")
                print("action distribution : ", np.unique(self.gate_control_list, return_counts=True))
                f.write("action distribution : {a} \n".format(
                    a=np.unique(self.gate_control_list, return_counts=True)))
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

            if (self.total_episode >= MAX_EPISODE / 2) and (self.reward_max < sum(rewards_all)):
                self.reward_max = sum(rewards_all)
                self.agent.model.save_model(
                    "./result/" + DATE + "/" + "[" + str(episode_num) + "]" + str(round(self.reward_max, 2)) + ".h5")
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
            "./result/" + DATE + "/" + "[" + str(MAX_EPISODE) + "]" + str(round(self.reward_max, 2)) + ".h5")
        self.log.to_csv("./result/" + DATE + "/log_last" + DATE + ".csv")
        self.delay.to_csv("./result/" + DATE + "/avg_delay_last" + DATE + ".csv")
        save_result_plot(self.log)
        end_time = time.time()
        duration = end_time - start_time
        f.write("total simulation time : {h} h {m} m \n".format(
            h=int(duration / 3600), m=int((duration / 3600) % 60)))
        f.close()

    def fifo(self, env):

        for episode_num in range(MAX_EPISODE):
            self.total_episode += 1
            env.process(self.generate_cc(env))
            env.process(self.generate_be(env))
            s = time.time()
            rewards_all = []

            while not self.done:
                self.timeslots += 1

                yield env.process(self.node.packet_FIFO_out(self.trans_list))
                # print("gcl, trans_list", gcl, self.trans_list.items)
                # print("node", time.time())
                env.process(self.sendTo_next_node(env))
                yield env.timeout(TIMESLOT_SIZE / 1000)
                rewards_all.append(self.reward)
                self.reward = 0

                _, self.done = self.step()

            self.end_time = env.now
            log_ = pd.DataFrame([(episode_num, self.end_time - self.start_time, self.timeslots, np.sum(rewards_all),
                                  0, 0, self.success[0], self.success[1])],
                                columns=['Episode', 'Duration', 'Slots', 'Score', 'Epsilon', 'min_loss',
                                         'p1', 'p2'])
            self.log = self.log.append(log_, ignore_index=True)
            e = time.time() - s

            print("소요시간 : %s 초, 예상소요시간 : %s 시간" % (
                round(e % 60, 2), round(e * (MAX_EPISODE - self.total_episode) / 3600, 2)))

            print("Episode {p}, Score: {s}, Final Step: {t}, Epsilon: {e} , Min loss: {m}, success: {l}, "
                  "avg_qdelay: {d}".format(
                p=episode_num,
                s=np.sum(rewards_all),
                t=self.timeslots,
                e=0,
                m=0,
                l=self.success,
                d=list(map(np.mean, self.avg_delay))))
            print("simulation ends")
            self.reset()

        self.log.to_csv("./result/" + DATE + "/log_last" + DATE + ".csv")
        save_result_plot(self.log)

    def step(self, states=None):
        qlen = states[0]
        max_et = states[1]
        state = np.zeros((PRIORITY_QUEUE, STATE))
        state[:, 0] = qlen
        state[:, 1] = max_et
        state = state.flatten()

        done = 0

        if MAXSLOT_MODE:
            if (self.received_packet == COMMAND_CONTROL + BEST_EFFORT) or (self.timeslots == MAXSLOTS):
                done = 1
        else:
            if self.received_packet == COMMAND_CONTROL + BEST_EFFORT:  # originally (CC + A + V + BE)
                done = 1

        return [state, done]

    def run(self):
        self.env.process(self.episode(self.env))
        # self.env.process(self.fifo(self.env))
        self.env.run(until=1000000)


if __name__ == "__main__":
    environment = GateControlSimulation()
    environment.run()
