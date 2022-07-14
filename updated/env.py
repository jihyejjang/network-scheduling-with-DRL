# ----------------------------------------
# Autor : Jihye Ryu, jh_r_1004@naver.com
# ----------------------------------------

# from re import S
import pandas as pd
import simpy
from src import Src
from node import Node
from agent import Agent
import warnings
import time
import tensorflow as tf
from parameter import *

tf.compat.v1.disable_eager_execution()
warnings.filterwarnings('ignore')


class TrainSchedulingSimulation:

    def __init__(self, args):
        self.single_node = args.env
        if self.single_node:
            self.output_port = 1
        else:
            self.output_port = 2
        self.work_conserving = args.workconserving
        self.save = args.save
        self.withflows = args.withflows
        self.env = simpy.Environment()
        self.start_time = self.env.now
        self.seq = random_sequence()
        self.source = Src(self.start_time, self.seq, self.single_node)
        self.agent = Agent()
        self.timeslots = 0
        state_shape = [np.zeros(INPUT_SIZE) for _ in range(OUTPUT_PORT)]
        self.done = False
        self.end_time = 0
        self.received_packet = 0
        self.score = 0 
        
        if self.single_node:
            self.node = Node(1, self.env, self.start_time, self.single_node, self.work_conserving)
            self.state = state_shape
            self.next_state = state_shape
            self.transmission = simpy.Store(self.env)
            self.reward = 0
            self.success = [0, 0]
        else:
            self.node = [Node(n + 1, self.env, self.start_time, self.single_node, self.work_conserving) for n in range(NODES)]
            self.state = [state_shape for _ in range(NODES)]
            self.next_state = [state_shape for _ in range(NODES)]
            self.transmission = {1: simpy.Store(self.env),
                       2: simpy.Store(self.env),
                       3: simpy.Store(self.env),
                       4: simpy.Store(self.env),
                       5: simpy.Store(self.env),
                       6: simpy.Store(self.env),
                       7: simpy.Store(self.env),
                       8: simpy.Store(self.env),
                       9: simpy.Store(self.env)}
            self.reward = [0 for _ in range(NODES)]
            self.success = [0 for _ in range(SRCES)]

        if self.save:
            self.logs = pd.DataFrame(
                columns=['Episode', 'Duration', 'Slots', 'Score', 'Epsilon', 'min_loss', 'p1', 'p2'])
            self.delay = pd.DataFrame(columns=['Episode', 'p1', 'p2'])
        
        self.estimated_delay = [[], []] # in topology, it traces node only 6(7) or 8(9) 
        self.queueing_delay = [[],[]]
        self.total_episode = 0
        self.reward_max = 0

    def reset(self):
        self.start_time = self.env.now
        self.seq = random_sequence()
        self.source = Src(self.start_time, self.seq, self.single_node)
        self.timeslots = 0
        self.score = 0 
        state_shape = [np.zeros(INPUT_SIZE) for _ in range(OUTPUT_PORT)]
        
        if self.single_node:
            self.node = Node(1, self.env, self.start_time)
            self.state = state_shape
            self.next_state = state_shape
            self.transmission = simpy.Store(self.env)
            self.reward = 0
            self.success = [0, 0]
        else:
            self.node = [Node(n + 1, self.env, self.start_time) for n in range(NODES)]
            state_shape = np.zeros((INPUT_SIZE))
            self.state = [state_shape for _ in range(NODES)]
            self.next_state = [state_shape for _ in range(NODES)]
            self.transmission = {1: simpy.Store(self.env),
                       2: simpy.Store(self.env),
                       3: simpy.Store(self.env),
                       4: simpy.Store(self.env),
                       5: simpy.Store(self.env),
                       6: simpy.Store(self.env),
                       7: simpy.Store(self.env),
                       8: simpy.Store(self.env),
                       9: simpy.Store(self.env)}
            self.reward = [0 for _ in range(NODES)]
            self.success = [0 for _ in range(SRCES)]

        self.done = False
        self.received_packet = 0
        self.end_time = 0
        self.seq = random_sequence()
        self.estimated_delay = [[], []]
        self.queueing_delay = [[],[]]
        eps = self.agent.reset()
        
        return eps

    def reward_single(self, packet):
        r = 0
        p = packet.priority_ - 1
        packet.current_delay_ += packet.queueing_delay_
        et = packet.random_delay_ + packet.current_delay_ + packet.remain_hops_ + 1
        packet.queueing_delay_ = 0
        packet.arrival_time_ = self.env.now - self.start_time
        self.estimated_delay[p].append(et)
        self.queueing_delay[p].append(packet.current_delay_)

        dl = packet.deadline_
        if et / dl <= 1:
            packet.met_ = 1
            self.success[p] += 1
            r += W[p] + A
        else:
            packet.met_ = 0
            r += A

        return r

    def sendTo_next_node(self, env):
        if self.single_node:
            # transmission completed immediatly
            flows = self.transmission
            if not flows.items:
                return
            for _ in range(len(flows.items)):
                flow = yield flows.get()
                self.received_packet += 1

                self.reward += self.reward_single(flow)
        else:
            transmission = self.transmission
            for i, pkts in transmission.items():
                if not pkts.items:
                    continue
                for _ in range(len(pkts.items)):
                    node = i - 1
                    self.reward[node] += A
                    packet = yield pkts.get()
                    h = packet.remain_hops_
                    l = len(packet.queueing_delay_)
                    q = packet.queueing_delay_[l - h - 2]
                    packet.current_delay_ += q + 1

                    p = packet.priority_ - 1
                    src = packet.src_
                    if packet.remain_hops_ < 0:
                        packet.remain_hops_ = 0
                    ET = packet.current_delay_ + packet.remain_hops_ + packet.random_delay_
                    
                    if node == 1 or node == 7: # competition node
                        self.queueing_delay[p].append(q)

                    # if ET <= packet.deadline_:
                        # self.reward[node] += W[p]
                        
                    if packet.route_ == [0]:
                        self.received_packet += 1
                        
                        if node == 6 or node == 8: # the edge nodes of flow 1, 2
                            self.avg_delay[p].append(ET)

                        if ET <= packet.deadline_:
                            packet.met_ = 1
                            self.reward[node] += W[p]
                            self.success[src - 1] += 1
                        else:
                            packet.met_ = 0
                    else:
                        r = packet.route_[0]
                        yield env.process(self.nodes[r - 1].packet_in(packet))


    def train_episode(self, env):
        start_time = time.time()
        epsilon = EPSILON_MAX

        for episode_num in range(MAX_EPISODE):
            self.total_episode += 1
            rewards_all = []
            action = [INITIAL_ACTION for _ in range(self.output_port)]

            env.process(self.source.send(env, self.node, 1))
            env.process(self.source.send(env, self.node, 4))
            
            s = time.time()
            loss = []
            schedulable_ports = [i for i in range(self.output_port)]

            while not self.done:
                self.timeslots += 1

                yield env.process(self.node.scheduling(self.transmission, 'ddqn'))
                yield env.process(self.sendTo_next_node(env))  # packet arrive and get reward
                yield env.timeout(TIMESLOT_SIZE * 0.001)

                self.next_state = self.node.state_observe()
                self.done = self.terminated()
                rewards_all.append(self.reward)

                # observe sample for schedulable ports
                for p in schedulable_ports:
                    s_ = np.array(self.state[p]).reshape(INPUT_SIZE)
                    ns_ = np.array(self.next_state[p]).reshape(INPUT_SIZE)
                    self.agent.observation(s_, action[p], self.reward, ns_, self.done)

                self.reward = 0
                self.state = self.next_state
                schedulable_ports = self.node.schedulable()
                for p in schedulable_ports:
                    action[p] = self.agent.choose_action(self.state[p])  # new state로 gcl 업데이트
                    self.node.action_update(action[p], p)
                    loss.append(self.agent.replay())  # train
            
            # The episode ends
            if self.total_episode % UPDATE == 0:
                print("Target models update")
                # print("action distribution : ", np.unique(self.gate_control_list, return_counts=True))
                # f.write("action distribution : {a} \n".format(
                #     a=np.unique(self.gate_control_list, return_counts=True)))
                # self.gate_control_list = []
                # print(self.success)
                self.agent.update_target_model()

            # Episode ends
            self.end_time = env.now
            log_ = pd.DataFrame([(episode_num, self.end_time - self.start_time, self.timeslots, np.sum(rewards_all),
                                  epsilon, min(loss), self.success[0], self.success[1])],
                                columns=['Episode', 'Duration', 'Slots', 'Score', 'Epsilon', 'min_loss',
                                         'p1', 'p2'])

            delay_ = pd.DataFrame([(episode_num, np.mean(self.estimated_delay[0]),
                                    np.mean(self.estimated_delay[1]))],
                                  columns=['Episode', 'p1', 'p2'])

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
                  "ET: {d}".format(
                p=episode_num,
                s=np.sum(rewards_all),
                t=self.timeslots,
                e=round(epsilon, 4),
                m=round(np.min(loss), 4),
                l=self.success,
                d=list(map(np.mean, self.estimated_delay))))

            epsilon = self.reset()

        #the simulation ends
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

    def terminated(self):

        done = False

        if MAXSLOT_MODE:
            if (self.received_packet == COMMAND_CONTROL + BEST_EFFORT) or (self.timeslots == MAXSLOTS):
                done = True
        else:
            if self.received_packet == COMMAND_CONTROL + BEST_EFFORT:  # originally (CC + A + V + BE)
                done = True

        return done

    # def fifo(self, env):
    #
    #     for episode_num in range(MAX_EPISODE):
    #         self.total_episode += 1
    #         env.process(self.generate_cc(env))
    #         env.process(self.generate_be(env))
    #         s = time.time()
    #         rewards_all = []
    #
    #         while not self.done:
    #             self.timeslots += 1
    #
    #             yield env.process(self.node.strict_priority(self.trans_list))
    #             # print("gcl, trans_list", gcl, self.trans_list.items)
    #             # print("node", time.time())
    #             env.process(self.sendTo_next_node(env))
    #             yield env.timeout(TIMESLOT_SIZE / 1000)
    #             rewards_all.append(self.reward)
    #             self.reward = 0
    #
    #             _, self.done = self.step()
    #
    #         self.end_time = env.now
    #         log_ = pd.DataFrame([(episode_num, self.end_time - self.start_time, self.timeslots, np.sum(rewards_all),
    #                               0, 0, self.success[0], self.success[1])],
    #                             columns=['Episode', 'Duration', 'Slots', 'Score', 'Epsilon', 'min_loss',
    #                                      'p1', 'p2'])
    #         self.log = self.log.append(log_, ignore_index=True)
    #         e = time.time() - s
    #
    #         print("소요시간 : %s 초, 예상소요시간 : %s 시간" % (
    #             round(e % 60, 2), round(e * (MAX_EPISODE - self.total_episode) / 3600, 2)))
    #
    #         print("Episode {p}, Score: {s}, Final Step: {t}, Epsilon: {e} , Min loss: {m}, success: {l}, "
    #               "avg_qdelay: {d}".format(
    #             p=episode_num,
    #             s=np.sum(rewards_all),
    #             t=self.timeslots,
    #             e=0,
    #             m=0,
    #             l=self.success,
    #             d=list(map(np.mean, self.avg_delay))))
    #         print("simulation ends")
    #         self.reset()
    #
    #     self.log.to_csv("./result/" + DATE + "/log_last" + DATE + ".csv")
    #     save_result_plot(self.log)

    # def step(self, qlen=None, max_et=None):
    #     state = np.zeros((PRIORITY_QUEUE, STATE))
    #     state[:, 0] = qlen
    #     state[:, 1] = max_et
    #     # state[:, 2] = max_qp
    #     state = state.flatten()
    #
    #     done = False
    #
    #     if MAXSLOT_MODE:
    #         if (self.received_packet == COMMAND_CONTROL + BEST_EFFORT) or (self.timeslots == MAXSLOTS):
    #             done = True
    #     else:
    #         if self.received_packet == COMMAND_CONTROL + BEST_EFFORT:  # originally (CC + A + V + BE)
    #             done = True
    #
    #     return [state, done]

    def run(self):
        self.env.process(self.train_episode(self.env))
        # self.env.process(self.fifo(self.env))
        self.env.run(until=1000000)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description = "The training environment of timeslot scheduling with DDQN")
    
    
    
    environment = TrainSchedulingSimulation()
    environment.run()
