# ----------------------------------------
# Autor : Jihye Ryu, jh_r_1004@naver.com
# ----------------------------------------

import pandas as pd
import simpy
from src import Src
from node import Node
from agent import Agent
import warnings
import time
import tensorflow as tf
from utils import *
import os
tf.compat.v1.disable_eager_execution()
warnings.filterwarnings('ignore')


class TrainSchedulingSimulation:

    def __init__(self, args):
        self.single_node = args.env
        if self.single_node:
            print("@@@@@@@@@@@@@@@@@@",args.env)
        self.work_conserving = args.workconserving
        self.save = args.save
        self.withflows = args.withflows
        self.seed = args.seed
        self.w = args.reward
        self.a = args.reward2
        self.max_episode = args.totalepisode
        self.np = args.numberofpackets
        self.rh = args.randomhops
        self.rcd = args.randomcurrentdelay

        if self.single_node:
            self.output_port = 1
        else:
            self.output_port = 2
        
        self.env = simpy.Environment()
        self.start_time = self.env.now
        self.seq = train_random_sequence(self.seed, self.np, self.rh, self.rcd)
        self.source = Src(self.start_time, self.seq, args)
        self.agent = Agent(args)
        self.timeslots = 0
        state_shape = [np.zeros(INPUT_SIZE) for _ in range(OUTPUT_PORT)]
        self.done = False
        self.end_time = 0
        self.received_packet = 0
        self.score = 0 
        
        if self.single_node:
            self.node = Node(1, self.env, self.start_time, args)
            self.state = state_shape
            self.next_state = state_shape
            self.transmission = simpy.Store(self.env)
            self.reward = 0
            self.success = [0, 0]
        else:
            self.node = [Node(n + 1, self.env, self.start_time, args) for n in range(NODES)]
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
            self.date = args.date
            if not os.path.exists("./result/" + self.date):
                os.makedirs("./result/" + self.date)
            self.folder = "./result/"+self.date+"/"
        
        self.log = pd.DataFrame(
                columns=['Episode', 'Duration', 'Slots', 'Score', 'Epsilon', 'min_loss', 'p1', 'p2'])
        self.delay = pd.DataFrame(columns=['Episode', 'p1', 'p2'])
        
        self.estimated_delay = [[], []] # in topology, it traces node only 6(7) or 8(9) 
        self.queueing_delay = [[],[]]
        self.total_episode = 0
        self.reward_max = 0

    def reset(self):
        self.start_time = self.env.now
        self.seq = train_random_sequence(self.seed, self.np, self.rh, self.rcd)
        self.source.reset(self.start_time, self.seq)
        self.timeslots = 0
        self.score = 0 
        state_shape = [np.zeros(INPUT_SIZE) for _ in range(OUTPUT_PORT)]
        
        if self.single_node:
            self.node.reset(self.start_time)
            self.state = state_shape
            self.next_state = state_shape
            self.transmission = simpy.Store(self.env)
            self.reward = 0
            self.success = [0, 0]
        else:
            for node in self.node:
                node.reset(self.start_time)
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
            r += self.w[p] + self.a
        else:
            packet.met_ = 0
            r += self.a

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
                    self.reward[node] += self.a
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
                        
                    if packet.route_ == [0]:
                        self.received_packet += 1
                        
                        if node == 6 or node == 8: # the edge nodes of flow 1, 2
                            self.avg_delay[p].append(ET)

                        if ET <= packet.deadline_:
                            packet.met_ = 1
                            self.reward[node] += self.w[p]
                            self.success[src - 1] += 1
                        else:
                            packet.met_ = 0
                    else:
                        r = packet.route_[0]
                        yield env.process(self.node[r - 1].route_modify(packet))

    def train_episode(self, env): # episode for singlenode
        start_time = time.time()
        epsilon = EPSILON_MAX

        for episode_num in range(self.max_episode):
            self.total_episode += 1
            rewards_all = []
            action = [INITIAL_ACTION for _ in range(self.output_port)]
            self.packet_generation(env)
            
            s = time.time()
            loss = []
            schedulable_ports = [i for i in range(self.output_port)]

            while not self.done:
                self.timeslots += 1
                
                # print(self.node)
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
                    loss.append(self.agent.replay())  # train


                self.reward = 0
                self.state = self.next_state
                schedulable_ports = self.node.schedulable()
                for p in schedulable_ports:
                    action[p] = self.agent.choose_action(self.state[p])  # new state로 gcl 업데이트
                    self.node.action_update(action[p], p)
            
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

            if self.save:
                if (self.total_episode >= self.max_episode / 2) and (self.reward_max < sum(rewards_all)):
                    self.reward_max = sum(rewards_all)
                    self.agent.model.save_model(
                        self.folder + "[" + str(episode_num) + "]" + str(round(self.reward_max, 2)) + ".h5")

            e = time.time() - s

            print("소요시간 : %s 초, 예상소요시간 : %s 시간" % (
                round(e % 60, 2), round(e * (self.max_episode - self.total_episode) / 3600, 2)))

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
        self.agent.model.save_model(self.folder + "[" + str(self.max_episode) + "]" + str(round(self.reward_max, 2)) + ".h5")
        self.log.to_csv(self.folder + "log_last" + self.date + ".csv")
        # self.delay.to_csv(self.folder "avg_et_last" + self.date + ".csv")
        save_result_plot(self.log,self.folder)
        end_time = time.time()
        duration = end_time - start_time
        print("total simulation time : {h} h {m} m \n".format(
            h=int(duration / 3600), m=int((duration / 3600) % 60)))


    def terminated(self):
        done = False
        if MAXSLOT_MODE:
            if (self.received_packet == self.np[0] + self.np[1]) or (self.timeslots == MAXSLOTS):
                done = True
        else:
            if self.received_packet == self.np[0] + self.np[1]:  
                done = True

        return done

    def packet_generation(self,env):
        if self.single_node:
            env.process(self.source.send(env, self.node, 1))
            env.process(self.source.send(env, self.node, 4))
        else:
            if self.withflows:
                for s in range(SRCES):
                    env.process(self.source[s].send(env, self.nodes, s + 1))
            else:
                env.process(self.source.send(env, self.nodes, 1))
                env.process(self.source.send(env, self.nodes, 2))
                env.process(self.source.send(env, self.nodes, 7))
                env.process(self.source.send(env, self.nodes, 8))
                

    def run(self):
        self.env.process(self.train_episode(self.env))
        self.env.run(until=1000000)


# if __name__ == "__main__":
#     import argparse
#     parser = argparse.ArgumentParser(description = "The training environment of timeslot scheduling with DDQN")
    
    
    
#     environment = TrainSchedulingSimulation()
#     environment.run()
