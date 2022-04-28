import pandas as pd
import simpy
from node import Node
from src import Src
import time
import tensorflow as tf
from parameter import *


# max_burst()

# utilization()


def operator(delay):
    delay = np.mean(delay * 1000)
    return delay


class GateControlTestSimulation:

    def __init__(self):
        self.model = tf.keras.models.load_model(WEIGHT_FILE)
        self.env = simpy.Environment()
        self.start_time = self.env.now
        self.nodes = [Node(n + 1, self.env, self.start_time) for n in range(NODES)]
        self.output = {1: simpy.Store(self.env),
                       2: simpy.Store(self.env),
                       3: simpy.Store(self.env),
                       4: simpy.Store(self.env),
                       5: simpy.Store(self.env),
                       6: simpy.Store(self.env),
                       7: simpy.Store(self.env),
                       8: simpy.Store(self.env),
                       9: simpy.Store(self.env)}
        self.seq = random_sequence()
        self.source = Src(1, self.start_time, self.seq)
        self.timeslots = 0
        state_shape = [np.zeros(INPUT_SIZE) for _ in range(OUTPUT_PORT)]
        self.state = [state_shape for _ in range(NODES)]
        self.reward = [0 for _ in range(NODES)]
        self.next_state = [state_shape for _ in range(NODES)]
        self.done = False
        self.end_time = 0
        self.received_packet = 0

        # save logs
        # self.ex = pd.DataFrame(columns=['timeslot', 'state', 'gcl'])
        # self.ap = pd.DataFrame(columns=['timeslot', 'state', 'gcl'])
        self.node0 = pd.DataFrame(columns=['node', 'timeslot', 'state', 'gcl'])
        self.node1 = pd.DataFrame(columns=['node', 'timeslot', 'state', 'gcl'])
        self.node2 = pd.DataFrame(columns=['node', 'timeslot', 'state', 'gcl'])
        self.node3 = pd.DataFrame(columns=['node', 'timeslot', 'state', 'gcl'])
        self.node5 = pd.DataFrame(columns=['node', 'timeslot', 'state', 'gcl'])
        self.node6 = pd.DataFrame(columns=['node', 'timeslot', 'state', 'gcl'])
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
        self.success = [0 for _ in range(SRCES)]
        # self.state_and_action = []

    def reset(self):  # initial state, new episode start
        self.start_time = self.env.now  # episode 시작
        self.nodes = [Node(n + 1, self.env, self.start_time) for n in range(NODES)]
        self.source = Src(1, self.start_time, self.seq)
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
        state_shape = [np.zeros(INPUT_SIZE) for _ in range(OUTPUT_PORT)]
        self.state = [state_shape for _ in range(NODES)]
        self.next_state = [state_shape for _ in range(NODES)]
        self.reward = [0 for _ in range(NODES)]
        self.done = False
        self.end_time = 0
        self.received_packet = 0
        self.success = [0 for _ in range(SRCES)]
        self.avg_delay = [[], []]
        self.node0 = pd.DataFrame(columns=['node', 'timeslot', 'state', 'gcl'])
        self.node1 = pd.DataFrame(columns=['node', 'timeslot', 'state', 'gcl'])
        self.node2 = pd.DataFrame(columns=['node', 'timeslot', 'state', 'gcl'])
        self.node3 = pd.DataFrame(columns=['node', 'timeslot', 'state', 'gcl'])
        self.node5 = pd.DataFrame(columns=['node', 'timeslot', 'state', 'gcl'])
        self.node6 = pd.DataFrame(columns=['node', 'timeslot', 'state', 'gcl'])
        self.node7 = pd.DataFrame(columns=['node', 'timeslot', 'state', 'gcl'])
        self.node8 = pd.DataFrame(columns=['node', 'timeslot', 'state', 'gcl'])

    def ddqn_predict(self, state):
        n = self.model.predict(state)
        id_ = np.argmax(n)
        # print (id_)
        # action = number_to_action(id_)
        return id_

    def sendTo_next_node(self, env, output):

        for i, pkts in output.items():
            if not len(pkts.items):
                continue
            for _ in range(len(pkts.items)):
                node = i - 1
                self.reward[node] += A
                packet = yield pkts.get()
                h = packet.remain_hops_
                l = len(packet.queueing_delay_)
                # print (h, l)
                q = packet.queueing_delay_[l - h - 2]
                packet.current_delay_ += q + 1

                p = packet.priority_ - 1
                src = packet.src_
                # print(packet.current_delay_)
                if packet.remain_hops_ < 0:
                    packet.remain_hops_ = 0
                # ET = packet.current_delay_ + packet.remain_hops
                ET = packet.current_delay_ + packet.remain_hops_ + packet.random_delay_
                # print (ET)
                if ET <= packet.deadline_:
                    self.reward[node] += W[p]
                # if node == 1:
                #     print("priority", p, "out")
                # transmission completed
                if packet.route_ == [0]:
                    self.received_packet += 1
                    if node == 6 or node == 8:
                        # self.avg_delay[p].append(packet.queueing_delay_)
                        self.avg_delay[p].append(ET)

                    if ET <= packet.deadline_:
                        packet.met_ = 1
                        # print (src - 1)
                        self.success[src - 1] += 1
                    else:
                        packet.met_ = 0
                else:
                    r = packet.route_[0]
                    yield env.process(self.nodes[r - 1].packet_in(packet))

    def src_send(self, env):
        env.process(self.source.send(env, self.nodes, 1))
        env.process(self.source.send(env, self.nodes, 2))
        env.process(self.source.send(env, self.nodes, 3))
        env.process(self.source.send(env, self.nodes, 4))
        env.process(self.source.send(env, self.nodes, 5))
        env.process(self.source.send(env, self.nodes, 6))
        env.process(self.source.send(env, self.nodes, 7))
        env.process(self.source.send(env, self.nodes, 8))

    def rl_agent(self, env):  # mainprocess
        s = time.time()
        rewards_all = []
        a = [INITIAL_ACTION for _ in range(OUTPUT_PORT)]
        action = [a for _ in range(NODES)]

        self.src_send(self.env)

        while not self.done:
            self.timeslots += 1

            for n in range(NODES):
                yield env.process(self.nodes[n].link(self.output[n + 1], 'ddqn'))
            yield env.process(self.sendTo_next_node(env, self.output))
            yield env.timeout(TIMESLOT_SIZE / 1000)

            for n in range(NODES):
                self.next_state[n] = self.nodes[n].step()
                self.done = self.terminated()
            rewards_all.extend(self.reward)
            self.reward = [0 for _ in range(NODES)]
            self.state = self.next_state

            # observe(생략)

            for n in range(NODES):
                p = self.nodes[n].schedulable()
                # p = [0, 1]
                for i in p:
                    # print("node:", n, self.state[n][i])
                    action[n][i] = self.ddqn_predict(
                        np.array(self.state[n][i]).reshape(1, INPUT_SIZE))
                    self.nodes[n].action_update(action[n][i], i)

        # Episode ends
        self.end_time = env.now
        e = time.time() - s
        print("DDQN 소요시간", e, "score", np.sum(rewards_all))
        # print("Score: {s}, Final Step: {t}, success: {l}, avg_qdelay: {d}".format(
        #     s=np.sum(rewards_all),
        #     t=self.timeslots,
        #     l=self.success,
        #     d=list(map(np.mean, self.avg_delay))))
        # log_ = pd.DataFrame([(self.timeslots, np.sum(rewards_all), self.success[0], self.success[1])],
        #                     columns=['Slots', 'Score', 'p1', 'p2'])

        # self.log1 = self.log1.append(log_, ignore_index=True)
        # self.node0.to_csv('result/test/node0.csv')
        # self.node1.to_csv('result/test/node1.csv')
        # self.node2.to_csv('result/test/node2.csv')
        # self.node3.to_csv('result/test/node3.csv')
        # self.node5.to_csv('result/test/node5.csv')
        # self.node6.to_csv('result/test/node6.csv')
        # self.node7.to_csv('result/test/node7.csv')
        # self.node8.to_csv('result/test/node8.csv')

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
        s = time.time()
        rewards_all = []

        self.src_send(self.env)

        while not self.done:  # 1회의 episode가 종료될 때 까지 cycle을 반복하는 MAIN process
            self.timeslots += 1
            # log0 = pd.DataFrame([('1', self.timeslots, self.state[0])],
            #                     columns=['node', 'timeslot', 'state'])
            # log1 = pd.DataFrame([('2', self.timeslots, self.state[1])],
            #                     columns=['node', 'timeslot', 'state'])
            # log2 = pd.DataFrame([('3', self.timeslots, self.state[2])],
            #                     columns=['node', 'timeslot', 'state'])
            # log3 = pd.DataFrame([('4', self.timeslots, self.state[3])],
            #                     columns=['node', 'timeslot', 'state']),
            # log5 = pd.DataFrame([('6', self.timeslots, self.state[5])],
            #                     columns=['node', 'timeslot', 'state'])
            # log6 = pd.DataFrame([('7', self.timeslots, self.state[6])],
            #                     columns=['node', 'timeslot', 'state'])
            # log7 = pd.DataFrame([('8', self.timeslots, self.state[7])],
            #                     columns=['node', 'timeslot', 'state'])
            # log8 = pd.DataFrame([('9', self.timeslots, self.state[8])],
            #                     columns=['node', 'timeslot', 'state'])

            for n in range(NODES):
                yield env.process(self.nodes[n].link(self.output[n + 1], 'sp'))
            yield env.process(self.sendTo_next_node(env, self.output))
            yield env.timeout(TIMESLOT_SIZE / 1000)

            for n in range(NODES):
                # self.next_state[n] = self.nodes[n].step()
                self.done = self.terminated()
            rewards_all.extend(self.reward)
            self.reward = [0 for _ in range(NODES)]
            # self.state = self.next_state

            # self.node0 = self.node0.append(log0, ignore_index=True)
            # self.node1 = self.node1.append(log1, ignore_index=True)
            # self.node2 = self.node2.append(log2, ignore_index=True)
            # self.node3 = self.node3.append(log3, ignore_index=True)
            # self.node5 = self.node5.append(log5, ignore_index=True)
            # self.node6 = self.node6.append(log6, ignore_index=True)
            # self.node7 = self.node7.append(log7, ignore_index=True)
            # self.node8 = self.node8.append(log8, ignore_index=True)

        e = time.time() - s
        print("SP 소요시간", e, "score", np.sum(rewards_all))
        # print("Score: {s}, Final Step: {t}, success: {l}, avg_qdelay: {d}".format(
        #     s=np.sum(rewards_all),
        #     t=self.timeslots,
        #     l=self.success,
        #     d=list(map(np.mean, self.avg_delay))))
        log_ = pd.DataFrame([(self.timeslots, np.sum(rewards_all), self.success[0], self.success[1])],
                            columns=['Slots', 'Score', 'p1', 'p2'])

        self.log3 = self.log3.append(log_, ignore_index=True)
        # self.node0.to_csv('result/test/SP/node0.csv')
        # self.node1.to_csv('result/test/SP/node1.csv')
        # self.node2.to_csv('result/test/SP/node2.csv')
        # self.node3.to_csv('result/test/SP/node3.csv')
        # self.node5.to_csv('result/test/SP/node5.csv')
        # self.node6.to_csv('result/test/SP/node6.csv')
        # self.node7.to_csv('result/test/SP/node7.csv')
        # self.node8.to_csv('result/test/SP/node8.csv')

    def model_simulation(self, i, model='ddqn'):
        if model == 'ddqn':
            self.env.process(self.rl_agent(self.env))
            self.env.run()
            D1 = self.avg_delay[0]
            D2 = self.avg_delay[1]
            d1 = pd.DataFrame(D1)
            d2 = pd.DataFrame(D2)
            d1.to_csv('result/test/delay1.csv')
            d2.to_csv('result/test/delay2.csv')
            if i > 0:
                print("DDQN+H , Final Step: {t}, success: {l}, avg_qdelay: {d}".format(
                    t=self.timeslots,
                    l=self.success,
                    d=list(map(operator, self.avg_delay))))
            self.reset()

        elif model == 'sp':
            self.env.process(self.SP(self.env))
            self.env.run()

            D1 = self.avg_delay[0]
            D2 = self.avg_delay[1]
            d1 = pd.DataFrame(D1)
            d2 = pd.DataFrame(D2)
            d1.to_csv('result/test/SP/delay1.csv')
            d2.to_csv('result/test/SP/delay2.csv')
            if i > 0:
                print("SP , Final Step: {t}, success: {l}, avg_qdelay: {d}".format(
                    t=self.timeslots,
                    l=self.success,
                    d=list(map(operator, self.avg_delay))))
            self.reset()

        elif model == 'reset':
            print("reset")
            self.seq = random_sequence()
            self.source = Src(1, self.start_time, self.seq)

    def simulation(self):
        iteration = 2
        for i in range(iteration):
            self.model_simulation(i, 'sp')
            self.model_simulation(i, 'ddqn')
            self.model_simulation(i, 'reset')
            # DDQN with heuristic simulation
            # self.env.process(self.rl_agent(self.env))
            # self.env.run()
            # D1 = self.avg_delay[0]
            # D2 = self.avg_delay[1]
            # d1 = pd.DataFrame(D1)
            # d2 = pd.DataFrame(D2)
            # d1.to_csv('result/test/delay1.csv')
            # d2.to_csv('result/test/delay2.csv')
            # self.reset()

            # SP

        # self.log1.to_csv("result/test/extract.csv")
        # self.log2.to_csv("result/test/apply.csv")
        # self.log3.to_csv("result/test/fifo.csv")

    def terminated(self):
        done = False

        if MAXSLOT_MODE:
            if (self.received_packet == COMMAND_CONTROL * 2 + BEST_EFFORT * 6) or (self.timeslots >= MAXSLOTS):
                done = True
        else:
            if self.received_packet == COMMAND_CONTROL * 2 + BEST_EFFORT * 6:  # originally (CC + A + V + BE)
                done = True

        return done


if __name__ == "__main__":
    test = GateControlTestSimulation()
    test.simulation()
