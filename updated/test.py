import pandas as pd
import simpy
from node import Node
import time
import tensorflow as tf
from parameter import *
from src import Src


# max_burst()

# utilization()
def operator(delay):
    delay = np.mean(delay * 1000)
    return delay

class TestSchedulingSimulation:

    def __init__(self):
        self.end_time = 0
        self.model = tf.keras.models.load_model(WEIGHT_FILE)
        self.env = simpy.Environment()
        self.start_time = self.env.now
        # self.sequence_p1, self.sequence_p2 = random_sequence()
        self.seq = test_random_sequence(SEED*2)
        self.source = Src(1, self.start_time, self.seq)
        self.node = Node(1, self.env, self.start_time)
        self.trans_list = simpy.Store(self.env)
        self.cnt1 = 0
        self.cnt4 = 0
        self.timeslots = 0  # progressed timeslots in a episode
        state_shape = [np.zeros(INPUT_SIZE) for _ in range(OUTPUT_PORT)]
        self.state = state_shape
        self.next_state = state_shape
        self.reward = 0
        self.done = False
        self.received_packet = 0  # 전송완료 패킷수
        # self.loss_min = 999999
        # self.gate_control_list = [0]  # action
        # self.state_list = []

        # save logs
        self.ex = pd.DataFrame(columns=['timeslot', 'state', 'gcl'])
        self.ap = pd.DataFrame(columns=['timeslot', 'state', 'gcl'])

        self.log1 = pd.DataFrame(
            columns=['Slots', 'Score', 'p1', 'p2'])  # extract
        self.log2 = pd.DataFrame(
            columns=['Slots', 'Score', 'p1', 'p2'])  # fifo(spq)
        # self.log3 = pd.DataFrame(
        #     columns=['Slots', 'Score', 'p1', 'p2'])  # apply
        # self.delay = pd.DataFrame(columns=['p1', 'p2'])

        self.success = [0, 0]  # deadline met
        self.qdelay = [[], []]
        # self.et = []
        # self.s = [[0, 0] for _ in range(PRIORITY_QUEUE)]
        # self.state_and_action = []

    def reset(self):  # initial state, new episode start
        self.start_time = self.env.now  # episode 시작
        self.node = Node(1, self.env, self.start_time)
        self.source = Src(1, self.start_time, self.seq)
        self.trans_list = simpy.Store(self.env)
        self.timeslots = 0
        self.cnt1 = 0
        self.cnt4 = 0
        state_shape = [np.zeros(INPUT_SIZE) for _ in range(OUTPUT_PORT)]
        self.state = state_shape
        self.next_state = state_shape
        self.reward = 0
        self.done = False

        self.received_packet = 0
        self.success = [0, 0]  # deadline met
        self.qdelay = [[], []]
        # self.et = []
        # self.delay = pd.DataFrame(columns=['p1', 'p2'])
        self.ex = pd.DataFrame(columns=['timeslot', 'state', 'gcl'])
        self.ap = pd.DataFrame(columns=['timeslot', 'state', 'gcl'])

    def ddqn_predict(self, state):
        n = self.model.predict(state)
        id_ = np.argmax(n)
        # action = number_to_action(id_)
        return id_

    def reward_func(self, packet):
        r = 0
        p = packet.priority_ - 1
        packet.current_delay_ += packet.queueing_delay_
        et = packet.random_delay_ + packet.current_delay_ + packet.remain_hops_ + 1
        # if et <= 4:
        #     print(packet.random_delay_, packet.current_delay_, packet.remain_hops_)
        # et = packet.current_delay_ + packet.remain_hops_ + 1
        # delay = (packet.queueing_delay_ + 1) * TIMESLOT_SIZE / 1000
        # self.qdelay.append([p, delay])
        packet.queueing_delay_ = 0
        packet.arrival_time_ = self.env.now - self.start_time
        self.qdelay[p].append(et * 0.001 * TIMESLOT_SIZE)
        # if p == 0:
            # print(packet.random_delay_,packet.remain_hops_)
            # print(et)
            # if et>5:
            #     print(packet.random_delay_,packet.current_delay_,packet.remain_hops_)
            #     print(et)

        dl = packet.deadline_
        # print(dl, et)

        if COMPLEX:
            if BOUND[p] <= et / dl <= 1:  # packet received within configured latency boundary
                packet.met_ = 1
                self.success[p] += 1
                r += W0[p]
                # self.reward += W[t]

            elif et / dl < BOUND[p]:
                packet.met_ = 1
                self.success[p] += 1
                r += 0.01

            elif 1 < et / dl <= LM:
                packet.met_ = 0
                r += W2[p]

            else:
                packet.met_ = 0
                r -= et / dl
        else:
            if et / dl <= 1:
                packet.met_ = 1
                self.success[p] += 1
                r += W[p] + A
            else:
                packet.met_ = 0
                r += A
        return r

    def sendTo_next_node(self, env):
        flows = self.trans_list

        if not (len(flows.items)):
            return

        # transmission completed immediatly
        for _ in range(len(flows.items)):
            flow = yield flows.get()
            self.received_packet += 1

            self.reward += self.reward_func(flow)

            # t = flow.priority_ - 1

            # h = flow.remain_hops_
            # l = len(flow.queueing_delay_)
            # # print (h, l)
            # q = flow.queueing_delay_[l - h - 2]
            # packet.current_delay_ += q + 1

            # flow.current_delay_ += flow.queueing_delay_
            # delay = (flow.queueing_delay_ + 1) * TIMESLOT_SIZE / 1000
            # self.qdelay.append([t, delay])
            # flow.queueing_delay_ = 0
            # flow.arrival_time_ = env.now - self.start_time
            # ET = (flow.random_delay_ + flow.current_delay_ + flow.remain_hops_ + 1) * TIMESLOT_SIZE / 1000
            # self.et.append([t, ET])

            # if ET <= flow.deadline_:
            #     flow.met_ = 1
            #     self.success[t] += 1
            #     self.reward += W[t]
            #
            # else:
            #     flow.met_ = 0

    def rl_agent(self, env):  # mainprocess
        s = time.time()
        rewards_all = []
        action = [INITIAL_ACTION for _ in range(OUTPUT_PORT)]

        env.process(self.source.send(env, self.node, 1))
        env.process(self.source.send(env, self.node, 4))

        while not self.done:
            self.timeslots += 1
            # log = pd.DataFrame([(self.timeslots, self.state, action)], columns=['timeslot', 'state', 'gcl'])
            yield env.process(self.node.scheduling(self.trans_list, 'ddqn'))
            yield env.process(self.sendTo_next_node(env))
            yield env.timeout(TIMESLOT_SIZE * 0.001)

            self.next_state = self.node.state_observe()
            self.done = self.terminated()
            rewards_all.append(self.reward)
            self.reward = 0
            self.state = self.next_state

            # observe(생략)

            p = self.node.schedulable()
            for i in p:
                action[i] = self.ddqn_predict(
                    np.array(self.state[i]).reshape(1, INPUT_SIZE))
                self.node.action_update(action[i], i)

            # self.ex = self.ex.append(log, ignore_index=True)

        # Episode ends
        self.end_time = env.now
        e = time.time() - s
        print("extract 소요시간", e)
        log_ = pd.DataFrame([(self.timeslots, np.sum(rewards_all), self.success[0], self.success[1])],
                            columns=['Slots', 'Score', 'p1', 'p2'])

        self.log1 = self.log1.append(log_, ignore_index=True)
        # print("DDQN: {s}, Final Step: {t}, success: {l}, qdelay: {d}".format(
        #     s=np.sum(rewards_all),
        #     t=self.timeslots,
        #     l=self.success,
        #     d=list(map(np.mean, self.qdelay))))
        # delay = pd.DataFrame(self.qdelay)
        # delay.to_csv('result/test/delay.csv')
        # est = pd.DataFrame(self.et)
        # est.to_csv('result/test/et.csv')

    def SP(self, env):
        rewards_all = []
        env.process(self.source.send(env, self.node, 1))
        env.process(self.source.send(env, self.node, 4))

        while not self.done:
            self.timeslots += 1
            yield env.process(self.node.scheduling(self.trans_list, 'sp'))
            yield env.process(self.sendTo_next_node(env))
            yield env.timeout(TIMESLOT_SIZE * 0.001)

            self.next_state = self.node.state_observe()
            self.done = self.terminated()
            rewards_all.append(self.reward)
            self.reward = 0
            self.state = self.next_state
            # self.node.gcl_update(gcl)

        # print("Sp: {s}, Final Step: {t}, success: {l}, avg_qdelay: {d}".format(
        #     s=np.sum(rewards_all),
        #     t=self.timeslots,
        #     l=self.success,
        #     d=list(map(np.mean, self.qdelay))))
        # log_ = pd.DataFrame([(self.timeslots, np.sum(rewards_all), self.success[0], self.success[1])],
        #                     columns=['Slots', 'Score', 'p1', 'p2'])

        # self.log3 = self.log3.append(log_, ignore_index=True)
        # delay = pd.DataFrame(self.qdelay)
        # delay.to_csv('result/test/SP/delay.csv')
        # est = pd.DataFrame(self.et)
        # est.to_csv('result/test/SP/et.csv')

    def model_simulation(self, i, model='ddqn'):
        if model == 'ddqn':
            self.env.process(self.rl_agent(self.env))
            self.env.run()
            D1 = self.qdelay[0]
            D2 = self.qdelay[1]
            d1 = pd.DataFrame(D1)
            d2 = pd.DataFrame(D2)
            d1.to_csv('result/test/single_delay1.csv')
            d2.to_csv('result/test/single_delay2.csv')
            if i > 0:
                print("DDQN+H , Final Step: {t}, success: {l}, avg_qdelay: {d}".format(
                    t=self.timeslots,
                    l=self.success,
                    d=list(map(operator, self.qdelay))))
            self.reset()

        elif model == 'sp':
            self.env.process(self.SP(self.env))
            self.env.run()
            D1 = self.qdelay[0]
            D2 = self.qdelay[1]
            d1 = pd.DataFrame(D1)
            d2 = pd.DataFrame(D2)
            d1.to_csv('result/test/SP/single_delay1.csv')
            d2.to_csv('result/test/SP/single_delay2.csv')
            if i > 0:
                print("SP , Final Step: {t}, success: {l}, avg_qdelay: {d}".format(
                    t=self.timeslots,
                    l=self.success,
                    d=list(map(operator, self.qdelay))))
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

    # def simulation(self):
    #     iter_ = 2
    #     for _ in range(iter_):
    #         self.env.process(self.rl_agent(self.env))
    #         self.env.run()
    #         self.reset()
    #
    #         # print("@@@@@@@@@GCL Extract 완료@@@@@@@@@")
    #
    #         # FIFO
    #         self.env.process(self.SP(self.env))
    #         self.env.run()
    #         # print("@@@@@@@@@FIFO 완료@@@@@@@@@")
    #         self.seq = random_sequence()
    #         self.source = Src(1, self.start_time, self.seq)
    #         self.reset()

        # self.log1.to_csv("result/test/extract.csv")
        # self.log2.to_csv("result/test/apply.csv")
        # self.log3.to_csv("result/test/fifo.csv")

    def terminated(self):
        done = False

        if MAXSLOT_MODE:
            if (self.received_packet == COMMAND_CONTROL + BEST_EFFORT) or (self.timeslots == MAXSLOTS):
                done = True
        else:
            if self.received_packet == COMMAND_CONTROL + BEST_EFFORT:  # originally (CC + A + V + BE)
                done = True

        return done


if __name__ == "__main__":
    test = GateControlTestSimulation()
    test.simulation()
