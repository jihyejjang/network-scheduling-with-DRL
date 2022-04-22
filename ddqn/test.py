import pandas as pd
import simpy
from node import Node
import time
import tensorflow as tf
from parameter import *
from src import Src


# max_burst()

# utilization()


class GateControlTestSimulation:

    def __init__(self):
        self.end_time = 0
        self.model = tf.keras.models.load_model(WEIGHT_FILE)
        self.env = simpy.Environment()
        self.start_time = self.env.now
        # self.sequence_p1, self.sequence_p2 = random_sequence()
        self.seq = random_sequence()
        self.source = Src(1, self.start_time, self.seq)
        self.node = Node(1, self.env, self.start_time)
        self.trans_list = simpy.Store(self.env)
        self.cnt1 = 0
        self.cnt4 = 0
        self.timeslots = 0  # progressed timeslots in a episode
        self.start_time = self.env.now  # episode 시작
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
        self.delay = pd.DataFrame(columns=['p1', 'p2'])

        self.success = [0, 0]  # deadline met
        self.qdelay = []
        self.et = []
        # self.s = [[0, 0] for _ in range(PRIORITY_QUEUE)]
        # self.state_and_action = []

    def reset(self):  # initial state, new episode start
        self.start_time = self.env.now  # episode 시작
        self.node = Node(1, self.env, self.start_time)
        self.source = Src(1, self.start_time, self.seq)
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
        self.qdelay = []
        self.et = []
        self.ex = pd.DataFrame(columns=['timeslot', 'state', 'gcl'])
        self.ap = pd.DataFrame(columns=['timeslot', 'state', 'gcl'])
        # self.delay = pd.DataFrame(columns=['p1', 'p2'])

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
        # delay = (packet.queueing_delay_ + 1) * TIMESLOT_SIZE / 1000
        # self.qdelay.append([p, delay])
        packet.queueing_delay_ = 0
        packet.arrival_time_ = self.env.now - self.start_time
        self.et.append([p, et])

        dl = packet.deadline_

        if BOUND[p] <= et / dl <= 1:  # packet received within configured latency boundary
            packet.met_ = 1
            self.success[p] += 1
            # self.reward += W[t]

        elif et/dl < BOUND[p]:
            packet.met_ = 1
            self.success[p] += 1
            r += W1[p]
        elif 1 < et/dl <= LM :
            packet.met_ = 0
            r += W2[p]
        else:
            packet.met_ = 0
            r += W3

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

    def gcl_extract(self, env):  # mainprocess
        s = time.time()
        rewards_all = []
        action = [INITIAL_ACTION for _ in range(OUTPUT_PORT)]

        env.process(self.source.send(env, self.node, 1))
        env.process(self.source.send(env, self.node, 4))

        while not self.done:
            self.timeslots += 1
            # log = pd.DataFrame([(self.timeslots, self.state, action)], columns=['timeslot', 'state', 'gcl'])
            yield env.process(self.node.link(self.trans_list, 'ddqn'))
            yield env.process(self.sendTo_next_node(env))
            yield env.timeout(TIMESLOT_SIZE * 0.001)

            self.next_state = self.node.step()
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
        print("Score: {s}, Final Step: {t}, success: {l}, qdelay: {d}".format(
            s=np.sum(rewards_all),
            t=self.timeslots,
            l=self.success,
            d=list(map(np.mean, self.qdelay))))

        # if sum(rewards_all) < 30 :
        # self.ex.to_csv('result/test/ex_analysis.csv')
        delay = pd.DataFrame(self.qdelay)
        delay.to_csv('result/test/delay.csv')
        est = pd.DataFrame(self.et)
        est.to_csv('result/test/et.csv')

    def SP(self, env):
        rewards_all = []
        env.process(self.source.send(env, self.node, 1))
        env.process(self.source.send(env, self.node, 4))

        while not self.done:
            self.timeslots += 1
            yield env.process(self.node.link(self.trans_list, 'sp'))
            yield env.process(self.sendTo_next_node(env))
            yield env.timeout(TIMESLOT_SIZE / 1000)

            self.next_state = self.node.step()
            self.done = self.terminated()
            rewards_all.append(self.reward)
            self.reward = 0
            self.state = self.next_state
            # self.node.gcl_update(gcl)

        print("Score: {s}, Final Step: {t}, success: {l}, avg_qdelay: {d}".format(
            s=np.sum(rewards_all),
            t=self.timeslots,
            l=self.success,
            d=list(map(np.mean, self.qdelay))))
        # log_ = pd.DataFrame([(self.timeslots, np.sum(rewards_all), self.success[0], self.success[1])],
        #                     columns=['Slots', 'Score', 'p1', 'p2'])

        # self.log3 = self.log3.append(log_, ignore_index=True)
        delay = pd.DataFrame(self.qdelay)
        delay.to_csv('result/test/SP/delay.csv')
        est = pd.DataFrame(self.et)
        est.to_csv('result/test/SP/et.csv')

    def simulation(self):
        iter_ = 1
        for _ in range(iter_):
            self.env.process(self.gcl_extract(self.env))
            self.env.run()
            self.reset()

            # print("@@@@@@@@@GCL Extract 완료@@@@@@@@@")

            # FIFO
            self.env.process(self.SP(self.env))
            self.env.run()
            # print("@@@@@@@@@FIFO 완료@@@@@@@@@")
            self.seq = random_sequence()
            self.source = Src(1, self.start_time, self.seq)
            self.reset()

        # self.log1.to_csv("result/test/extract.csv")
        # self.log2.to_csv("result/test/apply.csv")
        # self.log3.to_csv("result/test/fifo.csv")

    def terminated(self):
        # state = np.zeros((PRIORITY_QUEUE, STATE))
        # state[:, 0] = qlen
        # state[:, 1] = max_et
        # # state[:, 2] = max_qp
        # state = state.flatten()

        done = False

        if MAXSLOT_MODE:
            if (self.received_packet == COMMAND_CONTROL + BEST_EFFORT) or (self.timeslots == MAXSLOTS):
                done = True
        else:
            if self.received_packet == COMMAND_CONTROL + BEST_EFFORT:  # originally (CC + A + V + BE)
                done = True

        return done

    # def reward2(self, node):
    #     w = np.array([2, 3, 0.1, 0.01])
    #     # print("s:",self.s)
    #     r = round(sum(w * np.array(self.s[node - 5])), 1)
    #     # print ("r2:", r)
    #     return r


if __name__ == "__main__":
    test = GateControlTestSimulation()
    test.simulation()
