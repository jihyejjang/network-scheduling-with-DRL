#!/usr/bin/env python
# coding: utf-8

import simpy
from parameter import *
import warnings
warnings.filterwarnings('ignore')


class Node:

    def __init__(self, datapath_id, env):
        # TODO: 학습 노드 구현
        self.datapath_id = datapath_id
        self.class_based_queues = [simpy.Store(env) for _ in range(PRIORITY_QUEUE)]
        self.action = number_to_action(INITIAL_ACTION)
        self.start_time = 0

    def reset(self, env, start_time):  # initial state, new episode start
        self.start_time = start_time
        self.class_based_queues = [simpy.Store(env) for _ in range(PRIORITY_QUEUE)]
        self.action = number_to_action(INITIAL_ACTION)

    # def timestep_reset(self, time):
    #     self.start_time = time

    def packet_in(self, time, flow):
        yield self.class_based_queues[flow.priority_ - 1].put(flow)
        flow.node_arrival_time_ = time - self.start_time
        flow.hops_ -= 1

    def queue_info(self, now):
        #d = [0, 0] #data
        l = [0, 0] #length
        at = [[], []] #생성부터 현재까지 시간
        qd = [0, 0] #queueing delay
        # dl = [[], []]
        # min_dl = [0, 0]
        for q in range(PRIORITY_QUEUE):
            l[q] = len(self.class_based_queues[q].items)
            for flow in self.class_based_queues[q].items:
                #d[q] += int(flow.bits_ / 8)
                d = (now - self.start_time) - flow.node_arrival_time_
                at[q].append(d)
                #dl[q].append(flow.deadline_)
            if not at[q]:
                qd[q] = 0
            else:
                qd[q] = np.mean(at[q])  # 현재까지 queueing delay 평균

            # try:
            #     min_dl[q] = min(dl[q])
            # except:
            #     min_dl[q] = 0
        # print (d, qd, min_dl, l)
        # for q in range(PRIORITY_QUEUE):
        #     for flow in self.class_based_queues[q].items:
        #         l[flow.type_ - 1] += 1
        #         d[q] += flow.bits_/8
        return qd, l

    def gcl_update(self, gcl_):  # observe state and update GCL (cycle : 0.2*3)
        self.action = gcl_
        # print (gcl_)

    def packet_out(self, env, trans_dict, t):
        gcl = self.action[:, t]
        # print (gcl)
        #w= [1,0.1]
        q = 0
        l = 1
        bits_sum = 0
        while bits_sum <= MAX_BURST and l > 0:
            if gcl[q] == 1:
                if len(self.class_based_queues[q].items) >= 1:
                    #print(q, len(self.class_based_queues[q].items))
                    f = yield self.class_based_queues[q].get()
                    bits_sum += f.bits_
                    if bits_sum > MAX_BURST:
                        yield self.class_based_queues[q].put(f)
                        return
                    # departure_time = (bits_sum-f.bits_)/1000000.0 + env.now #- self.start_time
                    #print("now" , env.now)
                    #print("arrval", f.node_arrival_time_)
                    f.queueing_delay_ = (bits_sum - f.bits_) / 20000000.0 + (env.now-self.start_time) - f.node_arrival_time_  # departure_time - f.node_arrival_time_
                    yield trans_dict.put(f)
            q = (q + 1) % 2
            l = gcl[0] * len(self.class_based_queues[0].items) + gcl[1] * len(self.class_based_queues[1].items)

        '''
        strict preemption
        for q in range(PRIORITY_QUEUE):
            if gcl[q] == 1:
                while len(self.class_based_queues[q].items):
                    f = yield self.class_based_queues[q].get()
                    bits_sum += f.bits_
                    if bits_sum > MAX_BURST:
                        yield self.class_based_queues[q].put(f)
                        return
                    departure_time = env.now - self.start_time
                    #f.queueing_delay_ = departure_time - f.node_arrival_time_
                    f.queueing_delay_ = (bits_sum - f.bits_) / 20000000.0 + env.now - f.node_arrival_time_
                    # if deadline[q] > 0:
                    #     if f.queueing_delay_ < (deadline[q]-0.0006):
                    #         f.reward_ = w[q]
                    #     else : f.reward_ = -w[q]
                    # else:
                    #     f.reward_ = 0
                    yield trans_dict.put(f)
        '''

