#!/usr/bin/env python
# coding: utf-8

import simpy
from parameter import *
import warnings
import numpy as np

warnings.filterwarnings('ignore')


class Node:

    def __init__(self, datapath_id, env):
        self.datapath_id = datapath_id
        self.class_based_queues = [simpy.Store(env) for _ in range(PRIORITY_QUEUE)]
        self.action = INITIAL_ACTION
        self.start_time = 0

    def reset(self, env, start_time):  # initial state, new episode start
        self.start_time = start_time
        self.class_based_queues = [simpy.Store(env) for _ in range(PRIORITY_QUEUE)]
        self.action = INITIAL_ACTION

    # def timestep_reset(self, time):
    #     self.start_time = time

    def packet_in(self, timeslot, flow):
        yield self.class_based_queues[flow.priority_ - 1].put(flow)
        flow.node_arrival_time_ = timeslot
        flow.hops_ -= 1

    def queue_info(self):
        # state = queuelength, maxET, argmax

        l = [0, 0]  # length
        ET = [[], []]
        max_et = [0, 0]
        max_q_pos = [0, 0]
        for q in range(PRIORITY_QUEUE):
            flows = self.class_based_queues[q].items
            if not flows:
                continue
            l[q] = len(flows)
            for i, flow in enumerate(flows):
                ET[q].append(flow.queueing_delay_ + flow.current_delay_ + flow.hops_ + i)
            max_et[q] = max(ET[q])
            max_q_pos[q] = np.argmax(ET[q])
        return l, max_et, max_q_pos

    def gcl_update(self, gcl_):  # observe state and update GCL (cycle : 0.2*3)
        self.action = number_to_action(gcl_)
        # print (gcl_)

    def packet_out(self, trans_list):

        for q in range(PRIORITY_QUEUE):
            flows = self.class_based_queues[q].items
            for f in flows:
                f.queueing_delay_ += 1
            if (len(self.class_based_queues[q].items) >= 1) and (self.action[q] == 1):
                f = yield self.class_based_queues[q].get()
                yield trans_list.put(f)
                return

        # while bits_sum <= MAX_BURST and l > 0:
        #     if gcl[q] == 1:
        #         if len(self.class_based_queues[q].items) >= 1:
        #             #print(q, len(self.class_based_queues[q].items))
        #             f = yield self.class_based_queues[q].get()
        #             bits_sum += f.bits_
        #             if bits_sum > MAX_BURST:
        #                 yield self.class_based_queues[q].put(f)
        #                 return
        #             # departure_time = (bits_sum-f.bits_)/1000000.0 + env.now #- self.start_time
        #             #print("now" , env.now)
        #             #print("arrval", f.node_arrival_time_)
        #             f.queueing_delay_ = (bits_sum - f.bits_) / 20000000.0 + (env.now-self.start_time) - f.node_arrival_time_  # departure_time - f.node_arrival_time_
        #             yield trans_dict.put(f)
        #     q = (q + 1) % 2
        #     l = gcl[0] * len(self.class_based_queues[0].items) + gcl[1] * len(self.class_based_queues[1].items)

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
