#!/usr/bin/env python
# coding: utf-8
# import time

import simpy
from parameter import *
import warnings
# import numpy as np

warnings.filterwarnings('ignore')


class Node:

    def __init__(self, node, env):
        self.node = node
        self.class_based_queues = [simpy.Store(env) for _ in range(PRIORITY_QUEUE)]
        self.action = number_to_action(INITIAL_ACTION)
        # self.start_time = 0

    def reset(self, env):  # initial state, new episode start
        # self.start_time = start_time
        self.class_based_queues = [simpy.Store(env) for _ in range(PRIORITY_QUEUE)]
        self.action = number_to_action(INITIAL_ACTION)

    def packet_in(self, pk):
        yield self.class_based_queues[pk.priority_ - 1].put(pk)

    def queue_info(self):
        # state = queuelength, maxET, argmax

        l = [0, 0]  # length
        ET = [[], []]
        max_et = [0, 0]
        # max_q_pos = [0, 0]
        for q in range(PRIORITY_QUEUE):
            flows = self.class_based_queues[q].items
            if not flows:
                continue
            l[q] = len(flows)
            for i, flow in enumerate(flows):
                ET[q].append(flow.random_delay_ + flow.queueing_delay_ + flow.current_delay_ + flow.remain_hops_ + i)
            max_et[q] = max(ET[q])
            # max_q_pos[q] = np.argmax(ET[q])
        return l, max_et

    def gcl_update(self, gcl_):  # observe state and update GCL (cycle : 0.2*3)
        self.action = number_to_action(gcl_)

    def packet_out(self, output):

        if action_to_number(self.action) == 0:
            flows = self.class_based_queues[0].items
            for fl in flows:
                fl.queueing_delay_ += 1

            if len(self.class_based_queues[0].items) >= 1:
                fl = yield self.class_based_queues[0].get()
                fl.remain_hops_ -= 1
                fl.route_ = fl.route_[1:]
                yield output.put(fl)
        else:
            flows = self.class_based_queues[1].items
            for fl in flows:
                fl.queueing_delay_ += 1

            if len(self.class_based_queues[1].items) >= 1:
                fl = yield self.class_based_queues[1].get()
                fl.remain_hops_ -= 1
                fl.route_ = fl.route_[1:]
                yield output.put(fl)

    def strict_priority(self, output):
        pk_cnt = 0
        for q in range(PRIORITY_QUEUE):
            flows = self.class_based_queues[q].items
            for fl in flows:
                fl.queueing_delay_ += 1

            if len(self.class_based_queues[q].items) >= 1:
                fl = yield self.class_based_queues[q].get()
                fl.remain_hops_ -= 1
                fl.route_ = fl.route_[1:]
                yield output.put(fl)
                pk_cnt += 1

            if pk_cnt == 1:
                break
