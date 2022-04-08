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
        priority_queue = [simpy.Store(env) for _ in range(PRIORITY_QUEUE)]
        self.output_port = [priority_queue for _ in range(OUTPUT_PORT)]
        self.action = number_to_action(INITIAL_ACTION)
        self.port = -1
        # self.start_time = 0

    def reset(self, env):  # initial state, new episode start
        # self.start_time = start_time
        priority_queue = [simpy.Store(env) for _ in range(PRIORITY_QUEUE)]
        self.output_port = [priority_queue for _ in range(OUTPUT_PORT)]
        self.action = number_to_action(INITIAL_ACTION)
        self.port = -1

    def packet_in(self, pk):
        pt = 1

        if self.port == -1:
            if pk.src_:
                self.port = pk.src_
            else:
                self.port = 0

        if (pk.src_ == self.port) or (pk.src_ == [] and self.port == 0):
            pt = 0
        # print(self.output_port)
        yield self.output_port[pt][pk.priority_ - 1].put(pk)

    def link(self, output, scheduler='ddqn'):
        if scheduler == 'ddqn':
            for p in range(OUTPUT_PORT):
                self.packet_out_with_heuristic(output)
        elif scheduler == 'sp':
            self.strict_priority(output)
        else:
            self.packet_out(output)

    def queue_info(self):
        # state = queuelength, maxET, argmax

        l = [0, 0]  # length
        ET = [[], []]
        max_et = [0, 0]
        for q in range(PRIORITY_QUEUE):
            for p in range(OUTPUT_PORT):
                flows = self.output_port[p][q].items
                if not flows:
                    continue
                l[q] += len(flows)
                for i, flow in enumerate(flows):
                    ET[q].append(
                        flow.random_delay_ + flow.queueing_delay_ + flow.current_delay_ + flow.remain_hops_ + i)
                max_et[q] = max(ET[q])
            # max_q_pos[q] = np.argmax(ET[q])
        return l, max_et

    def gcl_update(self, gcl_):  # observe state and update GCL (cycle : 0.2*3)
        self.action = number_to_action(gcl_)

    def packet_out_with_heuristic(self, output, port):

        priority1 = self.output_port[port][0].items + self.output_port[1][0].items
        priority2 = self.output_port[port][1].items + self.output_port[1][1].items
        if not priority1:
            for p in range(OUTPUT_PORT):
                if len(self.output_port[p][1].items):
                    fl = yield self.output_port[p][1].get()
                    fl.remain_hops_ -= 1
                    fl.route_ = fl.route_[1:]
                    yield output.put(fl)
        elif not priority2:
            for p in range(OUTPUT_PORT):
                if len(self.output_port[p][0].items):
                    fl = yield self.output_port[p][0].get()
                    fl.remain_hops_ -= 1
                    fl.route_ = fl.route_[1:]
                    yield output.put(fl)
        else:
            if action_to_number(self.action) == 0:
                for p in range(OUTPUT_PORT):
                    if len(self.output_port[p][0].items):
                        fl = yield self.output_port[p][0].get()
                        fl.remain_hops_ -= 1
                        fl.route_ = fl.route_[1:]
                        yield output.put(fl)

            else:
                for p in range(OUTPUT_PORT):
                    if len(self.output_port[p][1].items):
                        fl = yield self.output_port[p][1].get()
                        fl.remain_hops_ -= 1
                        fl.route_ = fl.route_[1:]
                        yield output.put(fl)

        for p in range(OUTPUT_PORT):
            for q in range(PRIORITY_QUEUE):
                waiting = self.output_port[p][q].items
                # print (waiting)
                for w in waiting:
                    w.queueing_delay_ += 1

    def packet_out(self, output):

        if action_to_number(self.action) == 0:
            for p in range(OUTPUT_PORT):
                if len(self.output_port[p][0].items) >= 1:
                    fl = yield self.output_port[p][0].get()
                    fl.remain_hops_ -= 1
                    fl.route_ = fl.route_[1:]
                    yield output.put(fl)

        else:
            for p in range(OUTPUT_PORT):
                if len(self.output_port[p][1].items) >= 1:
                    fl = yield self.output_port[p][1].get()
                    fl.remain_hops_ -= 1
                    fl.route_ = fl.route_[1:]
                    yield output.put(fl)

        for p in range(OUTPUT_PORT):
            for q in range(PRIORITY_QUEUE):
                waiting = self.output_port[p][q].items
                # print (waiting)
                for w in waiting:
                    w.queueing_delay_ += 1

    def strict_priority(self, output):
        pk_cnt = 0

        for p in range(OUTPUT_PORT):
            for q in range(PRIORITY_QUEUE):
                waiting = self.output_port[p][q].items
                for w in waiting:
                    w.queueing_delay_ += 1

        for q in range(PRIORITY_QUEUE):
            for p in range(OUTPUT_PORT):
                if len(self.output_port[p][q].items) >= 1:
                    fl = yield self.output_port[p][q].get()
                    fl.remain_hops_ -= 1
                    fl.route_ = fl.route_[1:]
                    yield output.put(fl)
                    pk_cnt += 1

                if pk_cnt == 1:  # OUTPUT_PORT
                    break
