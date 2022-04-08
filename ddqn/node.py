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
        self.env = env
        self.output_port = [[simpy.Store(env), simpy.Store(env)] for _ in range(OUTPUT_PORT)]
        self.action = [number_to_action(INITIAL_ACTION) for _ in range(OUTPUT_PORT)]
        self.port = -1
        self.state = [np.zeros(INPUT_SIZE) for _ in range(OUTPUT_PORT)]
        # self.start_time = 0

    def reset(self, env):  # initial state, new episode start
        # self.start_time = start_time
        self.env = env
        self.output_port = [[simpy.Store(env), simpy.Store(env)] for _ in range(OUTPUT_PORT)]
        self.action = [number_to_action(INITIAL_ACTION) for _ in range(OUTPUT_PORT)]
        self.state = [np.zeros(INPUT_SIZE) for _ in range(OUTPUT_PORT)]
        self.port = -1

    def schedulable(self):
        port = []

        for p in range(OUTPUT_PORT):
            q1 = int(self.state[p][0])
            q2 = int(self.state[p][2])
            # print (q1, q2)
            if not (q1 + q2 == (q1 or q2)):
                port.append(p)

        return port

    def link(self, output, scheduler=None):
        if scheduler == 'ddqn':
            for p in range(OUTPUT_PORT):
                yield self.env.process(self.packet_out_with_heuristic(output, p))
        elif scheduler == 'sp':
            for p in range(OUTPUT_PORT):
                yield self.env.process(self.strict_priority(output, p))
        else:
            for p in range(OUTPUT_PORT):
                yield self.env.process(self.packet_out(output, p))

    def packet_in(self, pk):
        pt = 0
        if OUTPUT_PORT == 2:
            if self.port == -1:
                if pk.route_[0]:
                    self.port = pk.route_[0]
                else:
                    self.port = 0

            if not ((pk.route_[0] == self.port) or (pk.route_[0] == [] and self.port == 0)):
                pt = 1
        # print(self.output_port)
        yield self.output_port[pt][pk.priority_ - 1].put(pk)

    def step(self):

        for p in range(OUTPUT_PORT):
            qlen, max_et = self.queue_info(p)
            state = np.zeros((PRIORITY_QUEUE, STATE))
            state[:, 0] = qlen
            state[:, 1] = max_et
            self.state[p] = state.flatten()

            # if any(self.state[p]):
            #     print (self.state[p])
        return self.state

    def queue_info(self, port):
        # state = queuelength, maxET, argmax

        l = [0, 0]  # length
        ET = [[], []]
        max_et = [0, 0]
        for q in range(PRIORITY_QUEUE):
            # for p in range(OUTPUT_PORT):
            flows = self.output_port[port][q].items
            if not flows:
                continue
            l[q] += len(flows)
            for i, flow in enumerate(flows):
                ET[q].append(
                    flow.random_delay_ + flow.queueing_delay_ + flow.current_delay_ + flow.remain_hops_ + i)
            max_et[q] = max(ET[q])
            # max_q_pos[q] = np.argmax(ET[q])
        return l, max_et

    def gcl_update(self, action, port):  # observe state and update GCL (cycle : 0.2*3)
        self.action[port] = number_to_action(action)

    def packet_out_with_heuristic(self, output, port):

        priority1 = self.output_port[port][0].items
        priority2 = self.output_port[port][1].items

        if not priority1:
            # for p in range(OUTPUT_PORT):
            if len(self.output_port[port][1].items):
                fl = yield self.output_port[port][1].get()
                fl.remain_hops_ -= 1
                fl.route_ = fl.route_[1:]
                yield output.put(fl)
        elif not priority2:
            # for p in range(OUTPUT_PORT):
            if len(self.output_port[port][0].items):
                fl = yield self.output_port[port][0].get()
                fl.remain_hops_ -= 1
                fl.route_ = fl.route_[1:]
                yield output.put(fl)
        else:
            if action_to_number(self.action[port]) == 0:
                # for p in range(OUTPUT_PORT):
                if len(self.output_port[port][0].items):
                    fl = yield self.output_port[port][0].get()
                    fl.remain_hops_ -= 1
                    fl.route_ = fl.route_[1:]
                    yield output.put(fl)

            else:
                # for p in range(OUTPUT_PORT):
                if len(self.output_port[port][1].items):
                    fl = yield self.output_port[port][1].get()
                    fl.remain_hops_ -= 1
                    fl.route_ = fl.route_[1:]
                    yield output.put(fl)

        for q in range(PRIORITY_QUEUE):
            # for q in range(PRIORITY_QUEUE):
            waiting = self.output_port[port][q].items
            for w in waiting:
                w.queueing_delay_ += 1

    def packet_out(self, output, port):

        if action_to_number(self.action[port]) == 0:
            # for p in range(OUTPUT_PORT):
            if len(self.output_port[port][0].items) >= 1:
                fl = yield self.output_port[port][0].get()
                fl.remain_hops_ -= 1
                fl.route_ = fl.route_[1:]
                yield output.put(fl)

        else:
            # for p in range(OUTPUT_PORT):
            if len(self.output_port[port][1].items) >= 1:
                fl = yield self.output_port[port][1].get()
                fl.remain_hops_ -= 1
                fl.route_ = fl.route_[1:]
                yield output.put(fl)

        # for p in range(OUTPUT_PORT):
        for q in range(PRIORITY_QUEUE):
            waiting = self.output_port[port][q].items
            # print (waiting)
            for w in waiting:
                w.queueing_delay_ += 1

    def strict_priority(self, output, port):
        # pk_cnt = 0

        for q in range(PRIORITY_QUEUE):
            waiting = self.output_port[port][q].items
            for w in waiting:
                w.queueing_delay_ += 1

        for q in range(PRIORITY_QUEUE):
            # for p in range(OUTPUT_PORT):
            if len(self.output_port[port][q].items) >= 1:
                #print(self.output_port[port][q].items)
                fl = yield self.output_port[port][q].get()
                fl.remain_hops_ -= 1
                fl.route_ = fl.route_[1:]
                yield output.put(fl)
                break



