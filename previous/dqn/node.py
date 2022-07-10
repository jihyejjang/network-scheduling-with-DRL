#!/usr/bin/env python
# coding: utf-8

import numpy as np
import simpy
from dataclasses import dataclass
import warnings

warnings.filterwarnings('ignore')

MAX_BURST = 12000
PRIORITY_QUEUE = 2
GCL_LENGTH = 3
action_list = [56, 49, 35, 42, 28, 21, 14, 7]

@dataclass
class Flow:  # type(class1:cc,2:ad,3:vd,4:be),Num,deadline,generate_time,depart_time,bits
    type_: int = None
    num_: int = None
    deadline_: float = None  # millisecond 단위, arrival time - generated time < deadline 이어야 함
    generated_time_: float = None  # millisecond 단위
    queueing_delay_: float = None  # node departure time - node arrival time
    node_arrival_time_: float = None
    arrival_time_: float = None
    bits_: int = None
    met_: bool = None
    hops_: int = None
    priority_: int = None
    reward_: int = None


def action_to_number(action):
    action_ = action.flatten()
    bin = ''
    for a in action_:
        bin += str(a)
    return action_list.index(int(bin, 2))


def number_to_action(action_id):  # number -> binary gcl code
    #idx = action_list.index(action_id)
    b_id = format(action_list[action_id], '06b')
    action_ = np.array(list(map(int, b_id)))
    return action_.reshape((PRIORITY_QUEUE, GCL_LENGTH))


class Node:

    def __init__(self, datapath_id, env):
        self.datapath_id = datapath_id
        self.class_based_queues = [simpy.Store(env) for _ in range(PRIORITY_QUEUE)]
        self.action = number_to_action(0)
        self.start_time = 0

    def reset(self, env, start_time):  # initial state, new episode start
        self.start_time = start_time
        self.class_based_queues = [simpy.Store(env) for _ in range(PRIORITY_QUEUE)]
        self.action = number_to_action(0)

    def packet_in(self, time, flow):
        # print("packet_in, 노드",self.datapath_id)
        yield self.class_based_queues[flow.priority_ - 1].put(flow)
        # print('queue',self.class_based_queues[flow.type_-1].items)
        flow.node_arrival_time_ = time - self.start_time
        flow.hops_ -= 1

    def queue_info(self, time):
        d = [0, 0]
        l = [0, 0]
        at = [[], []]
        qd = [0, 0]
        dl = [[], []]
        min_dl = [0, 0]
        for q in range(PRIORITY_QUEUE):
            l[q] = len(self.class_based_queues[q].items)
            for flow in self.class_based_queues[q].items:
                d[q] += int(flow.bits_ / 8)
                at[q].append(flow.node_arrival_time_)
                dl[q].append(flow.deadline_)
            if not at[q]:
                qd[q] = 0
            else:
                qd[q] = time - np.mean(at[q])  # 현재까지 queueing delay 평균

            try:
                min_dl[q] = min(dl[q])
            except:
                min_dl[q] = 0
        # print (d, qd, min_dl, l)
        # for q in range(PRIORITY_QUEUE):
        #     for flow in self.class_based_queues[q].items:
        #         l[flow.type_ - 1] += 1
        #         d[q] += flow.bits_/8
        return d, qd, min_dl, l

    def gcl_update(self, gcl_):  # observe state and update GCL (cycle : 0.2*3)
        self.action = gcl_
        # print (gcl_)

    def packet_out(self, env, trans_dict, t, deadline):
        gcl = self.action[:, t]
        # print (gcl)
        w= [1,0.1]
        bits_sum = 0

        for q in range(PRIORITY_QUEUE):
            if gcl[q] == 1:
                while len(self.class_based_queues[q].items):
                    f = yield self.class_based_queues[q].get()
                    bits_sum += f.bits_
                    if bits_sum > MAX_BURST:
                        yield self.class_based_queues[q].put(f)
                        return
                    departure_time = env.now - self.start_time
                    f.queueing_delay_ = departure_time - f.node_arrival_time_
                    # if deadline[q] > 0:
                    #     if f.queueing_delay_ < (deadline[q]-0.0006):
                    #         f.reward_ = w[q]
                    #     else : f.reward_ = -w[q]
                    # else:
                    #     f.reward_ = 0
                    yield trans_dict.put(f)
