#!/usr/bin/env python
# coding: utf-8

import numpy as np
import simpy
from dataclasses import dataclass
import warnings

warnings.filterwarnings('ignore')

PRIORITY_QUEUES = 4
MAX_BURST = 364448  # maximum burst는 약 0.364448밀리초만에 전송됨

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


class Node:

    def __init__(self, datapath_id, env):
        # TODO: 학습 노드 구현
        self.datapath_id = datapath_id
        self.class_based_queues = [simpy.Store(env) for _ in range(PRIORITY_QUEUES)]
        self.actions = np.array([list(map(int, '1111111111')) for _ in range(PRIORITY_QUEUES)])
        self.start_time = 0

    def reset(self, env, start_time):  # initial state, new episode start
        self.start_time = start_time
        self.class_based_queues = [simpy.Store(env) for _ in range(PRIORITY_QUEUES)]
        self.actions = np.array([list(map(int, '1111111111')) for _ in range(PRIORITY_QUEUES)])

    def packet_in(self, time, flow):
        # print("packet_in, 노드",self.datapath_id)
        yield self.class_based_queues[flow.type_ - 1].put(flow)
        # print('queue',self.class_based_queues[flow.type_-1].items)
        flow.node_arrival_time_ = time - self.start_time
        flow.hops_ -= 1

    def queue_length(self):
        p = [0, 0, 0, 0]
        for q in range(len(self.class_based_queues)):
            p[q] = len(self.class_based_queues[q].items)
        return p

    def gcl_update(self, gcl_):  # observe state and update GCL (cycle : 0.4*10)
        gcl = [list(map(int, g)) for g in gcl_]
        self.actions = np.array(gcl)

    def packet_out(self, env, trans_dict, t):
        gcl = self.actions[:, t]
        #print (gcl)
        bits_sum = 0

        for q in range(PRIORITY_QUEUES):
            if (gcl[q] == 1) and (bits_sum < MAX_BURST):
                while len(self.class_based_queues[q].items):
                    f = yield self.class_based_queues[q].get()
                    #print (self.class_based_queues[0].items)
                    bits_sum += f.bits_
                    departure_time = env.now - self.start_time
                    f.queueing_delay_ = departure_time - f.node_arrival_time_
                    # f.node_departure_time_.append(env.now - self.start_time)
                    yield trans_dict.put(f)
