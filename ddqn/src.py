import time
import simpy
from parameter import *
import warnings
import numpy as np

warnings.filterwarnings('ignore')


class Src:
    def __init__(self, src, start_time, seq):
        self.src = src
        self.cnt = 0
        self.episode_start_time = start_time
        self.sequence_p1, self.sequence_p2 = seq

    def reset(self, src, start_time):
        self.src = src
        self.cnt = 0
        self.episode_start_time = start_time

    def flow_generator(self, now, src, fnum):
        flow = Flow()

        flow.priority_ = 1

        if 1 < src < 8:  # must got to be editted when network topology being changed
            flow.priority_ = 2

        flow.src_ = src
        flow.dst_ = src
        flow.num_ = fnum
        flow.generated_time_ = now - self.episode_start_time
        flow.current_delay_ = 0
        flow.met_ = -1

        if SINGLE_NODE:
            flow.route_ = []
            flow.queueing_delay_ = 0

            if flow.priority_ == 1:
                flow.remain_hops_ = self.sequence_p1[1][fnum]
                flow.random_delay_ = self.sequence_p1[0][fnum]
                flow.deadline_ = CC_DEADLINE
                flow.bits_ = CC_BYTE * 8
            else:
                flow.remain_hops_ = self.sequence_p2[1][fnum]
                flow.random_delay_ = self.sequence_p2[0][fnum]
                flow.deadline_ = BE_DEADLINE
                flow.bits_ = BE_BYTE * 8

        else:
            flow.route_ = route[src - 1]
            h = len(route[src - 1]) - 1
            flow.remain_hops_ = h - 1
            flow.queueing_delay_ = [0 for _ in range(h)]  # nodes to be passed packts

            if flow.priority_ == 1:
                flow.deadline_ = CC_DEADLINE
                flow.random_delay_ = self.sequence_p1[0][fnum]
                flow.bits_ = CC_BYTE * 8

            else:
                flow.deadline_ = BE_DEADLINE
                flow.random_delay_ = self.sequence_p2[0][fnum]
                flow.bits_ = BE_BYTE * 8

        return flow

    def send(self, env, nodes, src):
        if SINGLE_NODE:
            if not 1 < src < 8:  # must got to be editted when network topology being changed
                for i in range(COMMAND_CONTROL):
                    flow = self.flow_generator(env.now, src, i)
                    yield env.process(nodes.packet_in(flow))
                    yield env.timeout(TIMESLOT_SIZE * PERIOD_CC / 1000)

            else:
                for i in range(BEST_EFFORT):
                    flow = self.flow_generator(env.now, src, i)
                    yield env.process(nodes.packet_in(flow))
                    yield env.timeout(TIMESLOT_SIZE * PERIOD_BE / 1000)
        else:
            if not 1 < src < 8:  # must got to be editted when network topology being changed
                for i in range(COMMAND_CONTROL):
                    flow = self.flow_generator(env.now, src, i)
                    r = flow.route_[0]
                    yield env.process(nodes[r - 1].packet_in(flow))
                    yield env.timeout(TIMESLOT_SIZE * PERIOD_CC / 1000)

            else:
                for i in range(BEST_EFFORT):
                    flow = self.flow_generator(env.now, src, i)
                    r = flow.route_[0]
                    yield env.process(nodes[r - 1].packet_in(flow))
                    yield env.timeout(TIMESLOT_SIZE * PERIOD_BE / 1000)

    # def send(self, env, nodes):
    #     if self.p == 1:
    #         for i in range(COMMAND_CONTROL):
    #             flow = self.flow_generator(env.now, i)
    #             # print(i,":",flow.route_, flow.remain_hops_)
    #             print(flow.route_)
    #             r = flow.route_[0]
    #             yield env.process(nodes[r - 1].packet_in(flow))
    #             self.cnt += 1
    #             yield env.timeout(TIMESLOT_SIZE * PERIOD_CC / 1000)
    #
    #     else:
    #         for i in range(BEST_EFFORT):
    #             flow = self.flow_generator(env.now, i)
    #             print(flow.route_)
    #             yield env.process(nodes[flow.route_[0] - 1].packet_in(flow))
    #             self.cnt += 1
    #             yield env.timeout(TIMESLOT_SIZE * PERIOD_BE / 1000)
