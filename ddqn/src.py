import time
import simpy
from parameter import *
import warnings
import numpy as np

warnings.filterwarnings('ignore')


class Src:
    def __init__(self, src, start_time):
        self.src = src
        self.cnt = 0
        self.p = 1
        if src > 3:
            self.p = 2
        # self.sequence_p1, self.sequence_p2 = random_sequence()
        self.episode_start_time = start_time

    def reset(self, src, start_time):
        self.src = src
        self.cnt = 0
        self.p = 1
        if src > 3:
            self.p = 2
        # self.sequence_p1, self.sequence_p2 = random_sequence()
        self.episode_start_time = start_time
        # if not FIXED_SEQUENCE:
        #     self.sequence_p1, self.sequence_p2 = random_sequence()

    def flow_generator(self, time, fnum):
        flow = Flow()

        flow.src_ = self.src
        flow.dst_ = self.src
        flow.route_ = route[self.src - 1]
        flow.priority_ = self.p
        flow.num_ = fnum
        f.generated_time_ = time - self.episode_start_time
        flow.current_delay_ = 0
        flow.queueing_delay_ = 0
        flow.remain_hops_ = 0
        flow.met_ = -1

        if flow.priority_ == 1:
            flow.deadline_ = CC_DEADLINE * 0.001
            flow.bits_ = CC_BYTE * 8

        else:
            flow.deadline_ = BE_DEADLINE * 0.001
            flow.bits_ = BE_BYTE * 8

        return flow

    def send(self, env, nodes):
        if self.p == 1:
            for i in range(COMMAND_CONTROL):
                flow = self.flow_generator(env.now(), i)
                yield env.process(nodes[flow.route_[0]].packet_in(flow))
                self.cnt += 1
                yield env.timeout(TIMESLOT_SIZE * PERIOD_CC / 1000)

        else :
            for i in range(BEST_EFFORT):
                flow = self.flow_generator(env.now(), i)
                yield env.process(nodes[flow.route_[0]].packet_in(flow))
                self.cnt += 1
                yield env.timeout(TIMESLOT_SIZE * PERIOD_BE / 1000)
