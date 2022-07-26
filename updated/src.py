from utils import *
import warnings

warnings.filterwarnings('ignore')


class Src:
    def __init__(self, start_time, seq, args):
        self.single_node = args.env
        self.cnt = 0
        self.episode_start_time = start_time
        self.sequence_p1, self.sequence_p2 = seq
        self.pr = args.period
        self.np = args.numberofpackets
        self.dl = args.deadline

    def reset(self, start_time, seq):
        self.cnt = 0
        self.sequence_p1, self.sequence_p2 = seq
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

        if self.single_node:
            flow.route_ = []
            flow.queueing_delay_ = 0

            if flow.priority_ == 1:
                flow.remain_hops_ = self.sequence_p1[1][fnum]
                flow.random_delay_ = self.sequence_p1[0][fnum]
                flow.deadline_ = self.dl[0]
                flow.bits_ = CC_BYTE * 8
            else:
                flow.remain_hops_ = self.sequence_p2[1][fnum]
                flow.random_delay_ = self.sequence_p2[0][fnum]
                flow.deadline_ = self.dl[1]
                flow.bits_ = BE_BYTE * 8

        else:
            flow.route_ = route[src - 1]
            h = len(route[src - 1]) - 1
            flow.remain_hops_ = h - 1
            flow.queueing_delay_ = [0 for _ in range(h)]  # nodes to be passed packets

            if flow.priority_ == 1:
                flow.deadline_ = self.dl[0]
                flow.random_delay_ = self.sequence_p1[0][fnum]
                flow.bits_ = CC_BYTE * 8

            else:
                flow.deadline_ = self.dl[1]
                flow.random_delay_ = self.sequence_p2[0][fnum]
                flow.bits_ = BE_BYTE * 8

        return flow

    def send(self, env, nodes, src):
        if self.single_node:
            if not 1 < src < 8:  # must got to be editted when network topology being changed
                for i in range(self.np[0]):
                    flow = self.flow_generator(env.now, src, i)
                    yield env.process(nodes.route_modify(flow))
                    yield env.timeout(TIMESLOT_SIZE * self.pr[0] / 1000)

            else:
                for i in range(self.np[1]):
                    flow = self.flow_generator(env.now, src, i)
                    yield env.process(nodes.route_modify(flow))
                    yield env.timeout(TIMESLOT_SIZE * self.pr[1] / 1000)
        else:
            if not 1 < src < 8:  # must got to be editted when network topology being changed
                for i in range(self.np[0]):
                    flow = self.flow_generator(env.now, src, i)
                    r = flow.route_[0]
                    yield env.process(nodes[r - 1].route_modify(flow))
                    yield env.timeout(TIMESLOT_SIZE * self.pr[0] / 1000)

            else:
                for i in range(self.np[1]):
                    flow = self.flow_generator(env.now, src, i)
                    r = flow.route_[0]
                    yield env.process(nodes[r - 1].route_modify(flow))
                    yield env.timeout(TIMESLOT_SIZE * self.pr[1] / 1000)
