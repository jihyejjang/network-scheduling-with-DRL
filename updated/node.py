import simpy
from utils import *
import warnings

warnings.filterwarnings('ignore')


class Node:

    def __init__(self, node, env, start, args):
        self.single_node = args.env
        if self.single_node:
            self.op = 1
        else:
            self.op = OUTPUT_PORT
        self.wc = args.workconserving
        self.rrw = args.wrrweight
        self.node = node
        self.env = env
        self.start = start
        self.output_port = [[simpy.Store(env), simpy.Store(env)] for _ in range(self.op)]
        self.action = [number_to_action(INITIAL_ACTION) for _ in range(self.op)]
        self.port = -1
        self.schedulable_ports = []
        self.r = [0, 0]
        self.rr = 0
        self.state = [np.zeros(INPUT_SIZE) for _ in range(self.op)]

    def reset(self, start):  
        # self.env = env
        self.start = start
        self.output_port = [[simpy.Store(self.env), simpy.Store(self.env)] for _ in range(self.op)]
        self.action = [number_to_action(INITIAL_ACTION) for _ in range(self.op)]
        self.port = -1
        self.schedulable_ports = []
        self.r = [0, 0]
        self.rr = 0
        self.state = [np.zeros(INPUT_SIZE) for _ in range(self.op)]

    def scheduling(self, output, scheduler='sp'):
        if self.wc:
            if scheduler == 'ddqn':
                for p in range(self.op):
                    if p in self.schedulable_ports:
                        yield self.env.process(self.ddqn(output, p))
                    else:
                        yield self.env.process(self.work_conserving(output, p))
            elif scheduler == 'sp':
                for p in range(self.op):
                    yield self.env.process(self.strict_priority(output, p))
            elif scheduler == 'rr':
                for p in self.schedulable_ports:
                    yield self.env.process(self.round_robin(output, p))
                else:
                    yield self.env.process(self.work_conserving(output, p))
        else:
            if scheduler == 'ddqn':
                for p in range(self.op):
                    yield self.env.process(self.ddqn(output, p))
            elif scheduler == 'sp':
                for p in range(self.op):
                    yield self.env.process(self.strict_priority(output, p))
            elif scheduler == 'rr':
                for p in range(self.op):
                    yield self.env.process(self.round_robin(output, p))
    
    def schedulable(self):
        port = []
        self.schedulable_ports = []

        for p in range(self.op):
            q1 = int(self.state[p][0])
            q2 = int(self.state[p][2])
            # print (q1, q2)
            if not (q1 + q2 == (q1 or q2)):
                port.append(p)
                self.schedulable_ports.append(p)
                
        return port

    def route_modify(self, pk):
        pt = 0
        if self.op == 2:
            if self.port == -1:
                if pk.route_[1]:
                    self.port = pk.route_[1]
                else:
                    self.port = 0
            else:
                if self.port:
                    if not (pk.route_[1] == self.port):
                        pt = 1
        yield self.output_port[pt][pk.priority_ - 1].put(pk)

    def state_observe(self):

        for p in range(self.op):
            qlen, max_et = self.queue_info(p)
            state = np.zeros((PRIORITY_QUEUE, STATE))
            state[:, 0] = qlen
            state[:, 1] = max_et
            self.state[p] = state.flatten()

        return self.state

    def queue_info(self, port):
        l = [0, 0]  # state 1
        pod = [[], []]
        max_et = [0, 0]  # state 2
        for q in range(PRIORITY_QUEUE):
            flows = self.output_port[port][q].items
            if not flows:
                continue
            l[q] += len(flows)
            for i, flow in enumerate(flows):
                # The unit of estimated delay is T(timeslot)
                if self.single_node:
                    et = flow.random_delay_ + flow.current_delay_ + flow.queueing_delay_ + flow.remain_hops_ + i
                else:
                    et = flow.random_delay_ + sum(flow.queueing_delay_) + flow.remain_hops_ + i
                # pod_ = round(et/flow.deadline_,2)
                # pod[q].append(pod_)
                pod[q].append(et)
            max_et[q] = max(pod[q])
        return l, max_et

    def action_update(self, action, port):  # observe state and update GCL (cycle : 0.2*3)
        self.action[port] = number_to_action(action)

    def delay_for_topology(self, port):
        for q in range(PRIORITY_QUEUE):
            waiting = self.output_port[port][q].items
            for w in waiting:
                r = w.remain_hops_
                l = len(w.queueing_delay_)
                w.queueing_delay_[l - r - 1] += 1

    def delay_for_singlenode(self, port):
        for q in range(PRIORITY_QUEUE):
            waiting = self.output_port[port][q].items
            for w in waiting:
                w.queueing_delay_ += 1
    
    def work_conserving(self, output, port):
        priority1 = self.output_port[port][0].items
        priority2 = self.output_port[port][1].items

        if not priority1:
            # print("priority1 없음 - work conserving")
            if len(self.output_port[port][1].items):
                fl = yield self.output_port[port][1].get()
                fl.remain_hops_ -= 1
                fl.route_ = fl.route_[1:]
                yield output.put(fl)
        elif not priority2:
            if len(self.output_port[port][0].items):
                fl = yield self.output_port[port][0].get()
                fl.remain_hops_ -= 1
                fl.route_ = fl.route_[1:]
                yield output.put(fl)

        if self.single_node:
            self.delay_for_singlenode(port)
        else:
            self.delay_for_topology(port)

    def ddqn(self, output, port):

        if action_to_number(self.action[port]) == 0:
            fl = yield self.output_port[port][0].get()
            fl.remain_hops_ -= 1
            fl.route_ = fl.route_[1:]
            yield output.put(fl)
        else:
            fl = yield self.output_port[port][1].get()
            fl.remain_hops_ -= 1
            fl.route_ = fl.route_[1:]
            yield output.put(fl)

        if self.single_node:
            self.delay_for_singlenode(port)
        else:
            self.delay_for_topology(port)

    def round_robin(self, output, port):

        if self.r[1] == 1:
            fl = yield self.output_port[port][1].get()
            fl.remain_hops_ -= 1
            fl.route_ = fl.route_[1:]
            self.r = [0, 0]
            yield output.put(fl)
            
        elif self.r[0] < self.rrw:
            fl = yield self.output_port[port][0].get()
            fl.remain_hops_ -= 1
            fl.route_ = fl.route_[1:]
            yield output.put(fl)
            if self.r[0] == self.rrw - 1:
                self.r[1] = 1
            else:
                self.r[0] += 1

        if self.single_node:
            self.delay_for_singlenode(port)
        else:
            self.delay_for_topology(port)

    def strict_priority(self, output, port):

        for q in range(PRIORITY_QUEUE):
            if len(self.output_port[port][q].items):
                fl = yield self.output_port[port][q].get()
                fl.remain_hops_ -= 1
                fl.route_ = fl.route_[1:]
                yield output.put(fl)
                break

        if self.single_node:
            self.delay_for_singlenode(port)
        else:
            self.delay_for_topology(port)
