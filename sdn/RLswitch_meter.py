#TODO : meter 적용
import time
import pandas as pd
from ryu.lib import hub

from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import CONFIG_DISPATCHER, MAIN_DISPATCHER, HANDSHAKE_DISPATCHER
from ryu.controller.handler import set_ev_cls

from ryu.ofproto import ofproto_v1_3

from ryu.lib.packet import packet, ether_types
from ryu.lib.packet import ethernet
import numpy as np


def addr_table():  # address table dictionary is created manually
    H = ['00:00:00:00:00:0' + str(h) for h in range(1, 9)]  # hosts
    mac_to_port = {}  # switch는 6개, src는 8개가 존재
    port = [[1, 2, 3, 3, 3, 3, 3, 3],
            [3, 3, 1, 2, 3, 3, 3, 3],
            [1, 1, 2, 2, 3, 3, 3, 3],
            [1, 1, 1, 1, 2, 2, 3, 3],
            [1, 1, 1, 1, 2, 3, 1, 1],
            [1, 1, 1, 1, 1, 1, 2, 3]]  # custom topology에 해당하는 swith,port mapping 정보

    for s in range(1, 7):  # 6 switches
        mac_to_port.setdefault(s, {})
        for h in range(len(H)): #0~7
            mac_to_port[s][H[h]] = port[s - 1][h]
    return mac_to_port

class rl_switch(app_manager.RyuApp):
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

    def __init__(self, *args, **kwargs):
        super(rl_switch, self).__init__(*args, **kwargs)
        self.generated_log = pd.DataFrame(columns=['switch','class','number','time','queue'])
        self.terminal = False
        #self.start_time=datetime.now()
        self.first = True
        self.state=np.zeros((6,4))
        self.mac_to_port = addr_table()
        self.H = ['00:00:00:00:00:0' + str(h) for h in range(1, 9)]  # hosts
        self.dp={}
        self.queue = np.zeros((6,3,4)) #switch, (output)port, priority queue #TODO: queue 필요?
        self.timeslot_size = 0.5 #ms
        self.cycle = 10
        self.ts_cnt=0
        # self.gcl = {1: ['0000000000', '0000000000', '0000000000', '0000000000'],
        #             2: ['0000000000', '0000000000', '0000000000', '1111111111'],
        #             3: ['0000000000', '0000000000', '0000000000', '1111111111'],
        #             4: ['0000000000', '0000000000', '0000000000', '1111111111'],
        #             5: ['0000000000', '0000000000', '0000000000', '1111111111'],
        #             6: ['0000000000', '0000000000', '0000000000', '1111111111']} #최초 action
        self.gcl = {1: ['1111111111', '1111111111', '1111111111', '1111111111'],
                    2: ['1111111111', '1111111111', '1111111111', '1111111111'],
                    3: ['1111111111', '1111111111', '1111111111', '1111111111'],
                    4: ['1111111111', '1111111111', '1111111111', '1111111111'],
                    5: ['1111111111', '1111111111', '1111111111', '1111111111'],
                    6: ['1111111111', '1111111111', '1111111111', '1111111111']
                    } #최초 action

        self.gcl_={1: np.array([list(l) for l in self.gcl[1]]),
                   2: np.array([list(l) for l in self.gcl[2]]),
                   3: np.array([list(l) for l in self.gcl[3]]),
                   4: np.array([list(l) for l in self.gcl[4]]),
                   5: np.array([list(l) for l in self.gcl[5]]),
                   6: np.array([list(l) for l in self.gcl[6]])
                   }

        # flow attribute
        self.command_control = 20  # c&c flow number (Even)
        self.cc_cnt = 0
        self.cc_cnt2 = 0
        self.video = 2  # video flow number (Even)
        self.vd_cnt = 0
        self.vd_cnt2 = 0
        self.audio = 8  # audio flow number (Even)
        self.ad_cnt = 0
        self.ad_cnt2 = 0
        self.cc_period = 5  # to 80
        self.vd_period = 33
        self.ad_period = 1  # milliseconds

        self.timeslot_start = 0

    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def _switch_features_handler(self, ev):
        msg = ev.msg
        datapath = msg.datapath
        self.dp[datapath.id]=datapath
        self.logger.info("스위치 %s 연결" %  datapath.id)

        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser

        match = parser.OFPMatch()
        actions = [parser.OFPActionOutput(port=ofproto.OFPP_CONTROLLER,
                                          max_len=ofproto.OFPCML_NO_BUFFER)]
        inst = [parser.OFPInstructionActions(ofproto.OFPIT_APPLY_ACTIONS,
                                             actions)]
        self.add_flow(datapath, 0, match, 0, inst)
        self.add_meter(datapath, 1)
        self.add_meter(datapath, 2)
        self.add_meter(datapath, 3)


        if len(self.dp)==6:
            hub.sleep(3)
            self.timeslot_start = time.time()
            #self.action_thread = hub.spawn(self.gcl_cycle)
            self.action_1 = hub.spawn(self.gcl_3)
            self.action_2 = hub.spawn(self.gcl_4)
            self.action_3 = hub.spawn(self.gcl_5)
            self.action_4 = hub.spawn(self.gcl_6)
            self.cc_thread = hub.spawn(self._cc_gen1)
            self.cc_thread2 = hub.spawn(self._cc_gen2)
            self.ad_thread = hub.spawn(self._ad_gen1)
            self.ad_thread2 = hub.spawn(self._ad_gen2)
            self.vd_thread =  hub.spawn(self._vd_gen1)
            self.vd_thread2 = hub.spawn(self._vd_gen2)

    #TODO: GCL 적용하기
    def timeslot(self, time):  # timeslot 진행 횟수를 알려주는 함수
        msec = round((time - self.timeslot_start)*1000,1)
        slots = int(msec / self.timeslot_size)
        cyc = int(slots / self.cycle)
        clk = cyc % self.cycle
        return cyc, clk

    def gcl_3(self):
        hub.sleep(3.05)
        datapath = self.dp[3]
        gate = np.array([1,1,1,1])

        while True:
            old_gate = gate
            _,clk = self.timeslot(time.time()) #TODO : clock을 컴퓨터시간에 맞추면 GCl을 순서대로 참조하지 못할 수도 있음
            gate = np.array(list(map(int,self.gcl_[datapath.id][:,clk])))
            change = old_gate - gate

            #class 1
            if gate[0] == 0 :
                self._add_meter(datapath, 1, 0)
                # print("set 0 kbps rate for class 1")
            else :
                self._add_meter(datapath, 1, 1024)

            # class 2
            if gate[1] == 0:
                self._add_meter(datapath, 2, 0)
                #print("set 0 kbps rate for class 2")
            else :
                self._add_meter(datapath, 2, 1024)

            # class 3
            if gate[2] == 0:
                self._add_meter(datapath, 3, 0)
                # print("set 0 kbps rate for class 2")
            else:
                self._add_meter(datapath, 3, 1024)

            #TODO : class 4

            hub.sleep(0.0004)

            if self.terminal == 6:
                self.generated_log.to_csv('switchlog0818_generated.csv')

    def gcl_4(self):
        hub.sleep(3.05)
        datapath = self.dp[4]
        gate = np.array([1,1,1,1])
        while True:
            old_gate = gate
            _, clk = self.timeslot(time.time())  # TODO : clock을 컴퓨터시간에 맞추면 GCl을 순서대로 참조하지 못할 수도 있음
            gate = np.array(list(map(int, self.gcl_[datapath.id][:, clk])))
            change = old_gate - gate

            # class 1
            if gate[0] == 0:
                self._add_meter(datapath, 1, 0)
                #print("set 0 kbps rate for class 1")
            else:
                self._add_meter(datapath, 1, 1024)

            # class 2
            if gate[1] == 0:
                self._add_meter(datapath, 2, 0)
                #print("set 0 kbps rate for class 2")
            else:
                self._add_meter(datapath, 2, 1024)

            # class 3
            if gate[2] == 0:
                self._add_meter(datapath, 3, 0)
                # print("set 0 kbps rate for class 2")
            else:
                self._add_meter(datapath, 3, 1024)

            #TODO : class 4

            hub.sleep(0.0004)

    def gcl_5(self):
        hub.sleep(3.05)
        datapath = self.dp[5]
        gate = np.array([1,1,1,1])
        while True:
            old_gate = gate
            _, clk = self.timeslot(time.time())  # TODO : clock을 컴퓨터시간에 맞추면 GCl을 순서대로 참조하지 못할 수도 있음
            gate = np.array(list(map(int, self.gcl_[datapath.id][:, clk])))
            change = old_gate - gate

            # class 1
            if gate[0] == 0:
                self._add_meter(datapath, 1, 0)
                #print("set 0 kbps rate for class 1")
            else:
                self._add_meter(datapath, 1, 1024)

            # class 2
            if gate[1] == 0:
                self._add_meter(datapath, 2, 0)
                #print("set 0 kbps rate for class 2")
            else:
                self._add_meter(datapath, 2, 1024)

            # class 3
            if gate[2] == 0:
                self._add_meter(datapath, 3, 0)
                # print("set 0 kbps rate for class 2")
            else:
                self._add_meter(datapath, 3, 1024)

            #TODO : class 4

            hub.sleep(0.0004)

    def gcl_6(self):
        hub.sleep(3.05)
        datapath = self.dp[6]
        gate = np.array([1, 1, 1, 1])
        while True:
            old_gate = gate
            _, clk = self.timeslot(time.time())  # TODO : clock을 컴퓨터시간에 맞추면 GCl을 순서대로 참조하지 못할 수도 있음
            gate = np.array(list(map(int, self.gcl_[datapath.id][:, clk])))
            change = old_gate - gate

            # class 1
            if gate[0] == 0:
                self._add_meter(datapath, 1, 0)
                # print("set 0 kbps rate for class 1")
            else:
                self._add_meter(datapath, 1, 1024)

            # class 2
            if gate[1] == 0:
                self._add_meter(datapath, 2, 0)
                # print("set 0 kbps rate for class 2")
            else:
                self._add_meter(datapath, 2, 1024)

            # class 3
            if gate[2] == 0:
                self._add_meter(datapath, 3, 0)
                # print("set 0 kbps rate for class 2")
            else:
                self._add_meter(datapath, 3, 1024)

            # TODO : class 4

            hub.sleep(0.0004)


    # def gcl_cycle(self):
    #     time.sleep(0.005)
    #
    #     while True:
    #         time.sleep(0.001 * self.timeslot_size * 9)
    #         #state 관측
    #         #TODO: state 수정
    #         for switch in range(len(self.state)):
    #             for queue in range(len(self.state[0])): #switch 별 state : len(state[0]) = 4
    #                 self.state[switch][queue] = sum(self.queue[switch, :, queue])
    #
    #         for s in range(len(self.dp)):
    #             self.gcl[s] = [format(np.argmax(self.model1.predict(self.state[s].reshape(-1,4))), '010b'),
    #                    format(np.argmax(self.model2.predict(self.state[s].reshape(-1,4))), '010b'),
    #                    format(np.argmax(self.model3.predict(self.state[s].reshape(-1,4))), '010b'),
    #                    format(np.argmax(self.model4.predict(self.state[s].reshape(-1,4))), '010b')]
    #             print(self.gcl[s])
    #
    #flow entry modification message
    def add_flow_(self, datapath, priority, match, tableid, inst):
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        mod = parser.OFPFlowMod(datapath=datapath, table_id = tableid, priority=priority, command = ofproto.OFPFC_MODIFY,
                                    match=match, instructions = inst)
        datapath.send_msg(mod)

    def add_flow(self, datapath, priority, match, tableid, inst):
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        mod = parser.OFPFlowMod(datapath=datapath, table_id = tableid, priority=priority, command = ofproto.OFPFC_ADD,
                                    match=match, instructions = inst)
        datapath.send_msg(mod)

    def add_meter(self, datapath, m_id):
        parser = datapath.ofproto_parser
        ofproto = datapath.ofproto
        band = parser.OFPMeterBandDrop(rate = 0, burst_size = 1024) #TODO: drop으로 인해 packet drop이 일어날 수도 있음
        mod = parser.OFPMeterMod(datapath, ofproto.OFPMC_ADD, ofproto.OFPMF_KBPS, m_id, [band])
        datapath.send_msg(mod)

    def _add_meter(self, datapath, m_id , rate):
        parser = datapath.ofproto_parser
        ofproto = datapath.ofproto
        band = parser.OFPMeterBandDrop(rate = rate, burst_size = 1024) #TODO: drop으로 인해 packet drop이 일어날 수도 있음
        mod = parser.OFPMeterMod(datapath, ofproto.OFPMC_MODIFY, ofproto.OFPMF_KBPS, m_id, [band])
        datapath.send_msg(mod)

    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def _packet_in_handler(self, ev):
        msg = ev.msg
        datapath = msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        in_port = msg.match['in_port']
        switchid = datapath.id
        pkt = packet.Packet(msg.data)
        eth = pkt.get_protocols(ethernet.ethernet)[0]
        eth_type_ = eth.ethertype
        dst = eth.dst
        src = eth.src
        class_ = 4
        match = parser.OFPMatch(in_port = in_port, eth_dst=dst)

        if (dst in self.H) and (src in self.H):
            #print("dd")
            if eth_type_ == ether_types.ETH_TYPE_IEEE802_3:
                class_ = 1
            elif eth_type_ == ether_types.ETH_TYPE_8021AD:
                class_ = 2
            elif eth_type_ == ether_types.ETH_TYPE_8021AH:
                class_ = 3
        else : #best effort
            self.add_flow(datapath, 1, match, 0, [])
            return

        if dst in self.mac_to_port[switchid]:
            out_port = self.mac_to_port[switchid][dst]
        else:
            out_port = ofproto.OFPP_FLOOD

        actions = [parser.OFPActionOutput(out_port)]
        #actions.insert(0,parser.OFPActionSetQueue(out_port))
        inst = [parser.OFPInstructionActions(ofproto.OFPIT_APPLY_ACTIONS, actions),
                 parser.OFPInstructionMeter(class_, ofproto.OFPIT_METER)]
        goto = parser.OFPInstructionGotoTable(class_)

        self.add_flow(datapath, 100, match, class_, inst)
        self.add_flow(datapath, 100, match, 0, [goto])

        out = parser.OFPPacketOut(datapath=datapath, buffer_id=msg.buffer_id,
                                  in_port=in_port, actions=actions, data=pkt.data)

        datapath.send_msg(out)
        if class_ != 4:
            self.logger.info("[in] %f : 스위치 %s, class %s 패킷" % \
                                 (time.time(), switchid, class_))

    #flow generating thread
    def _cc_gen1(self):
        hub.sleep(3)
        datapath = self.dp[1]
        pkt = packet.Packet()
        pkt.add_protocol(ethernet.ethernet(ethertype=ether_types.ETH_TYPE_IEEE802_3,
                                           dst=self.H[5],
                                           src=self.H[1]))

        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser

        while True:
            self.cc_cnt += 1
            #match = parser.OFPMatch(in_port=2, eth_type=0x05dc, eth_dst=self.H[5])
            actions = [parser.OFPActionOutput(3)]
            #inst = parser.OFPInstructionActions(ofproto.OFPIT_APPLY_ACTIONS, actions)
            #self.add_flow(datapath, 1000, match, 0, [inst])
            pkt.serialize()
            out = parser.OFPPacketOut(datapath=datapath,
                                      buffer_id=ofproto.OFP_NO_BUFFER,
                                      in_port = 2,
                                      actions=actions, data=pkt.data)
            datapath.send_msg(out)
            self.logger.info("%f : %s번 째 C&C1 generated in switch%s " % (time.time(), self.cc_cnt, datapath.id))
            df = pd.DataFrame([(datapath.id, 1, self.cc_cnt, time.time(), 'x')],
                              columns=['switch', 'class', 'number', 'time', 'queue'])
            self.generated_log = self.generated_log.append(df)
            hub.sleep(self.cc_period / 1000)

            if (self.cc_cnt >= self.command_control):
                self.terminal += 1
                if self.terminal == 6:
                    self.generated_log.to_csv('switchlog0818_generated.csv')
                break

    def _cc_gen2(self):
        hub.sleep(3)
        datapath = self.dp[2]
        pkt = packet.Packet()
        pkt.add_protocol(ethernet.ethernet(ethertype=ether_types.ETH_TYPE_IEEE802_3,
                                           dst=self.H[6],
                                           src=self.H[2]))

        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser

        while True:
            self.cc_cnt2 += 1
            match = parser.OFPMatch(in_port=1, eth_type=0x05dc, eth_dst=self.H[6])
            actions = [parser.OFPActionOutput(3)]
            inst = parser.OFPInstructionActions(ofproto.OFPIT_APPLY_ACTIONS, actions)
            self.add_flow(datapath, 1000, match, 0, [inst])
            pkt.serialize()
            out = parser.OFPPacketOut(datapath=datapath,
                                      buffer_id=ofproto.OFP_NO_BUFFER,
                                      in_port=1,
                                      actions=actions, data=pkt.data)
            datapath.send_msg(out)
            self.logger.info("%f : %s번 째 C&C2 generated in switch%s " % (time.time(), self.cc_cnt2, datapath.id))
            df = pd.DataFrame([(datapath.id, 1, self.cc_cnt2, time.time(), 'x')],
                              columns=['switch', 'class', 'number', 'time', 'queue'])
            self.generated_log = self.generated_log.append(df)
            hub.sleep(self.cc_period / 1000)

            if (self.cc_cnt2 >= self.command_control):
                self.terminal += 1
                if self.terminal == 6:
                    self.generated_log.to_csv('switchlog0818_generated.csv')
                break

    def _ad_gen1(self):
        hub.sleep(3)
        datapath = self.dp[1]
        pkt = packet.Packet()
        pkt.add_protocol(ethernet.ethernet(ethertype=ether_types.ETH_TYPE_8021AD,
                                           dst=self.H[4],
                                           src=self.H[0]))

        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser

        while True:
            self.ad_cnt += 1
            #match = parser.OFPMatch(in_port=1, eth_type=0x88a8, eth_dst=self.H[4])
            actions = [parser.OFPActionOutput(3)]
            #inst = parser.OFPInstructionActions(ofproto.OFPIT_APPLY_ACTIONS, actions)
            #self.add_flow(datapath, 1000, match, 0, [inst])
            pkt.serialize()
            out = parser.OFPPacketOut(datapath=datapath,
                                      buffer_id=ofproto.OFP_NO_BUFFER,
                                      in_port=1,
                                      actions=actions, data=pkt.data)
            datapath.send_msg(out)
            self.logger.info("%f : %s번 째 Audio1 generated in switch%s " % (time.time(), self.ad_cnt, datapath.id))
            df = pd.DataFrame([(datapath.id, 2, self.ad_cnt, time.time(), 'x')],
                              columns=['switch', 'class', 'number', 'time', 'queue'])
            self.generated_log = self.generated_log.append(df)
            hub.sleep(self.ad_period / 1000)

            if (self.ad_cnt >= self.audio):
                self.terminal += 1
                break

    def _ad_gen2(self):
        hub.sleep(3)
        datapath = self.dp[2]
        pkt = packet.Packet()
        pkt.add_protocol(ethernet.ethernet(ethertype=ether_types.ETH_TYPE_8021AD,
                                           dst=self.H[7],
                                           src=self.H[3]))

        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser

        while True:
            self.ad_cnt2 += 1
            match = parser.OFPMatch(in_port=2, eth_type=0x88a8, eth_dst=self.H[7])
            actions = [parser.OFPActionOutput(3)]
            inst = parser.OFPInstructionActions(ofproto.OFPIT_APPLY_ACTIONS, actions)
            self.add_flow(datapath, 1000, match, 0, [inst])
            pkt.serialize()
            out = parser.OFPPacketOut(datapath=datapath,
                                      buffer_id=ofproto.OFP_NO_BUFFER,
                                      in_port=2,
                                      actions=actions, data=pkt.data)
            datapath.send_msg(out)
            self.logger.info("%f : %s번 째 Audio2 generated in switch%s " % (time.time(), self.ad_cnt2, datapath.id))
            df = pd.DataFrame([(datapath.id, 2, self.ad_cnt2, time.time(), 'x')],
                              columns=['switch', 'class', 'number', 'time', 'queue'])
            self.generated_log = self.generated_log.append(df)
            hub.sleep(self.ad_period / 1000)

            if (self.ad_cnt2 >= self.audio):
                self.terminal += 1
                break

    def _vd_gen1(self):
        hub.sleep(3)
        datapath = self.dp[1]
        pkt = packet.Packet()
        pkt.add_protocol(ethernet.ethernet(ethertype=ether_types.ETH_TYPE_8021AH,
                                           dst=self.H[4],
                                           src=self.H[0]))

        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser

        while True:
            self.vd_cnt += 1
            match = parser.OFPMatch(in_port=1, eth_type=0x88e7, eth_dst=self.H[4])
            actions = [parser.OFPActionOutput(3)]
            inst = parser.OFPInstructionActions(ofproto.OFPIT_APPLY_ACTIONS, actions)
            self.add_flow(datapath, 1000, match, 0, [inst])
            pkt.serialize()
            out = parser.OFPPacketOut(datapath=datapath,
                                      buffer_id=ofproto.OFP_NO_BUFFER,
                                      in_port=1,
                                      actions=actions, data=pkt.data)
            datapath.send_msg(out)
            self.logger.info("%f : %s번 째 Video1 generated in switch%s " % (time.time(), self.vd_cnt, datapath.id))
            df = pd.DataFrame([(datapath.id, 3, self.vd_cnt, time.time(), 'x')],
                              columns=['switch', 'class', 'number', 'time', 'queue'])
            self.generated_log = self.generated_log.append(df)
            hub.sleep(self.vd_period / 1000)

            if (self.vd_cnt >= self.video):
                self.terminal += 1
                break

    def _vd_gen2(self):
        hub.sleep(3)
        datapath = self.dp[2]
        pkt = packet.Packet()
        pkt.add_protocol(ethernet.ethernet(ethertype=ether_types.ETH_TYPE_8021AH,
                                           dst=self.H[7],
                                           src=self.H[3]))

        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser

        while True:
            self.vd_cnt2 += 1
            match = parser.OFPMatch(in_port=2, eth_type=0x88e7, eth_dst=self.H[7])
            actions = [parser.OFPActionOutput(3)]
            inst = parser.OFPInstructionActions(ofproto.OFPIT_APPLY_ACTIONS, actions)
            self.add_flow(datapath, 1000, match, 0, [inst])
            pkt.serialize()
            out = parser.OFPPacketOut(datapath=datapath,
                                      buffer_id=ofproto.OFP_NO_BUFFER,
                                      in_port=2,
                                      actions=actions, data=pkt.data)
            datapath.send_msg(out)
            self.logger.info("%f : %s번 째 Video2 generated in switch%s " % (time.time(), self.vd_cnt2, datapath.id))
            df = pd.DataFrame([(datapath.id, 3, self.vd_cnt2, time.time(), 'x')],
                              columns=['switch', 'class', 'number', 'time', 'queue'])
            self.generated_log = self.generated_log.append(df)
            hub.sleep(self.vd_period / 1000)

            if (self.vd_cnt2 >= self.video):
                self.terminal += 1
                break
