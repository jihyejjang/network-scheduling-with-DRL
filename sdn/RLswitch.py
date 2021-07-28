
import time
from datetime import datetime
import pandas as pd
from ryu.lib import hub

from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import CONFIG_DISPATCHER, MAIN_DISPATCHER
from ryu.controller.handler import set_ev_cls

from ryu.ofproto import ofproto_v1_5

from ryu.lib.packet import packet, ether_types
from ryu.lib.packet import ethernet
import numpy as np

from tensorflow import keras

# deadline 구현 -> Latency(flow 별 전송시간)구하기 : 모든 packet들이 다 전송되는 데 걸리는 시간
# TODO: dqn model 연결
# openflow 1.3 -> 1.5 : buffer 미지원
# 모든 flow들이 다 전송되면 프로그램을 종료

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
    OFP_VERSIONS = [ofproto_v1_5.OFP_VERSION]

    def __init__(self, *args, **kwargs):
        super(rl_switch, self).__init__(*args, **kwargs)

        self.cc_thread = None
        self.switch_log = pd.DataFrame(columns=['switch','class','arrival','queue'])#{'switch','class','arrival time','queue'}

        self.terminal = False
        #self.start_time=datetime.now()
        self.first = True
        self.state=np.zeros((6,4))
        self.mac_to_port = addr_table()
        self.H = ['00:00:00:00:00:0' + str(h) for h in range(1, 9)]  # hosts
        self.dp={}
        self.queue = np.zeros((6,3,4)) #switch, (output)port, priority queue
        self.timeslot_size = 0.5 #ms
        self.cycle = 10
        self.ts_cnt=0
        self.gcl = {1: ['1111111111', '1111111111', '1111111111', '1111111111'],
                    2: ['1111111111', '1111111111', '1111111111', '1111111111'],
                    3: ['1111111111', '1111111111', '1111111111', '1111111111'],
                    4: ['1111111111', '1111111111', '1111111111', '1111111111'],
                    5: ['1111111111', '1111111111', '1111111111', '1111111111'],
                    6: ['1111111111', '1111111111', '1111111111', '1111111111'],
                    } #스위치 첫 연결 시 action은 FIFO

        # flow attribute
        #self.best_effort = 30  # best effort traffic (Even)
        #self.cnt1 = 0  # 전송된 flow 개수 카운트
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
        # controller에 전송하고 flow entry modify하는 명령 : 빈 매치를 modify해서 flow-miss를 controller로 보내는 명령
        actions = [parser.OFPActionOutput(port=ofproto.OFPP_CONTROLLER,
                                          max_len=ofproto.OFPCML_NO_BUFFER)]
        self.add_flow(datapath,0,match,actions,ofproto.OFP_NO_BUFFER)

        #switch가 모두 연결됨과 동시에 flow들을 주기마다 생성, queue state 요청 메세지
        #동시 실행인지, 순차적 실행인지..? - multithreading이기 때문에 동시실행으로 추측
        if len(self.dp)==6:
            self.timeslot_start = datetime.now()
            self.action_thread = hub.spwan(self.gcl_cycle)
            self.first = False
            self.cc_thread = hub.spawn(self._cc_gen1)
            self.cc_thread2 = hub.spawn(self._cc_gen2)
            self.ad_thread = hub.spawn(self._ad_gen1)
            self.ad_thread2 = hub.spawn(self._ad_gen2)
            self.vd_thread = hub.spawn(self._vd_gen1)
            self.vd_thread2 = hub.spawn(self._vd_gen2)


    # self.queue 구현해서 대기중인 flow 구하고, gcl 함수호출로 실행, 스위치 첫연결시 gcl은 FIFO
    # 0.5밀리초마다 타임슬롯 함수를 실행하는게 아니라 절대시간을 보고 몇번째 timeslot인지 계산한다. gcl도 마찬가지로,0.5ms*9에 갱신한다.
    def timeslot(self, time):  # timeslot 진행 횟수를 알려주는 함수
        sec = (time - self.timeslot_start).seconds
        ms = round((time - self.timeslot_start).microseconds / 1000, 1)  # 소수점 첫째자리까지 출력
        slots = int((sec + 0.001 * ms) / (0.001 * self.timeslot_size))
        cyc = int(slots / self.cycle)  # 몇번째 cycle인지
        clk = cyc % self.cycle  # 몇번째 timeslot인지
        return cyc, clk

    def gcl_cycle(self):
        if self.first == True:
            return

        _,clk = self.timeslot(datetime.now())

        if clk != 9 :
            return

        #state 관측
        for switch in range(len(self.state)):
            for queue in range(len(self.state[0])): #switch 별 state : len(state[0]) = 4
                self.state[switch][queue] = sum(self.queue[switch, :, queue])

        #TODO: model dqn 추가하면 이부분을 수정(아랫부분을 주석처리 하면 gcl은 FIFO역할을 하게 됨)

        for s in range(len(self.dp)):
            self.gcl[s] = [format(np.argmax(self.model1.predict_one(self.state[s])), '010b'),
                   format(np.argmax(self.model2.predict_one(self.state[s])), '010b'),
                   format(np.argmax(self.model3.predict_one(self.state[s])), '010b'),
                   format(np.argmax(self.model4.predict_one(self.state[s])), '010b')]

    # flow table entry 업데이트 - timeout(설정)
    def add_flow(self, datapath, priority, match, actions,buffer_id=None):
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        inst = [parser.OFPInstructionActions(ofproto.OFPIT_APPLY_ACTIONS,
                                             actions)]

        mod = parser.OFPFlowMod(datapath=datapath, priority=priority,
                                    match=match, instructions=inst,buffer_id=buffer_id)
        datapath.send_msg(mod)

    # packet-in handler에서는 gcl의 Open정보와 현재 timeslot cnt를 비교하여 delay후 전송한다.
    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def _packet_in_handler(self, ev):
        #gcl을 참조하여 dealy 계산
        _,clk = self.timeslot(datetime.now())
        msg = ev.msg
        datapath = msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        in_port = msg.match['in_port']

        switchid = datapath.id
        bufferid = msg.buffer_id

        pkt = packet.Packet(msg.data)
        eth = pkt.get_protocols(ethernet.ethernet)[0]
        dst = eth.dst
        src = eth.src
        class_ = 4 #best effort
        if eth.ethertype == ether_types.ETH_TYPE_LLDP:
            # ignore lldp packet
            return
        # flow generator check(debug)
        if (dst in self.H) and (src in self.H):

            if eth.ethertype == ether_types.ETH_TYPE_IEEE802_3:
                class_ = 1
                #self.logger.info("class %s packet" % (class_))
            elif eth.ethertype == ether_types.ETH_TYPE_8021AD:
                class_ = 2
                #self.logger.info("class %s packet" % (class_))
            elif eth.ethertype == ether_types.ETH_TYPE_8021AH:
                class_ = 3
                #self.logger.info("class %s packet" % (class_))

        if class_ != 4:
            self.logger.info("[in] %s초 %0.1f : 스위치 %s, 패킷 in class %s,clk %s, buffer %s " % \
                             ((datetime.now() - self.timeslot_start).seconds,
                              (datetime.now() - self.timeslot_start).microseconds / 1000, switchid, class_, clk,
                              bufferid))
        # queue에 진입, ts_cnt와 GCl을 보고 대기
        # queue에서 대기(하고있다고 가정)중인 패킷 증가

        # mac table에 없는 source 추가
        if not (src in self.mac_to_port[switchid]):
            self.mac_to_port[switchid][src] = in_port
            # input port와 연결되어있는 source mac 학습테이블에 저장 (Q: input port가 host가 아닌 switch랑 연결되어있어도..?)

        if dst in self.mac_to_port[switchid]:  # dst의 mac이 테이블에 저장되어있는 경우 그쪽으로 나가면 되지만 아니라면 flooding
            out_port = self.mac_to_port[switchid][dst]
            self.queue[switchid - 1][out_port - 1][class_ - 1] += 1
        else:
            out_port = ofproto.OFPP_FLOOD


        # mac address table에 따라 output port 지정
        actions = [parser.OFPActionOutput(out_port)]
        # 들어온 패킷에 대해 해당하는 Match를 생성하고, flow entry에 추가하는 작업 (꼭 필요한 작업인가?, 내가 생성해야하는 플로우들만 flow entry에 추가해야하는가?)
        if out_port != ofproto.OFPP_FLOOD:
            match = parser.OFPMatch(in_port=in_port, eth_dst=dst)
            self.add_flow(datapath, 1,match, actions,ofproto.OFP_NO_BUFFER)
            # # verify if we have a valid buffer_id, if yes avoid to send both
            # # flow_mod & packet_out
            if msg.buffer_id != ofproto.OFP_NO_BUFFER:  # 버퍼가 존재하는 패킷이면 return? 전송하지 않음..?
                self.add_flow(datapath, 1, match, actions, msg.buffer_id)
                self.logger.info("buffer가 존재합니다.")
                #return
            # else:
            #     self.add_flow(datapath, 1, match, actions)

        data = None #buffer 존재하면 data는 None이어야 함 (이유는 모름..)
        if msg.buffer_id == ofproto.OFP_NO_BUFFER:
            data = msg.data #buffer가 없으면 data를 할당받음
            #self.logger.info("No buffer")
        #뇌피셜 : buffer가 있으면 data를 보낼 수 없으니 데이터는 전송하지 않고 플로우 정보만 전송해주는것이 아닐까 하는 생각

        while True:
            try:
                delay = (self.gcl[switchid][class_-1][clk - 1:].index('1')) * self.timeslot_size  # gate가 open되기까지의 시간을 계산 (만약 열려있으면 바로 전송)
                break
            except:
                print("다음 cycle까지 기다리기 : 현재 사이클에 OPEN예정이 없음")
                time.sleep(self.timeslot_size/1000)

        time.sleep(delay/1000) #delay
        # self.logger.info("sleep")
        match = parser.OFPMatch(in_port = in_port)
        #flow가 match와 일치하면 match생성시에 지정해준 action으로 packet out한다.
        out = parser.OFPPacketOut(datapath=datapath, buffer_id=msg.buffer_id,
                                  match=match, actions=actions, data=data)
        datapath.send_msg(out)

        #TODO : start time 변경 - deadline이 ms단위인데 엑셀 타임스템프의 범위가 벗어남
        if (1 <= out_port <= 3) and ((switchid == 5) or (switchid == 6)):
            self.queue[switchid-1][out_port-1][class_-1] -= 1
            df = pd.DataFrame([(switchid, class_, datetime.now()-self.timeslot_start, self.queue[switchid-1][out_port-1][class_-1])], columns=['switch','class','arrival','queue'])
            self.switch_log = self.switch_log.append(df)

        # if class_ != 4:
        #     self.logger.info("%s초 %0.1f : 스위치 %s, 패킷 out class %s,clk %s " % \
        #                      ((datetime.now() - self.start_time).seconds,
        #                       (datetime.now() - self.start_time).microseconds / 1000, switchid, class_, clk))

        if self.terminal == True:
            # for d in range(len(self.dp)):
            #     self.send_flow_stats_request(self.dp[d+1])
            self.logger.info("simulation terminated, duration %s.%0.1f" % ((datetime.now() - self.timeslot_start).seconds,(datetime.now() - self.timeslot_start).microseconds / 1000))
            self.switch_log.to_csv('switchlog0728_1.csv')
            #self.terminal = False

    def _cc_gen1(self):
        time.sleep(1)
        datapath = self.dp[1]
        pkt = packet.Packet()
        pkt.add_protocol(ethernet.ethernet(ethertype=ether_types.ETH_TYPE_IEEE802_3,
                                           dst=self.H[5],
                                           src=self.H[1]))

        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        pkt.serialize()

        match = parser.OFPMatch(in_port=2, eth_dst=self.H[5])
        actions = [parser.OFPActionOutput(3)]
        self.add_flow(datapath, 1, match, actions, ofproto.OFP_NO_BUFFER)
        match = parser.OFPMatch(in_port=2)
        data = pkt.data

        out = parser.OFPPacketOut(datapath=datapath,
                                  buffer_id=ofproto.OFP_NO_BUFFER,
                                  match=match,
                                  actions=actions, data=data)

        while True:
            self.cc_cnt += 1
            datapath.send_msg(out)
            hub.sleep(self.cc_period/1000)
            self.logger.info("%s.%0.1f : C&C1 generated %s, 스위치%s " % \
                                     ((datetime.now() - self.timeslot_start).seconds,
                              (datetime.now() - self.timeslot_start).microseconds / 1000, self.cc_cnt, datapath.id))

            df = pd.DataFrame([(datapath.id, 1, datetime.now() - self.timeslot_start, 'x')],
                              columns=['switch', 'class', 'arrival', 'queue'])
            self.switch_log = self.switch_log.append(df)

            if (self.cc_cnt >= self.command_control) and (self.cc_cnt2 >= self.command_control) and (self.ad_cnt >= self.audio) \
                                     and (self.ad_cnt2 >= self.audio) and (self.vd_cnt >= self.video) and (self.vd_cnt2 >= self.video):
                self.terminal = True
                break

    def _cc_gen2(self):
        time.sleep(1)
        datapath = self.dp[2]
        pkt = packet.Packet()
        pkt.add_protocol(ethernet.ethernet(ethertype=ether_types.ETH_TYPE_IEEE802_3,
                                           dst=self.H[6],
                                           src=self.H[2]))

        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        pkt.serialize()

        match = parser.OFPMatch(in_port=1, eth_dst=self.H[6])
        actions = [parser.OFPActionOutput(3)]
        self.add_flow(datapath, 1, match, actions, ofproto.OFP_NO_BUFFER)
        match = parser.OFPMatch(in_port=1)
        data = pkt.data

        out = parser.OFPPacketOut(datapath=datapath,
                                  buffer_id=ofproto.OFP_NO_BUFFER,
                                  match=match,
                                  actions=actions, data=data)
        while True:
            self.cc_cnt2 += 1
            datapath.send_msg(out)
            hub.sleep(self.cc_period/1000)
            self.logger.info("%s.%0.1f : C&C2 generated %s, 스위치%s " % \
                                     ((datetime.now() - self.timeslot_start).seconds,
                              (datetime.now() - self.timeslot_start).microseconds / 1000, self.cc_cnt2, datapath.id))

            df = pd.DataFrame([(datapath.id, 1, datetime.now() - self.timeslot_start, 'x')],
                              columns=['switch', 'class', 'arrival', 'queue'])
            self.switch_log = self.switch_log.append(df)

            if (self.cc_cnt >= self.command_control) and (self.cc_cnt2 >= self.command_control) and (
                    self.ad_cnt >= self.audio) \
                    and (self.ad_cnt2 >= self.audio) and (self.vd_cnt >= self.video) and (self.vd_cnt2 >= self.video):
                self.terminal = True
                break

    def _ad_gen1(self):
        time.sleep(1)
        datapath = self.dp[1]
        pkt = packet.Packet()
        pkt.add_protocol(ethernet.ethernet(ethertype=ether_types.ETH_TYPE_8021AD,
                                           dst=self.H[4],
                                           src=self.H[0]))

        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        pkt.serialize()

        match = parser.OFPMatch(in_port=1, eth_dst=self.H[4])
        actions = [parser.OFPActionOutput(3)]
        self.add_flow(datapath, 1, match, actions, ofproto.OFP_NO_BUFFER)
        match = parser.OFPMatch(in_port=1)
        data = pkt.data

        out = parser.OFPPacketOut(datapath=datapath,
                                  buffer_id=ofproto.OFP_NO_BUFFER,
                                  match=match,
                                  actions=actions, data=data)
        while True:
            self.ad_cnt += 1
            datapath.send_msg(out)
            hub.sleep(self.ad_period/1000)
            self.logger.info("%s.%0.1f : Audio1 generated %s, 스위치%s " % \
                                     ((datetime.now() - self.timeslot_start).seconds,
                              (datetime.now() - self.timeslot_start).microseconds / 1000, self.ad_cnt, datapath.id))

            df = pd.DataFrame([(datapath.id, 2, datetime.now() - self.timeslot_start, 'x')],
                              columns=['switch', 'class', 'arrival', 'queue'])
            self.switch_log = self.switch_log.append(df)

            if (self.cc_cnt >= self.command_control) and (self.cc_cnt2 >= self.command_control) and (
                    self.ad_cnt >= self.audio) \
                    and (self.ad_cnt2 >= self.audio) and (self.vd_cnt >= self.video) and (self.vd_cnt2 >= self.video):
                self.terminal = True
                break

    def _ad_gen2(self):
        time.sleep(1)
        datapath = self.dp[2]
        pkt = packet.Packet()
        pkt.add_protocol(ethernet.ethernet(ethertype=ether_types.ETH_TYPE_8021AD,
                                           dst=self.H[7],
                                           src=self.H[3]))

        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        pkt.serialize()

        match = parser.OFPMatch(in_port=2, eth_dst=self.H[7])
        actions = [parser.OFPActionOutput(3)]
        self.add_flow(datapath, 1, match, actions, ofproto.OFP_NO_BUFFER)
        match = parser.OFPMatch(in_port=2)
        data = pkt.data

        out = parser.OFPPacketOut(datapath=datapath,
                                  buffer_id=ofproto.OFP_NO_BUFFER,
                                  match=match,
                                  actions=actions, data=data)
        while True:
            self.ad_cnt2 += 1
            datapath.send_msg(out)
            hub.sleep(self.ad_period/1000)
            self.logger.info("%s.%0.1f : Audio1 generated %s, 스위치%s " % \
                                     ((datetime.now() - self.timeslot_start).seconds,
                              (datetime.now() - self.timeslot_start).microseconds / 1000, self.ad_cnt2, datapath.id))

            df = pd.DataFrame([(datapath.id, 2, datetime.now() - self.timeslot_start, 'x')],
                              columns=['switch', 'class', 'arrival', 'queue'])
            self.switch_log = self.switch_log.append(df)

            if (self.cc_cnt >= self.command_control) and (self.cc_cnt2 >= self.command_control) and (
                    self.ad_cnt >= self.audio) \
                    and (self.ad_cnt2 >= self.audio) and (self.vd_cnt >= self.video) and (self.vd_cnt2 >= self.video):
                self.terminal = True
                break

    def _vd_gen1(self):
        time.sleep(1)
        datapath = self.dp[1]
        pkt = packet.Packet()
        pkt.add_protocol(ethernet.ethernet(ethertype=ether_types.ETH_TYPE_8021AH,
                                           dst=self.H[4],
                                           src=self.H[0]))

        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        pkt.serialize()

        match = parser.OFPMatch(in_port=1, eth_dst=self.H[4])
        actions = [parser.OFPActionOutput(3)]
        self.add_flow(datapath, 1, match, actions, ofproto.OFP_NO_BUFFER)
        match = parser.OFPMatch(in_port=1)
        data = pkt.data

        out = parser.OFPPacketOut(datapath=datapath,
                                  buffer_id=ofproto.OFP_NO_BUFFER,
                                  match=match,
                                  actions=actions, data=data)
        while True:
            self.vd_cnt += 1
            datapath.send_msg(out)
            hub.sleep(self.vd_period/1000)
            self.logger.info("%s.%0.1f : Audio1 generated %s, 스위치%s " % \
                                     ((datetime.now() - self.timeslot_start).seconds,
                              (datetime.now() - self.timeslot_start).microseconds / 1000, self.vd_cnt, datapath.id))

            df = pd.DataFrame([(datapath.id, 3, datetime.now() - self.timeslot_start, 'x')],
                              columns=['switch', 'class', 'arrival', 'queue'])
            self.switch_log = self.switch_log.append(df)

            if (self.cc_cnt >= self.command_control) and (self.cc_cnt2 >= self.command_control) and (
                    self.ad_cnt >= self.audio) \
                    and (self.ad_cnt2 >= self.audio) and (self.vd_cnt >= self.video) and (self.vd_cnt2 >= self.video):
                self.terminal = True
                break

    def _vd_gen2(self):
        time.sleep(1)
        datapath = self.dp[2]
        pkt = packet.Packet()
        pkt.add_protocol(ethernet.ethernet(ethertype=ether_types.ETH_TYPE_8021AH,
                                           dst=self.H[7],
                                           src=self.H[3]))

        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        pkt.serialize()

        match = parser.OFPMatch(in_port=2, eth_dst=self.H[7])
        actions = [parser.OFPActionOutput(3)]
        self.add_flow(datapath, 1, match, actions, ofproto.OFP_NO_BUFFER)
        match = parser.OFPMatch(in_port=2)
        data = pkt.data

        out = parser.OFPPacketOut(datapath=datapath,
                                  buffer_id=ofproto.OFP_NO_BUFFER,
                                  match=match,
                                  actions=actions, data=data)
        while True:
            self.vd_cnt2 += 1
            datapath.send_msg(out)
            hub.sleep(self.vd_period/1000)
            self.logger.info("%s.%0.1f : Audio1 generated %s, 스위치%s " % \
                                     ((datetime.now() - self.timeslot_start).seconds,
                              (datetime.now() - self.timeslot_start).microseconds / 1000, self.vd_cnt2, datapath.id))

            df = pd.DataFrame([(datapath.id, 3, datetime.now() - self.timeslot_start, 'x')],
                              columns=['switch', 'class', 'arrival', 'queue'])
            self.switch_log = self.switch_log.append(df)

            if (self.cc_cnt >= self.command_control) and (self.cc_cnt2 >= self.command_control) and (
                    self.ad_cnt >= self.audio) \
                    and (self.ad_cnt2 >= self.audio) and (self.vd_cnt >= self.video) and (self.vd_cnt2 >= self.video):
                self.terminal = True
                break
