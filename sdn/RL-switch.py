from threading import Timer
from model.dqn import DQN

from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import CONFIG_DISPATCHER, MAIN_DISPATCHER
from ryu.controller.handler import set_ev_cls

from ryu.ofproto import ofproto_v1_3

from ryu.lib.packet import packet, ether_types
from ryu.lib.packet import ethernet
from collections import deque
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
        for h in range(len(H)):
            mac_to_port[s][H[h]] = port[s - 1][h]
    return mac_to_port

class rl_switch(app_manager.RyuApp):
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

    def __init__(self, *args, **kwargs):
        super(rl_switch, self).__init__(*args, **kwargs)

        self.model = DQN(4,10)
        self.model.test('~/src/RYU project/weight files/<built-in function time>.h5')
        self.queue_dp1 = {1: deque(), 2: deque(), 3: deque()}
        self.queue_dp2 = {1: deque(), 2: deque(), 3: deque()}
        self.queue_dp3 = {1: deque(), 2: deque(), 3: deque()}
        self.queue_dp4 = {1: deque(), 2: deque(), 3: deque()}
        self.queue_dp5 = {1: deque(), 2: deque(), 3: deque()}
        self.queue_dp6 = {1: deque(), 2: deque(), 3: deque()}

        self.state=[]

        self.mac_to_port = addr_table()
        self.H = ['00:00:00:00:00:0' + str(h) for h in range(1, 9)]  # hosts
        self.dp={}
        self.timeslot_size = 0.5 #ms
        self.cycle = 10
        self.ts_cnt=0
        self.gcl={}

        # flow attribute
        # self.best_effort = 30  # best effort traffic (Even)
        # self.cnt1 = 0  # 전송된 flow 개수 카운트
        self.command_control = 20  # c&c flow number (Even)
        self.cc_cnt = 0
        self.cc_cnt2 = 0
        self.video = 2  # video flow number (Even)
        self.vd_cnt = 0
        self.vd_cnt2 = 0
        self.audio = 8  # audio flow number (Even)
        self.ad_cnt = 0
        self.ad_cnt2 = 0

        # self.be_period = 3
        self.cc_period = 5  # to 80
        self.vd_period = 33
        self.ad_period = 1  # milliseconds

        # switch address도 알아야 하는지? / mininet에서 항상 고정형 mac주소를 구현하는 방법?(autoMac=True 하면 됨)
        # 패킷을 생성해도 address table이 완성되지 않으면 flooding하는건가..?

    # 스위치 최초연결시 : flow modify
    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def _switch_features_handler(self, ev):
        msg = ev.msg
        datapath = msg.datapath
        self.dp[datapath.id]=datapath
        #self.logger.info("%s"%(self.dp))
        self.logger.info("스위치 %s 연결" % datapath.id)
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser

        # controller에 전송하고 flow entry modify하는 명령 : 빈 매치를 modify해서 flow-miss를 controller로 보내는 명령
        actions = [parser.OFPActionOutput(port=ofproto.OFPP_CONTROLLER,
                                          max_len=ofproto.OFPCML_NO_BUFFER)]
        inst = [parser.OFPInstructionActions(type_=ofproto.OFPIT_APPLY_ACTIONS,
                                             actions=actions)]
        mod = parser.OFPFlowMod(datapath=datapath,
                                priority=0,
                                match=parser.OFPMatch(),
                                instructions=inst)
        datapath.send_msg(mod)

        #switch가 모두 연결됨과 동시에 flow들을 주기마다 생성, queue state 요청 메세지
        #TODO: 동시 실행인지, 순차적 실행인지..? - multithreading이기 때문에 동시실행으로 추측
        if len(self.dp)==6:
            self.timeslot() #switch5개가 연결되면 timeslot 시작
            self.cc_generator1()
            self.ad_generator1()
            self.vd_generator1()
            self.cc_generator2()
            self.ad_generator2()
            self.vd_generator2()



    #TODO: Queue를 그냥 내가 구현하자.. self.queue 구현해서 대기중인 flow 구하고, state_observe는 데코레이터가 아니라 그냥 함수호출로 실행, 스위치 첫연결시 gcl?
    #@set_ev_cls(ofp_event.EventOFPQueueStatsReply, MAIN_DISPATCHER)
    def timeslot(self):
        if self.first: #스위치 첫 연결시 action
            return
        # msg = ev.msg
        # datapath = msg.datapath
        #
        # queues=[]
        #
        # for stat in ev.msg.body:
        #     self.queue[datapath.id][stat.port_no][stat.queue_id] = stat.tx_packets
        #
        # #mininet에서 실험해보고 주석처리
        # for stat in ev.msg.body:
        #     queues.append('port_no=%d queue_id=%d '
        #                   'tx_bytes=%d tx_packets=%d tx_errors=%d '
        #                   'duration_sec=%d duration_nsec=%d' %
        #                   (stat.port_no, stat.queue_id,
        #                    stat.tx_bytes, stat.tx_packets, stat.tx_errors,
        #                    stat.duration_sec, stat.duration_nsec))
        #     self.logger.debug('QueueStats: %s', queues)

        t = Timer((self.timeslot_size / 1000), self.timeslot)  # timeslot
        t.start()

        if self.ts_cnt >= self.cycle:
            t.cancel()
            self.state_observe()

    def state_observe(self):
        # TODO:queue_dp 딕셔너리를 통해 적절한 대기중인 packet의 수 관측
        self.state = [] #6*4*10
        for i in range(len(self.state)):#datapath 수만큼 반복
            gcl = format(np.argmax(self.model.predict_one(self.state[i])), '0' + str(self.tdm_cycle) + 'b')
            self.gcl[i+1]=gcl

        # for i in range(len(gcl[self.ts_cnt])):
        #     if gcl[i][self.ts_cnt]==1:
        #
        #         #어떡하지....
        #         return

        # t = Timer((self.timeslot_size / 1000), self.timeslot) #0.5ms마다 실행하고 10회실행하면 종료
        # t.start()
        #
        # if self.ts_cnt >= self.cycle:
        #     t.cancel()
        #     self.ts_cnt = 0

        self.ts_cnt=0
        self.timeslot() #cycle재시작



    # flow table entry 업데이트 - timeout(설정)
    def add_flow(self, datapath, priority, class_, match, actions, buffer_id=None):
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        timeout = 0

        inst = [parser.OFPInstructionActions(ofproto.OFPIT_APPLY_ACTIONS,
                                             actions)]

        # TODO: timeout이 아닌 실제 전송시간으로 Deadline내에 전송하는 지 확인하는 코드가 필요함 - timeout은 초단위이고, 스위치별로 존재하기 때문
        # if class_ == 1 : #c&c flow
        #     timeout = 5
        # elif class_ == 2 : #audio flow
        #     timeout = 4
        # elif class_ == 3 : #video flow
        #     timeout = 30
        # else:
        #     timeout = 10 #very loose timeout

        # TODO: buffer_id가 없는 경우와 있는 경우의 차이?
        if buffer_id:
            mod = parser.OFPFlowMod(datapath=datapath, buffer_id=buffer_id,
                                    priority=priority, match=match,
                                    instructions=inst)
        else:
            mod = parser.OFPFlowMod(datapath=datapath, priority=priority,
                                    match=match, instructions=inst)
        datapath.send_msg(mod)

    # packet-in 처리
    # TODO:packet-in handler에서는 gcl의 Open정보와 현재 timeslot cnt를 비교하여 delay후 전송한다.
    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def _packet_in_handler(self, ev):
        class_ = 0
        # 에러 발생 감지
        if ev.msg.msg_len < ev.msg.total_len:
            self.logger.debug("packet truncated: only %s of %s bytes",
                              ev.msg.msg_len, ev.msg.total_len)

        msg = ev.msg
        datapath = msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser

        in_port = msg.match['in_port']
        switchid = datapath.id
        bufferid = msg.buffer_id

        pkt = packet.Packet(data=msg.data)
        eth = pkt.get_protocols(ethernet.ethernet)[0]
        dst = eth.dst
        src = eth.src

        # flow generator check(debug)
        if (dst in self.H) and (src in self.H):
            # self.logger.info("packet-in , 이더넷 %s" % (pkt_ethernet) )
            self.logger.info("packet-in , 스위치 %s" % (switchid))
            self.logger.info("packet-in, buffer_id %s" % (bufferid))
            self.logger.info("스위치 %s에 source %s destination %s 패킷이 input port %s로 들어옴" % (switchid, src, dst, in_port))

        if eth.ethertype == ether_types.ETH_TYPE_LLDP:
            # ignore lldp packet
            return
        elif eth.ethertype == ether_types.ETH_TYPE_IEEE802_3:
            class_ = 1
            self.logger.info("class %s packet" % (class_))
        elif eth.ethertype == ether_types.ETH_TYPE_8021AD:
            class_ = 2
            self.logger.info("class %s packet" % (class_))
        elif eth.ethertype == ether_types.ETH_TYPE_8021AH:
            class_ = 3
            self.logger.info("class %s packet" % (class_))

        # mac table에 없는 source 추가
        if not (src in self.mac_to_port[switchid]):
            self.mac_to_port[switchid][src] = in_port
            # input port와 연결되어있는 source mac 학습테이블에 저장 (Q: input port가 host가 아닌 switch랑 연결되어있어도..?)

        if dst in self.mac_to_port[switchid]:  # dst의 mac이 테이블에 저장되어있는 경우 그쪽으로 나가면 되지만 아니라면 flooding
            out_port = self.mac_to_port[switchid][dst]
        else:
            out_port = ofproto.OFPP_FLOOD

        # # mac address table에 따라 output port 지정
        # actions = [parser.OFPActionOutput(out_port)]

        #TODO: class에 따라서 setqueue 지정 - flow generate시에도 해당. port지정 후 queue지정으로 바꿔주기
        actions = [parser.OFPActionOutput(out_port),parser.OFPActionSetQueue(queue_id=class_)]

        # 들어온 패킷에 대해 해당하는 Match를 생성하고, flow entry에 추가하는 작업 (꼭 필요한 작업인가?, 내가 생성해야하는 플로우들만 flow entry에 추가해야하는가?)
        if out_port != ofproto.OFPP_FLOOD:
            match = parser.OFPMatch(in_port=in_port, eth_dst=dst, eth_src=src)
            # verify if we have a valid buffer_id, if yes avoid to send both
            # flow_mod & packet_out
            if msg.buffer_id != ofproto.OFP_NO_BUFFER:  # 버퍼가 존재하는 패킷이면 return? 전송하지 않음..?
                self.add_flow(datapath, 1, class_, match, actions, msg.buffer_id)
                return
            else:
                self.add_flow(datapath, 1, class_, match, actions)

        # 왜 buffer가 존재하는 flood는 왜 data를 none으로 할까?
        data = None
        if msg.buffer_id == ofproto.OFP_NO_BUFFER:
            data = msg.data

        #flow가 match와 일치하면 match생성시에 지정해준 action으로 packet out한다.
        out = parser.OFPPacketOut(datapath=datapath, buffer_id=msg.buffer_id,
                                  in_port=in_port, actions=actions, data=data)
        datapath.send_msg(out)


    #TODO: setoutputport -> setqueue
    def cc_generator1(self):  # protocol을 추가?
        datapath = self.dp[1]
        #timer는 내부에서 실행해야 계속 재귀호출을 하면서 반복실행될 수 있음.
        self.cc_cnt += 1
        self.logger.info("%s번째 cc1" % (self.cc_cnt))

        priority = 1

        pkt = packet.Packet()
        pkt.add_protocol(ethernet.ethernet(ethertype=ether_types.ETH_TYPE_IEEE802_3,
                                           dst=self.H[5],
                                           src=self.H[1]))  # 패킷 생성 매커니즘, ethertype을 내가 설정해주어야 할듯

        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser

        pkt.serialize()
        self.logger.info("packet 정보", pkt)
        self.logger.info("c&c 패킷 객체 생성, 스위치%s" % (datapath.id))

        data = pkt.data
        actions = [parser.OFPActionOutput(port=3)]  # switch 1과 2의 3번 포트로 출력하기 때문에
        out = parser.OFPPacketOut(datapath=datapath,
                                 buffer_id=ofproto.OFP_NO_BUFFER,  # buffer id?
                                 in_port=ofproto.OFPP_CONTROLLER,
                                 # controller에서 들어온 패킷 (생성된 패킷이기 때문에? host자체에서 생성은 하지 못하는듯)
                                 actions=actions,
                                 data=data)

        datapath.send_msg(out)

        t = Timer((self.cc_period/1000), self.cc_generator1)
        t.start()

        if self.cc_cnt >= self.command_control:
            t.cancel()

    def cc_generator2(self):  # protocol을 추가?
        datapath = self.dp[2]
        self.cc_cnt2 += 1
        self.logger.info("%s번째 cc2" % (self.cc_cnt2))

        priority = 1

        pkt = packet.Packet()
        # pkt_ethernet = pkt.get_protocol(ethernet.ethernet)
        pkt.add_protocol(ethernet.ethernet(ethertype=ether_types.ETH_TYPE_IEEE802_3,
                                           dst=self.H[6],
                                           src=self.H[2]))  # 패킷 생성 매커니즘, ethertype을 내가 설정해주어야 할듯

        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser

        pkt.serialize()

        self.logger.info("c&c 패킷 객체 생성, 스위치%s" % (datapath.id))

        data = pkt.data
        actions = [parser.OFPActionOutput(port=3)]  # switch 1과 2의 3번 포트로 출력하기 때문에
        out = parser.OFPPacketOut(datapath=datapath,
                                 buffer_id=ofproto.OFP_NO_BUFFER,  # buffer id?
                                 in_port=ofproto.OFPP_CONTROLLER,
                                 # controller에서 들어온 패킷 (생성된 패킷이기 때문에? host자체에서 생성은 하지 못하는듯)
                                 actions=actions,
                                 data=data)

        datapath.send_msg(out)

        t = Timer((self.cc_period/1000), self.cc_generator2)
        t.start()

        if self.cc_cnt >= self.command_control:
            t.cancel()

    def ad_generator1(self):  # protocol을 추가?
        datapath = self.dp[1]
        # timer는 내부에서 실행해야 계속 재귀호출을 하면서 반복실행될 수 있음.
        self.ad_cnt += 1
        self.logger.info("%s번째 ad1" % (self.ad_cnt))

        priority = 2

        pkt = packet.Packet()
        pkt.add_protocol(ethernet.ethernet(ethertype=ether_types.ETH_TYPE_8021AD,
                                           dst=self.H[4],
                                           src=self.H[0]))  # 패킷 생성 매커니즘, ethertype을 내가 설정해주어야 할듯

        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser

        pkt.serialize()

        self.logger.info("audio 패킷 객체 생성, 스위치%s" % (datapath.id))

        data = pkt.data
        actions = [parser.OFPActionOutput(port=3)]  # switch 1과 2의 3번 포트로 출력하기 때문에
        out = parser.OFPPacketOut(datapath=datapath,
                                  buffer_id=ofproto.OFP_NO_BUFFER,  # buffer id?
                                  in_port=ofproto.OFPP_CONTROLLER,
                                  # controller에서 들어온 패킷 (생성된 패킷이기 때문에? host자체에서 생성은 하지 못하는듯)
                                  actions=actions,
                                  data=data)

        datapath.send_msg(out)

        t = Timer((self.ad_period / 1000), self.ad_generator1)
        t.start()

        if self.ad_cnt >= self.audio:
            t.cancel()

    def ad_generator2(self):  # protocol을 추가?
        datapath = self.dp[2]
        self.ad_cnt2 += 1
        self.logger.info("%s번째 ad2" % (self.ad_cnt2))

        priority = 2

        pkt = packet.Packet()
        # pkt_ethernet = pkt.get_protocol(ethernet.ethernet)
        pkt.add_protocol(ethernet.ethernet(ethertype=ether_types.ETH_TYPE_8021AD,
                                           dst=self.H[7],
                                           src=self.H[3]))  # 패킷 생성 매커니즘, ethertype을 내가 설정해주어야 할듯

        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser

        pkt.serialize()

        self.logger.info("audio 패킷 객체 생성, 스위치%s" % (datapath.id))

        data = pkt.data
        actions = [parser.OFPActionOutput(port=3)]  # switch 1과 2의 3번 포트로 출력하기 때문에
        out = parser.OFPPacketOut(datapath=datapath,
                                  buffer_id=ofproto.OFP_NO_BUFFER,  # buffer id?
                                  in_port=ofproto.OFPP_CONTROLLER,
                                  # controller에서 들어온 패킷 (생성된 패킷이기 때문에? host자체에서 생성은 하지 못하는듯)
                                  actions=actions,
                                  data=data)

        datapath.send_msg(out)

        t = Timer((self.ad_period / 1000), self.ad_generator2)
        t.start()

        if self.ad_cnt >= self.audio:
            t.cancel()

    def vd_generator1(self):  # protocol을 추가?
        datapath = self.dp[1]
        # timer는 내부에서 실행해야 계속 재귀호출을 하면서 반복실행될 수 있음.
        self.vd_cnt += 1
        self.logger.info("%s번째 vd1" % (self.ad_cnt))

        priority = 3

        pkt = packet.Packet()
        pkt.add_protocol(ethernet.ethernet(ethertype=ether_types.ETH_TYPE_8021AH,
                                           dst=self.H[4],
                                           src=self.H[0]))  # 패킷 생성 매커니즘, ethertype을 내가 설정해주어야 할듯

        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser

        pkt.serialize()

        self.logger.info("video 패킷 객체 생성, 스위치%s" % (datapath.id))

        data = pkt.data
        actions = [parser.OFPActionOutput(port=3)]  # switch 1과 2의 3번 포트로 출력하기 때문에
        out = parser.OFPPacketOut(datapath=datapath,
                                  buffer_id=ofproto.OFP_NO_BUFFER,  # buffer id?
                                  in_port=ofproto.OFPP_CONTROLLER,
                                  # controller에서 들어온 패킷 (생성된 패킷이기 때문에? host자체에서 생성은 하지 못하는듯)
                                  actions=actions,
                                  data=data)

        datapath.send_msg(out)

        t = Timer((self.vd_period / 1000), self.vd_generator1)
        t.start()

        if self.vd_cnt >= self.video:
            t.cancel()

    def vd_generator2(self):  # protocol을 추가?
        datapath = self.dp[2]
        self.vd_cnt2 += 1
        self.logger.info("%s번째 vd2" % (self.vd_cnt2))

        priority = 3

        pkt = packet.Packet()
        # pkt_ethernet = pkt.get_protocol(ethernet.ethernet)
        pkt.add_protocol(ethernet.ethernet(ethertype=ether_types.ETH_TYPE_8021AH,
                                           dst=self.H[7],
                                           src=self.H[3]))  # 패킷 생성 매커니즘, ethertype을 내가 설정해주어야 할듯

        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser

        pkt.serialize()

        self.logger.info("video 패킷 객체 생성, 스위치%s" % (datapath.id))

        data = pkt.data
        actions = [parser.OFPActionOutput(port=3)]  # switch 1과 2의 3번 포트로 출력하기 때문에
        out = parser.OFPPacketOut(datapath=datapath,
                                  buffer_id=ofproto.OFP_NO_BUFFER,  # buffer id?
                                  in_port=ofproto.OFPP_CONTROLLER,
                                  # controller에서 들어온 패킷 (생성된 패킷이기 때문에? host자체에서 생성은 하지 못하는듯)
                                  actions=actions,
                                  data=data)

        datapath.send_msg(out)

        t = Timer((self.vd_period / 1000), self.vd_generator2)
        t.start()

        if self.vd_cnt >= self.video:
            t.cancel()
