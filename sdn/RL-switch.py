import os
import sys
import time
from datetime import datetime
from threading import Timer
import pandas as pd
#from model.dqn import DQN

from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import CONFIG_DISPATCHER, MAIN_DISPATCHER
from ryu.controller.handler import set_ev_cls

from ryu.ofproto import ofproto_v1_3

from ryu.lib.packet import packet, ether_types
from ryu.lib.packet import ethernet
from collections import deque
import numpy as np

# TODO: deadline 구현 -> Latency(flow 별 전송시간)구하기 : 모든 packet들이 다 전송되는 데 걸리는 시간
# TODO: dqn model 연결
# TODO: 모든 flow들이 다 전송되면 프로그램을 종료

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

        #self.model = DQN(4,10)
        #self.model.test('~/src/RYU project/weight files/<built-in function time>.h5')
        self.terminal = False
        self.packet_log=pd.DataFrame()
        self.start_time=datetime.now()
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
        self.logger.info("%s.%s : 스위치 %s 연결" % ((datetime.now() - self.start_time).seconds, (datetime.now() - self.start_time).microseconds/100 , datapath.id))

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
        #동시 실행인지, 순차적 실행인지..? - multithreading이기 때문에 동시실행으로 추측
        if len(self.dp)==6:
            #wait 5sec for pingALL
            time.sleep(5)
            #then controller starts timeslot
            self.timeslot() #
            self.cc_generator1()
            self.ad_generator1()
            self.vd_generator1()
            self.cc_generator2()
            self.ad_generator2()
            self.vd_generator2()

    # self.queue 구현해서 대기중인 flow 구하고, gcl 함수호출로 실행, 스위치 첫연결시 gcl은 FIFO
    def timeslot(self):
        if self.first: #스위치 첫 연결시 action
            self.first = False

        self.ts_cnt+=1

        t = Timer((self.timeslot_size / 1000), self.timeslot)  # timeslot
        t.start()

        if self.ts_cnt >= self.cycle:
            t.cancel()
            self.gcl_cycle() #Gcl update

    #TODO: 맞는 지 확인 필요
    def gcl_cycle(self):
        #대기중인 패킷 수
        for switch in range(len(self.state)):
            for queue in range(len(self.state[0])): #switch 별 state : len(state[0]) = 4
                self.state[switch][queue] = sum(self.queue[switch, :, queue])

        #TODO: model dqn 추가하면 이부분을 수정(아랫부분을 주석처리 하면 gcl은 FIFO역할을 하게 됨)
        #for i in range(len(self.state)):#datapath 수만큼 반복
            #gcl = format(np.argmax(self.model.predict_one(self.state[i])), '010b')  #model predict부분
            #gcl =
            #self.gcl[i+1] = gcl

        self.ts_cnt=0
        self.timeslot() #cycle재시작



    # flow table entry 업데이트 - timeout(설정)
    def add_flow(self, datapath, priority, class_, match, actions, buffer_id=None):
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
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

        # TODO: buffer_id가 없는 경우와 있는 경우의 차이? : 대기중인 flow들이 buffer에서 대기하는지?
        if buffer_id:
            mod = parser.OFPFlowMod(datapath=datapath, buffer_id=buffer_id,
                                    priority=priority, match=match,
                                    instructions=inst)
        else:
            mod = parser.OFPFlowMod(datapath=datapath, priority=priority,
                                    match=match, instructions=inst)
        datapath.send_msg(mod)

    # packet-in handler에서는 gcl의 Open정보와 현재 timeslot cnt를 비교하여 delay후 전송한다.
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
        # if (dst in self.H) and (src in self.H):
        #     self.logger.info("스위치 %s의 %s 버퍼에 source %s destination %s 패킷이 input port %s로 들어옴 클래스는" % (switchid,bufferid, src, dst, in_port))

        if eth.ethertype == ether_types.ETH_TYPE_LLDP:
            # ignore lldp packet
            return
        elif eth.ethertype == ether_types.ETH_TYPE_IEEE802_3:
            class_ = 1
            #self.logger.info("class %s packet" % (class_))
        elif eth.ethertype == ether_types.ETH_TYPE_8021AD:
            class_ = 2
            #self.logger.info("class %s packet" % (class_))
        elif eth.ethertype == ether_types.ETH_TYPE_8021AH:
            class_ = 3
            #self.logger.info("class %s packet" % (class_))
        else :
            class_ = 4 #best effort

        # queue에 진입, ts_cnt와 GCl을 보고 대기
        # queue에서 대기(하고있다고 가정)중인 패킷 증가
        self.queue[switchid -1][in_port -1][class_ -1] += 1

        # mac table에 없는 source 추가
        if not (src in self.mac_to_port[switchid]):
            self.mac_to_port[switchid][src] = in_port
            # input port와 연결되어있는 source mac 학습테이블에 저장 (Q: input port가 host가 아닌 switch랑 연결되어있어도..?)

        if dst in self.mac_to_port[switchid]:  # dst의 mac이 테이블에 저장되어있는 경우 그쪽으로 나가면 되지만 아니라면 flooding
            out_port = self.mac_to_port[switchid][dst]
        else:
            out_port = ofproto.OFPP_FLOOD

        # mac address table에 따라 output port 지정
        actions = [parser.OFPActionOutput(out_port)]

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

        #gcl을 참조하여 dealy 계산
        clk = self.ts_cnt

        if class_ != 4:
            self.logger.info("%s.%s : 스위치 %s, 패킷 in class %s,clk %s " % \
                             ((datetime.now()-self.start_time).seconds,(datetime.now()-self.start_time).microseconds/100,switchid, class_,clk))

        while True:
            try:
                delay = (self.gcl[switchid][class_-1][clk - 1:].index('1')) * self.timeslot_size  # gate가 open되기까지의 시간을 계산 (만약 열려있으면 바로 전송)
                break
            except:
                print("다음 cycle까지 기다리기 : 현재 사이클에 OPEN예정이 없음")
                time.sleep(self.timeslot_size/1000)

        time.sleep(delay/1000) #delay
        #flow가 match와 일치하면 match생성시에 지정해준 action으로 packet out한다.
        out = parser.OFPPacketOut(datapath=datapath, buffer_id=msg.buffer_id,
                                  in_port=in_port, actions=actions, data=data)
        datapath.send_msg(out)

        self.queue[switchid-1][in_port-1][class_-1] -= 1
        self.logger.info("%s.%s : 스위치 %s, 패킷 out class %s,clk %s " % \
                         ((datetime.now() - self.start_time).seconds,
                          (datetime.now() - self.start_time).microseconds / 100, switchid, class_, clk))

        if self.terminal:
            self.logger.info("simulation terminated, duration %s.%s" % ((datetime.now() - self.start_time).seconds,
                                                                        (datetime.now() - self.start_time).microseconds / 100))
            #sys.exit()
            #os.exit()
            exit()

    def cc_generator1(self):  # protocol을 추가?
        datapath = self.dp[1]
        #timer는 내부에서 실행해야 계속 재귀호출을 하면서 반복실행될 수 있음.
        self.cc_cnt += 1
        #self.logger.info("%s번째 cc1" % (self.cc_cnt))

        priority = 1

        pkt = packet.Packet()
        pkt.add_protocol(ethernet.ethernet(ethertype=ether_types.ETH_TYPE_IEEE802_3,
                                           dst=self.H[5],
                                           src=self.H[1]))  # 패킷 생성 매커니즘, ethertype을 내가 설정해주어야 할듯

        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser

        pkt.serialize()
        #self.logger.info("packet 정보", pkt)
        #self.logger.info("c&c 패킷 객체 생성, 스위치%s" % (datapath.id))
        self.logger.info("%s.%s : C&C1 %s, 스위치%s " % \
                         ((datetime.now() - self.start_time).seconds,
                          (datetime.now() - self.start_time).microseconds / 100, self.cc_cnt, datapath.id))

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
            if (self.cc_cnt2 >= self.command_control) and (self.ad_cnt >= self.audio) \
                    and (self.ad_cnt2 >= self.audio) and (self.vd_cnt >= self.video) and (self.vd_cnt2 >= self.video):
                self.terminal = True


    def cc_generator2(self):  # protocol을 추가?
        datapath = self.dp[2]
        self.cc_cnt2 += 1
        #self.logger.info("%s번째 cc2" % (self.cc_cnt2))

        priority = 1

        pkt = packet.Packet()
        # pkt_ethernet = pkt.get_protocol(ethernet.ethernet)
        pkt.add_protocol(ethernet.ethernet(ethertype=ether_types.ETH_TYPE_IEEE802_3,
                                           dst=self.H[6],
                                           src=self.H[2]))  # 패킷 생성 매커니즘, ethertype을 내가 설정해주어야 할듯

        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser

        pkt.serialize()

        #self.logger.info("c&c 패킷 객체 생성, 스위치%s" % (datapath.id))
        self.logger.info("%s.%s : C&C2 %s, 스위치%s " % \
                         ((datetime.now() - self.start_time).seconds,
                          (datetime.now() - self.start_time).microseconds / 100, self.cc_cnt2, datapath.id))

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

        if self.cc_cnt2 >= self.command_control:
            t.cancel()
            if (self.cc_cnt >= self.command_control) and (self.ad_cnt >= self.audio) \
                    and (self.ad_cnt2 >= self.audio) and (self.vd_cnt >= self.video) and (self.vd_cnt2 >= self.video):
                self.terminal = True

    def ad_generator1(self):  # protocol을 추가?
        datapath = self.dp[1]
        # timer는 내부에서 실행해야 계속 재귀호출을 하면서 반복실행될 수 있음.
        self.ad_cnt += 1
        #self.logger.info("%s번째 ad1" % (self.ad_cnt))

        priority = 2

        pkt = packet.Packet()
        pkt.add_protocol(ethernet.ethernet(ethertype=ether_types.ETH_TYPE_8021AD,
                                           dst=self.H[4],
                                           src=self.H[0]))  # 패킷 생성 매커니즘, ethertype을 내가 설정해주어야 할듯

        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser

        pkt.serialize()

        #self.logger.info("audio 패킷 객체 생성, 스위치%s" % (datapath.id))
        self.logger.info("%s.%s : AD %s, 스위치%s " % \
                         ((datetime.now() - self.start_time).seconds,
                          (datetime.now() - self.start_time).microseconds / 100, self.ad_cnt, datapath.id))

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
            if (self.cc_cnt >= self.command_control) and (self.cc_cnt2 >= self.command_control) \
                    and (self.ad_cnt2 >= self.audio) and (self.vd_cnt >= self.video) and (self.vd_cnt2 >= self.video):
                self.terminal = True

    def ad_generator2(self):  # protocol을 추가?
        datapath = self.dp[2]
        self.ad_cnt2 += 1
        #self.logger.info("%s번째 ad2" % (self.ad_cnt2))

        priority = 2

        pkt = packet.Packet()
        # pkt_ethernet = pkt.get_protocol(ethernet.ethernet)
        pkt.add_protocol(ethernet.ethernet(ethertype=ether_types.ETH_TYPE_8021AD,
                                           dst=self.H[7],
                                           src=self.H[3]))  # 패킷 생성 매커니즘, ethertype을 내가 설정해주어야 할듯

        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser

        pkt.serialize()

        #self.logger.info("audio 패킷 객체 생성, 스위치%s" % (datapath.id))
        self.logger.info("%s.%s : AD2 %s, 스위치%s " % \
                         ((datetime.now() - self.start_time).seconds,
                          (datetime.now() - self.start_time).microseconds / 100, self.ad_cnt2, datapath.id))

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

        if self.ad_cnt2 >= self.audio:
            t.cancel()
            if (self.cc_cnt >= self.command_control) and (self.cc_cnt2 >= self.command_control) \
                    and (self.ad_cnt >= self.audio) and (self.vd_cnt >= self.video) and (self.vd_cnt2 >= self.video):
                self.terminal = True

    def vd_generator1(self):  # protocol을 추가?
        datapath = self.dp[1]
        # timer는 내부에서 실행해야 계속 재귀호출을 하면서 반복실행될 수 있음.
        self.vd_cnt += 1
        #self.logger.info("%s번째 vd1" % (self.ad_cnt))

        priority = 3

        pkt = packet.Packet()
        pkt.add_protocol(ethernet.ethernet(ethertype=ether_types.ETH_TYPE_8021AH,
                                           dst=self.H[4],
                                           src=self.H[0]))  # 패킷 생성 매커니즘, ethertype을 내가 설정해주어야 할듯

        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser

        pkt.serialize()

        #self.logger.info("video 패킷 객체 생성, 스위치%s" % (datapath.id))
        self.logger.info("%s.%s : VD %s, 스위치%s " % \
                         ((datetime.now() - self.start_time).seconds,
                          (datetime.now() - self.start_time).microseconds / 100, self.vd_cnt, datapath.id))

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
            if (self.cc_cnt >= self.command_control) and (self.cc_cnt2 >= self.command_control) \
                    and (self.ad_cnt >= self.audio) and (self.ad_cnt2 >= self.video) and (self.vd_cnt2 >= self.video):
                self.terminal = True

    def vd_generator2(self):  # protocol을 추가?
        datapath = self.dp[2]
        self.vd_cnt2 += 1
        #self.logger.info("%s번째 vd2" % (self.vd_cnt2))

        priority = 3

        pkt = packet.Packet()
        # pkt_ethernet = pkt.get_protocol(ethernet.ethernet)
        pkt.add_protocol(ethernet.ethernet(ethertype=ether_types.ETH_TYPE_8021AH,
                                           dst=self.H[7],
                                           src=self.H[3]))  # 패킷 생성 매커니즘, ethertype을 내가 설정해주어야 할듯

        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser

        pkt.serialize()

        #self.logger.info("video 패킷 객체 생성, 스위치%s" % (datapath.id))
        self.logger.info("%s.%s : VD2 %s, 스위치%s " % \
                         ((datetime.now() - self.start_time).seconds,
                          (datetime.now() - self.start_time).microseconds / 100, self.vd_cnt2, datapath.id))

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

        if self.vd_cnt2 >= self.video:
            t.cancel()
            if (self.cc_cnt >= self.command_control) and (self.cc_cnt2 >= self.command_control) \
                    and (self.ad_cnt >= self.audio) and (self.ad_cnt2 >= self.video) and (self.vd_cnt >= self.video):
                self.terminal = True
