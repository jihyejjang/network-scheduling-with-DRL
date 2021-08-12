
import time
# from datetime import datetime
import pandas as pd
from ryu.lib import hub
from multiprocessing import Process

from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import CONFIG_DISPATCHER, MAIN_DISPATCHER
from ryu.controller.handler import set_ev_cls

from ryu.ofproto import ofproto_v1_5

from ryu.lib.packet import packet, ether_types, in_proto
from ryu.lib.packet import ethernet, icmp, ipv4, ipv6
import numpy as np

#from tensorflow import keras

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

        #self.model1 = keras.models.load_model("agent17.900466698629316e-07.h5")
        #self.model2 = keras.models.load_model("agent27.900466698629316e-07.h5")
        #self.model3 = keras.models.load_model("agent37.900466698629316e-07.h5")
        #self.model4 = keras.models.load_model("agent47.900466698629316e-07.h5")

        self.generated_log = pd.DataFrame(columns=['switch','class','number','time','queue'])#{'switch','class','arrival time','queue'}
        self.received_log = pd.DataFrame(columns=['arrival time','switch','class','number','delay','queue'])
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
        self.action_thread = Process(target=self.gcl_cycle)

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
            self.timeslot_start = time.time()
            #self.action_thread.start()
            #self.action_thread = hub.spawn(self.gcl_cycle)
            # self.cc_thread = hub.spawn(self._cc_gen1)
            # self.cc_thread2 = hub.spawn(self._cc_gen2)
            # self.ad_thread = hub.spawn(self._ad_gen1)
            # self.ad_thread2 = hub.spawn(self._ad_gen2)
            # self.vd_thread = hub.spawn(self._vd_gen1)
            # self.vd_thread2 = hub.spawn(self._vd_gen2)


    # self.queue 구현해서 대기중인 flow 구하고, gcl 함수호출로 실행, 스위치 첫연결시 gcl은 FIFO
    # 0.5밀리초마다 타임슬롯 함수를 실행하는게 아니라 절대시간을 보고 몇번째 timeslot인지 계산한다. gcl도 마찬가지로,0.5ms*9에 갱신한다.
    def timeslot(self, time):  # timeslot 진행 횟수를 알려주는 함수
        msec = round((time - self.timeslot_start)*1000,1)
        slots = int(msec / self.timeslot_size)
        cyc = int(slots / self.cycle)  # 몇번째 cycle인지
        clk = cyc % self.cycle  # 몇번째 timeslot인지
        return cyc, clk

    def gcl_cycle(self):
        time.sleep(0.005)

        while True:
            print("dqn 관측@@@@@@@@@@@@@@@@@@@@@")
            time.sleep(0.001 * self.timeslot_size * 9)
            #state 관측
            for switch in range(len(self.state)):
                for queue in range(len(self.state[0])): #switch 별 state : len(state[0]) = 4
                    self.state[switch][queue] = sum(self.queue[switch, :, queue])

            #TODO: model dqn 추가하면 이부분을 수정(아랫부분을 주석처리 하면 gcl은 FIFO역할을 하게 됨)

            for s in range(len(self.dp)):
                self.gcl[s] = [format(np.argmax(self.model1.predict(self.state[s].reshape(-1,4))), '010b'),
                       format(np.argmax(self.model2.predict(self.state[s].reshape(-1,4))), '010b'),
                       format(np.argmax(self.model3.predict(self.state[s].reshape(-1,4))), '010b'),
                       format(np.argmax(self.model4.predict(self.state[s].reshape(-1,4))), '010b')]
                print(self.gcl[s])



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
        delay_start_time = time.time()
        _,clk = self.timeslot(delay_start_time)

        msg = ev.msg
        datapath = msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        in_port = msg.match['in_port']

        switchid = datapath.id
        #bufferid = msg.buffer_id

        pkt = packet.Packet(msg.data)
        eth = pkt.get_protocols(ethernet.ethernet)[0]
        ipv6_ = pkt.get_protocol(ipv6.ipv6)
        if ipv6_!= None:
            return
        dst = eth.dst
        src = eth.src

        # icmp_packet = pkt.get_protocol(icmp.icmp)
        # print (icmp_packet)
        # payload = icmp_packet.data
        # info = payload.split(';')

        class_ = 4 #best effort

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


        # if not (src in self.mac_to_port[switchid]):
        #     self.mac_to_port[switchid][src] = in_port

        if dst in self.mac_to_port[switchid]:
            out_port = self.mac_to_port[switchid][dst]
            self.queue[switchid - 1][out_port - 1][class_ - 1] += 1
        else:
            out_port = ofproto.OFPP_FLOOD


        actions = [parser.OFPActionOutput(out_port)]
        if out_port != ofproto.OFPP_FLOOD:
            match = parser.OFPMatch(in_port=in_port, eth_dst=dst)
            self.add_flow(datapath, 1,match, actions,ofproto.OFP_NO_BUFFER)
            # # verify if we have a valid buffer_id, if yes avoid to send both
            # # flow_mod & packet_out
            if msg.buffer_id != ofproto.OFP_NO_BUFFER:
                self.add_flow(datapath, 1, match, actions, msg.buffer_id)
                return

        # while True:
        #     try:
        delay = (self.gcl[switchid][class_-1][clk - 1:].index('1')) * self.timeslot_size  # gate가 open되기까지의 시간을 계산 (만약 열려있으면 바로 전송)
                # break
            # except:
            #     print("다음 cycle까지 기다리기 : 현재 사이클에 OPEN예정이 없음")
            #     time.sleep(self.timeslot_size/1000)

        #print("delay", delay/1000)

        hub.sleep(delay/1000) #delay

        match = parser.OFPMatch(in_port = in_port)
        #flow가 match와 일치하면 match생성시에 지정해준 action으로 packet out한다.
        delay_end_time = 0
        if msg.buffer_id == ofproto.OFP_NO_BUFFER:
            delay_end_time = time.time()

        out = parser.OFPPacketOut(datapath=datapath, buffer_id=msg.buffer_id,
                                  match=match, actions=actions, data=msg.data)

        datapath.send_msg(out)


        if (1 <= out_port <= 3):
            self.queue[switchid-1][out_port-1][class_-1] -= 1
            df = pd.DataFrame([(delay_end_time, switchid, class_, '-', delay_end_time - delay_start_time,
                                self.queue[switchid - 1][out_port - 1][class_ - 1])],
                              columns=['arrival time', 'switch', 'class', 'number', 'delay', 'queue'])
            self.received_log = self.received_log.append(df)

            if class_ != 4:
                self.logger.info("length %s",len(msg.data))
                self.logger.info("[in] %f : 스위치 %s, class %s 의 %s번째 패킷,clk %s" % \
                                 (time.time(), switchid, class_, '-', clk))



        if self.terminal == 6:
            # for d in range(len(self.dp)):
            #     self.send_flow_stats_request(self.dp[d+1])
            #self.logger.info("simulation terminated, duration %s.%0.1f" % ((datetime.now() - self.timeslot_start).seconds,(datetime.now() - self.timeslot_start).microseconds / 1000))
            self.generated_log.to_csv('switchlog0810_generated.csv')
            self.received_log.to_csv('switchlog0810_received.csv')
            #self.terminal = False

    def _cc_gen1(self):
        datapath = self.dp[1]
        pkt = packet.Packet()
        pkt.add_protocol(ethernet.ethernet(ethertype=ether_types.ETH_TYPE_IEEE802_3,
                                           dst=self.H[5],
                                           src=self.H[1]))

        # pkt.add_protocol(ipv4.ipv4(proto=in_proto.IPPROTO_ICMP,
        #                            src='10.0.0.2',
        #                            dst='10.0.0.6'))

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
            # payload = str('%d;%f' % (self.cc_cnt, time.time())).encode('ascii')
            # print ("payload",payload)
            # payload_ = icmp.echo(data=payload)
            #pkt.add_protocol(icmp.icmp(data=payload_))
            # pkt.serialize()
            # out = parser.OFPPacketOut(datapath=datapath,
            #                           buffer_id=ofproto.OFP_NO_BUFFER,
            #                           match=match,
            #                           actions=actions, data=pkt.data)
            datapath.send_msg(out)

            self.logger.info("%f : C&C1 generated %s, 스위치%s " % (time.time(), self.cc_cnt, datapath.id))

            df = pd.DataFrame([(datapath.id, 1, self.cc_cnt, time.time(), 'x')],
                              columns=['switch', 'class', 'number', 'time', 'queue'])
            self.generated_log = self.generated_log.append(df)

            hub.sleep(self.cc_period / 1000)

            if (self.cc_cnt >= self.command_control):
                self.terminal += 1
                break

    def _cc_gen2(self):
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

            self.logger.info("%f : C&C2 generated %s, 스위치%s " % (time.time(), self.cc_cnt2, datapath.id))

            df = pd.DataFrame([(datapath.id, 1, self.cc_cnt2, time.time(), 'x')],
                              columns=['switch', 'class', 'number', 'time', 'queue'])
            self.generated_log = self.generated_log.append(df)

            hub.sleep(self.cc_period / 1000)


            if (self.cc_cnt2 >= self.command_control):
                self.terminal += 1
                break

    def _ad_gen1(self):
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
            self.logger.info("%f : Audio1 generated %s, 스위치%s " % (time.time(), self.ad_cnt, datapath.id))

            df = pd.DataFrame([(datapath.id, 2, self.ad_cnt, time.time(), 'x')],
                              columns=['switch', 'class', 'number', 'time', 'queue'])
            self.generated_log = self.generated_log.append(df)
            hub.sleep(self.ad_period / 1000)

            if (self.ad_cnt >= self.audio):
                self.terminal += 1
                break

    def _ad_gen2(self):
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
            self.logger.info("%f : Audio2 generated %s, 스위치%s " % (time.time(), self.ad_cnt2, datapath.id))

            df = pd.DataFrame([(datapath.id, 2, self.ad_cnt2, time.time(), 'x')],
                              columns=['switch', 'class', 'number', 'time', 'queue'])
            self.generated_log = self.generated_log.append(df)
            hub.sleep(self.ad_period / 1000)

            if (self.ad_cnt2 >= self.audio):
                self.terminal+=1
                break


    def _vd_gen1(self):
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

            self.logger.info("%f : video1 generated %s, 스위치%s " % (time.time(), self.vd_cnt, datapath.id))

            df = pd.DataFrame([(datapath.id, 3, self.vd_cnt, time.time(), 'x')],
                              columns=['switch', 'class', 'number', 'time', 'queue'])
            self.generated_log = self.generated_log.append(df)
            hub.sleep(self.vd_period / 1000)

            if (self.vd_cnt >= self.video):
                self.terminal+=1
                break

    def _vd_gen2(self):
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
            self.logger.info("%f : video2 generated %s, 스위치%s " % (time.time(), self.vd_cnt2, datapath.id))

            df = pd.DataFrame([(datapath.id, 3, self.vd_cnt2, time.time(), 'x')],
                              columns=['switch', 'class', 'number', 'time', 'queue'])
            self.generated_log = self.generated_log.append(df)
            hub.sleep(self.vd_period / 1000)
            if (self.vd_cnt2 >= self.video):
                self.terminal+=1
                break
                # if (self.cc_cnt >= self.command_control) and (self.cc_cnt2 >= self.command_control) and (self.ad_cnt >= self.audio) \
                #         and (self.ad_cnt2 >= self.audio) and (self.vd_cnt >= self.video):
                #     self.terminal=True
                #     break

            #
            # if (self.cc_cnt >= self.command_control) and (self.cc_cnt2 >= self.command_control) and (
            #         self.ad_cnt >= self.audio) \
            #         and (self.ad_cnt2 >= self.audio) and (self.vd_cnt >= self.video) and (self.vd_cnt2 >= self.video):
            #     self.terminal = True
            #     break
