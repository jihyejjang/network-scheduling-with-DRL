# Copyright (C) 2011 Nippon Telegraph and Telephone Corporation.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from operator import attrgetter
from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import CONFIG_DISPATCHER, MAIN_DISPATCHER
from ryu.controller.handler import set_ev_cls
from ryu.ofproto import ofproto_v1_5
from ryu.lib.packet import packet, ether_types, in_proto
from ryu.lib.packet import ethernet,ipv4,icmp
from ryu.lib import hub
import time

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

class SimpleSwitch15(app_manager.RyuApp):
    OFP_VERSIONS = [ofproto_v1_5.OFP_VERSION]

    def __init__(self, *args, **kwargs):
        super(SimpleSwitch15, self).__init__(*args, **kwargs)
        self.mac_to_port = addr_table()
        self.H = ['00:00:00:00:00:0' + str(h) for h in range(1, 9)]  # hosts
        self.ip = ['10.0.0.'+str(h) for h in range(1,9)]
        self.dp = {}
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
        self.dp[datapath.id] = datapath
        self.logger.info("스위치 %s 연결" % datapath.id)

        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser

        match = parser.OFPMatch()
        actions = [parser.OFPActionOutput(port=ofproto.OFPP_CONTROLLER,
                                          max_len=ofproto.OFPCML_NO_BUFFER)]
        self.add_flow(datapath, 0, match, actions, ofproto.OFP_NO_BUFFER)

        if len(self.dp) == 6:
            print ("simulation started")
            self.timeslot_start = time.time()
            # self.action_thread = hub.spawn(self.gcl_cycle)
            self.cc_thread = hub.spawn(self._cc_gen1)
            # self.cc_thread2 = hub.spawn(self._cc_gen2)
            # self.ad_thread = hub.spawn(self._ad_gen1)
            # self.ad_thread2 = hub.spawn(self._ad_gen2)
            # self.vd_thread = hub.spawn(self._vd_gen1)
            # self.vd_thread2 = hub.spawn(self._vd_gen2)

    def add_flow(self, datapath, priority, match, actions, buffer_id=None):
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        inst = [parser.OFPInstructionActions(ofproto.OFPIT_APPLY_ACTIONS,
                                             actions)]

        mod = parser.OFPFlowMod(datapath=datapath, priority=priority,
                                match=match, instructions=inst, buffer_id=buffer_id)
        datapath.send_msg(mod)

    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def _packet_in_handler(self, ev):
        msg = ev.msg
        datapath = msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        in_port = msg.match['in_port']
        #print("msg.data",msg.data)
        #packet=msg.data[ICMP]

        pkt = packet.Packet(msg.data)
        #print ("pkt",pkt)

        eth = pkt.get_protocols(ethernet.ethernet)[0]

        # if eth.ethertype == ether_types.ETH_TYPE_LLDP:
        #     # ignore lldp packet
        #     return
        dst = eth.dst
        src = eth.src
        # icmp_ = pkt.get_protocol(icmp.icmp)
        # if icmp_ != None:
        #     print("@icmp@",icmp_)

        dpid = datapath.id

        if dst in self.mac_to_port[dpid]:
            out_port = self.mac_to_port[dpid][dst]
            self.logger.info("%s packet in 스위치%s 출발%s 도착%s %s,buffer %s",time.time(), dpid, src, dst, in_port,msg.buffer_id)
        else:
            #out_port = ofproto.OFPP_FLOOD
            return

        actions = [parser.OFPActionOutput(out_port)]

        match = parser.OFPMatch()
        # install a flow to avoid packet_in next time
        if eth.ethertype == ether_types.ETH_TYPE_IP:
            ip = pkt.get_protocol(ipv4.ipv4)
            match = parser.OFPMatch(in_port= in_port, eth_type=ether_types.ETH_TYPE_IP, ipv4_src=ip.src,
                                    ipv4_dst=ip.dst)
            self.add_flow(datapath, 10000, match, actions)

        # if out_port != ofproto.OFPP_FLOOD:
        #     match = parser.OFPMatch(in_port=in_port, eth_dst=dst)
        #     self.add_flow(datapath, 1, match, actions)

        data = None
        if msg.buffer_id == ofproto.OFP_NO_BUFFER:
            data = msg.data

        #match = parser.OFPMatch(in_port=in_port)

        out = parser.OFPPacketOut(datapath=datapath, buffer_id=msg.buffer_id,
                                  match=match, actions=actions, data=data)

        datapath.send_msg(out)


    def _cc_gen1(self):
        hub.sleep(3)
        datapath = self.dp[1]
        pkt = packet.Packet()
        pkt.add_protocol(ethernet.ethernet(ethertype=ether_types.ETH_TYPE_IP,
                                           dst=self.H[5],
                                           src=self.H[1]))

        pkt.add_protocol(ipv4.ipv4(proto=in_proto.IPPROTO_ICMP,
                                   src=self.ip[1],
                                   dst=self.ip[5]))

        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        pkt.serialize()

        match = parser.OFPMatch(in_port = eth_type = ether_types.ETH_TYPE_IP, ipv4_src=self.ip[1],
                                ipv4_dst=self.ip[5])

        actions = [parser.OFPActionOutput(3)]

        self.add_flow(datapath, 10000, match, actions, ofproto.OFP_NO_BUFFER)
        #match = parser.OFPMatch(in_port=2)
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

            # df = pd.DataFrame([(datapath.id, 1, self.cc_cnt, time.time(), 'x')],
            #                   columns=['switch', 'class', 'number', 'time', 'queue'])
            # self.generated_log = self.generated_log.append(df)

            hub.sleep(self.cc_period / 1000)

            if (self.cc_cnt >= self.command_control):
                # self.terminal += 1
                break
    #
    # #traffic monitoring
    # def _request_stats(self, datapath):
    #     self.logger.debug('send stats request: %016x', datapath.id)
    #     ofproto = datapath.ofproto
    #     parser = datapath.ofproto_parser
    #
    #     req = parser.OFPFlowStatsRequest(datapath)
    #     datapath.send_msg(req)
    #
    #     req = parser.OFPPortStatsRequest(datapath, 0, ofproto.OFPP_ANY)
    #     datapath.send_msg(req)
    #
    # @set_ev_cls(ofp_event.EventOFPPortStatsReply, MAIN_DISPATCHER)
    # def _port_stats_reply_handler(self, ev):
    #     body = ev.msg.body
    #     self.logger.info('port stats : dp %s', ev.msg.datapath.id)
    #     self.logger.info('datapath         port     '
    #                      'rx-pkts  rx-bytes rx-error '
    #                      'tx-pkts  tx-bytes tx-error')
    #     self.logger.info('---------------- -------- '
    #                      '-------- -------- -------- '
    #                      '-------- -------- --------')
    #     for stat in sorted(body, key=attrgetter('port_no')):
    #         self.logger.info('%016x %8x %8d %8d %8d %8d %8d %8d',
    #                          ev.msg.datapath.id, stat.port_no,
    #                          stat.rx_packets, stat.rx_bytes, stat.rx_errors,
    #                          stat.tx_packets, stat.tx_bytes, stat.tx_errors)
    #
    # @set_ev_cls(ofp_event.EventOFPFlowStatsReply, MAIN_DISPATCHER)
    # def _flow_stats_reply_handler(self, ev):
    #     body = ev.msg.body
    #
    #     self.logger.info('port stats : dp %s', ev.msg.datapath.id)
    #
    #     self.logger.info('datapath         '
    #                      'in-port  eth-dst           '
    #                      'out-port packets  bytes')
    #     self.logger.info('---------------- '
    #                      '-------- ----------------- '
    #                      '-------- -------- --------')
    #     for stat in sorted([flow for flow in body if flow.priority == 1],
    #                        key=lambda flow: (flow.match['in_port'],
    #                                          flow.match['eth_dst'])):
    #         self.logger.info('%016x %8x %17s %8x %8d %8d',
    #                          ev.msg.datapath.id,
    #                          stat.match['in_port'], stat.match['eth_dst'],
    #                          stat.instructions[0].actions[0].port,
    #                          stat.packet_count, stat.byte_count)