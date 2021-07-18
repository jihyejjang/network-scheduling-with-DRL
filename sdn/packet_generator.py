from RLswitch import rl_switch
import time
from datetime import datetime
# from threading import Timer
# from ryu.controller import ofp_event
# from ryu.controller.handler import MAIN_DISPATCHER, DEAD_DISPATCHER
# from ryu.controller.handler import set_ev_cls
from ryu.lib import hub
#
# from ryu.base import app_manager
# from ryu.controller import ofp_event
# from ryu.controller.handler import CONFIG_DISPATCHER, MAIN_DISPATCHER
# from ryu.controller.handler import set_ev_cls

from ryu.lib.packet import packet, ether_types
from ryu.lib.packet import ethernet

from ryu.ofproto import ofproto_v1_5

class FlowGenerator(rl_switch):
    OFP_VERSIONS = [ofproto_v1_5.OFP_VERSION]

    def __init__(self, *args, **kwargs):
        super(FlowGenerator, self).__init__(*args, **kwargs)
        self.cc_thread = hub.spawn(self._cc_gen1)

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

        match = parser.OFPMatch(in_port=1, eth_dst=self.H[5])
        actions = [parser.OFPActionOutput(3)]
        self.add_flow(datapath, 1, match, actions, ofproto.OFP_NO_BUFFER)
        match = parser.OFPMatch(in_port=1)
        data = pkt.data

        out = parser.OFPPacketOut(datapath=datapath,
                                  buffer_id=ofproto.OFP_NO_BUFFER,
                                  match=match,
                                  actions=actions, data=data)

        for i in range(len(self.cc_cnt)):
            self.cc_cnt += 1
            datapath.send_msg(out)
            hub.sleep(self.cc_period)
            self.logger.info("%s.%0.1f : C&C1 generated %s, 스위치%s " % \
                                     ((datetime.now() - self.start_time).seconds,
                              (datetime.now() - self.start_time).microseconds / 1000, self.cc_cnt, datapath.id))
        self.terminal == True
        print("전송 끝!!!!@@@@@")

    # def cc_generator1(self):
    #     datapath = self.dp[1]
    #     self.cc_cnt += 1
    #
    #     pkt = packet.Packet()
    #     pkt.add_protocol(ethernet.ethernet(ethertype=ether_types.ETH_TYPE_IEEE802_3,
    #                                        dst=self.H[5],
    #                                        src=self.H[1]))
    #
    #     ofproto = datapath.ofproto
    #     parser = datapath.ofproto_parser
    #     pkt.serialize()
    #
    #     match = parser.OFPMatch(in_port=1, eth_dst=self.H[5])
    #     actions = [parser.OFPActionOutput(3)]
    #     self.add_flow(datapath, 1, match, actions, ofproto.OFP_NO_BUFFER)
    #     match = parser.OFPMatch(in_port=1)
    #     data = pkt.data
    #     out = parser.OFPPacketOut(datapath=datapath,
    #                               buffer_id=ofproto.OFP_NO_BUFFER,
    #                               match=match,
    #                               actions=actions, data=data)
    #
    #     datapath.send_msg(out)
    #     self.logger.info("%s.%0.1f : C&C1 generated %s, 스위치%s " % \
    #                      ((datetime.now() - self.start_time).seconds,
    #                       (datetime.now() - self.start_time).microseconds / 1000, self.cc_cnt, datapath.id))
    #
    #     t = Timer((self.cc_period / 1000), self.cc_generator1)
    #     t.start()
    #
    #     if self.cc_cnt >= self.command_control:
    #         t.cancel()
    #         print("전송 끝!")
    #         time.sleep(1)
    #         self.terminal = True