#!/usr/bin/env python
# coding: utf-8

from mininet.topo import Topo
from mininet.net import Mininet
from mininet.util import irange, dumpNodeConnections
from mininet.log import setLogLevel
from mininet.cli import CLI
from mininet.node import RemoteController, OVSSwitch
from mininet.link import TCLink
from scapy.all import sendp, send,  Ether, ICMP
import time
import concurrent.futures


#수신측에서 packet[icmp].time -> timestamp, packet[icmp].payload 로 데이터 확인 가능
# def gen_(src_,dst_):
#     n=1
#     cnt=1
#     period=0.005
#     #"00:00:00:00:00:01","00:00:00:00:00:06","10.0.0.1","10.0.0.6"
#     for i in range(40):
#         packet = Ether(src=src_,dst=dst_)/ICMP()/str("class"+str(n)+";"+str(cnt)+";")
#         send(packet)
#         print("packet class1 전송",cnt)
#         cnt+=1
#         time.sleep(period)
class MyTopo(Topo):

    def __init__(self):
        "Create custom loop topo."

        # Initialize topology
        Topo.__init__(self)
        
        # Add hosts and switches
        c1 = self.addController('c1', controller = RemoteController)
        host1 = self.addHost('h1')
        host2 = self.addHost('h2')
        host3 = self.addHost('h3')
        host4 = self.addHost('h4')
        host5 = self.addHost('h5')
        host6 = self.addHost('h6')
        host7 = self.addHost('h7')
        host8 = self.addHost('h8')
        
        switch1 = self.addSwitch("s1")
        switch2 = self.addSwitch("s2")
        switch3 = self.addSwitch("s3")
        switch4 = self.addSwitch("s4")
        switch5 = self.addSwitch("s5")
        switch6 = self.addSwitch("s6")

        # Add links
        self.addLink(switch1, host1, cls=TCLink, bw = 1000)
        self.addLink(switch1, host2, cls=TCLink, bw = 1000)
        self.addLink(switch1, switch3, cls=TCLink, bw = 1000)
        self.addLink(switch2, host3, cls=TCLink, bw = 1000)
        self.addLink(switch2, host4, cls=TCLink, bw = 1000)
        self.addLink(switch2, switch3, cls=TCLink, bw = 1000)
        self.addLink(switch3, switch4, cls=TCLink, bw = 1000)
        self.addLink(switch4, switch5, cls=TCLink, bw = 1000)
        self.addLink(switch5, host5, cls=TCLink, bw = 1000)
        self.addLink(switch5, host6, cls=TCLink, bw = 1000)
        self.addLink(switch4, switch6, cls=TCLink, bw = 1000)
        self.addLink(switch6, host7, cls=TCLink, bw = 1000)
        self.addLink(switch6, host8, cls=TCLink, bw = 1000)

        self.addLink(c1, switch1, cls=TCLink, bw=1000)
        self.addLink(switch2, c1, cls=TCLink, bw=1000)
        self.addLink(switch3, c1, cls=TCLink, bw=1000)
        self.addLink(switch4, c1, cls=TCLink, bw=1000)
        self.addLink(switch5, c1, cls=TCLink, bw=1000)
        self.addLink(switch6, c1, cls=TCLink, bw=1000)


        
        # Add controller
	#cont1 = self.addController('c1', controller=Controller, port=6624)
	
def runMyTopo(): #activate mininet topology after ping test
     	
     topo = MyTopo()

     net = Mininet (topo=topo, controller=RemoteController, switch=OVSSwitch, autoSetMacs=True)
     net.start()
     time.sleep(1)
     #net.pingAll()
     # for i in range(40):
     #    packet = Ether(src="00:00:00:00:00:01", dst="00:00:00:00:00:05") / ICMP() / str("class" + str(1) + ";" + str(1) + ";")
     #    print (packet[Ether].time,packet[ICMP].payload)
     #    sendp(packet)
     #    time.sleep(1)

     CLI(net)
        
if __name__ == '__main__':
    setLogLevel('info')
    runMyTopo()
   
topos ={'topo':MyTopo}
