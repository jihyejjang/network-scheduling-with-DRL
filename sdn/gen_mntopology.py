#!/usr/bin/env python
# coding: utf-8

from mininet.topo import Topo
from mininet.net import Mininet
from mininet.util import irange, dumpNodeConnections
from mininet.log import setLogLevel
from mininet.cli import CLI
from mininet.node import RemoteController, OVSSwitch, Controller
from mininet.link import TCLink
from scapy.all import sendp, send, IP, Ether, ICMP
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


def mnNetwork():
    net = Mininet (topo=None, controller = RemoteController , switch=OVSSwitch, autoSetMacs=True)
    #net = Mininet(topo=None,build=False, switch=OVSSwitch, autoSetMacs=True)
    c1 = net.addController('c1', controller = RemoteController)
    c2 = net.addController('c2', controller=RemoteController)
    host1 = net.addHost('h1')
    host2 = net.addHost('h2')
    host3 = net.addHost('h3')
    host4 = net.addHost('h4')
    host5 = net.addHost('h5')
    host6 = net.addHost('h6')
    host7 = net.addHost('h7')
    host8 = net.addHost('h8')

    switch1 = net.addSwitch("s1")
    switch2 = net.addSwitch("s2")
    switch3 = net.addSwitch("s3")
    switch4 = net.addSwitch("s4")
    switch5 = net.addSwitch("s5")
    switch6 = net.addSwitch("s6")

    net.addLink(switch1, host1, cls=TCLink, bw=1000)
    net.addLink(switch1, host2, cls=TCLink, bw=1000)
    net.addLink(switch1, switch3, cls=TCLink, bw=1000)
    net.addLink(switch2, host3, cls=TCLink, bw=1000)
    net.addLink(switch2, host4, cls=TCLink, bw=1000)
    net.addLink(switch2, switch3, cls=TCLink, bw=1000)
    net.addLink(switch3, switch4, cls=TCLink, bw=1000)
    net.addLink(switch4, switch5, cls=TCLink, bw=1000)
    net.addLink(switch5, host5, cls=TCLink, bw=10)
    net.addLink(switch5, host6, cls=TCLink, bw=1000)
    net.addLink(switch4, switch6, cls=TCLink, bw=1000)
    net.addLink(switch6, host7, cls=TCLink, bw=1000)
    net.addLink(switch6, host8, cls=TCLink, bw=1000)
    net.build()
    c1.start()
    c2.start()
    net.start()

    #gen_("00:00:00:00:00:01","00:00:00:00:00:06")

    CLI(net)

if __name__=='__main__':
    setLogLevel('info')
    mnNetwork()
    # with concurrent.futures.ProcessPoolExecutor() as executor:
    #      cc1_process = [executor.submit(gen_,"00:00:00:00:00:01","00:00:00:00:00:06","10.0.0.1","10.0.0.6")]


