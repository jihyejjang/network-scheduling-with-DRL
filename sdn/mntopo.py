#!/usr/bin/env python
# coding: utf-8

from mininet.topo import Topo
from mininet.net import Mininet
from mininet.log import setLogLevel
from mininet.cli import CLI
from mininet.node import RemoteController, OVSSwitch
from mininet.link import TCLink



class mnTopo(Topo):

    def __init__(self):
        "Create custom loop topo."

        # Initialize topology
        Topo.__init__(self)
        
        # Add hosts and switches
        self.host1 = self.addHost('h1')
        self.host2 = self.addHost('h2')
        self.host3 = self.addHost('h3')
        self.host4 = self.addHost('h4')
        self.host5 = self.addHost('h5')
        self.host6 = self.addHost('h6')
        self.host7 = self.addHost('h7')
        self.host8 = self.addHost('h8')
        
        self.switch1 = self.addSwitch("s1")
        self.switch2 = self.addSwitch("s2")
        self.switch3 = self.addSwitch("s3")
        self.switch4 = self.addSwitch("s4")
        self.switch5 = self.addSwitch("s5")
        self.switch6 = self.addSwitch("s6")

        # Add links
        self.addLink(self.switch1, self.host1, cls=TCLink, bw = 10)
        self.addLink(self.switch1, self.host2, cls=TCLink, bw = 10)
        self.addLink(self.switch1, self.switch3, cls=TCLink, bw = 10)
        self.addLink(self.switch2, self.host3, cls=TCLink, bw = 10)
        self.addLink(self.switch2, self.host4, cls=TCLink, bw = 10)
        self.addLink(self.switch2, self.switch3, cls=TCLink, bw = 10)
        self.addLink(self.switch3, self.switch4, cls=TCLink, bw = 10)
        self.addLink(self.switch4, self.switch5, cls=TCLink, bw = 10)
        self.addLink(self.switch5, self.host5, cls=TCLink, bw = 10)
        self.addLink(self.switch5, self.host6, cls=TCLink, bw = 10)
        self.addLink(self.switch4, self.switch6, cls=TCLink, bw = 10)
        self.addLink(self.switch6, self.host7, cls=TCLink, bw = 10)
        self.addLink(self.switch6, self.host8, cls=TCLink, bw = 10)
        
        # Add controller
	#cont1 = self.addController('c1', controller=Controller, port=6624)
	
def runMyTopo(): #activate mininet topology after ping test
     	
     topo = MyTopo()
     #net = Mininet (topo=topo, controller = RemoteController , switch=OVSSwitch, autoSetMacs=True)
     net = Mininet (topo=topo, switch=OVSSwitch, autoSetMacs=True)
     net.start()
     net.pingAll()

     net.iperf()
     #net.iperf((net.host1, topo.host5), l4Type = 'UDP')

     net.pingPairFull()

     #net.stop()
     #dumpNodeConnections(net.hosts)
     #dumpNodeConnections(net.switches) #연결 정보 확인
     #net.pingAll()
     #print("ping test completed")
     CLI(net)
        
if __name__ == '__main__':
    setLogLevel('info')
    runMyTopo()
   
topos ={'topo':MyTopo}
