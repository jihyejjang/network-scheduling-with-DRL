#!/usr/bin/env python
# coding: utf-8

from mininet.topo import Topo
from mininet.net import Mininet
from mininet.util import irange, dumpNodeConnections
from mininet.log import setLogLevel
from mininet.cli import CLI
from mininet.node import RemoteController, OVSSwitch

class MyTopo(Topo):

    def __init__(self):
        "Create custom loop topo."

        # Initialize topology
        Topo.__init__(self)
        
        # Add hosts and switches
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
        self.addLink(switch1, host1)
        self.addLink(switch1, host2)
        self.addLink(switch1, switch3)
        self.addLink(switch2, host3)
        self.addLink(switch2, host4)
        self.addLink(switch2, switch3)
        self.addLink(switch3, switch4)
        self.addLink(switch4, switch5)
        self.addLink(switch5, host5)
        self.addLink(switch5, host6)
        self.addLink(switch4, switch6)
        self.addLink(switch6, host7)
        self.addLink(switch6, host8)
        
        # Add controller
	#cont1 = self.addController('c1', controller=Controller, port=6624)
	
def runMyTopo(): #activate mininet topology after ping test
     	
     topo = MyTopo()
     net = Mininet (topo=topo, controller = RemoteController , switch=OVSSwitch, autoSetMacs=True)
     net.start()

     dumpNodeConnections(net.hosts)
     dumpNodeConnections(net.switches)
     net.pingAll()
     print("ping test completed")	
     CLI(net)
        
if __name__ == '__main__':
    setLogLevel('info')
    runMyTopo()
   
topos ={'topo':MyTopo}
