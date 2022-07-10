from dataclasses import dataclass
import time
import os

DATE = '1223'
if not os.path.exists("./result/" + DATE):
    os.makedirs("./result/" + DATE)
PRIORITY_QUEUE = 2
STATE = 3
STATE_SIZE = STATE * PRIORITY_QUEUE
GCL_LENGTH = 3
ACTION_SIZE = 2 ** (GCL_LENGTH * PRIORITY_QUEUE)
MAX_EPISODE = 2000
COMMAND_CONTROL = 40
AUDIO = 8
VIDEO_FRAME = 30
VIDEO = 2 * VIDEO_FRAME
BEST_EFFORT = 100
CC_PERIOD = 5  # milliseconds #TODO: originally 5 (simulation duration을 위해 잠시 줄임)
AD_PERIOD = 1
VD_PERIOD = 1.1
BE_PERIOD = 1
TIMESLOT_SIZE = 0.6
NODES = 6  # switch의 수
UPDATE = 30000
W = [6, 0.05]

@dataclass
class Flow:  # type(class1:cc,2:ad,3:vd,4:be),Num,deadline,generate_time,depart_time,bits
    type_: int = None
    num_: int = None
    deadline_: float = None  # millisecond 단위, arrival time - generated time < deadline 이어야 함
    generated_time_: float = None  # millisecond 단위
    queueing_delay_: float = None  # node departure time - node arrival time
    node_arrival_time_: float = None
    # node_departure_time_: list = None
    arrival_time_: float = None
    bits_: int = None
    met_: bool = None
    hops_: int = None
    priority_: int = None

def Flow1():
    f = Flow()
    f.type_ = 1
    f.priority_ = 1
    f.num_ = fnum
    f.deadline_ = CC_PERIOD * 0.001
    f.generated_time_ = time - self.start_time
    f.queueing_delay_ = 0
    f.node_arrival_time_ = 0
    f.arrival_time_ = 0
    f.bits_ = 300 * 8  # originally random.randrange(53, 300)
    f.met_ = -1
    f.hops_ = 4  # it means how many packet-in occur



    def flow_generator(self, time, type_num, fnum):  # flow structure에 맞게 flow생성, timestamp등 남길 수 있음

        f = Flow()

        if type_num == 1:  # c&c
            f.type_ = 1
            f.priority_ = 1
            f.num_ = fnum
            f.deadline_ = CC_PERIOD * 0.001
            f.generated_time_ = time - self.start_time
            f.queueing_delay_ = 0
            f.node_arrival_time_ = 0
            f.arrival_time_ = 0
            f.bits_ = 300 * 8 #originally random.randrange(53, 300)
            f.met_ = -1
            f.hops_ = 4  # it means how many packet-in occur

        elif type_num == 2:  # audio
            f.type_ = 2
            f.priority_ = 1
            f.num_ = fnum
            f.deadline_ = 6 * 0.001  # originally random.choice([4, 10]) * 0.001
            f.generated_time_ = time - self.start_time
            f.queueing_delay_ = 0
            f.node_arrival_time_ = 0
            f.arrival_time_ = 0
            f.bits_ = 256 * 8 #originally random.choice([128, 256])
            f.met_ = -1
            f.hops_ = 4

        elif type_num == 3:  # video
            f.type_ = 3
            f.priority_ = 1
            f.num_ = fnum
            f.deadline_ = 0.030  # originally 30ms
            f.generated_time_ = time - self.start_time
            f.queueing_delay_ = 0
            f.node_arrival_time_ = 0
            f.arrival_time_ = 0
            f.bits_ = 1500 * 8
            f.met_ = -1
            f.hops_ = 4

        else:  # best effort
            f.type_ = 4
            f.priority_ = 2
            f.num_ = fnum
            f.deadline_ = 0.3
            f.generated_time_ = time - self.start_time
            f.queueing_delay_ = 0
            f.node_arrival_time_ = 0
            f.arrival_time_ = 0
            f.bits_ = 1024 * 8
            f.met_ = -1
            f.hops_ = 4

        return f