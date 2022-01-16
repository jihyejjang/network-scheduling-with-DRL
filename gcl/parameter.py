import os
from dataclasses import dataclass
import numpy as np
import random

# RL agent
PRIORITY_QUEUE = 2
# STATE = 4
INPUT_SIZE = 4
GCL_LENGTH = 3
OUTPUT_SIZE = 8
LEARNING_RATE = 0.00001
ALPHA = 0.1
PRIORITY_QUEUE = 2
INITIAL_ACTION = 0
ACTION_LIST = [56, 7, 63, 49, 35, 57, 59, 51]
ACTION_SIZE = len(ACTION_LIST)
UPDATE = 5000
BATCH = 128
EPSILON_MAX = 1
EPSILON_DECAY = 0.99995
EPSILON_MIN = 0.01
DISCOUNT_FACTOR = 0.99

# Environment
MAX_EPISODE = 20000
COMMAND_CONTROL = 40
AUDIO = 8
VIDEO_FRAME = 30
VIDEO = 2 * VIDEO_FRAME
BEST_EFFORT = 100
CC_PERIOD = 10
AD_PERIOD = 6
VD_PERIOD = 8
BE_PERIOD = 4  # PERIOD는 Utilization을 위해 조절해야 할 듯
CC_DEADLINE = 7
AD_DEADLINE = 8
VD_DEADLINE = 30
BE_DEADLINE = 500
CC_BYTE = 300
AD_BYTE = 256
VD_BYTE = 1500
BE_BYTE = 1024
TIMESLOT_SIZE = 0.6
BANDWIDTH = 20000  # bits per msec (20Mbps)
MAX_BURST = 12000
NODES = 6
W = [10,10,1,0.1]

# Save
DATE = '0116'
#FILENAME = '[1999]0.011379198171198368.h5'  # weight file name
#WEIGHT_FILE = './result/' + DATE + '/' + FILENAME
if not os.path.exists("./result/" + DATE):
    os.makedirs("./result/" + DATE)


# Functions
def max_burst():
    burst = BANDWIDTH * TIMESLOT_SIZE
    print("burst", burst)


def utilization():
    f1 = (CC_BYTE * 8) / CC_PERIOD  # bits per ms
    f2 = (AD_BYTE * 8) / AD_PERIOD
    f3 = (VD_BYTE * 8) / VD_PERIOD
    f4 = (BE_BYTE * 8) / BE_PERIOD
    utilization1 = (f1 + f2 + f3) / BANDWIDTH
    print("utilization without BE ", round(utilization1, 2))
    utilization2 = f4 / BANDWIDTH
    print("utilization of BE ", round(utilization2, 2))


# def initial_state(u1, u2):
#     hops = [3, 3, 2, 1, 0, 0]


def util_calculation(u1, u2):
    cc_period = round(CC_PERIOD / u1, 1)
    ad_period = round(AD_PERIOD / u1, 1)
    vd_period = round(VD_PERIOD / u1, 1)
    be_period = round(BE_PERIOD / u2, 1)
    return cc_period, ad_period, vd_period, be_period


def action_to_number(action):
    action_ = action.flatten()
    bin_ = ''
    for a in action_:
        bin_ += str(a)
    return ACTION_LIST.index(int(bin_, 2))


def number_to_action(action_id):  # number -> binary gcl code
    b_id = format(ACTION_LIST[action_id], '06b')
    action_ = np.array(list(map(int, b_id)))
    return action_.reshape((PRIORITY_QUEUE, GCL_LENGTH))


# Data Structure

@dataclass
class Flow:  # type(class1:cc,2:ad,3:vd,4:be),Num,deadline,generate_time,depart_time,bits
    type_: int = None
    num_: int = None
    deadline_: float = None  # millisecond 단위, arrival time - generated time < deadline 이어야 함
    generated_time_: float = None  # millisecond 단위
    queueing_delay_: list = None  # node departure time - node arrival time
    node_arrival_time_: float = None
    # node_departure_time_: list = None
    arrival_time_: float = None
    bits_: int = None
    met_: bool = None
    hops_: int = None
    priority_: int = None
