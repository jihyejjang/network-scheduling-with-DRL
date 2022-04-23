import os
from dataclasses import dataclass
import numpy as np
import random

# parameters
SINGLE_NODE = True #False
OUTPUT_PORT = 1 #2
SRCES = 8  # 8

#Reward
BOUND = [0.5, 0.6]
W0 = [0.1, 0.03]
W1 = [0.01, 0.01]
# W2 = [-0.6, -0.2]
W3 = -1
LM = 1.5

W = [0.5, 0.6] #? 왜 이게 더 잘되지
A = 0.01


# HOP_WEIGHT = 4
RANDOM_HOP = 4 # 0
RANDOM_CURRENT_DELAY_CC = 1  # originally 0 unit : T
RANDOM_CURRENT_DELAY_BE = [0, 3]  # originally [0,1] unit : T
PERIOD_CC = 2  # T
PERIOD_BE = 2
COMMAND_CONTROL = 40  # 40
BEST_EFFORT = 40  # 100
CC_DEADLINE = 10 # 5 (8 T), least 5T, unit : T (if not, just multiply TIMESLOT_SIZE)
BE_DEADLINE = 12 # 50 ( 75 T )
FIXED_SEQUENCE = False
FIRST_TRAIN = True
MAXSLOT_MODE = True
MAXSLOTS = 300 # 250
LEARNING_RATE = 0.005 #0.0001
UPDATE = 500 #500
EPSILON_DECAY = 0.9998 #0.9998

# Save
DATE = '0423'
FILENAME = 'result/0220/[15963]0.001464993692934513.h5'  # weight file name
WEIGHT_FILE = FILENAME

# RL agent
PRIORITY_QUEUE = 2
STATE = 2 # for simulation with different utilizations(periods), it has to be editted to 3
INPUT_SIZE = 4
# GCL_LENGTH = 3
OUTPUT_SIZE = 2
ALPHA = 0.1
INITIAL_ACTION = 0
ACTION_LIST = [0, 1]
ACTION_SIZE = len(ACTION_LIST)
BATCH = 64
EPSILON_MAX = 1
EPSILON_MIN = 0.01
DISCOUNT_FACTOR = 0.99

# Environment
MAX_EPISODE = 15000

# CC_PERIOD = 10
# AD_PERIOD = 6
# VD_PERIOD = 8
# BE_PERIOD = 4  # PERIOD는 Utilization을 위해 조절해야 할 듯

CC_BYTE = 1500
# AD_BYTE = 256
# VD_BYTE = 1500
BE_BYTE = 1500
TIMESLOT_SIZE = 0.6
BANDWIDTH = 20000  # bits per msec (20Mbps)
MAX_BURST = 12000
NODES = 9

# random parameters

# W = [10,10,1,0.1]


if not os.path.exists("./result/" + DATE):
    os.makedirs("./result/" + DATE)

f = open("./result/" + DATE + "/parameters.txt", 'w')
d = "DATE : {p} \nFIXED_SEQUENDE MODE : {f} \nMAX_SLOTS MODE : {ms} \nLEARNING_RATE: {s} \nMAX_EPISODE: {t} \n \
    EPSILON_DECAY: {e} \nWeight: {m} \nalpha: {l} \n".format(
    p=DATE,
    f=FIXED_SEQUENCE,
    ms=MAXSLOT_MODE,
    s=LEARNING_RATE,
    t=MAX_EPISODE,
    e=EPSILON_DECAY,
    m=W,
    l=A)

f.write(d)
# f.close()
import matplotlib.pyplot as plt


def save_result_plot(df):
    x = range(int(len(df['Episode'])))
    y = df['Score']
    plt.scatter(x, y, s=3)
    plt.savefig("./result/" + DATE + "/sum of reward.png", dpi=300)


# Functions
def max_burst():
    burst = BANDWIDTH * TIMESLOT_SIZE
    print("burst", burst)


def random_sequence():
    p1 = [[], []]
    p2 = [[], []]

    for i in range(COMMAND_CONTROL):
        hop = random.randint(0, RANDOM_HOP) + 1
        cd = hop + random.randint(0,RANDOM_CURRENT_DELAY_CC)
        p1[0].append(cd)
        p1[1].append(hop)

    for i in range(BEST_EFFORT):
        hop = random.randint(0, RANDOM_HOP) + 1
        cd = hop + random.randint(RANDOM_CURRENT_DELAY_BE[0], RANDOM_CURRENT_DELAY_BE[1])
        p2[0].append(cd)
        p2[1].append(hop)

    return p1, p2


# def utilization():
#     f1 = (CC_BYTE * 8) / CC_PERIOD  # bits per ms
#     f2 = (AD_BYTE * 8) / AD_PERIOD
#     f3 = (VD_BYTE * 8) / VD_PERIOD
#     f4 = (BE_BYTE * 8) / BE_PERIOD
#     utilization1 = (f1 + f2 + f3) / BANDWIDTH
#     print("utilization without BE ", round(utilization1, 2))
#     utilization2 = f4 / BANDWIDTH
#     print("utilization of BE ", round(utilization2, 2))


# def util_calculation(u1, u2):
#     cc_period = round(CC_PERIOD / u1, 1)
#     ad_period = round(AD_PERIOD / u1, 1)
#     vd_period = round(VD_PERIOD / u1, 1)
#     be_period = round(BE_PERIOD / u2, 1)
#     return cc_period, ad_period, vd_period, be_period


def action_to_number(action):
    # action_ = action.flatten()
    bin_ = ''
    for a in action:
        bin_ += str(a)
    return ACTION_LIST.index(int(bin_, 2))


def number_to_action(action_id):  # number -> binary gcl code
    b_id = format(ACTION_LIST[action_id], '02b')
    action_ = np.array(list(map(int, b_id)))
    return action_


route_ = [[1], [4, 5, 2], [7, 8, 9, 6, 3],
          [7, 4, 1, 2, 3], [8, 5, 6], [9]]

route = [[1, 2, 5, 6, 9, 0], [3, 2, 5, 4, 7, 0], [4, 1, 2, 0], [4, 7, 8, 0],
         [6, 3, 2, 0], [6, 9, 8, 0], [7, 8, 5, 6, 3, 0], [9, 8, 5, 4, 1, 0]]


# 첫 번째, 마지막 루트만 priority 1

# packet structure

@dataclass
class Flow:
    src_: int = None
    dst_: int = None
    route_: list = None
    type_: int = None
    priority_: int = None
    num_: int = None
    deadline_: float = None
    generated_time_: float = None
    queueing_delay_: list = None
    current_delay_: float = None
    bits_: int = None
    met_: bool = None
    remain_hops_: int = None
    random_delay_: float = None
