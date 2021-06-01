from collections import deque
from threading import Timer
import time
import numpy as np

cnt=0
a=None
state=[1,2,1,2]

# def packet_handler():
#     global a
#     print("패킷 도착",time.time())
#
#
#     print("랜덤 초 후에 전송")
#
#     print("패킷 전송")

# queue_dp = np.zeros((6,3,4))
#
# queue_dp[0][0][0] += 1 #1번 스위치 1번 포트의 1번큐
# queue_dp[0][1][0] += 1
# queue_dp[0][2][3] += 1
#
# state=np.zeros((6,4))
#
# for i in range(len(state[0])):
#     state[0][i] = sum(queue_dp[0, :, i])
#
# print ("스위치1 큐",queue_dp[0])
# print ("스위치1 state",state)
# state = np.zeros((6, 4))
# print(len(state))
#
# def cycle():
#     global cnt
#     if cnt==0:
#         print("state", state)
#
#     cnt+=1
#     print (cnt,"초")
#
#     t=Timer(1,cycle)#timeslot
#     t.start()
#
#     if cnt>=10:
#         t.cancel()
#         timeslot_start()
#
# def timeslot_start():
#     global cnt
#     global state
#     state=[1,1,2,3]
#     cnt=0
#     cycle() #cycle재시작
#
# cycle()

cnt = 3

gcl='00001111'

print(gcl[cnt-1:].find('1'))
#print(gcl.index('1'))
#gcl[cnt]

time.sleep(gcl[cnt-1:].find('1')/1000)

print("ㅇㅇ")