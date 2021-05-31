from threading import Timer
import time
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

def cycle():
    global cnt
    if cnt==0:
        print("state", state)

    cnt+=1
    print (cnt,"초")

    t=Timer(1,cycle)#timeslot
    t.start()

    if cnt>=10:
        t.cancel()
        timeslot_start()

def timeslot_start():
    global cnt
    global state
    state=[1,1,2,3]
    cnt=0
    cycle() #cycle재시작

cycle()