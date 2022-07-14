# ----------------------------------------
# Autor : Jihye Ryu, jh_r_1004@naver.com
# ----------------------------------------

import argparse
from env import TrainSchedulingSimulation
from test import TestSchedulingSimulation
  
if __name__ == "__main__":
    # date, env, mode, algorithm, wc, save
    parser = argparse.ArgumentParser(description='The environment of timeslot scheduling with DDQN')
    # parser.add_argument("date")
    parser.add_argument("-m","--mode", help="Train or test mode, True for train", required=True, type=str, choices=list(['train','test']))
    parser.add_argument("-e", "--env", help="Single or topology env, True for single",required=True, type=bool, default=False) 
    parser.add_argument("-f","--withflows",help="with 3,4,5,6 flows",type=bool,default=False)
    parser.add_argument("-al","--algo", help="sp,rr,ddqn", type=list, default=['ddqn'])
    parser.add_argument("-w", "--workconserving", help="Work-conserving mode", type=bool, default=True)
    parser.add_argument("-s","--save", help="save model or results", type=bool, default=False)
    
    #save folder 정보 추가
    args = parser.parse_args()
    if args.mode == 'train':
        environment = TrainSchedulingSimulation(args.env, args.workconserving, args.save)
        environment.run()
    elif args.mode == 'test':
        environment = TestSchedulingSimulation(args.env, args.workconserving, args.save, args.algo)
