# ----------------------------------------
# Autor : Jihye Ryu, jh_r_1004@naver.com
# ----------------------------------------

import argparse
from env import TrainSchedulingSimulation


# from test import TestSchedulingSimulation

def main(args_):
    # test source code

    if args_.mode == 'train':
        environment = TrainSchedulingSimulation(args_)
        environment.run()
    # elif args.mode == 'test':
    #     environment = TestSchedulingSimulation(args)
    #     environment.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='The environment of timeslot scheduling with DDQN')

    # Environment setting
    parser.add_argument("-m", "--mode", help="Train or test mode, True for train", type=str,default='train',
                        choices=list(['train', 'test']))
    parser.add_argument("-e", "--env", help="Single or topology env, True for single",default=True, type=bool)
    # parser.add_argument("-m", "--mode", help="Train or test mode, True for train", required=True, type=str,
    #                     choices=list(['train', 'test']))
    # parser.add_argument("-e", "--env", help="Single or topology env, True for single", required=True, type=bool)
    parser.add_argument("-f", "--withflows", help="with 3,4,5,6 flows", type=bool, default=False)
    parser.add_argument("-w", "--workconserving", help="Work-conserving mode", type=bool, default=True)
    parser.add_argument("-al", "--algo", help="sp,rr,ddqn", type=list, default=['sp', 'rr', 'ddqn'])

    # Save setting
    parser.add_argument("-s", "--save", help="save model or results", type=bool, default=False)
    parser.add_argument("-d", "--date", help="save folder name", type=str, default=None)
    parser.add_argument("-wf", "--weightfile", help='weight file', type=str, default=None)

    # Parameter setting
    parser.add_argument("-r", "--reward", default=[0.6, 0.1])  # W
    parser.add_argument("-r2", "--reward2", default=0.1)  # A
    parser.add_argument("-sd", "--seed", default=7)  # SEED
    parser.add_argument("-lr", "--learningrate", default=0.001)
    parser.add_argument("-wrr", "--wrrweight", default=3)
    parser.add_argument("-ep", "--totalepisode", default=20000)

    # Flow setting
    parser.add_argument("-pr", "--period", default=[2, 2])
    parser.add_argument("-np", "--numberofpackets", default=[40, 40])
    parser.add_argument("-dl", "--deadline", default=[8, 8])

    args = parser.parse_args()
    # Flow setting by env
    if args.env:  # single node
        parser.add_argument("-rh", "--randomhops", default=4)
        parser.add_argument("-rcd", "--randomcurrentdelay", default=[7, [0, 1]])
    else:
        parser.add_argument("-rh", "--randomhops", default=0)
        parser.add_argument("-rcd", "--randomcurrentdelay", default=[1, [0, 2]])

    max_reward = args.numberofpackets[0] * args.reward[0] + \
                 args.numberofpackets[1] * args.reward[1] + args.reward2 * sum(args.numberofpackets)
    print(f'available maximum reward is {max_reward}')

    args = parser.parse_args()
    main(args)

# single node train
# python run.py -m "train" -e True -s False
