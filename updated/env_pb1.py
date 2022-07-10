#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
from node import Node, Packet, Source, Destination, FIFO, RoundRobin
import simpy
import random
import matplotlib.pyplot as plt
from ddqn_agent import Agent

import time
import os
import warnings

warnings.filterwarnings('ignore')

EPISODES = 3000
STATE_SIZE = 7
ACTION_SIZE = 2
LEARNING_RATE = 0.002
LEARNING_RATE_DECAY = 0.5
GAMMA = 0.99 #discount factor
EPSILON = 1.0
EPSILON_MIN = 0.001
EPSILON_DECAY = 0.997
UPDATE_CYCLE = 500
MEMORIZE_PERIOD = 500


class PortBased1(object):
	
  def __init__(self, env, repeat = 0):
    self.env = env
    self.agent = [DQNAgent(STATE_SIZE, ACTION_SIZE, GAMMA, EPSILON, 
                  EPSILON_MIN, EPSILON_DECAY, LEARNING_RATE),
                  DQNAgent(STATE_SIZE, ACTION_SIZE, GAMMA, EPSILON, 
                  EPSILON_MIN, EPSILON_DECAY, LEARNING_RATE),
                  DQNAgent(STATE_SIZE, ACTION_SIZE, GAMMA, EPSILON, 
                  EPSILON_MIN, EPSILON_DECAY, LEARNING_RATE),
                  DQNAgent(STATE_SIZE, ACTION_SIZE, GAMMA, EPSILON, 
                  EPSILON_MIN, EPSILON_DECAY, LEARNING_RATE)]
    self.batch_size = 64
    self.num_node = 4
    self.num_src = 5

    self.state_and_action = []
    self.reward_list = []
    self.loss_list = []
    self.step_list = []
    self.delay_list = []
    self.delay_max = []
    self.delay_mean = []
    self.step_num = 0
    self.fname = "port_based_1"
    if not os.path.exists("./result/" + self.fname):
      os.makedirs("./result/" + self.fname)
    if not os.path.exists("./result/" + self.fname + "/csv"):
      os.makedirs("./result/" + self.fname + "/csv")
    if not os.path.exists("./result/" + self.fname + "/model"):
      os.makedirs("./result/" + self.fname + "/model")
    self.repeat = str(repeat)
    
    self.links = {"src0_to_node0": simpy.Store(self.env),
                  "src1_to_node0": simpy.Store(self.env),
                  "src2_to_node1": simpy.Store(self.env),
                  "src3_to_node2": simpy.Store(self.env),
                  "src4_to_node3": simpy.Store(self.env),
                  "node0_to_node1": simpy.Store(self.env),
                  "node1_to_node2": simpy.Store(self.env),
                  "node2_to_node3": simpy.Store(self.env),
                  "node0_to_dst1": simpy.Store(self.env),
                  "node1_to_dst2": simpy.Store(self.env),
                  "node2_to_dst3": simpy.Store(self.env),
                  "node3_to_dst4": simpy.Store(self.env),
                  "node3_to_dst0": simpy.Store(self.env)}
    self.srcs = {0: Source(0, self.env, self.links["src0_to_node0"], 1, 5, 2, 0),
                 1: Source(1, self.env, self.links["src1_to_node0"], 1, 2, 2, 0),
                 2: Source(2, self.env, self.links["src2_to_node1"], 1, 2, 2, 1),
                 3: Source(3, self.env, self.links["src3_to_node2"], 1, 2, 2, 2),
                 4: Source(4, self.env, self.links["src4_to_node3"], 1, 2, 2, 3)}
    self.dsts = {0: Destination(0, self.env, self.links["node3_to_dst0"]),
                 1: Destination(1, self.env, self.links["node0_to_dst1"]),
                 2: Destination(2, self.env, self.links["node1_to_dst2"]),
                 3: Destination(3, self.env, self.links["node2_to_dst3"]),
                 4: Destination(4, self.env, self.links["node3_to_dst4"])}
    self.nodes = {0: FIFO(0, self.env, 2,
                     [self.links["src0_to_node0"], self.links["src1_to_node0"]],
                     [self.links["node0_to_node1"], self.links["node0_to_dst1"]]),
                  1: FIFO(1, self.env, 2,
                     [self.links["node0_to_node1"], self.links["src2_to_node1"]],
                     [self.links["node1_to_node2"], self.links["node1_to_dst2"]]),
                  2: FIFO(2, self.env, 2,
                     [self.links["node1_to_node2"], self.links["src3_to_node2"]],
                     [self.links["node2_to_node3"], self.links["node2_to_dst3"]]),
                  3: FIFO(3, self.env, 2,
                     [self.links["node2_to_node3"], self.links["src4_to_node3"]],
                     [self.links["node3_to_dst0"], self.links["node3_to_dst4"]])}
    self.simulation = env.process(self.episode())

  def reset(self):
    self.links = {"src0_to_node0": simpy.Store(self.env),
                  "src1_to_node0": simpy.Store(self.env),
                  "src2_to_node1": simpy.Store(self.env),
                  "src3_to_node2": simpy.Store(self.env),
                  "src4_to_node3": simpy.Store(self.env),
                  "node0_to_node1": simpy.Store(self.env),
                  "node1_to_node2": simpy.Store(self.env),
                  "node2_to_node3": simpy.Store(self.env),
                  "node0_to_dst1": simpy.Store(self.env),
                  "node1_to_dst2": simpy.Store(self.env),
                  "node2_to_dst3": simpy.Store(self.env),
                  "node3_to_dst4": simpy.Store(self.env),
                  "node3_to_dst0": simpy.Store(self.env)}
    self.srcs = {0: Source(0, self.env, self.links["src0_to_node0"], 1, 5, 2, 0),
                 1: Source(1, self.env, self.links["src1_to_node0"], 1, 2, 2, 0),
                 2: Source(2, self.env, self.links["src2_to_node1"], 1, 2, 2, 1),
                 3: Source(3, self.env, self.links["src3_to_node2"], 1, 2, 2, 2),
                 4: Source(4, self.env, self.links["src4_to_node3"], 1, 2, 2, 3)}
    self.dsts = {0: Destination(0, self.env, self.links["node3_to_dst0"]),
                 1: Destination(1, self.env, self.links["node0_to_dst1"]),
                 2: Destination(2, self.env, self.links["node1_to_dst2"]),
                 3: Destination(3, self.env, self.links["node2_to_dst3"]),
                 4: Destination(4, self.env, self.links["node3_to_dst4"])}
    self.nodes = {0: FIFO(0, self.env, 2,
                     [self.links["src0_to_node0"], self.links["src1_to_node0"]],
                     [self.links["node0_to_node1"], self.links["node0_to_dst1"]]),
                  1: FIFO(1, self.env, 2,
                     [self.links["node0_to_node1"], self.links["src2_to_node1"]],
                     [self.links["node1_to_node2"], self.links["node1_to_dst2"]]),
                  2: FIFO(2, self.env, 2,
                     [self.links["node1_to_node2"], self.links["src3_to_node2"]],
                     [self.links["node2_to_node3"], self.links["node2_to_dst3"]]),
                  3: FIFO(3, self.env, 2,
                     [self.links["node2_to_node3"], self.links["src4_to_node3"]],
                     [self.links["node3_to_dst0"], self.links["node3_to_dst4"]])}

  def save_plot(self):
    plt.plot(self.reward_list)
    plt.xlim(0, EPISODES)
    plt.savefig('./result/' + self.fname + '/reward_' + self.repeat +'.png')
    plt.clf()
    plt.plot(self.step_list)
    plt.xlim(0, EPISODES)
    plt.savefig('./result/' + self.fname + '/step_' + self.repeat +'.png')
    plt.clf()
    plt.plot(self.loss_list)
    plt.xlim(0, EPISODES)
    plt.savefig('./result/' + self.fname + '/loss_' + self.repeat +'.png')
    plt.clf()
    average_reward = np.convolve(self.reward_list, np.ones(20), 'valid') / 20
    plt.plot(average_reward)
    plt.xlim(0, EPISODES)
    plt.savefig('./result/' + self.fname + '/average_' + self.repeat +'.png')
    plt.clf()
    plt.plot(self.delay_max)
    plt.plot(self.delay_mean)
    plt.xlim(0, EPISODES)
    plt.savefig('./result/' + self.fname + '/delay_' + self.repeat +'.png')
    plt.clf()
			
  def episode(self):
    for episode in range(EPISODES):
      self.step_num = 0
      reward = 0
      min_loss = 0
      done = False
      self.delay_list = np.zeros(self.num_src)
      states = np.zeros((self.num_node, STATE_SIZE))
      actions = np.zeros((self.num_node), dtype = np.int32)

      for i in range(self.num_src):
        self.env.process(self.srcs[i].send())
      for i in range(self.num_node):
        yield self.env.process(self.nodes[i].receive())
      for i in range(self.num_node):
        states[i][0:6] = self.nodes[i].get_state()
        states[i][-1] = self.step_num
      while not done:
        self.step_num += 1

        #for i in range(self.num_node):
        #  actions[i] = self.agent[i].act(np.reshape(states[i], [1, STATE_SIZE]))

        next_states, rewards, dones = yield self.env.process(self.step(actions))
        reward += sum(rewards)
        #for i in range(self.num_node):
        #  if any(states[i][:6]):
        #    reward += rewards[i]

        if self.step_num==100:
          for i in range(self.num_node):
            dones[i] = True

        '''for i in range(self.num_node):
          if not dones[i] or any(states[i][:6]):
            self.agent[i].memorize(np.reshape(states[i], [1, STATE_SIZE]), actions[i], rewards[i], 
                                np.reshape(next_states[i], [1, STATE_SIZE]), dones[i])
          if (episode % 100 == 0) or (episode == EPISODES - 1):
              self.state_and_action.append([episode, states[i], actions[i]])'''
        
        if all(dones):
          done = True

        '''if episode >= MEMORIZE_PERIOD:
          for i in range(self.num_node):
            min_loss += self.agent[i].replay(self.batch_size)
            if done:
              if episode % UPDATE_CYCLE == 0:
                self.agent[i].update_target_model()'''
        states = next_states
        if self.step_num == 100:
          done = True

      '''if episode >= MEMORIZE_PERIOD:
        for i in range(self.num_node):
          min_loss += self.agent[i].replay(self.batch_size, ACTION_SIZE)
          if done:
            if episode % UPDATE_CYCLE == 0:
              self.agent[i].update_target_model(LEARNING_RATE_DECAY)'''

      print("Episode %s: {epsilon: %.4f, step_num: %s, min_loss: %.4f, reward: %.4f, min(delay): %s, max(delay): %s, mean(delay): %.4f}" %
            (episode, self.agent[0].epsilon, self.step_num, min_loss, reward, min(self.delay_list), max(self.delay_list), np.mean(self.delay_list)))
            
      self.reward_list.append(reward)
      self.step_list.append(self.step_num)
      self.loss_list.append(min_loss)
      if self.step_num == 100:
        self.delay_max.append(self.step_num)
        self.delay_mean.append((self.step_num+sum(self.delay_list))/(len(self.delay_list)+1))
      else:
        self.delay_max.append(max(self.delay_list))
        self.delay_mean.append(sum(self.delay_list)/len(self.delay_list))

      if episode%50==0:
        self.save_plot()
      self.reset()
      if episode >= MEMORIZE_PERIOD:
        if self.agent[0].epsilon >= EPSILON_MIN:
          for i in range(self.num_node):
            self.agent[i].epsilon *= EPSILON_DECAY

    for i in range(self.num_node):
      self.agent[i].model.save("./result/" + self.fname + "/model/model_" + self.repeat + 
                               "_" + str(i) + ".h5")
    
    df = pd.DataFrame({"reward": self.reward_list, "loss": self.loss_list,
                       "step": self.step_list})
    df.to_csv("./result/" + self.fname + "/csv/result_" + self.repeat +
              ".csv")
            
  def step(self, actions):
    next_states = np.zeros((self.num_node, STATE_SIZE))
    rewards = np.zeros(self.num_node)
    dones = np.zeros(self.num_node, dtype=bool)

    for i in range(self.num_node):
      c = self.nodes[i].count
      if len(self.nodes[i].queues[actions[i]].items):
        l = len(self.nodes[i].queues[actions[i]].items)
        packet = self.nodes[i].queues[actions[i]].items[0]
        delay = self.env.now - packet.generated_time
        rewards[i] = 1 + (packet.hops + l + delay) / (self.step_num**2)
      elif c!=0 and any(self.nodes[i].get_state()[:6]):
        rewards[i] = -1

    for i in range(self.num_node):
      self.env.process(self.nodes[i].send(actions[i]))
    yield self.env.timeout(1)

    for i in range(self.num_node):
      yield self.env.process(self.nodes[i].receive())
    for i in range(self.num_node):
      next_states[i][0:6] = self.nodes[i].get_state()
      next_states[i][-1] = self.step_num
      if self.nodes[i].count == 0:
        dones[i] = True
    
    for i in range(self.num_src):
      yield self.env.process(self.dsts[i].receive())
      delay = self.dsts[i].get_delay()
      if delay > 0:
        self.delay_list[i] = delay

    return [next_states, rewards, dones]