#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import random
from collections import deque


class Memory:

    def __init__(self):
        self.memory = deque(maxlen=self.capacity)

    def remember(self, sample):
        self.memory.append(sample)

    def sample(self, n):
        sample_batch = random.sample(self.memory, n)
        return sample_batch

