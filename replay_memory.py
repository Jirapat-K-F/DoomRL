import collections
from datamodel import Transition
import random

class ReplayMemory:

    def __init__(self , capacity) :
        self.memory = collections.deque(maxlen=capacity)

    def push(self,*args) :
        self.memory.append(Transition(*args))

    def sample(self,batch_size) :
        return random.sample(self.memory,batch_size)
    
    def __len__(self) :
        return len(self.memory)