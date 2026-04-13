import collections
import dataclasses
import pandas as pd
from typing import Tuple , List
Transition = collections.namedtuple('Transition',('state','action','next_state','reward'))

ActionRes = collections.namedtuple('ActionRes',('step','logits'))

@dataclasses.dataclass
class TrainingInfo :
    learning_step = 0
    eval_mean_rewards : List[Tuple[int,float]] = dataclasses.field(default_factory=list)

    def to_csv(self,path) :
        df = pd.DataFrame(self.eval_mean_rewards,columns=['t','mean_reward'])
        df.to_csv(path) 

training_info = TrainingInfo()