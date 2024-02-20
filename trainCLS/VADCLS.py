import torch
import torch.nn as nn
import torch.nn.functional as F
from trainCLS.baseCLS import BasicCLS

class VADCLS(BasicCLS):
    def __init__(self,config):
        super().__init__(config=config)