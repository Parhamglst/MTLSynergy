import torch
import numpy as np
from Models import MTLSynergy


mtlSynergyModel = MTLSynergy()
mtlSynergyModel.load_state_dict(torch.load('save/MTLSynergy/fold_4.pth'))

output = mtlSynergyModel()