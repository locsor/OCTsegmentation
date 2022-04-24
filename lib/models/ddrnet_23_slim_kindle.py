import torch
import torch.nn as nn
import kindle
from kindle import Model
from kindle.utils.torch_utils import count_model_params
from kindle.generator import GeneratorAbstract
import numpy as np
from typing import Any, Dict, List, Union

model = Model("./configs/ddrnet_23_slim_base.yaml",verbose=True)