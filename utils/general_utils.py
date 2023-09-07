import logging
import torch
import math
from torch import nn
from transformers import (
    BertTokenizer,
    ElectraTokenizer,
    RobertaTokenizer,
    MPNetTokenizer,
    # RobertaForMaskedLM,
    # ElectraForMaskedLM,
)

from data_reader import (
    MLMDataset,
    MTLDataset,
    PromptTuningProceesor,
    GRASPv2Processor,
    DreProcessor
)


from models import GRASPModel, GRASPModelMinit


def move_to_cuda(maybe_tensor, device):
    if torch.is_tensor(maybe_tensor):
        return maybe_tensor.cuda(device)
    elif isinstance(maybe_tensor, dict):
        return {
            key: move_to_cuda(value, device)
            for key, value in maybe_tensor.items()
        }
    elif isinstance(maybe_tensor, list):
        return [move_to_cuda(x, device) for x in maybe_tensor]
    else:
        return maybe_tensor
    
class GELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
 
MODEL_CLASSES = {
    "GRASP": GRASPModel,
    "GRASP-minit": GRASPModelMinit,
}

ADVISOR_MODEL_PROCESSORS = {
    "GRASP": GRASPv2Processor,
    "GRASP-minit": GRASPv2Processor,
}

TOKENIZER_CLASSES = {
    "GRASP": RobertaTokenizer,
    "GRASP-minit": RobertaTokenizer,
}

ADVISOR_DATASET_CLASSES = {
    "GRASP": MTLDataset,
    "GRASP-minit": MTLDataset,
}

DATA_PROCESSORS = {
    "dre": DreProcessor,
}

TYPE_WORD_RATIOS = {
    "person": 0.7955080323339814,
    "name": 0.08587434769262253,
    "title": 0.08587434769262253,
    "place": 0.018418090657935128,
    "organization": 0.009976465773048193,
    "date": 0.0021743579248951193,
    "age": 0.0021743579248951193,
}

