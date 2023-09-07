import torch
import torch.nn as nn
from copy import deepcopy
from transformers import (
    AutoModelForMaskedLM,
)

from utils.bert_utils import multilabel_categorical_crossentropy
from .GRASP import GRASPModel


class GRASPModelMinit(GRASPModel):

    def __init__(self, args, tokenizer, device, model_processor):
        super(GRASPModelMinit, self).__init__(args, tokenizer, device, model_processor)

    def prepare_verbalizer(self, rel_ids_list, tr_class_ids_list):
        self.set_rel_verbalizer(rel_ids_list)
        self.set_tr_verbalizer(tr_class_ids_list)
        self.tr_manually_initalize()

    def tr_manually_initalize(self):
        tr_init_tokens = {
            "[outside]": "this is outside",
            "[subject]": "this is subject",
            "[object]": "this is object",
            "[trigger]": "this is trigger"
        }
        class_init_ids_list = []
        for class_token in tr_init_tokens:
            init_tokens = tr_init_tokens[class_token]
            init_token_ids = self.tokenizer(init_tokens, add_special_tokens=False)['input_ids']
            class_init_ids_list.append(init_token_ids)

        with torch.no_grad():
            word_embeddings = self.bert_model.get_input_embeddings()
            continuous_class_ids_list = self.model_processor.project_to_tr_class_idx
            for i, (continuous_tr_class_ids, class_init_ids) in enumerate(
                    zip(continuous_class_ids_list, class_init_ids_list)):
                word_embeddings.weight[continuous_tr_class_ids] = torch.mean(word_embeddings.weight[class_init_ids],
                                                                             dim=0)
