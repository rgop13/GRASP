import torch
import torch.nn as nn
from copy import deepcopy
from transformers import (
    AutoModelForMaskedLM,
)

from utils.bert_utils import multilabel_categorical_crossentropy


class GRASPModel(nn.Module):

    def __init__(self, args, tokenizer, device, model_processor):
        super(GRASPModel, self).__init__()
        self.args = args
        self.tokenizer = tokenizer
        self.model_processor = model_processor
        self.device = device

        self.bert_model = AutoModelForMaskedLM.from_pretrained(self.args.bert_model).to(device)
        self.config = self.bert_model.config

        self.softmax = nn.Softmax(dim=-1)

        self.rel_loss_func = multilabel_categorical_crossentropy
        self.tr_loss_func = nn.CrossEntropyLoss()

        # Verbalizer
        self.project_to_rel_idx = None
        self.project_to_tr_class_idx = None

    def forward(self, input_ids, token_type_ids, attention_mask, labels, tr_labels, **kwargs):
        forward_args = {
            "input_ids": input_ids,
            "token_type_ids": token_type_ids,
            "attention_mask": attention_mask,
        }
        if "roberta" in self.args.bert_model:
            forward_args.pop("token_type_ids")
        outputs = self.bert_model(**forward_args, return_dict=True)

        tr_logits = outputs.logits  # [bsz, max_seq, model_vocab_size]
        tr_logits = self.project_to_tr_classes(tr_logits)  # [bsz, max_seq, 4]
        tr_loss = self.tr_loss_func(tr_logits.view(-1, len(self.project_to_tr_class_idx)), tr_labels.view(-1))
        rcd_scores = self.softmax(tr_logits)  # [bsz, seq, 4]
        rcd_preds = torch.argmax(rcd_scores, dim=-1)  # [bsz, max_seq] 0, 1, 2, 3(trigger)

        trigger_masks = (rcd_preds == self.model_processor.tr_class_to_id[self.model_processor.tr_classes[-1]]).to(
            self.device)

        rel_input_ids, rel_token_type_ids, rel_attention_masks = self._reconstruct_input_ids(
            input_ids, token_type_ids, attention_mask, trigger_masks
        )
        rel_forward_args = {
            "input_ids": rel_input_ids,
            "token_type_ids": rel_token_type_ids,
            "attention_mask": rel_attention_masks,
        }
        if "roberta" in self.args.bert_model:
            rel_forward_args.pop("token_type_ids")
        rel_outputs = self.bert_model(**rel_forward_args, return_dict=True)
        rel_logits = rel_outputs.logits
        rel_logits = self.project_to_relations(rel_logits, input_ids)
        rel_loss = self.rel_loss_func(rel_logits, labels)

        final_loss = self.args.tr_loss_ratio * tr_loss + self.args.rel_loss_ratio * rel_loss
        final_loss.backward()

        return final_loss, None

    def _reconstruct_input_ids(self, input_ids, token_type_ids, attention_masks, trigger_masks):
        new_ids_list = []
        new_token_type_ids, new_attention_masks = [], []
        for i, batch_input_ids in enumerate(input_ids):
            token_list = self.tokenizer.convert_ids_to_tokens(batch_input_ids)
            new_token_list = []
            new_tti, new_am = [], []
            for ti, token in enumerate(token_list):
                if ti - 1 >= 0 and not trigger_masks[i][ti - 1] and trigger_masks[i][ti]:
                    if self.args.APM_type_for_trigger == "first_token" or self.args.APM_type_for_trigger == "surround":
                        new_token_list.append(self.args.front_trigger_marker_token)
                        new_tti.append(token_type_ids[i][ti])
                        new_am.append(attention_masks[i][ti])
                    # Normalize Space
                    if self.args.front_trigger_marker_token == 'Ġ' and 'Ġ' == token[0]:
                        token = token[1:]
                new_token_list.append(token)
                new_tti.append(token_type_ids[i][ti])
                new_am.append(attention_masks[i][ti])
            if self.args.APM_type_for_trigger == "surround":
                new_token_list.append(self.args.back_trigger_marker_token)
                new_tti.append(new_tti[-1])
                new_am.append(new_am[-1])

            new_token_list = new_token_list[:self.args.max_seq_len]
            new_tti = new_tti[:self.args.max_seq_len]
            new_am = new_am[:self.args.max_seq_len]

            new_ids_list.append(self.tokenizer.convert_tokens_to_ids(new_token_list))
            new_token_type_ids.append(new_tti)
            new_attention_masks.append(new_am)
        rel_input_ids = torch.tensor(new_ids_list).long().to(self.device)
        rel_token_type_ids = torch.tensor(new_token_type_ids).long().to(self.device)
        rel_attention_masks = torch.tensor(new_attention_masks).long().to(self.device)
        return rel_input_ids, rel_token_type_ids, rel_attention_masks

    def set_rel_verbalizer(self, rel_ids):
        if rel_ids is not None:
            self.project_to_rel_idx = torch.tensor(rel_ids).to(self.device)

    def set_tr_verbalizer(self, project_to_tr_class_idx):
        if project_to_tr_class_idx is not None:
            self.project_to_tr_class_idx = torch.tensor(project_to_tr_class_idx).to(self.device)

    def inference(self, input_ids, token_type_ids, attention_mask, **kwargs):
        forward_args = {
            "input_ids": input_ids,
            "token_type_ids": token_type_ids,
            "attention_mask": attention_mask,
        }
        if "roberta" in self.args.bert_model:
            forward_args.pop("token_type_ids")
        outputs = self.bert_model(**forward_args, return_dict=True)

        tr_logits = outputs.logits  # [bsz, max_seq, model_vocab_size]
        tr_logits = self.project_to_tr_classes(tr_logits)  # [bsz, max_seq, 4]
        rcd_scores = self.softmax(tr_logits)  # [bsz, seq, 4]
        rcd_preds = torch.argmax(rcd_scores, dim=-1)  # [bsz, max_seq]

        trigger_masks = (rcd_preds == self.model_processor.tr_class_to_id[self.model_processor.tr_classes[-1]]).to(
            self.device)
        rel_input_ids, rel_token_type_ids, rel_attention_masks = self._reconstruct_input_ids(
            input_ids, token_type_ids, attention_mask, trigger_masks
        )
        rel_forward_args = {
            "input_ids": rel_input_ids,
            "token_type_ids": rel_token_type_ids,
            "attention_mask": rel_attention_masks,
        }
        if "roberta" in self.args.bert_model:
            rel_forward_args.pop("token_type_ids")
        rel_outputs = self.bert_model(**rel_forward_args, return_dict=True)
        rel_logits = rel_outputs.logits
        rel_logits = self.project_to_relations(rel_logits, input_ids)

        return rel_logits, rcd_scores, None

    def project_to_relations(self, logits, input_ids):
        # convert the [batch_size, seq_len, vocab_size] => [batch_size, num_labels]
        _, mask_idx = (input_ids == self.tokenizer.mask_token_id).nonzero(as_tuple=True)
        bs = input_ids.shape[0]
        mask_output = logits[torch.arange(bs).to(self.device), mask_idx]
        assert mask_idx.shape[0] == bs, "only one mask in sequence!"
        final_output = mask_output[:, self.project_to_rel_idx]

        return final_output

    def project_to_tr_classes(self, logits):
        final_output = logits[:, :, self.project_to_tr_class_idx]
        return final_output

    def prepare_verbalizer(self, rel_ids_list, tr_class_ids_list):
        self.set_rel_verbalizer(rel_ids_list)
        self.set_tr_verbalizer(tr_class_ids_list)

    def get_true_predictions(
            self,
            input_ids_list,
            tr_logits_list,
            tr_labels_list,
    ):
        true_preds = []
        true_labels = []
        for i, (batch_pred, batch_label) in enumerate(zip(tr_logits_list, tr_labels_list)):
            for j, (token_pred, token_label) in enumerate(zip(batch_pred, batch_label)):
                if token_label != -100:
                    true_preds.append(  # 0,1,2,3
                        self.model_processor.pred_mapping[token_pred]  # int (class_id) to string
                    )
                    true_labels.append(
                        self.model_processor.label_mapping[token_label]  # int (class_id) to string
                    )
        return true_preds, true_labels
