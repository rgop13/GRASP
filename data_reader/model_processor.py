import os
import logging
import numpy as np

import torch
from torch.utils.data import Dataset

from tqdm import tqdm
from .data_processor import DreModelProcessor
from utils.data_utils import APMTokenize
from utils.data_utils import (
    get_target_mask,
    tokenize_vanilla
)

logger = logging.getLogger()


class MLMDataset(Dataset):
    def __init__(self, args, tokenizer, processor, model_processor, mode):
        super().__init__()
        self.args = args
        self.tokenizer = tokenizer
        self.processor = processor
        self.mode = mode
        self.model_processor = model_processor

        data_version = os.path.split(self.args.data_dir)[-1]
        # make cached directory under the task directory
        os.makedirs(os.path.join(args.data_dir, args.task, "cached"), exist_ok=True)
        is_f1c = "f1c" if self.args.conversational_setting else "f1"
        is_apm = "apm" if self.args.APM else "no_apm"
        if "/" in args.bert_model:
            bert_model = args.bert_model.replace("/", "-")
            cached_file_name = f"{args.model}-{args.task_desc}-{mode}-{bert_model}-{args.max_seq_len}-{is_f1c}-{is_apm}-{data_version}"
        else:
            bert_model = args.bert_model
            cached_file_name = f"{args.model}-{args.task_desc}-{mode}-{bert_model}-{args.max_seq_len}-{is_f1c}-{is_apm}-{data_version}"

        cached_features_file = os.path.join(args.data_dir, args.task, "cached", cached_file_name)
        
        self.features = self.model_processor.cache_load_examples(cached_features_file, self.mode)
        print(f"Total number of features for {mode}ing: {len(self.features)}")
        self.pad_token = self.tokenizer.pad_token_id
        print(f"Pad Token index: {self.pad_token}")
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, index):
        cur_example = self.features[index]
        feature = dict()
        feature["dial_id"] = cur_example.dial_id
        feature["subject_ids"] = cur_example.subject_ids
        feature["object_ids"] = cur_example.object_ids
        feature["input_ids"] = torch.tensor(cur_example.input_ids).long()
        feature["token_type_ids"] = torch.tensor(cur_example.token_type_ids).long()
        feature["attention_mask"] = torch.tensor(cur_example.attention_mask).long()
        feature["labels"] = torch.tensor(cur_example.labels).long()
        feature["rel_id"] = cur_example.rel_id
        
        return feature
    
    def collate_fn(self, batch):
        merged_batch = {key: [d[key] for d in batch] for key in batch[0]}
        max_src = max([len(e) for e in merged_batch["input_ids"]])
        
        for key in merged_batch:
            if key in ['input_ids', 'attention_mask', 'token_type_ids']:
                for batch_idx, features in enumerate(merged_batch[key]):
                    pad_token = self.pad_token if key == 'input_ids' else 0
                    pad_features = pad_token + torch.zeros(max_src - len(features)).long()
                    merged_batch[key][batch_idx] = torch.cat((features, pad_features), axis=0)
                merged_batch[key] = torch.stack(merged_batch[key], axis=0)
            elif key in ['labels']:
                for batch_idx, features in enumerate(merged_batch[key]):
                    merged_batch[key][batch_idx] = torch.tensor(features)
                merged_batch[key] = torch.stack(merged_batch[key], axis=0)
            else:
                pass

        return merged_batch


class MTLDataset(MLMDataset):

    def __init__(self, args, tokenizer, processor, model_processor, mode):
        super(MTLDataset, self).__init__(args, tokenizer, processor, model_processor, mode)

    def __getitem__(self, index):
        cur_example = self.features[index]
        feature = dict()
        feature["dial_id"] = cur_example.dial_id
        feature["subject_ids"] = cur_example.subject_ids
        feature["object_ids"] = cur_example.object_ids
        feature["input_ids"] = torch.tensor(cur_example.input_ids).long()
        feature["token_type_ids"] = torch.tensor(cur_example.token_type_ids).long()
        feature["attention_mask"] = torch.tensor(cur_example.attention_mask).long()
        feature["labels"] = torch.tensor(cur_example.labels).long()
        feature["rel_id"] = cur_example.rel_id
        feature["tr_labels"] = torch.tensor(cur_example.tr_labels).long()

        return feature

    def collate_fn(self, batch):
        merged_batch = {key: [d[key] for d in batch] for key in batch[0]}
        max_src = max([len(e) for e in merged_batch["input_ids"]])

        for key in merged_batch:
            if key in ['input_ids', 'attention_mask', 'token_type_ids']:
                for batch_idx, features in enumerate(merged_batch[key]):
                    pad_token = self.pad_token if key == 'input_ids' else 0
                    pad_features = pad_token + torch.zeros(max_src - len(features)).long()
                    merged_batch[key][batch_idx] = torch.cat((features, pad_features), axis=0)
                merged_batch[key] = torch.stack(merged_batch[key], axis=0)
            elif key in ['tr_labels']:
                for batch_idx, features in enumerate(merged_batch[key]):
                    pad_token = -100
                    pad_features = pad_token + torch.zeros(max_src - len(features)).long()
                    merged_batch[key][batch_idx] = torch.cat((features, pad_features), axis=0)
                merged_batch[key] = torch.stack(merged_batch[key], axis=0)
            elif key in ['tr_labels_v2']:
                for batch_idx, features in enumerate(merged_batch[key]):
                    pad_token = 0
                    pad_features = pad_token + torch.zeros(max_src - len(features)).long()
                    merged_batch[key][batch_idx] = torch.cat((features, pad_features), axis=0)
                merged_batch[key] = torch.stack(merged_batch[key], axis=0)
            elif key in ['labels']:
                for batch_idx, features in enumerate(merged_batch[key]):
                    merged_batch[key][batch_idx] = torch.tensor(features)
                merged_batch[key] = torch.stack(merged_batch[key], axis=0)
            else:
                pass

        return merged_batch


class PromptTuningProceesor(DreModelProcessor):

    def __init__(self, args, tokenizer, data_processor, prompt_size):
        super().__init__(args, tokenizer, data_processor, prompt_size)
        self.args = args
        self.prompt_size = prompt_size
        self.tokenizer = tokenizer
        self.processor = data_processor
        self.num_labels = len(self.processor.get_labels())
        self.relations = self.processor.get_relations()

        self.class_list = [f"[class{i}]" for i in range(1, self.num_labels + 1)]
        num_added_tokens = self.tokenizer.add_special_tokens({'additional_special_tokens': self.class_list})
        logger.info(
            "Vocab_size After add_special_tokens: {}, num_added_tokens: {}".format(len(tokenizer), num_added_tokens))
        self.unused_list = [f"[unused{i}]" for i in range(1, 10)]
        num_added_tokens = self.tokenizer.add_special_tokens({'additional_special_tokens': self.unused_list})
        logger.info(
            "Vocab_size After add_special_tokens: {}, num_added_tokens: {}".format(len(tokenizer), num_added_tokens))
        self.prompt_list = ["[subj]", "[obj]"]
        num_added_tokens = self.tokenizer.add_special_tokens({'additional_special_tokens': self.prompt_list})
        logger.info(
            "Vocab_size After add_special_tokens: {}, num_added_tokens: {}".format(len(tokenizer), num_added_tokens))

        self.tokenize_func = None
        if self.args.APM:
            self.tokenize_func = APMTokenize
        else:
            self.tokenize_func = tokenize_vanilla
        logger.info("Tokenization Function is {}".format(self.tokenize_func.__name__))
        self.bpe_space_tok = self.tokenizer.tokenize("  .")[0]

    def get_tr_class_ids_list(self):
        return None

    def get_encoded_data(self, dre_data, mode):
        encoded_data = []
        for data in tqdm(dre_data):
            enc = {}

            dialogue_input_ids = self.tokenizer.convert_tokens_to_ids(
                self.tokenize_func(
                    data.dialog, self.tokenizer, data.subject, data.object, None, self.bpe_space_tok
                )
            )
            x_input_ids = self.tokenizer.convert_tokens_to_ids(
                self.tokenize_func(
                    data.subject, self.tokenizer, data.subject, data.object, None, self.bpe_space_tok
                )
            )
            y_input_ids = self.tokenizer.convert_tokens_to_ids(
                self.tokenize_func(
                    data.object, self.tokenizer, data.subject, data.object, None, self.bpe_space_tok
                )
            )
            trigger_ids_list = [
                self.tokenizer.convert_tokens_to_ids(
                    self.tokenize_func(
                        trigger, self.tokenizer, data.subject, data.object, None, self.bpe_space_tok
                    )
                )
                for trigger in data.triggers
            ]

            enc['dial_id'] = data.dial_id
            enc['dialog'] = dialogue_input_ids
            enc['subject'] = x_input_ids
            enc['object'] = y_input_ids
            enc['raw_subject'] = data.subject
            enc['raw_object'] = data.object
            enc['rel_id'] = data.rel_id
            enc['labels'] = data.rel_labels
            enc['trigger_ids_list'] = trigger_ids_list
            enc['s_id'] = data.s_id
            enc['o_id'] = data.o_id

            encoded_data.append(enc)

        return encoded_data

    def _truncate_seq_tuple(self, tokens_a, tokens_b, tokens_c, max_length):
        """Truncates a sequence tuple in place to the maximum length."""
    
        # This is a simple heuristic which will always truncate the longer sequence
        # one token at a time. This makes more sense than truncating an equal percent
        # of tokens from each, since if one sequence is very short then each token
        # that's truncated likely contains more information than a longer sequence.
        while True:
            total_length = len(tokens_a) + len(tokens_b) + len(tokens_c)
            if total_length <= max_length:
                break
            if len(tokens_a) >= len(tokens_b) and len(tokens_a) >= len(tokens_c):
                tokens_a.pop()
            elif len(tokens_b) >= len(tokens_a) and len(tokens_b) >= len(tokens_c):
                tokens_b.pop()
            else:
                tokens_c.pop()
        return tokens_a, tokens_b, tokens_c

    def get_encoded_feature(self, example):
        inputs = {}
        tokens_a, tokens_b, tokens_c = self._truncate_seq_tuple(
            example["dialog"],
            example["subject"],
            example["object"],
            self.args.max_seq_len - 4 - self.prompt_size - 7
        )
        input_ids = [self.tokenizer.cls_token_id]
        input_ids += tokens_a
        input_ids += [self.tokenizer.sep_token_id]
        token_type_ids = [0 for _ in range(len(input_ids))]

        input_ids += [self.tokenizer.convert_tokens_to_ids("[subj]")] + \
                     [self.tokenizer.convert_tokens_to_ids("[unused3]")] + \
                     tokens_b + \
                     [self.tokenizer.convert_tokens_to_ids("[unused4]")] + \
                     [self.tokenizer.convert_tokens_to_ids("[subj]")]

        input_ids += [self.tokenizer.convert_tokens_to_ids("[unused7]")] +\
                     [self.tokenizer.convert_tokens_to_ids("[unused8]")] +\
                     [self.tokenizer.mask_token_id] +\
                     [self.tokenizer.convert_tokens_to_ids("[unused9]")]

        input_ids += [self.tokenizer.convert_tokens_to_ids("[obj]")] + \
                     [self.tokenizer.convert_tokens_to_ids("[unused5]")] + \
                     tokens_c + \
                     [self.tokenizer.convert_tokens_to_ids("[unused6]")] + \
                     [self.tokenizer.convert_tokens_to_ids("[obj]")]
        input_ids += [self.tokenizer.sep_token_id]
        previous_seg_size = len(token_type_ids)
        token_type_ids += [1 for _ in range(len(input_ids) - previous_seg_size)]

        attention_mask = [1 for _ in range(len(input_ids))]
        assert len(input_ids) == len(token_type_ids) == len(attention_mask)
        padding_length = self.args.max_seq_len - len(input_ids)
        if padding_length > 0:
            input_ids = input_ids + ([self.tokenizer.pad_token_id] * padding_length)
            attention_mask = attention_mask + ([0] * padding_length)
            token_type_ids = token_type_ids + ([0] * padding_length)
        else:
            input_ids = input_ids[:self.args.max_seq_len]
            attention_mask = attention_mask[:self.args.max_seq_len]
            token_type_ids = token_type_ids[:self.args.max_seq_len]

        assert len(input_ids) == self.args.max_seq_len
        assert len(attention_mask) == self.args.max_seq_len
        assert len(token_type_ids) == self.args.max_seq_len

        inputs["input_ids"] = input_ids
        inputs["subject_ids"] = example["subject"]
        inputs["object_ids"] = example["object"]
        inputs["attention_mask"] = attention_mask
        inputs["token_type_ids"] = token_type_ids
        inputs["labels"] = example["labels"]
        inputs["dial_id"] = example["dial_id"]
        inputs["raw_subject"] = example["raw_subject"]
        inputs["raw_object"] = example["raw_object"]
        inputs["rel_id"] = example["rel_id"]
        inputs["s_id"] = example["s_id"]
        inputs["o_id"] = example["o_id"]

        return inputs
    
    def convert_data_to_features(self, encoded_datas, typ):
        """Creates examples for the training and dev sets. / json files"""
        features = []

        for i, data_e in tqdm(enumerate(encoded_datas)):
            inputs = self.get_encoded_feature(data_e)
            features.append(InputFeatures(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                token_type_ids=inputs["token_type_ids"],
                labels=inputs["labels"],
                dial_id=inputs["dial_id"],
                subject_ids=inputs["subject_ids"],
                object_ids=inputs["object_ids"],
                rel_id=inputs["rel_id"],
                s_id=inputs["s_id"],
                o_id=inputs["o_id"],
                ))
            if i < 3:
                logger.info("Feature Examples...")
                logger.info("{}th Feature input_ids: {}".format(i, features[-1].input_ids))
                logger.info("{}th Feature input_text: {}".format(i, self.tokenizer.decode(features[-1].input_ids)))

        return features

    def prepare_relation_labels(self):
        label_ids_list = []
        for label in self.relations:
            label = label.lower()
            label = label.replace("per:", "person ").replace("org:", "organization ").replace("gpe:", "geopolitical ")
            label = label.replace("_", " ").replace("/", " ")
            label_ids = self.tokenizer(label, add_special_tokens=False)['input_ids']
            label_ids_list.append(label_ids)
        return label_ids_list

    def prepare_marker_words(self):
        return None


class GRASPv1Processor(PromptTuningProceesor):

    def __init__(self, args, tokenizer, data_processor, prompt_size):
        super().__init__(args, tokenizer, data_processor, prompt_size)

        self.tr_classes = ["[subject]", "[object]", "[trigger]"]
        num_added_tokens = self.tokenizer.add_special_tokens({'additional_special_tokens': self.tr_classes})
        logger.info(
            "Vocab_size After add_special_tokens: {}, num_added_tokens: {}".format(len(tokenizer), num_added_tokens))

        self.tr_subject_class_id = self.tokenizer.convert_tokens_to_ids("[subject]")
        self.tr_object_class_id = self.tokenizer.convert_tokens_to_ids("[object]")
        self.tr_trigger_class_id = self.tokenizer.convert_tokens_to_ids("[trigger]")

        self.project_to_tr_class_idx = [
            self.tr_subject_class_id, self.tr_object_class_id, self.tr_trigger_class_id
        ]

        self.tr_class_to_vocab_id = {
            self.tr_classes[0]: self.tr_subject_class_id,
            self.tr_classes[1]: self.tr_object_class_id,
            self.tr_classes[2]: self.tr_trigger_class_id,
        }

        self.tr_class_to_id = dict()
        for i, tr_class in enumerate(self.tr_classes):
            self.tr_class_to_id[tr_class] = i
            # [subject]: 0, [object]: 1, [trigger]: 2

        self.pred_mapping = dict()
        for i, tr_class in enumerate(self.tr_classes):
            self.pred_mapping[i] = tr_class

        self.label_mapping = dict()
        for i, tr_vocab_id in enumerate(self.project_to_tr_class_idx):
            self.label_mapping[tr_vocab_id] = self.tr_classes[i]

        self.vocab_class_id_to_real_class = dict()
        for i, vocab_class_id in enumerate(self.project_to_tr_class_idx):
            self.vocab_class_id_to_real_class[vocab_class_id] = self.tr_classes[i]

        trigger_markers = [self.args.front_trigger_marker_token, self.args.back_trigger_marker_token]
        num_added_tokens = self.tokenizer.add_special_tokens(
            {'additional_special_tokens': trigger_markers}
        )
        logger.info(
            "[Trigger marker] Vocab_size After add_special_tokens: {}, num_added_tokens: {}".format(
                len(tokenizer), num_added_tokens
            )
        )

    def get_encoded_data(self, dre_data, mode):
        encoded_data = []
        for data in tqdm(dre_data):
            enc = {}
            dialogue_input_ids = self.tokenizer.convert_tokens_to_ids(
                self.tokenize_func(
                    data.dialog, self.tokenizer, data.subject, data.object, None, self.bpe_space_tok
                )
            )
            x_input_ids = self.tokenizer.convert_tokens_to_ids(
                self.tokenize_func(
                    data.subject, self.tokenizer, data.subject, data.object, None, self.bpe_space_tok
                )
            )
            y_input_ids = self.tokenizer.convert_tokens_to_ids(
                self.tokenize_func(
                    data.object, self.tokenizer, data.subject, data.object, None, self.bpe_space_tok
                )
            )
            trigger_ids_list = [
                self.tokenizer.convert_tokens_to_ids(
                    self.tokenize_func(
                        trigger, self.tokenizer, data.subject, data.object, None, self.bpe_space_tok
                    )
                )
                for trigger in data.triggers
            ]

            enc['dial_id'] = data.dial_id
            enc['dialog'] = dialogue_input_ids
            enc['subject'] = x_input_ids
            enc['object'] = y_input_ids
            enc['raw_subject'] = data.subject
            enc['raw_object'] = data.object
            enc['raw_triggers'] = data.triggers
            enc['rel_id'] = data.rel_id
            enc['labels'] = data.rel_labels
            enc['trigger_ids_list'] = trigger_ids_list
            enc['s_id'] = data.s_id
            enc['o_id'] = data.o_id

            encoded_data.append(enc)

        return encoded_data

    def get_tr_class_ids_list(self):
        return self.project_to_tr_class_idx

    def get_encoded_feature(self, example):
        inputs = {}
        tokens_a, tokens_b, tokens_c = self._truncate_seq_tuple(
            example["dialog"],
            example["subject"],
            example["object"],
            self.args.max_seq_len - 4 - self.prompt_size - 7 - 2
        )
        input_ids = [self.tokenizer.cls_token_id]
        input_ids += tokens_a
        input_ids += [self.tokenizer.sep_token_id]
        token_type_ids = [0 for _ in range(len(input_ids))]

        input_ids += [self.tokenizer.convert_tokens_to_ids("[subj]")] + \
                     [self.tokenizer.convert_tokens_to_ids("[unused3]")] + \
                     tokens_b + \
                     [self.tokenizer.convert_tokens_to_ids("[unused4]")] + \
                     [self.tokenizer.convert_tokens_to_ids("[subj]")]

        input_ids += [self.tokenizer.convert_tokens_to_ids("[unused7]")] + \
                     [self.tokenizer.convert_tokens_to_ids("[unused8]")] + \
                     [self.tokenizer.mask_token_id] + \
                     [self.tokenizer.convert_tokens_to_ids("[unused9]")]

        input_ids += [self.tokenizer.convert_tokens_to_ids("[obj]")] + \
                     [self.tokenizer.convert_tokens_to_ids("[unused5]")] + \
                     tokens_c + \
                     [self.tokenizer.convert_tokens_to_ids("[unused6]")] + \
                     [self.tokenizer.convert_tokens_to_ids("[obj]")]
        input_ids += [self.tokenizer.sep_token_id]
        previous_seg_size = len(token_type_ids)
        token_type_ids += [1 for _ in range(len(input_ids) - previous_seg_size)]

        attention_mask = [1 for _ in range(len(input_ids))]
        assert len(input_ids) == len(token_type_ids) == len(attention_mask)
        padding_length = self.args.max_seq_len - len(input_ids)
        if padding_length > 0:
            input_ids = input_ids + ([self.tokenizer.pad_token_id] * padding_length)
            attention_mask = attention_mask + ([0] * padding_length)
            token_type_ids = token_type_ids + ([0] * padding_length)
        else:
            input_ids = input_ids[:self.args.max_seq_len]
            attention_mask = attention_mask[:self.args.max_seq_len]
            token_type_ids = token_type_ids[:self.args.max_seq_len]

        assert len(input_ids) == self.args.max_seq_len
        assert len(attention_mask) == self.args.max_seq_len
        assert len(token_type_ids) == self.args.max_seq_len

        # Create subject/object/trigger masks
        subj_mask = np.asarray(get_target_mask(
            input_ids, example["subject"], example["raw_subject"], self.tokenizer.pad_token_id, self.tokenizer
        ))
        obj_mask = np.asarray(get_target_mask(
            input_ids, example["object"], example["raw_object"], self.tokenizer.pad_token_id, self.tokenizer
        ))
        trigger_mask_list = []
        for ti, trigger_ids in enumerate(example["trigger_ids_list"]):
            trigger_mask = np.asarray(get_target_mask(
                input_ids, trigger_ids, example["raw_triggers"][ti], self.tokenizer.pad_token_id, self.tokenizer)
            )
            trigger_mask_list.append(trigger_mask)

        tr_labels = np.full(len(input_ids), fill_value=-100)
        tr_labels[subj_mask == 1] = self.tr_subject_class_id
        tr_labels[obj_mask == 1] = self.tr_object_class_id
        for ti, trigger_mask in enumerate(trigger_mask_list):
            tr_labels[trigger_mask == 1] = self.tr_trigger_class_id

        inputs["input_ids"] = input_ids
        inputs["subject_ids"] = example["subject"]
        inputs["object_ids"] = example["object"]
        inputs["attention_mask"] = attention_mask
        inputs["token_type_ids"] = token_type_ids
        inputs["labels"] = example["labels"]
        inputs["dial_id"] = example["dial_id"]
        inputs["raw_subject"] = example["raw_subject"]
        inputs["raw_object"] = example["raw_object"]
        inputs["rel_id"] = example["rel_id"]
        inputs["tr_labels"] = tr_labels
        inputs["s_id"] = example["s_id"]
        inputs["o_id"] = example["o_id"]

        return inputs

    def convert_data_to_features(self, encoded_datas, typ):
        """Creates examples for the training and dev sets. / json files"""
        features = []
        for i, data_e in tqdm(enumerate(encoded_datas)):
            inputs = self.get_encoded_feature(data_e)
            features.append(InputFeatures(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                token_type_ids=inputs["token_type_ids"],
                labels=inputs["labels"],
                dial_id=inputs["dial_id"],
                subject_ids=inputs["subject_ids"],
                object_ids=inputs["object_ids"],
                rel_id=inputs["rel_id"],
                tr_labels=inputs["tr_labels"],
                s_id=inputs["s_id"],
                o_id=inputs["o_id"],
            ))
            if i < 3:
                logger.info("Feature Examples...")
                logger.info("{}th Feature input_ids: {}".format(i, features[-1].input_ids))
                logger.info("{}th Feature input_text: {}".format(i, self.tokenizer.decode(features[-1].input_ids)))
                logger.info("{}th Feature TR labels: {}".format(i, features[-1].tr_labels))

        return features

    def prepare_relation_labels(self):
        label_ids_list = []
        for label in self.relations:
            label = label.lower()
            label = label.replace("per:", "person ").replace("org:", "organization ").replace("gpe:", "geopolitical ")
            label = label.replace("_", " ").replace("/", " ")
            label_ids = self.tokenizer(label, add_special_tokens=False)['input_ids']
            label_ids_list.append(label_ids)
        return label_ids_list

    def prepare_marker_words(self):
        return None


class GRASPv2Processor(GRASPv1Processor):

    def __init__(self, args, tokenizer, data_processor, prompt_size):
        super().__init__(args, tokenizer, data_processor, prompt_size)

        self.tr_classes = ["[outside]", "[subject]", "[object]", "[trigger]"]
        num_added_tokens = self.tokenizer.add_special_tokens({'additional_special_tokens': self.tr_classes})
        logger.info(
            "Vocab_size After add_special_tokens: {}, num_added_tokens: {}".format(len(tokenizer), num_added_tokens))

        self.tr_outside_class_id = self.tokenizer.convert_tokens_to_ids("[outside]")
        self.tr_subject_class_id = self.tokenizer.convert_tokens_to_ids("[subject]")
        self.tr_object_class_id = self.tokenizer.convert_tokens_to_ids("[object]")
        self.tr_trigger_class_id = self.tokenizer.convert_tokens_to_ids("[trigger]")

        self.project_to_tr_class_idx = [
            self.tr_outside_class_id, self.tr_subject_class_id, self.tr_object_class_id, self.tr_trigger_class_id
        ]

        self.tr_class_to_vocab_id = {
            self.tr_classes[0]: self.tr_outside_class_id,
            self.tr_classes[1]: self.tr_subject_class_id,
            self.tr_classes[2]: self.tr_object_class_id,
            self.tr_classes[3]: self.tr_trigger_class_id,
        }

        self.tr_class_to_id = dict()
        for i, tr_class in enumerate(self.tr_classes):
            self.tr_class_to_id[tr_class] = i
            # [outside]: 0, [subject]: 1, [object]: 2, [trigger]: 3

        self.pred_mapping = dict()
        for i, tr_class in enumerate(self.tr_classes):
            self.pred_mapping[i] = tr_class

        self.label_mapping = dict()
        for i, tr_class in enumerate(self.tr_classes):
            self.label_mapping[i] = tr_class
            # vocab_id([subject]): [subject], vocab_id([object]): [object]

        self.tr_vocab_id_to_id = dict()
        for i, tr_vocab_id in enumerate(self.project_to_tr_class_idx):
            self.tr_vocab_id_to_id[tr_vocab_id] = i
            # vocab_id([outside]): 0, vocab_id([subject]): 1, ...

    def get_encoded_feature(self, example):
        inputs = {}
        tokens_a, tokens_b, tokens_c = self._truncate_seq_tuple(
            example["dialog"],
            example["subject"],
            example["object"],
            self.args.max_seq_len - 4 - self.prompt_size - 7 - 2
        )
        input_ids = [self.tokenizer.cls_token_id]
        input_ids += tokens_a
        input_ids += [self.tokenizer.sep_token_id]
        token_type_ids = [0 for _ in range(len(input_ids))]

        input_ids += [self.tokenizer.convert_tokens_to_ids("[subj]")] + \
                     [self.tokenizer.convert_tokens_to_ids("[unused3]")] + \
                     tokens_b + \
                     [self.tokenizer.convert_tokens_to_ids("[unused4]")] + \
                     [self.tokenizer.convert_tokens_to_ids("[subj]")]

        input_ids += [self.tokenizer.convert_tokens_to_ids("[unused7]")] + \
                     [self.tokenizer.convert_tokens_to_ids("[unused8]")] + \
                     [self.tokenizer.mask_token_id] + \
                     [self.tokenizer.convert_tokens_to_ids("[unused9]")]

        input_ids += [self.tokenizer.convert_tokens_to_ids("[obj]")] + \
                     [self.tokenizer.convert_tokens_to_ids("[unused5]")] + \
                     tokens_c + \
                     [self.tokenizer.convert_tokens_to_ids("[unused6]")] + \
                     [self.tokenizer.convert_tokens_to_ids("[obj]")]
        input_ids += [self.tokenizer.sep_token_id]
        previous_seg_size = len(token_type_ids)
        token_type_ids += [1 for _ in range(len(input_ids) - previous_seg_size)]

        attention_mask = [1 for _ in range(len(input_ids))]
        assert len(input_ids) == len(token_type_ids) == len(attention_mask)
        padding_length = self.args.max_seq_len - len(input_ids)
        if padding_length > 0:
            input_ids = input_ids + ([self.tokenizer.pad_token_id] * padding_length)
            attention_mask = attention_mask + ([0] * padding_length)
            token_type_ids = token_type_ids + ([0] * padding_length)
        else:
            input_ids = input_ids[:self.args.max_seq_len]
            attention_mask = attention_mask[:self.args.max_seq_len]
            token_type_ids = token_type_ids[:self.args.max_seq_len]

        assert len(input_ids) == self.args.max_seq_len
        assert len(attention_mask) == self.args.max_seq_len
        assert len(token_type_ids) == self.args.max_seq_len

        subj_mask = np.asarray(get_target_mask(
            input_ids, example["subject"], example["raw_subject"], self.tokenizer.pad_token_id, self.tokenizer
        ))
        obj_mask = np.asarray(get_target_mask(
            input_ids, example["object"], example["raw_object"], self.tokenizer.pad_token_id, self.tokenizer
        ))
        trigger_mask_list = []
        for ti, trigger_ids in enumerate(example["trigger_ids_list"]):
            trigger_mask = np.asarray(get_target_mask(
                input_ids, trigger_ids, example["raw_triggers"][ti], self.tokenizer.pad_token_id, self.tokenizer)
            )
            trigger_mask_list.append(trigger_mask)

        tr_labels = np.full(len(input_ids), fill_value=-100)
        tr_labels_v2 = np.full(len(input_ids), fill_value=self.tr_class_to_id["[outside]"])
        tr_labels[subj_mask == 1] = self.tr_subject_class_id
        tr_labels_v2[subj_mask == 1] = self.tr_class_to_id["[subject]"]
        tr_labels[obj_mask == 1] = self.tr_object_class_id
        tr_labels_v2[obj_mask == 1] = self.tr_class_to_id["[object]"]
        for ti, trigger_mask in enumerate(trigger_mask_list):
            tr_labels[trigger_mask == 1] = self.tr_trigger_class_id
            tr_labels_v2[trigger_mask == 1] = self.tr_class_to_id["[trigger]"]
        tr_labels_v2[input_ids == self.tokenizer.pad_token_id] = -100
        tr_labels_v2[input_ids == self.tokenizer.mask_token_id] = -100
        tr_labels_v2[input_ids == self.tokenizer.cls_token_id] = -100
        tr_labels_v2[input_ids == self.tokenizer.sep_token_id] = -100
        tr_labels_v2[input_ids == self.tokenizer.convert_tokens_to_ids("[subj]")] = -100
        tr_labels_v2[input_ids == self.tokenizer.convert_tokens_to_ids("[obj]")] = -100

        inputs["input_ids"] = input_ids
        inputs["subject_ids"] = example["subject"]
        inputs["object_ids"] = example["object"]
        inputs["attention_mask"] = attention_mask
        inputs["token_type_ids"] = token_type_ids
        inputs["labels"] = example["labels"]
        inputs["dial_id"] = example["dial_id"]
        inputs["raw_subject"] = example["raw_subject"]
        inputs["raw_object"] = example["raw_object"]
        inputs["rel_id"] = example["rel_id"]
        inputs["tr_labels"] = tr_labels
        inputs["tr_labels_v2"] = tr_labels_v2
        inputs["s_id"] = example["s_id"]
        inputs["o_id"] = example["o_id"]

        return inputs

    def convert_data_to_features(self, encoded_datas, typ):
        """Creates examples for the training and dev sets. / json files"""
        features = []
        for i, data_e in tqdm(enumerate(encoded_datas)):
            inputs = self.get_encoded_feature(data_e)
            features.append(InputFeatures(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                token_type_ids=inputs["token_type_ids"],
                labels=inputs["labels"],
                dial_id=inputs["dial_id"],
                subject_ids=inputs["subject_ids"],
                object_ids=inputs["object_ids"],
                rel_id=inputs["rel_id"],
                tr_labels=inputs["tr_labels_v2"],
                s_id=inputs["s_id"],
                o_id=inputs["o_id"],
            ))
            if i < 3:
                logger.info("Feature Examples...")
                logger.info("{}th Feature input_ids: {}".format(i, features[-1].input_ids))
                logger.info("{}th Feature input_text: {}".format(i, self.tokenizer.decode(features[-1].input_ids)))
                logger.info("{}th Feature TR labels: {}".format(i, features[-1].tr_labels))

        return features


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(
            self,
            input_ids,
            attention_mask,
            token_type_ids,
            labels,
            dial_id,
            subject_ids,
            object_ids,
            rel_id,
            s_id=None,
            o_id=None,
            tr_labels=None,
    ):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.labels = labels
        self.dial_id = dial_id
        self.subject_ids = subject_ids
        self.object_ids = object_ids
        self.rel_id = rel_id
        self.s_id = s_id
        self.o_id = o_id
        self.tr_labels = tr_labels
