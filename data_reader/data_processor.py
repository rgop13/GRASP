import os
import random
import re

import six
import copy
import torch
import json
import logging
import warnings
from collections import defaultdict
from utils.data_utils import rename

warnings.filterwarnings("ignore")
logger = logging.getLogger()


def convert_to_unicode(text):
    """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
    if six.PY3:
        if isinstance(text, str):
            return text
        elif isinstance(text, bytes):
            return text.decode("utf-8", "ignore")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    elif six.PY2:
        if isinstance(text, str):
            return text.decode("utf-8", "ignore")
        elif isinstance(text, unicode):
            return text
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    else:
        raise ValueError("Not running on Python2 or Python 3?")


class InputExample(object):
    def __init__(self, dial_id, dialog, subject, object, rel_labels, rel_id, triggers, s_id=None, o_id=None, relation=None):
        self.dial_id = dial_id
        self.dialog = dialog
        self.subject = subject
        self.object = object
        self.rel_labels = rel_labels
        self.rel_id = rel_id
        self.triggers = triggers
        self.s_id = s_id
        self.o_id = o_id
        self.relation = relation


class DataProcessor(object):
    def __init__(self, args, **kwargs):
        self.args = args
        self.is_marker = False
        self.na_num = 0
        self.rel_num = 0


class DreProcessor(DataProcessor):
    def read_data_files(self, data_dir, typ):
        '''

        :param typ: 'train' or 'dev' or 'test'
        :return:
        '''
        dial_data = []
        datas = self.get_data(typ)

        for di, data in enumerate(datas):
            for j in range(len(data[1])):
                if self.args.conversational_setting and typ != "train":
                    d_list = self.create_conversational_input_example(data, di, j)
                    dial_data += d_list
                else:
                    d = self.create_input_example(data, di, j)
                    dial_data.append(d)
        if self.args.conversational_setting and typ != "train":
            logger.info("[DreProcessor] Note that this setting is \'conversational setting\' for F1c evaluation.")
        logger.info("[DreProcessor/DRE_{}_dataset] read_data is done! size: {}".format(typ, len(dial_data)))
        for i in range(3):
            logger.info("[DreProcessor/DRE_{}_dataset] Example_{} ------*\nDial_id:{}\nDialog:{}\nrel_id:{}\nlabels:{}\nSubject:{}\nObject:{}".format(
                typ,
                i,
                dial_data[i].dial_id,
                dial_data[i].dialog,
                dial_data[i].rel_id,
                dial_data[i].rel_labels,
                dial_data[i].subject,
                dial_data[i].object,
            ))
        return dial_data

    def get_data(self, typ):
        print("Reading %s" % typ)
        if typ == "train":
            with open(os.path.join(self.args.data_dir, "train.json"), "r", encoding="utf-8") as reader:
                examples = json.load(reader)
        elif typ == "dev":
            with open(os.path.join(self.args.data_dir, "dev.json"), "r", encoding="utf-8") as reader:
                examples = json.load(reader)
        else:
            with open(os.path.join(self.args.data_dir, "test.json"), "r", encoding="utf-8") as reader:
                examples = json.load(reader)
                
        return examples
    
    def create_input_example(self, data, dial_id, j):
        rel_labels = []
        triggers = []
        rel_id = j
        for trigger in data[1][j]['t']:
            if trigger == "":
                continue
            trigger = trigger.lower()
            triggers.append(trigger)
        for k in range(36):
            if k + 1 in data[1][j]["rid"]:
                rel_labels += [1]
            else:
                rel_labels += [0]
        dialogue, subject, object = rename(' '.join(data[0]).lower(), data[1][j]["x"].lower(), data[1][j]["y"].lower())
        d = InputExample(
            dial_id=dial_id,
            dialog=dialogue,
            subject=subject,
            object=object,
            rel_labels=rel_labels,
            rel_id=rel_id,
            triggers=triggers,
            s_id=0,
            o_id=0,
        )
        return d

    def create_conversational_input_example(self, data, dial_id, j):
        d_list = []
        rel_labels = []
        triggers = []
        rel_id = j
        for trigger in data[1][j]['t']:
            if trigger == "":
                continue
            trigger = trigger.lower()
            triggers.append(trigger)
        for k in range(36):
            if k + 1 in data[1][j]["rid"]:
                rel_labels += [1]
            else:
                rel_labels += [0]
        for l in range(1, len(data[0])+1):
            dialogue, subject, object = rename(' '.join(data[0][:l]).lower(), data[1][j]["x"].lower(), data[1][j]["y"].lower())
            d = InputExample(
                dial_id=dial_id,
                dialog=dialogue,
                subject=subject,
                object=object,
                rel_labels=rel_labels,
                rel_id=rel_id,
                triggers=triggers,
                s_id=0,
                o_id=0,
            )
            d_list += [d]
        return d_list

    def get_relations(self):
        return ["per:positive_impression", "per:negative_impression", "per:acquaintance", "per:alumni", "per:boss", "per:subordinate", "per:client", "per:dates", "per:friends", "per:girl/boyfriend", "per:neighbor", "per:roommate", "per:children", "per:other_family", "per:parents", "per:siblings", "per:spouse", "per:place_of_residence", "per:place_of_birth", "per:visited_place", "per:origin", "per:employee_or_member_of", "per:schools_attended", "per:works", "per:age", "per:date_of_birth", "per:major", "per:place_of_work", "per:title", "per:alternate_names", "per:pet", "gpe:residents_of_place", "gpe:births_in_place", "gpe:visitors_of_place", "org:employees_or_members", "org:students"]

    def get_labels(self):
        """

        :return: DialogueRE 데이터셋의 관계 수
        """
        return [str(x) for x in range(len(self.get_relations()))]


class DreModelProcessor(object):

    def __init__(self, args, tokenizer, data_processor, prompt_size):
        self.args = args
        self.prompt_size = prompt_size
        self.tokenizer = tokenizer
        self.processor = data_processor
        self.num_labels = len(self.processor.get_labels())
        self.relations = self.processor.get_relations()

    def cache_load_examples(self, cached_features_file, typ):
        if not os.path.exists(cached_features_file):
            print("Creating features from dataset file at %s/%s" % (self.args.data_dir, self.args.task))
            try:
                dre_datas = self.processor.read_data_files(
                    os.path.join(self.args.data_dir, self.args.task), typ)
                encoded_datas = self.get_encoded_data(dre_datas, typ)
                features = self.convert_data_to_features(encoded_datas, typ)
        
                print("Saving features into cached file %s", cached_features_file)
                torch.save(features, cached_features_file)
            except ValueError:
                print("For mode, only train, dev, test is available")
                raise
        else:
            print("Cached Features already exists in %s", cached_features_file)
            features = torch.load(cached_features_file)

        return features

    def get_encoded_data(self, qa_data, mode):
        raise NotImplementedError

    def convert_data_to_features(self, qa_data, typ):
        raise NotImplementedError
