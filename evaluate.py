import copy
import csv
import json
import logging
import numpy as np
import os
import pickle
import torch
from collections import defaultdict, Counter
from sklearn.metrics import f1_score, classification_report
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.general_utils import ADVISOR_DATASET_CLASSES

logger = logging.getLogger()


class Evaluation(object):
    def __init__(self, args, tokenizer, data_processor, model_processor):

        self.args = args
        self.tokenizer = tokenizer
        self.processor = data_processor
        self.model_processor = model_processor
        self.device = torch.device("cuda:" + str(self.args.gpu_ids[0]) if torch.cuda.is_available() else "cpu")
        self.dev_dataset = ADVISOR_DATASET_CLASSES[self.args.model](
            self.args, self.tokenizer, self.processor, self.model_processor, mode="dev"
        )
        self.dev_dataloader = DataLoader(
            self.dev_dataset,
            batch_size=self.args.valid_batch_size,
            num_workers=self.args.cpu_workers,
            shuffle=False,
            drop_last=False,
            collate_fn=self.dev_dataset.collate_fn
        )

        self.test_dataset = ADVISOR_DATASET_CLASSES[self.args.model](
            self.args, self.tokenizer, self.processor, self.model_processor, mode="test"
        )
        self.test_dataloader = DataLoader(
            self.test_dataset,
            batch_size=self.args.valid_batch_size,
            num_workers=self.args.cpu_workers,
            shuffle=False,
            drop_last=False,
            collate_fn=self.test_dataset.collate_fn
        )
        self.data_map = {
            "dev": self.dev_dataloader,
            "test": self.test_dataloader
        }
        self.dataset_map = {
            "dev": self.dev_dataset,
            "test": self.test_dataset
        }
        self.na_num = self.processor.na_num
        self.rel_num = self.processor.rel_num

    def get_predict(self, result, T1=0.5, T2=0.4):
        for i in range(len(result)):
            r = []
            maxl, maxj = -1, -1
            for j in range(len(result[i])):
                if result[i][j] > T1:
                    r += [j]
                if result[i][j] > maxl:
                    maxl = result[i][j]
                    maxj = j
            if len(r) == 0:
                if maxl <= T2:
                    r = [36]
                else:
                    r += [maxj]
            result[i] = r
        return result

    def get_evaluate(self, devp, data):
        index = 0
        correct_sys, all_sys = 0, 0
        correct_gt = 0

        for i in range(len(data)):
            for j in range(len(data[i][1])):
                for id in data[i][1][j]["rid"]:
                    if id != 36:
                        correct_gt += 1
                        if id in devp[index]:
                            correct_sys += 1
                for id in devp[index]:
                    if id != 36:
                        all_sys += 1
                index += 1

        precision = correct_sys / all_sys if all_sys != 0 else 1
        recall = correct_sys / correct_gt if correct_gt != 0 else 0
        f_1 = 2 * precision * recall / (precision + recall) if precision + recall != 0 else 0

        return precision, recall, f_1

    def get_evaluate_f1c(self, devp, data):
        index = 0
        precisions = []
        recalls = []

        for i in range(len(data)):
            for j in range(len(data[i][1])):
                correct_sys, all_sys = 0, 0
                correct_gt = 0

                x = data[i][1][j]["x"].lower().strip()
                y = data[i][1][j]["y"].lower().strip()
                t = {}
                for k in range(len(data[i][1][j]["rid"])):
                    if data[i][1][j]["rid"][k] != 36:
                        t[data[i][1][j]["rid"][k]] = data[i][1][j]["t"][k].lower().strip()

                l = set(data[i][1][j]["rid"]) - set([36])

                ex, ey = False, False
                et = {}
                for r in range(36):
                    et[r] = r not in l

                for k in range(len(data[i][0])):
                    o = set(devp[index]) - set([36])
                    e = set()
                    if x in data[i][0][k].lower():
                        ex = True
                    if y in data[i][0][k].lower():
                        ey = True
                    if k == len(data[i][0]) - 1:
                        ex = ey = True
                        for r in range(36):
                            et[r] = True
                    for r in range(36):
                        if r in t:
                            if t[r] != "" and t[r] in data[i][0][k].lower():
                                et[r] = True
                        if ex and ey and et[r]:
                            e.add(r)
                    correct_sys += len(o & l & e)
                    all_sys += len(o & e)
                    correct_gt += len(l & e)
                    index += 1

                precisions += [correct_sys / all_sys if all_sys != 0 else 1]
                recalls += [correct_sys / correct_gt if correct_gt != 0 else 0]

        precision = sum(precisions) / len(precisions)
        recall = sum(recalls) / len(recalls)
        f_1 = 2 * precision * recall / (precision + recall) if precision + recall != 0 else 0

        return precision, recall, f_1

    def evaluate(self, model, epoch, typ='dev'):
        logger.info("Starting Evaluation {}".format(typ))
        model.eval()
        if self.args.conversational_setting:
            qualitative_path = os.path.join(self.args.save_dirpath, 'f1c', self.args.task)
        else:
            qualitative_path = os.path.join(self.args.save_dirpath, self.args.task)
        os.makedirs(qualitative_path, exist_ok=True)
        with open(os.path.join(qualitative_path, f'qualitative_results_{epoch}.csv'), 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
            tqdm_batch_iterator = tqdm(self.data_map[typ])

            best_p, best_r, best_f = -1., -1., -1.
            best_t = 0.
            input_ids_list = []
            rel_logits_list = []
            tr_logits_list = []
            rel_labels_list = []
            tr_labels_list = []
            preds_list = []
            attentions_list = []
            inputs_list = []
            for batch_idx, batch in enumerate(tqdm_batch_iterator):
                with torch.no_grad():
                    dial_id = batch.pop("dial_id")
                    rel_id = batch.pop("rel_id")
                    labels = batch.pop("labels")
                    if "tr_labels" in batch:
                        tr_labels = batch.pop("tr_labels")
                    else:
                        tr_labels = None
                    for b_k in batch:
                        if b_k in ["input_ids", "token_type_ids", "attention_mask"]:
                            batch[b_k] = batch[b_k].to(self.device)
                    rel_logits, tr_logits, _ = model.inference(**batch)
                    rel_logits = rel_logits.detach().cpu().numpy().tolist()
                    labels = labels.numpy().tolist()
                    if tr_logits is not None:
                        tr_logits = tr_logits.detach().cpu()
                        tr_logits_list += torch.argmax(tr_logits, dim=-1).numpy().tolist()  # [bsz, seq]
                        tr_labels_list += tr_labels.numpy().tolist()  # [bsz, seq]

                    rel_labels_list += labels
                    rel_logits_list += rel_logits

                    detached_inputs = batch["input_ids"].detach().cpu().numpy().tolist()
                    input_ids_list += detached_inputs
                    inputs_list += self.tokenizer.batch_decode(detached_inputs)

            rel_logits_list = np.asarray(rel_logits_list)
            rel_logits_list = list(1 / (1 + np.exp(-rel_logits_list)))  # Softmax
            with open(os.path.join(self.args.data_dir, f"{typ}.json"), "r", encoding="utf8") as f:
                dataeval = json.load(f)
                for i in range(len(dataeval)):
                    for j in range(len(dataeval[i][1])):
                        for k in range(len(dataeval[i][1][j]["rid"])):
                            dataeval[i][1][j]["rid"][k] -= 1
            for i in range(0, 51):
                preds_list = self.get_predict(copy.deepcopy(rel_logits_list), T2=i / 100.0)
                if not self.args.conversational_setting:
                    precision, recall, f_1 = self.get_evaluate(preds_list, dataeval)
                else:
                    precision, recall, f_1 = self.get_evaluate_f1c(preds_list, dataeval)
                if f_1 > best_f:
                    best_f, best_p, best_r = f_1, precision, recall
                    best_t = i

            tr_f1_micro = None
            trigger_f1 = None
            if len(tr_logits_list) != 0:
                true_preds, true_labels = model.get_true_predictions(
                    input_ids_list=input_ids_list,
                    tr_logits_list=tr_logits_list,
                    tr_labels_list=tr_labels_list,
                )
                tr_f1_micro = f1_score(y_true=true_labels, y_pred=true_preds, average='micro')
                report = classification_report(y_true=true_labels, y_pred=true_preds, output_dict=True)
                logger.info(report)
                trigger_f1 = report['[trigger]']['f1-score']
                logger.info("{} RCD F1 : {:2.5f} Trigger F1 : {:2.5f}".format(typ, tr_f1_micro, trigger_f1))

            for id in range(len(rel_labels_list)):
                csvwriter.writerow(
                    [
                        preds_list[id],
                        rel_labels_list[id],
                        self.tokenizer.convert_ids_to_tokens(input_ids_list[id]),
                        tr_logits_list[id] if len(tr_logits_list) != 0 else None,
                        tr_labels_list[id] if len(tr_labels_list) != 0 else None,
                    ]
                )
        logger.info('%s Precision: %2.5f | Recall: %2.5f | F1 : %2.5f' % (typ, best_p, best_r, best_f))

        return best_p, best_r, best_f, tr_f1_micro, trigger_f1
