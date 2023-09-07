import argparse
import glob
import json
import logging
import numpy as np
import os
import pickle
import sys
import torch
import torch.nn as nn
from attrdict import AttrDict
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from setproctitle import setproctitle
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AdamW, get_linear_schedule_with_warmup

from evaluate import Evaluation
from utils.eval_utils import CheckpointManager, load_checkpoint
from utils.general_utils import (
    TOKENIZER_CLASSES,
    ADVISOR_DATASET_CLASSES,
    DATA_PROCESSORS,
    ADVISOR_MODEL_PROCESSORS,
    TYPE_WORD_RATIOS,
    MODEL_CLASSES,
)


class GRASPTrainer(object):

    def __init__(self, args):
        self.args = args
        setproctitle(self.args.process_name)
        timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        if self.args.save_dirpath == 'checkpoints/':
            self.save_dirpath = os.path.join(self.args.root_dirpath, self.args.task, self.args.model_type,
                                             "%s/" % timestamp, self.args.save_dirpath)
        else:
            self.save_dirpath = self.args.save_dirpath
        if not os.path.exists(self.save_dirpath):
            os.makedirs(self.save_dirpath, exist_ok=True)
        formatter = logging.Formatter('[%(asctime)s - %(levelname)s - %(name)s]  >>  %(message)s')
        file_handler = logging.FileHandler(self.save_dirpath + "/log")
        file_handler.setFormatter(formatter)
        logging.basicConfig(
            format='[%(asctime)s - %(levelname)s - %(name)s]  >>  %(message)s',
            datefmt='%m/%d/%Y %H:%M:%S',
            level=logging.INFO
        )
        logging.root.addHandler(file_handler)
        self._logger = logging.getLogger()

        np.random.seed(args.random_seed)
        torch.manual_seed(args.random_seed)
        torch.cuda.manual_seed_all(args.random_seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

        self.device = torch.device("cuda:" + str(self.args.gpu_ids[0]) if torch.cuda.is_available() else "cpu")

        self._logger.info("Training/evaluation parameters {}".format(self.args))
        self.build_dataloader()
        self.build_model()
        self.setup_training()
        self.evaluation = Evaluation(args, self.tokenizer, self.processor, self.model_processor)

    def build_dataloader(self):
        # =============================================================================
        #   SETUP DATASET, DATALOADER
        # =============================================================================
        self.tokenizer = TOKENIZER_CLASSES[args.model].from_pretrained(args.bert_model)
        self.original_vocab_size = deepcopy(len(self.tokenizer))

        self.processor = DATA_PROCESSORS[self.args.task](self.args)
        self.model_processor = ADVISOR_MODEL_PROCESSORS[self.args.model](self.args, self.tokenizer, self.processor, 4)
        self.train_dataset = ADVISOR_DATASET_CLASSES[self.args.model](self.args, self.tokenizer, self.processor,
                                                                      self.model_processor, mode="train")
        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.args.batch_size,
            num_workers=self.args.cpu_workers,
            shuffle=True,
            drop_last=True,
            collate_fn=self.train_dataset.collate_fn
        )

        self.train_dataloader_advisor = DataLoader(
            self.train_dataset,
            batch_size=self.args.batch_size,
            num_workers=self.args.cpu_workers,
            collate_fn=self.train_dataset.collate_fn
        )

        self._logger.info(
            """
            # -------------------------------------------------------------------------
            #   BUILD DATALOADER DONE
            # -------------------------------------------------------------------------
            """
        )

    def build_model(self):
        self.model = MODEL_CLASSES[args.model](self.args, self.tokenizer, self.device, self.model_processor)
        self.model.to(self.device)

        # =============================================================================
        #   vocabulary resize for the added prompt/tokens and initialize them
        # =============================================================================
        self.model.bert_model.resize_token_embeddings(len(self.tokenizer))
        resized_vocab_size = deepcopy(len(self.tokenizer))
        self._logger.info("original_vocab_size: {}, resized_vocab_size: {}".format(
            self.original_vocab_size, resized_vocab_size)
        )
        assert self.original_vocab_size != resized_vocab_size

        if self.args.manual_prompt_init:
            rel_ids_list = self._init_prompt_embeddings()
        else:
            rel_ids_list = [
                a[0] for a in self.tokenizer(self.model_processor.class_list, add_special_tokens=False)['input_ids']
            ]

        tr_class_ids_list = self.model_processor.get_tr_class_ids_list()
        self.model.prepare_verbalizer(rel_ids_list, tr_class_ids_list)

        # Use Multi-GPUs
        if -1 not in self.args.gpu_ids and len(self.args.gpu_ids) > 1:
            self.model = nn.DataParallel(self.model, self.args.gpu_ids)

        self._logger.info(
            """
            # -------------------------------------------------------------------------
            #   BUILD MODEL DONE
            # -------------------------------------------------------------------------
            """
        )

    def setup_training(self):
        # =============================================================================
        #   optimizer / scheduler
        # =============================================================================

        self.iterations = len(self.train_dataset) // self.args.virtual_batch_size

        no_decay = ['bias', 'LayerNorm.weight']
        param_optimizer = self.model.named_parameters()
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay': self.args.weight_decay},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}
        ]
        self.optimizer = AdamW(
            optimizer_grouped_parameters, lr=self.args.learning_rate, eps=self.args.adam_epsilon
        )

        if self.args.scheduler == "lambda":
            lr_lambda = lambda epoch: self.args.lr_decay ** epoch
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_lambda)
        elif self.args.scheduler == "warmup":
            num_training_steps = self.iterations * self.args.num_epochs
            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer, num_warmup_steps=self.args.warmup_steps,
                num_training_steps=num_training_steps
            )

        # =============================================================================
        #   checkpoint_manager / tensorboard summary_writer
        # =============================================================================
        timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')

        tb_prefix = self.args.model_type + "_" + self.args.task_desc
        self.tb_writer = SummaryWriter(log_dir=f'./runs/{self.args.task}/' + timestamp + tb_prefix, comment=tb_prefix)

        self.checkpoint_manager = CheckpointManager(self.model, self.optimizer, self.save_dirpath)

        # If loading from checkpoint, adjust start epoch and load parameters.X
        if self.args.load_pthpath == "":
            self.start_epoch = 1
        else:
            # "path/to/checkpoint_xx.pth" -> xx
            self.start_epoch = int(self.args.load_pthpath.split("_")[-1][:-4])
            self.start_epoch += 1
            model_state_dict, optimizer_state_dict = load_checkpoint(
                self.args.load_pthpath,
                self.device,
            )
            if isinstance(self.model, nn.DataParallel):
                self.model.module.load_state_dict(model_state_dict)
            else:
                self.model.load_state_dict(model_state_dict)
            self.optimizer.load_state_dict(optimizer_state_dict)
            self.previous_model_path = self.args.load_pthpath
            self._logger.info("Loaded model from {}".format(self.args.load_pthpath))

        self._logger.info(
            """
            # -------------------------------------------------------------------------
            #   SETUP TRAINING DONE
            # -------------------------------------------------------------------------
            """
        )

    def train(self):

        start_time = datetime.now().strftime('%H:%M:%S')
        self._logger.info("Start train model at {}".format(start_time))

        train_begin = datetime.utcnow()
        global_iteration_step = 0
        early_stop_count = self.args.early_stop_count
        best_dev_f1 = -1.0

        # evaluation results writer
        self.eval_result_writer = open(os.path.join(self.checkpoint_manager.ckpt_dirpath, "eval_results.txt"), "w",
                                       encoding='utf-8')
        self._logger.info("Saving Evaluation Results in {}".format(
            os.path.join(self.checkpoint_manager.ckpt_dirpath, "eval_results.txt")))

        self._logger.info("Total number of iterations per epoch: {}".format(self.iterations))
        self._logger.info("Start Training...")

        for epoch in range(self.start_epoch, self.args.num_epochs + 1):
            accu_loss, accu_cnt = 0, 0

            self.model.train()
            tqdm_batch_iterator = tqdm(self.train_dataloader)
            accu_batch = 0

            for batch_idx, batch in enumerate(tqdm_batch_iterator):
                dial_id = batch.pop("dial_id")
                rel_id = batch.pop("rel_id")
                for b_k in batch:
                    if b_k in ["input_ids", "token_type_ids", "attention_mask", "labels", "tr_labels"]:
                        batch[b_k] = batch[b_k].to(self.device)

                loss, _ = self.model(**batch)
                accu_loss += loss.item()
                accu_cnt += 1
                accu_batch += batch["input_ids"].shape[0]

                if (self.args.virtual_batch_size == accu_batch) or (
                        batch_idx == (len(self.train_dataset) // self.args.batch_size)):  # last batch

                    nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_gradient_norm)
                    self.optimizer.step()
                    if self.args.scheduler == "warmup":
                        self.scheduler.step()
                    self.optimizer.zero_grad()

                    accu_batch = 0
                    global_iteration_step += 1
                    description = "[{}][Epoch: {:3d}][Iter: {:6d}][Loss: {:6f}][lr: {:7f}]".format(
                        datetime.utcnow() - train_begin,
                        epoch,
                        global_iteration_step, accu_loss / accu_cnt,
                        self.optimizer.param_groups[0]['lr'])

                    tqdm_batch_iterator.set_description(description)

                    self.tb_writer.add_scalar('Training_loss', accu_loss / accu_cnt, global_iteration_step)

            # -------------------------------------------------------------------------
            #   ON EPOCH END  (checkpointing and validation)
            # -------------------------------------------------------------------------
            self._logger.info("Evaluation after {} epoch".format(epoch))
            self._logger.info('Start Evaluating...')
            dev_precision, dev_recall, dev_f_1, dev_rcd_f1, dev_tr_f1 = self.evaluation.evaluate(
                model=self.model,
                epoch=epoch,
                typ='dev',
            )

            self.eval_result_writer.write(
                f"dev {epoch} | dev precision: {dev_precision} | dev recall: {dev_recall} | dev f1: {dev_f_1} | dev rcd f1: {dev_rcd_f1} | dev Trigger f1: {dev_tr_f1}\n")
            self.tb_writer.add_scalar('Dev_Precision', dev_precision, epoch)
            self.tb_writer.add_scalar('Dev_Recall', dev_recall, epoch)
            self.tb_writer.add_scalar('Dev_F1', dev_f_1, epoch)
            if dev_rcd_f1 is not None:
                self.tb_writer.add_scalar('Dev_rcd_F1', dev_rcd_f1, epoch)
                self.tb_writer.add_scalar('Dev_rcd_Trigger_F1', dev_tr_f1, epoch)

            if dev_f_1 > best_dev_f1:
                # remove previous checkpoint
                for ckpt in glob.glob(os.path.join(self.checkpoint_manager.ckpt_dirpath, "*.pth")):
                    os.remove(ckpt)

                self.checkpoint_manager.step(epoch)
                self.previous_model_path = os.path.join(self.checkpoint_manager.ckpt_dirpath,
                                                        "checkpoint_%d.pth" % (epoch))

                self._logger.info(self.previous_model_path)
                best_dev_f1 = dev_f_1
                early_stop_count = self.args.early_stop_count  # reset early stop count

            else:
                if self.args.scheduler == "lambda":
                    self.scheduler.step()  # learning rate decay

                # early stopping
                early_stop_count -= 1
                if early_stop_count == 0:
                    break

        model_state_dict, optimizer_state_dict = load_checkpoint(
            glob.glob(os.path.join(self.checkpoint_manager.ckpt_dirpath, "*.pth"))[0],
            self.device
        )
        if isinstance(self.model, nn.DataParallel):
            self.model.module.load_state_dict(model_state_dict)
        else:
            self.model.load_state_dict(model_state_dict)

    def evaluate(self):
        start_time = datetime.now().strftime('%H:%M:%S')
        if self.args.conversational_setting:
            eval_result_path = os.path.join(self.checkpoint_manager.ckpt_dirpath, "f1c")
            if not os.path.exists(eval_result_path):
                os.makedirs(eval_result_path, exist_ok=True)
        else:
            eval_result_path = self.checkpoint_manager.ckpt_dirpath
        self.eval_result_writer = open(
            os.path.join(eval_result_path, "eval_results.txt"), "w", encoding='utf-8'
        )
        self._logger.info("Start evaluating model at %s".format(start_time))
        self._logger.info('Start Evaluating...')
        self._logger.info(glob.glob(os.path.join(self.checkpoint_manager.ckpt_dirpath, "*.pth")))
        model_state_dict, optimizer_state_dict = load_checkpoint(
            glob.glob(os.path.join(self.checkpoint_manager.ckpt_dirpath, "*.pth"))[0],
            self.device
        )
        if isinstance(self.model, nn.DataParallel):
            self.model.module.load_state_dict(model_state_dict)
        else:
            self.model.load_state_dict(model_state_dict)

        dev_precision, dev_recall, dev_f_1, dev_rcd_f, dev_tr_f1 = self.evaluation.evaluate(
            self.model,
            epoch="dev",
            typ="dev",
        )
        test_precision, test_recall, test_f_1, test_rcd_f, test_tr_f1 = self.evaluation.evaluate(
            self.model,
            epoch="test",
            typ="test",
        )
        self.eval_result_writer.write(
            f"dev | dev precision: {dev_precision} | dev recall: {dev_recall} | dev f1: {dev_f_1} | dev RCD f1: {dev_rcd_f} | dev Trigger f1: {dev_tr_f1}\n")
        self.eval_result_writer.write(
            f"test | test precision: {test_precision} | test recall: {test_recall} | test f1: {test_f_1} | test RCD f1: {test_rcd_f} | test Trigger f1: {test_tr_f1}\n")
        self.tb_writer.add_scalar('test_Precision', test_precision, 0)
        self.tb_writer.add_scalar('test_Recall', test_recall, 0)
        self.tb_writer.add_scalar('test_F1', test_f_1, 0)
        if test_rcd_f is not None:
            self.tb_writer.add_scalar('test_rcd_F1', test_rcd_f, 0)
            self.tb_writer.add_scalar('test_rcd_Trigger_F1', test_tr_f1, 0)

    def _init_prompt_embeddings(self):
        relation_ids_list = self.model_processor.prepare_relation_labels()
        with torch.no_grad():
            word_embeddings = self.model.bert_model.get_input_embeddings()
            continuous_relation_ids_list = [
                a[0] for a in self.tokenizer(self.model_processor.class_list, add_special_tokens=False)['input_ids']
            ]
            for i, (continuous_relation_ids, relation_ids) in enumerate(
                    zip(continuous_relation_ids_list, relation_ids_list)):
                word_embeddings.weight[continuous_relation_ids] = torch.mean(word_embeddings.weight[relation_ids],
                                                                             dim=0)

            prompt_id_list = [
                a[0] for a in self.tokenizer(self.model_processor.prompt_list, add_special_tokens=False)['input_ids']
            ]
            semantic_id_list = [
                a[0] for a in self.tokenizer(
                    list(TYPE_WORD_RATIOS.keys()), add_special_tokens=False
                )['input_ids']
            ]
            for i, (prompt_id, semantic_id) in enumerate(zip(prompt_id_list, semantic_id_list)):
                ratio_matrix = torch.tensor(list(TYPE_WORD_RATIOS.values())).unsqueeze(0)
                ratio_matrix = ratio_matrix.to(self.device)
                embedding_matrix = word_embeddings.weight[semantic_id_list].to(self.device)
                word_embeddings.weight[prompt_id] = torch.mm(ratio_matrix, embedding_matrix).squeeze()
            assert torch.equal(self.model.bert_model.get_input_embeddings().weight, word_embeddings.weight)
            assert torch.equal(self.model.bert_model.get_input_embeddings().weight,
                               self.model.bert_model.get_output_embeddings().weight)

        return continuous_relation_ids_list


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description="Multiple Choice (PyTorch)")
    arg_parser.add_argument("--config_dir", dest="config_dir", type=str, default="config", help="Config Directory")
    arg_parser.add_argument("--config_file", dest="config_file", type=str, default="bert-base-cased",
                            help="Config json file")
    arg_parser.add_argument("--task_desc", type=str, default=None, help="Task Description")
    arg_parser.add_argument("--task", dest="task", type=str, default=None, help="Task")
    arg_parser.add_argument("--process_name", dest="process_name", type=str, default="DRE", help="process_name")
    arg_parser.add_argument("--mode", dest="mode", type=str, default="train")
    arg_parser.add_argument("--max_seq_len", dest="maximum sequence length", type=int, default=512)
    arg_parser.add_argument("--gpu_ids", dest="gpu_ids", type=str, default=None)
    arg_parser.add_argument("--random_seed", type=int, default=None)
    arg_parser.add_argument("--data_dir", dest="data_dir", type=str, default=None)
    arg_parser.add_argument("--save_dirpath", dest="save_dirpath", type=str, default="", help="Save path")
    arg_parser.add_argument("--root_dirpath", dest="root_dirpath", type=str, default="", help="Root directory path")
    arg_parser.add_argument("--load_pthpath", dest="load_pthpath", type=str, default="", help="Checkpoint path")
    arg_parser.add_argument("--tr_loss_ratio", type=float, default=None, help="Trigger loss ratio")
    arg_parser.add_argument("--rel_loss_ratio", type=float, default=None, help="Relation loss ratio")
    arg_parser.add_argument("--batch_size", type=int, default=None, help="batch_size")
    arg_parser.add_argument("--valid_batch_size", type=int, default=None, help="batch_size")
    arg_parser.add_argument("--virtual_batch_size", type=int, default=None, help="Virtual batch size")
    arg_parser.add_argument("--dropout", type=float, default=None, help="dropout")
    arg_parser.add_argument("--early_stop_count", type=int, default=None, help="early_stop_count")
    arg_parser.add_argument(
        "--conversational_setting",
        action='store_true',
        default=None,
        help="Whether you want to enable the model to a conversational setting."
    )

    parsed_args = arg_parser.parse_args()

    with open(os.path.join(parsed_args.config_dir, "{}.json".format(parsed_args.config_file))) as f:
        args = AttrDict(json.load(f))
        args.update({"root_dirpath": parsed_args.root_dirpath})
        args.update({"load_pthpath": parsed_args.load_pthpath})
        args.update({"save_dirpath": parsed_args.save_dirpath})
        if parsed_args.task is not None:  # dre, tacred, docred
            args.update({"task": parsed_args.task})
        if parsed_args.task_desc is not None:
            args.update({"task_desc": parsed_args.task_desc})
        if parsed_args.gpu_ids is not None:
            args.update({"gpu_ids": [parsed_args.gpu_ids]})
        if parsed_args.data_dir is not None:
            args.update({"data_dir": parsed_args.data_dir})
        if parsed_args.process_name is not None:
            args.update({"process_name": parsed_args.process_name})
        if parsed_args.mode is not None:
            args.update({"mode": parsed_args.mode})
        if parsed_args.tr_loss_ratio is not None:
            args.update({"tr_loss_ratio": parsed_args.tr_loss_ratio})
        if parsed_args.rel_loss_ratio is not None:
            args.update({"rel_loss_ratio": parsed_args.rel_loss_ratio})
        if parsed_args.batch_size is not None:
            args.update({"batch_size": parsed_args.batch_size})
        if parsed_args.valid_batch_size is not None:
            args.update({"valid_batch_size": parsed_args.valid_batch_size})
        if parsed_args.virtual_batch_size is not None:
            args.update({"virtual_batch_size": parsed_args.virtual_batch_size})
        if parsed_args.random_seed is not None:
            args.update({"random_seed": parsed_args.random_seed})
        if parsed_args.conversational_setting is not None:
            logging.info("*** Conversational setting is enabled! ***")
            args.update({"conversational_setting": parsed_args.conversational_setting})
        if parsed_args.early_stop_count is not None:
            args.update({"early_stop_count": parsed_args.early_stop_count})
    # print("Training/evaluation parameters {}".format(args))
    trainer = GRASPTrainer(args)
    if parsed_args.mode == 'train':
        trainer.train()
    else:
        trainer.evaluate()
