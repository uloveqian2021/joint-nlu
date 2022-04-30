import os
import torch
import logging
import argparse
import numpy as np
import torch.nn as nn
from tqdm import tqdm, trange
import torch.nn.functional as F
from optimization import BertAdam
from transformers import BertTokenizer
from data import load_and_cache_examples
from torch.nn import CrossEntropyLoss, MSELoss
from models import JointBERT, JointTinyBert, JointTinyBert2
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from utils import compute_metrics, get_intent_labels, get_slot_labels, init_logger, set_seed

init_logger()
logger = logging.getLogger(__name__)


def soft_cross_entropy(predicts, targets):
    student_likelihood = F.log_softmax(predicts, dim=-1)
    targets_prob = F.softmax(targets, dim=-1)
    return (- targets_prob * student_likelihood).mean()


class Trainer(object):
    def __init__(self, args, train_dataset=None, dev_dataset=None, test_dataset=None):
        self.args = args
        self.train_dataset = train_dataset
        self.dev_dataset = dev_dataset
        self.test_dataset = test_dataset

        self.intent_label_lst = get_intent_labels(args)
        self.slot_label_lst = get_slot_labels(args)

        # Use cross entropy ignore index as padding label id so that only real label ids contribute to the loss later
        self.pad_token_label_id = args.ignore_index
        self.device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"

        if args.stage == '2.0':
            self.teacher_model = JointBERT.from_pretrained(args.bert_path,
                                                           args=args,
                                                           intent_label_lst=self.intent_label_lst,
                                                           slot_label_lst=self.slot_label_lst)
        else:
            self.teacher_model = JointTinyBert.from_pretrained(args.teacher_model,
                                                               args=args,
                                                               intent_label_lst=self.intent_label_lst,
                                                               slot_label_lst=self.slot_label_lst)
            if args.stage == '2.1':
                self.student_model = JointTinyBert2.from_pretrained2(args.tinybert, args, self.intent_label_lst,
                                                                     self.slot_label_lst)
            elif args.stage == '2.2':
                self.student_model = JointTinyBert2.from_pretrained(args.student_model, args, self.intent_label_lst,
                                                                    self.slot_label_lst)
            self.student_model.to(self.device)
        self.teacher_model.to(self.device)

    def train(self):
        train_sampler = RandomSampler(self.train_dataset)
        train_dataloader = DataLoader(self.train_dataset, sampler=train_sampler, batch_size=self.args.train_batch_size)

        if self.args.max_steps > 0:
            t_total = self.args.max_steps
            self.args.num_train_epochs = self.args.max_steps // \
                                         (len(train_dataloader) // self.args.gradient_accumulation_steps) + 1
        else:
            t_total = len(train_dataloader) // self.args.gradient_accumulation_steps * self.args.num_train_epochs

        # Train!
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(self.train_dataset))
        logger.info("  Num Epochs = %d", self.args.num_train_epochs)
        logger.info("  Total train batch size = %d", self.args.train_batch_size)
        logger.info("  Gradient Accumulation steps = %d", self.args.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", t_total)
        logger.info("  Logging steps = %d", self.args.logging_steps)
        logger.info("  Save steps = %d", self.args.save_steps)

        if self.args.stage == '2.0':
            # Prepare optimizer and schedule (linear warmup and decay)
            no_decay = ['bias', 'LayerNorm.weight']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in self.teacher_model.named_parameters() if not any(nd in n for nd in no_decay)],
                 'weight_decay': self.args.weight_decay},
                {'params': [p for n, p in self.teacher_model.named_parameters() if any(nd in n for nd in no_decay)],
                 'weight_decay': 0.0}]
            schedule = 'warmup_linear'
            optimizer = BertAdam(optimizer_grouped_parameters,
                                 schedule=schedule,
                                 lr=self.args.learning_rate,
                                 warmup=self.args.warmup_proportion,
                                 t_total=t_total)

            global_step = 0
            tr_loss = 0.0
            best_dev_acc = 0.0
            self.teacher_model.zero_grad()

            train_iterator = trange(int(self.args.num_train_epochs), desc="Epoch")
            for _ in train_iterator:
                epoch_iterator = tqdm(train_dataloader, desc="Iteration")
                for step, batch in enumerate(epoch_iterator):
                    self.teacher_model.train()
                    batch = tuple(t.to(self.device) for t in batch)  # GPU or CPU

                    inputs = {'input_ids': batch[0],
                              'attention_mask': batch[1],
                              'intent_label_ids': batch[3],
                              'slot_labels_ids': batch[4]}
                    if self.args.model_type != 'distilbert':
                        inputs['token_type_ids'] = batch[2]
                    outputs = self.teacher_model(**inputs)
                    loss = outputs[0]

                    if self.args.gradient_accumulation_steps > 1:
                        loss = loss / self.args.gradient_accumulation_steps

                    loss.backward()

                    tr_loss += loss.item()
                    if (step + 1) % self.args.gradient_accumulation_steps == 0:
                        torch.nn.utils.clip_grad_norm_(self.teacher_model.parameters(), self.args.max_grad_norm)

                        optimizer.step()
                        self.teacher_model.zero_grad()
                        global_step += 1

                        if self.args.logging_steps > 0 and global_step % self.args.logging_steps == 0:
                            result = self.evaluate("dev", self.teacher_model)

                            save_model = False
                            if result['slot_f1'] > best_dev_acc:
                                best_dev_acc = result['slot_f1']
                                save_model = True

                            if save_model:
                                print("***** Save model *****")
                                self.save_model(self.teacher_model)

                    if 0 < self.args.max_steps < global_step:
                        epoch_iterator.close()
                        break

                if 0 < self.args.max_steps < global_step:
                    train_iterator.close()
                    break
        else:
            # Prepare optimizer and schedule (linear warmup and decay)
            param_optimizer = list(self.student_model.named_parameters())
            no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                 'weight_decay': 0.01},
                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]

            schedule = 'warmup_linear'
            if not self.args.pred_distill:
                schedule = 'none'
            optimizer = BertAdam(optimizer_grouped_parameters,
                                 schedule=schedule,
                                 lr=self.args.learning_rate,
                                 warmup=self.args.warmup_proportion,
                                 t_total=t_total)

            loss_mse = MSELoss()
            global_step = 0
            best_dev_acc = 0.0
            output_eval_file = os.path.join(self.args.output_dir, "eval_results.txt")

            for epoch_ in trange(int(self.args.num_train_epochs), desc="Epoch"):
                tr_loss = 0.
                tr_att_loss = 0.
                tr_rep_loss = 0.
                tr_cls_loss = 0.

                self.student_model.train()
                nb_tr_examples, nb_tr_steps = 0, 0
                for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration", ascii=True)):
                    batch = tuple(t.to(self.device) for t in batch)  # GPU or CPU

                    input_ids, input_mask, segment_ids, intent_label_ids, slot_labels_ids = batch
                    if input_ids.size()[0] != self.args.train_batch_size:
                        continue
                    att_loss = 0.
                    rep_loss = 0.

                    student_intent_logits, student_slot_logits, \
                    student_atts, student_reps = self.student_model(input_ids, segment_ids, input_mask, is_student=True)
                    with torch.no_grad():
                        teacher_intent_logits, teacher_slot_logits, teacher_atts, teacher_reps = self.teacher_model(
                            input_ids, segment_ids, input_mask)

                    if not self.args.pred_distill:
                        teacher_layer_num = len(teacher_atts)
                        student_layer_num = len(student_atts)
                        assert teacher_layer_num % student_layer_num == 0
                        layers_per_block = int(teacher_layer_num / student_layer_num)
                        new_teacher_atts = [teacher_atts[i * layers_per_block + layers_per_block - 1]
                                            for i in range(student_layer_num)]

                        for student_att, teacher_att in zip(student_atts, new_teacher_atts):
                            student_att = torch.where(student_att <= -1e2,
                                                      torch.zeros_like(student_att).to(self.device),
                                                      student_att)
                            teacher_att = torch.where(teacher_att <= -1e2,
                                                      torch.zeros_like(teacher_att).to(self.device),
                                                      teacher_att)

                            tmp_loss = loss_mse(student_att, teacher_att)
                            att_loss += tmp_loss

                        new_teacher_reps = [teacher_reps[i * layers_per_block] for i in range(student_layer_num + 1)]
                        new_student_reps = student_reps
                        for student_rep, teacher_rep in zip(new_student_reps, new_teacher_reps):
                            tmp_loss = loss_mse(student_rep, teacher_rep)
                            rep_loss += tmp_loss

                        loss = rep_loss + att_loss
                        tr_att_loss += att_loss.item()
                        tr_rep_loss += rep_loss.item()
                    else:
                        cls_intent_loss = soft_cross_entropy(student_intent_logits / self.args.temperature,
                                                             teacher_intent_logits / self.args.temperature)

                        active_loss = input_mask.view(-1) == 1
                        active_student_logits = student_slot_logits.view(-1, self.student_model.num_slot_labels)[
                            active_loss]
                        active_teacher_logits = teacher_slot_logits.view(-1, self.student_model.num_slot_labels)[
                            active_loss]

                        cls_slot_loss = soft_cross_entropy(active_student_logits / self.args.temperature,
                                                           active_teacher_logits / self.args.temperature)

                        loss = cls_intent_loss + cls_slot_loss
                        tr_cls_loss += cls_intent_loss.item() + cls_slot_loss.item()

                    loss.backward()

                    tr_loss += loss.item()
                    nb_tr_examples += intent_label_ids.size(0)
                    nb_tr_steps += 1

                    if (step + 1) % self.args.gradient_accumulation_steps == 0:
                        optimizer.step()
                        # scheduler.step()  # Update learning rate schedule
                        optimizer.zero_grad()
                        global_step += 1

                    if (global_step + 1) % self.args.eval_step == 0:
                        logger.info("***** Running evaluation *****")
                        logger.info("  Epoch = {} iter {} step".format(epoch_, global_step))
                        logger.info("  Batch size = %d", self.args.eval_batch_size)

                        self.student_model.eval()

                        loss = tr_loss / (step + 1)
                        cls_loss = tr_cls_loss / (step + 1)
                        att_loss = tr_att_loss / (step + 1)
                        rep_loss = tr_rep_loss / (step + 1)

                        result = {}
                        if self.args.pred_distill:
                            result = self.evaluate("dev", self.student_model)
                        result['global_step'] = global_step
                        result['cls_loss'] = cls_loss
                        result['att_loss'] = att_loss
                        result['rep_loss'] = rep_loss
                        result['loss'] = loss
                        for key in sorted(result.keys()):
                            print("  %s = %s", key, str(result[key]))

                        if not self.args.pred_distill:
                            save_model = True
                        else:
                            save_model = False

                            if result['slot_f1'] > best_dev_acc:
                                best_dev_acc = result['slot_f1']
                                save_model = True

                        if save_model:
                            logger.info("***** Save model *****")
                            self.save_model(self.student_model)

                        self.student_model.train()

    def evaluate(self, mode, model):
        if mode == 'test':
            dataset = self.test_dataset
        elif mode == 'dev':
            dataset = self.dev_dataset
        else:
            raise Exception("Only dev and test dataset available")

        eval_sampler = SequentialSampler(dataset)
        eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=self.args.eval_batch_size)

        # Eval!
        logger.info("***** Running evaluation on %s dataset *****", mode)
        logger.info("  Num examples = %d", len(dataset))
        logger.info("  Batch size = %d", self.args.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        intent_preds = None
        slot_preds = None
        out_intent_label_ids = None
        out_slot_labels_ids = None

        model.eval()

        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            batch = tuple(t.to(self.device) for t in batch)
            with torch.no_grad():
                if self.args.stage == '2.0':
                    inputs = {'input_ids': batch[0],
                              'attention_mask': batch[1],
                              'intent_label_ids': batch[3],
                              'slot_labels_ids': batch[4]}
                    if self.args.model_type != 'distilbert':
                        inputs['token_type_ids'] = batch[2]
                    outputs = model(**inputs)
                    tmp_eval_loss, (intent_logits, slot_logits) = outputs[:2]

                    eval_loss += tmp_eval_loss.mean().item()
                else:
                    input_ids, input_mask, segment_ids, intent_label_ids, slot_labels_ids = batch
                    intent_logits, slot_logits, _, _ = model(input_ids, segment_ids, input_mask)
                    loss_fct = CrossEntropyLoss()
                    slot_loss_fct = nn.CrossEntropyLoss(ignore_index=self.args.ignore_index)
                    tmp_intent_eval_loss = loss_fct(intent_logits.view(-1, model.num_intent_labels),
                                                    intent_label_ids.view(-1))
                    if input_mask is not None:
                        active_loss = input_mask.view(-1) == 1
                        active_logits = slot_logits.view(-1, model.num_slot_labels)[active_loss]
                        active_labels = slot_labels_ids.view(-1)[active_loss]
                        slot_loss = slot_loss_fct(active_logits, active_labels)
                    else:
                        slot_loss = slot_loss_fct(slot_logits.view(-1, model.num_slot_labels),
                                                  slot_labels_ids.view(-1))

                    eval_loss += tmp_intent_eval_loss.mean().item()
                    eval_loss += slot_loss.mean().item()

            nb_eval_steps += 1

            if self.args.stage == '2.0':
                # Intent prediction
                if intent_preds is None:
                    intent_preds = intent_logits.detach().cpu().numpy()
                    out_intent_label_ids = inputs['intent_label_ids'].detach().cpu().numpy()
                else:
                    intent_preds = np.append(intent_preds, intent_logits.detach().cpu().numpy(), axis=0)
                    out_intent_label_ids = np.append(
                        out_intent_label_ids, inputs['intent_label_ids'].detach().cpu().numpy(), axis=0)

                # Slot prediction
                if slot_preds is None:
                    slot_preds = slot_logits.detach().cpu().numpy()
                    out_slot_labels_ids = inputs["slot_labels_ids"].detach().cpu().numpy()
                else:
                    slot_preds = np.append(slot_preds, slot_logits.detach().cpu().numpy(), axis=0)
                    out_slot_labels_ids = np.append(out_slot_labels_ids,
                                                    inputs["slot_labels_ids"].detach().cpu().numpy(), axis=0)
            else:
                # Intent prediction
                if intent_preds is None:
                    intent_preds = intent_logits.detach().cpu().numpy()
                    out_intent_label_ids = intent_label_ids.detach().cpu().numpy()
                else:
                    intent_preds = np.append(intent_preds, intent_logits.detach().cpu().numpy(), axis=0)
                    out_intent_label_ids = np.append(
                        out_intent_label_ids, intent_label_ids.detach().cpu().numpy(), axis=0)

                # Slot prediction
                if slot_preds is None:
                    slot_preds = slot_logits.detach().cpu().numpy()
                    out_slot_labels_ids = slot_labels_ids.detach().cpu().numpy()
                else:
                    slot_preds = np.append(slot_preds, slot_logits.detach().cpu().numpy(), axis=0)
                    out_slot_labels_ids = np.append(
                        out_slot_labels_ids, slot_labels_ids.detach().cpu().numpy(), axis=0)

        eval_loss = eval_loss / nb_eval_steps
        results = {"loss": eval_loss}

        # Intent result
        intent_preds = np.argmax(intent_preds, axis=1)

        # Slot result
        slot_preds = np.argmax(slot_preds, axis=2)
        slot_label_map = {i: label for i, label in enumerate(self.slot_label_lst)}
        out_slot_label_list = [[] for _ in range(out_slot_labels_ids.shape[0])]
        slot_preds_list = [[] for _ in range(out_slot_labels_ids.shape[0])]

        if self.args.stage == '2.0':
            for i in range(out_slot_labels_ids.shape[0]):
                for j in range(out_slot_labels_ids.shape[1]):
                    if out_slot_labels_ids[i, j] != self.pad_token_label_id:
                        if out_slot_labels_ids[i, j] == 2:
                            if out_slot_labels_ids[i, j] != slot_preds[i, j]:
                                out_slot_label_list[i].append(slot_label_map[out_slot_labels_ids[i][j]])
                                slot_preds_list[i].append(slot_label_map[slot_preds[i][j]])
                        else:
                            out_slot_label_list[i].append(slot_label_map[out_slot_labels_ids[i][j]])
                            slot_preds_list[i].append(slot_label_map[slot_preds[i][j]])
        else:
            for i in range(out_slot_labels_ids.shape[0]):
                for j in range(out_slot_labels_ids.shape[1]):
                    if out_slot_labels_ids[i, j] != self.pad_token_label_id:
                        out_slot_label_list[i].append(slot_label_map[out_slot_labels_ids[i][j]])
                        slot_preds_list[i].append(slot_label_map[slot_preds[i][j]])

        total_result = compute_metrics(intent_preds, out_intent_label_ids, slot_preds_list, out_slot_label_list)
        results.update(total_result)

        logger.info("***** Eval results *****")
        for key in sorted(results.keys()):
            logger.info("  %s = %s", key, str(results[key]))

        return results

    def save_model(self, model):
        # Save model checkpoint (Overwrite)
        if not os.path.exists(self.args.output_dir):
            os.makedirs(self.args.output_dir)
        output_model_file = os.path.join(self.args.output_dir, 'pytorch_model.bin')
        output_config_file = os.path.join(self.args.output_dir, 'config.json')
        model_to_save = model.module if hasattr(model, 'module') else model
        torch.save(model_to_save.state_dict(), output_model_file)
        model_to_save.config.to_json_file(output_config_file)

        # Save training arguments together with the trained model
        torch.save(self.args, os.path.join(self.args.output_dir, 'training_args.bin'))
        print("Saving model checkpoint to %s", self.args.output_dir)

    def load_model(self):
        # Check whether model exists
        if not os.path.exists(self.args.output_dir):
            raise Exception("Model doesn't exists! Train first!")

        try:
            if self.args.stage == '2.0':
                self.teacher_model = JointBERT.from_pretrained(self.args.output_dir,
                                                               args=self.args,
                                                               intent_label_lst=self.intent_label_lst,
                                                               slot_label_lst=self.slot_label_lst)
                self.teacher_model.to(self.device)
                logger.info("***** Model Loaded *****")
                return self.teacher_model
            else:
                self.student_model = JointTinyBert2.from_pretrained(self.args.output_dir,
                                                                    args=self.args,
                                                                    intent_label_lst=self.intent_label_lst,
                                                                    slot_label_lst=self.slot_label_lst)
                self.student_model.to(self.device)
                logger.info("***** Model Loaded *****")
                return self.student_model
        except:
            raise Exception("Some model files might be missing...")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1234, help='Random seed for initialization')
    parser.add_argument('--no_cuda', action='store_true', help='Avoid using CUDA when available')
    parser.add_argument('--task', default='snips', type=str, help='The name of the task to train')
    parser.add_argument('--stage', default='2.1', help='stage 2.0 or 2.1 or 2.2')
    parser.add_argument('--do_train', default=True, help='Whether to run training.')
    parser.add_argument('--do_eval', default=True, help='Whether to run eval on the test set.')
    parser.add_argument('--pred_distill', action='store_true', help='Whether to distill prediction')
    parser.add_argument('--model_type', default='bert', type=str)

    parser.add_argument('--data_dir', default='./data', type=str, help='The input data dir')
    parser.add_argument('--bert_path', type=str, default='data/bert-base-uncased')
    parser.add_argument('--tinybert', default='./data/tinybert', type=str, help='The tiny model dir.')
    parser.add_argument('--teacher_model', default='./data/models/snips_teacher', type=str,
                        help='The teacher model dir.')
    parser.add_argument('--student_model', default='./data/models/snips_student_tmp', type=str,
                        help='The student model dir.')
    parser.add_argument('--output_dir', default='./data/models/snips_student_tmp', type=str,
                        help='Path to save, load model')

    parser.add_argument('--ignore_index', default=0, type=int,
                        help='Specifies a target value that is ignored and does not contribute to the input gradient')
    parser.add_argument('--max_seq_len', default=50, type=int,
                        help='The maximum total input sequence length after tokenization.')

    parser.add_argument('--train_batch_size', default=32, type=int, help='Batch size for training.')
    parser.add_argument('--eval_batch_size', default=64, type=int, help='Batch size for evaluation.')
    parser.add_argument('--learning_rate', default=5e-5, type=float, help='The initial learning rate for Adam.')
    parser.add_argument('--num_train_epochs', default=100.0, type=float,
                        help='Total number of training epochs to perform.')
    parser.add_argument('--weight_decay', default=0.0, type=float, help='Weight decay if we apply some.')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help='Number of updates steps to accumulate before performing a backward/update pass.')
    parser.add_argument('--adam_epsilon', default=1e-8, type=float, help='Epsilon for Adam optimizer.')
    parser.add_argument('--max_grad_norm', default=1.0, type=float, help='Max gradient norm.')
    parser.add_argument('--max_steps', default=-1, type=int,
                        help='If > 0: set total number of training steps to perform. Override num_train_epochs.')
    parser.add_argument('--warmup_steps', default=0, type=int, help='Linear warmup over warmup_steps.')
    parser.add_argument('--dropout_rate', default=0.1, type=float, help='Dropout for fully-connected layers')

    parser.add_argument('--logging_steps', type=int, default=200, help='Log every X updates steps.')
    parser.add_argument('--save_steps', type=int, default=200, help='Save checkpoint every X updates steps.')

    parser.add_argument('--intent_label_file', default='intent_label.txt', type=str, help='Intent Label file')
    parser.add_argument('--slot_label_file', default='slot_label.txt', type=str, help='Slot Label file')
    parser.add_argument('--slot_loss_coef', type=float, default=1.0, help='Coefficient for the slot loss.')
    parser.add_argument('--slot_pad_label', default='PAD', type=str,
                        help='Pad token for slot label pad (to be ignore when calculate loss)')
    parser.add_argument('--warmup_proportion', default=0.1, type=float,
                        help='Proportion of training to perform linear learning rate warmup for. '
                             'E.g., 0.1 = 10%% of training.')
    parser.add_argument('--eval_step', type=int, default=50)
    parser.add_argument('--temperature', type=float, default=1.)

    args = parser.parse_args()
    set_seed(args)
    tokenizer = BertTokenizer.from_pretrained('./data/bert-base-uncased', do_lower_case=True)

    train_dataset = load_and_cache_examples(args, tokenizer, mode="train")
    dev_dataset = load_and_cache_examples(args, tokenizer, mode="dev")
    test_dataset = load_and_cache_examples(args, tokenizer, mode="test")
    trainer = Trainer(args, train_dataset, dev_dataset, test_dataset)

    if args.do_train:
        trainer.train()

    if args.do_eval:
        model = trainer.load_model()
        trainer.evaluate('test', model)
