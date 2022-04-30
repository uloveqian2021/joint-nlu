import torch
import logging
import argparse
import numpy as np
from tqdm import tqdm
from transformers import BertTokenizer
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from utils import init_logger, get_intent_labels, get_slot_labels, get_args, load_model

logger = logging.getLogger(__name__)


def read_input_file(pred_config):
    lines = []
    with open(pred_config.input_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            words = line.split()
            lines.append(words)

    return lines


def convert_input_file_to_tensor_dataset(lines,
                                         args,
                                         tokenizer,
                                         pad_token_label_id,
                                         cls_token_segment_id=0,
                                         pad_token_segment_id=0,
                                         sequence_a_segment_id=0,
                                         mask_padding_with_zero=True):
    # Setting based on the current model type
    cls_token = '[CLS]'
    sep_token = '[SEP]'
    unk_token = '[UNK]'
    pad_token_id = 0

    all_input_ids = []
    all_attention_mask = []
    all_token_type_ids = []
    all_slot_label_mask = []

    for words in lines:
        tokens = []
        slot_label_mask = []
        for word in words:
            word_tokens = tokenizer.tokenize(word)
            if not word_tokens:
                word_tokens = [unk_token]  # For handling the bad-encoded word
            tokens.extend(word_tokens)
            # Use the real label id for the first token of the word, and padding ids for the remaining tokens
            slot_label_mask.extend([pad_token_label_id + 1] + [pad_token_label_id] * (len(word_tokens) - 1))

        # Account for [CLS] and [SEP]
        special_tokens_count = 2
        if len(tokens) > args.max_seq_len - special_tokens_count:
            tokens = tokens[: (args.max_seq_len - special_tokens_count)]
            slot_label_mask = slot_label_mask[:(args.max_seq_len - special_tokens_count)]

        # Add [SEP] token
        tokens += [sep_token]
        token_type_ids = [sequence_a_segment_id] * len(tokens)
        slot_label_mask += [pad_token_label_id]

        # Add [CLS] token
        tokens = [cls_token] + tokens
        token_type_ids = [cls_token_segment_id] + token_type_ids
        slot_label_mask = [pad_token_label_id] + slot_label_mask

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = args.max_seq_len - len(input_ids)
        input_ids = input_ids + ([pad_token_id] * padding_length)
        attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
        token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)
        slot_label_mask = slot_label_mask + ([pad_token_label_id] * padding_length)

        all_input_ids.append(input_ids)
        all_attention_mask.append(attention_mask)
        all_token_type_ids.append(token_type_ids)
        all_slot_label_mask.append(slot_label_mask)

    # Change to Tensor
    if len(lines) == 1:
        all_input_ids = torch.tensor(all_input_ids, dtype=torch.long)
        all_attention_mask = torch.tensor(all_attention_mask, dtype=torch.long)
        all_token_type_ids = torch.tensor(all_token_type_ids, dtype=torch.long)
        all_slot_label_mask = torch.tensor(all_slot_label_mask, dtype=torch.long)
        return all_input_ids, all_attention_mask, all_token_type_ids, all_slot_label_mask

    all_input_ids = torch.tensor(all_input_ids, dtype=torch.long)
    all_attention_mask = torch.tensor(all_attention_mask, dtype=torch.long)
    all_token_type_ids = torch.tensor(all_token_type_ids, dtype=torch.long)
    all_slot_label_mask = torch.tensor(all_slot_label_mask, dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_slot_label_mask)

    return dataset


def predict(pred_config):
    # load model and args
    args = get_args(pred_config.model_dir)
    device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    model = load_model(pred_config.model_dir, args, device)
    logger.info(args)

    intent_label_lst = get_intent_labels(args)
    slot_label_lst = get_slot_labels(args)

    # Convert input file to TensorDataset
    pad_token_label_id = args.ignore_index
    tokenizer = BertTokenizer.from_pretrained(args.bert_path)
    lines = read_input_file(pred_config)
    dataset = convert_input_file_to_tensor_dataset(lines, args, tokenizer, pad_token_label_id)

    # Predict
    sampler = SequentialSampler(dataset)
    data_loader = DataLoader(dataset, sampler=sampler, batch_size=pred_config.batch_size)

    all_slot_label_mask = None
    intent_preds = None
    slot_preds = None

    for batch in tqdm(data_loader, desc="Predicting"):
        batch = tuple(t.to(device) for t in batch)
        with torch.no_grad():
            input_ids, input_mask, segment_ids, _ = batch
            intent_logits, slot_logits, _, _ = model(input_ids, segment_ids, input_mask)

            # Intent Prediction
            if intent_preds is None:
                intent_preds = intent_logits.detach().cpu().numpy()
            else:
                intent_preds = np.append(intent_preds, intent_logits.detach().cpu().numpy(), axis=0)

            # Slot prediction
            if slot_preds is None:
                slot_preds = slot_logits.detach().cpu().numpy()
                all_slot_label_mask = batch[3].detach().cpu().numpy()
            else:
                slot_preds = np.append(slot_preds, slot_logits.detach().cpu().numpy(), axis=0)
                all_slot_label_mask = np.append(all_slot_label_mask, batch[3].detach().cpu().numpy(), axis=0)

    intent_preds = np.argmax(intent_preds, axis=1)
    slot_preds = np.argmax(slot_preds, axis=2)
    slot_label_map = {i: label for i, label in enumerate(slot_label_lst)}
    slot_preds_list = [[] for _ in range(slot_preds.shape[0])]

    for i in range(slot_preds.shape[0]):
        for j in range(slot_preds.shape[1]):
            if all_slot_label_mask[i, j] != pad_token_label_id:
                slot_preds_list[i].append(slot_label_map[slot_preds[i][j]])

    # Write to output file
    with open(pred_config.output_file, "w", encoding="utf-8") as f:
        for words, slot_preds, intent_pred in zip(lines, slot_preds_list, intent_preds):
            line = ""
            for word, pred in zip(words, slot_preds):
                if pred == 'O':
                    line = line + word + " "
                else:
                    line = line + "[{}:{}] ".format(word, pred)
            f.write("<{}> -> {}\n".format(intent_label_lst[intent_pred], line.strip()))

    logger.info("Prediction Done!")


def predict_single_query(pred_config):
    # load model and args
    args = get_args(pred_config.model_dir)
    device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    model = load_model(pred_config.model_dir, args, device)
    logger.info(args)

    intent_label_lst = get_intent_labels(args)
    slot_label_lst = get_slot_labels(args)

    # Convert input file to TensorDataset
    pad_token_label_id = args.ignore_index
    tokenizer = BertTokenizer.from_pretrained(args.bert_path)
    dataset = convert_input_file_to_tensor_dataset([pred_config.input_query.split()], args, tokenizer,
                                                   pad_token_label_id)
    dataset = tuple(t.to(device) for t in dataset)

    input_ids, input_mask, segment_ids, _ = dataset
    intent_logits, slot_logits, _, _ = model(input_ids, segment_ids, input_mask)

    intent_preds = intent_logits.detach().cpu().numpy()
    slot_preds = slot_logits.detach().cpu().numpy()
    all_slot_label_mask = dataset[3].detach().cpu().numpy()

    intent_preds = np.argmax(intent_preds, axis=1)

    slot_preds = np.argmax(slot_preds, axis=2)
    slot_label_map = {i: label for i, label in enumerate(slot_label_lst)}
    slot_preds_list = [[] for _ in range(slot_preds.shape[0])]

    for i in range(slot_preds.shape[0]):
        for j in range(slot_preds.shape[1]):
            if all_slot_label_mask[i, j] != pad_token_label_id:
                slot_preds_list[i].append(slot_label_map[slot_preds[i][j]])

    # Write to output file
    with open(pred_config.output_file, "w", encoding="utf-8") as f:
        # for words, slot_preds, intent_pred in zip(lines, slot_preds_list, intent_preds):
        line = ""
        for word, pred in zip(pred_config.input_query.split(), slot_preds_list[0]):
            if pred == 'O':
                line = line + word + " "
            else:
                line = line + "[{}:{}] ".format(word, pred)
        f.write("<{}> -> {}\n".format(intent_label_lst[intent_preds[0]], line.strip()))
        print(line)

    logger.info("Prediction Done!")


if __name__ == "__main__":
    init_logger()
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_file", default='data/snips/test/seq.in', type=str, help="Input file for prediction")
    parser.add_argument("--input_query", default='add sabrina salerno to the grime instrumentals playlist', type=str,
                        help="Input query for prediction")
    parser.add_argument("--output_file", default="./data/snips/sample_pred_out.txt", type=str,
                        help="Output file for prediction")
    parser.add_argument("--model_dir", default="./data/models/snips_student", type=str,
                        help="Path to save, load model")

    parser.add_argument("--batch_size", default=32, type=int, help="Batch size for prediction")
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")

    _pred_config = parser.parse_args()
    if _pred_config.input_file is not None:
        predict(_pred_config)
    # if _pred_config.input_query is not None:
    #     predict_single_query(_pred_config)
