import os
import torch
import argparse
import numpy as np
import tensorflow as tf
# from onnx_tf.converter import convert
from transformers import BertTokenizer
from data import load_and_cache_examples
from torch.utils.data import DataLoader, SequentialSampler
from utils import init_logger, get_slot_labels, compute_metrics, load_model, get_args


def convert_examples_to_features(text, max_seq_len, tokenizer):
    tokens = tokenizer.tokenize(text)
    tokens = tokens[:max_seq_len - 2]
    tokens = ["[CLS]"] + tokens + ["[SEP]"]
    segment_ids = [0] * len(tokens)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_ids)
    padding = [0] * (max_seq_len - len(input_ids))
    input_ids += padding
    input_mask += padding
    segment_ids += padding

    input_ids = torch.tensor([input_ids], dtype=torch.long)
    input_mask = torch.tensor([input_mask], dtype=torch.long)
    segment_ids = torch.tensor([segment_ids], dtype=torch.long)

    return input_ids, input_mask, segment_ids


def torch_to_onnx(sample, max_seq_len, model, tokenizer, output_path):
    input_ids, input_mask, segment_ids = convert_examples_to_features(sample, max_seq_len, tokenizer)
    input_names = ["input_ids", 'token_type_ids', 'attention_mask']
    output_names = ["output", "output2"]
    dummy_input = (input_ids, segment_ids, input_mask)

    traced_script_module = torch.jit.trace(model, dummy_input)
    # test = traced_script_module(input_ids, segment_ids, input_mask)
    traced_script_module.save('data/models/snips_convert/model.pt')
    torch.onnx.export(model, dummy_input, output_path, input_names=input_names,
                      output_names=output_names, verbose=False, export_params=True)

    net = torch.jit.load('data/models/snips_convert/model.pt')
    test = net(input_ids, segment_ids, input_mask)
    print(test)


def saved_model_to_tflite(input_path, output_path):
    converter = tf.compat.v1.lite.TFLiteConverter.from_saved_model(input_path,
                                                                   input_arrays=['serving_default_input_ids',
                                                                                 'serving_default_token_type_ids',
                                                                                 'serving_default_attention_mask'],
                                                                   output_arrays=['PartitionedCall',
                                                                                  'PartitionedCall:1'])
    converter.allow_custom_ops = True
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tf_lite_model = converter.convert()

    with open(output_path, 'wb') as f:
        f.write(tf_lite_model)


def test_tflite(args, tokenizer, model_path):
    test_dataset = load_and_cache_examples(args, tokenizer, mode="test")
    eval_sampler = SequentialSampler(test_dataset)
    eval_dataloader = DataLoader(test_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    interpreter = tf.compat.v1.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    intent_preds = None
    slot_preds = None
    out_slot_labels_ids = None
    out_intent_label_ids = None

    for batch_ in eval_dataloader:
        input_ids, input_mask, segment_ids, intent_label_ids, slot_labels_ids = batch_

        for i in range(len(input_ids)):
            input_id = input_ids[i, :].unsqueeze(0).detach().cpu().numpy()
            input_m = input_mask[i, :].unsqueeze(0).detach().cpu().numpy()
            segment_id = segment_ids[i, :].unsqueeze(0).detach().cpu().numpy()
            interpreter.set_tensor(input_details[1]['index'], input_id)
            interpreter.set_tensor(input_details[2]['index'], segment_id)
            interpreter.set_tensor(input_details[0]['index'], input_m)
            interpreter.invoke()
            intent_logits = interpreter.get_tensor(output_details[0]['index'])
            slot_logits = interpreter.get_tensor(output_details[1]['index'])

            if intent_preds is None:
                intent_preds = intent_logits
            else:
                intent_preds = np.append(intent_preds, intent_logits, axis=0)

            if slot_preds is None:
                slot_preds = slot_logits
            else:
                slot_preds = np.append(slot_preds, slot_logits, axis=0)
        if out_intent_label_ids is None:
            out_intent_label_ids = intent_label_ids.detach().cpu().numpy()
        else:
            out_intent_label_ids = np.append(
                out_intent_label_ids, intent_label_ids.detach().cpu().numpy(), axis=0)
        if out_slot_labels_ids is None:
            out_slot_labels_ids = slot_labels_ids.detach().cpu().numpy()
        else:
            out_slot_labels_ids = np.append(out_slot_labels_ids, slot_labels_ids.detach().cpu().numpy(), axis=0)

    intent_preds = np.argmax(intent_preds, axis=1)
    slot_preds = np.argmax(slot_preds, axis=2)
    slot_label_lst = get_slot_labels(args)
    slot_label_map = {i: label for i, label in enumerate(slot_label_lst)}
    out_slot_label_list = [[] for _ in range(out_slot_labels_ids.shape[0])]
    slot_preds_list = [[] for _ in range(out_slot_labels_ids.shape[0])]

    for i in range(out_slot_labels_ids.shape[0]):
        for j in range(out_slot_labels_ids.shape[1]):
            if out_slot_labels_ids[i, j] != 0:
                out_slot_label_list[i].append(slot_label_map[out_slot_labels_ids[i][j]])
                slot_preds_list[i].append(slot_label_map[slot_preds[i][j]])

    total_result = compute_metrics(intent_preds, out_intent_label_ids, slot_preds_list, out_slot_label_list)
    print(total_result)


if __name__ == '__main__':
    init_logger()
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', default='./data/models/snips_student', type=str, help='Path to save, load model')
    parser.add_argument('--data_dir', default='./data', type=str, help='The input data dir')
    parser.add_argument('--convert_dir', default='./data/models/snips_convert', type=str)
    parser.add_argument('--max_seq_len', default=50, type=int, help='Batch size for prediction')
    parser.add_argument('--sample', default='add sabrina salerno to the grime instrumentals playlist', type=str)
    convert_config = parser.parse_args()
    _args = get_args(convert_config.model_dir)

    _model = load_model(convert_config.model_dir, _args, "cpu")

    _tokenizer = BertTokenizer.from_pretrained('data/bert-base-uncased', do_lower_case=True)

    convert_dir = convert_config.convert_dir
    # onnx
    # if not os.path.exists(convert_dir):
    #     os.makedirs(convert_dir)
    # onnx_path = os.path.join(convert_dir, 'model.onnx')
    # torch_to_onnx(convert_config.sample, convert_config.max_seq_len, _model, _tokenizer, onnx_path)

    # # pip install git+https://github.com/onnx/onnx-tensorflow.git el-1.8.0'
    # onnx-tf convert -i "model.onnx" -o  "saved_model"
    saved_model_path = os.path.join(convert_dir, 'saved_model')
    # convert(onnx_path, saved_model_path)
    #
    tflite_path = os.path.join(convert_dir, 'model.tflite')
    saved_model_to_tflite(saved_model_path, tflite_path)

    test_tflite(_args, _tokenizer, tflite_path)
