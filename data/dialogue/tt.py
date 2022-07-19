# -*- coding:utf-8 -*-
"""
@author : wang bq
@email  : 
@time   :21-7-23 上午10:27
@IDE    :PyCharm
@document   :tt.py
"""
import logging
import os
import json
import copy
logger = logging.getLogger(__name__)


def get_intent_labels():
    return [
        label.strip()
        for label in open('intent_label.txt', "r", encoding="utf-8")
    ]


def get_slot_labels():
    return [
        label.strip()
        for label in open('slot_label.txt', "r", encoding="utf-8")
    ]


class InputExample(object):
    """
    A single training/test example for simple sequence classification.

    Args:
        guid: Unique id for the example.
        words: list. The words of the sequence.
        intent_label: (Optional) string. The intent label of the example.
        slot_labels: (Optional) list. The slot labels of the example.
    """

    def __init__(self, guid, words, intent_label=None, slot_labels=None):
        self.guid = guid
        self.words = words
        self.intent_label = intent_label
        self.slot_labels = slot_labels

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, attention_mask, token_type_ids, intent_label_id, slot_labels_ids):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.intent_label_id = intent_label_id
        self.slot_labels_ids = slot_labels_ids

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


def read_file(input_file, quotechar=None):
    """Reads a tab separated value file."""
    with open(input_file, "r", encoding="utf-8") as f:
        lines = []
        for line in f:
            lines.append(line.strip())
        return lines


def create_examples(texts, intents, slots, set_type):
    """Creates examples for the training and dev sets."""
    intent_labels = get_intent_labels()
    slot_labels = get_slot_labels()
    print(slot_labels)
    examples = []
    for i, (text, intent, slot) in enumerate(zip(texts, intents, slots)):
        guid = "%s-%s" % (set_type, i)
        # 1. input_text
        words = text.split(' ')  # Some are spaced twice
        # 2. intent
        intent_label = (
            intent_labels.index(intent) if intent in intent_labels else intent_labels.index("UNK")
        )
        # 3. slot
        slot_label = []
        for s in slot.split(' '):
            slot_label.append(
                slot_labels.index(s) if s in slot_labels else slot_labels.index("UNK")
            )

        if len(words) != len(slot_label):
            print(words)
            print(slot_label)
            print(len(words), len(slot_label))
            continue
        # assert len(words) == len(slot_labels)
        examples.append(InputExample(guid=guid, words=words, intent_label=intent_label, slot_labels=slot_label))
    return examples


def get_examples(mode):
    """
    Args:
        mode: train, dev, test
    """
    return create_examples(
        texts=read_file('train/seq.in'),
        intents=read_file('train/label'),
        slots=read_file('train/seq.out'),
        set_type=mode,
    )


res = get_examples('train')

