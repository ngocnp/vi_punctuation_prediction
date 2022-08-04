from datasets import load_dataset, load_metric
from datasets import Dataset, Features, ClassLabel, Value
from dataset import load, data_augmentation
from transformers import AutoTokenizer, Adafactor, BertTokenizer, RobertaTokenizerFast
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer, BertForTokenClassification
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import numpy as np
import numpy.ma as ma
from transformers import DataCollatorForTokenClassification
from tqdm import tqdm
import datetime
from pprint import pprint

model_checkpoint = "trituenhantaoio/bert-base-vietnamese-uncased" # "bert-base-uncased"   #"trituenhantaoio/bert-base-vietnamese-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

label_2_id = {"O": 0, ".": 1, ",": 2, "?": 3, "!": 4, ";": 5, ":": 6}


def tokenize_and_align_data(data, stride=0):
    tokenizer_settings = {'is_split_into_words': True, #'return_offsets_mapping': True,
                          'padding': False, 'truncation': True, 'stride': stride,
                          'max_length': 510, 'return_overflowing_tokens': True}  # tokenizer.model_max_length
    tokenized_inputs = tokenizer(data[0], **tokenizer_settings)

    labels = []
    for i, document in enumerate(tokenized_inputs.encodings):
        doc_encoded_labels = []
        last_word_id = None
        for word_id in document.word_ids:
            if word_id is None or last_word_id == word_id:
                doc_encoded_labels.append(-100)
            else:
                # document_id = tokenized_inputs.overflow_to_sample_mapping[i]
                # label = examples[task][document_id][word_id]
                label = data[1][word_id]
                doc_encoded_labels.append(label_2_id[label])
            last_word_id = word_id
        labels.append(doc_encoded_labels)

    tokenized_inputs["labels"] = labels

    return tokenized_inputs


def to_dataset(data, stride=0):
    labels, token_type_ids, input_ids, attention_masks = [], [], [], []
    for item in tqdm(data):
        result = tokenize_and_align_data(item, stride=stride)
        labels += result['labels']
        token_type_ids += result['token_type_ids']
        input_ids += result['input_ids']
        attention_masks += result['attention_mask']
    d = {'labels': labels, 'token_type_ids': token_type_ids, 'input_ids': input_ids, 'attention_mask': attention_masks}
    for k, v in d.items():
        print(k)
        for e in v:
            print(e)
        print()
    return Dataset.from_dict(
        {'labels': labels, 'token_type_ids': token_type_ids, 'input_ids': input_ids, 'attention_mask': attention_masks})


train_data = load("data/data_test")
tokenized_dataset_train = to_dataset(train_data, stride=100)

