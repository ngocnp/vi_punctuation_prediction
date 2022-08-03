import os.path
import pathlib
import random
from collections import Counter
from underthesea import word_tokenize
from sklearn.model_selection import train_test_split


from dataset import load


RAND_SEED = 42
random.seed(RAND_SEED)
RELEVANT_PUNCT = [".", ",", ":", "-", "?", "!", ";"]


def tokenize_sentence_into_words(text):
    tokens = word_tokenize(text)
    return tokens


def _create_directories(filename):
    directory = os.path.dirname(os.path.abspath(filename))
    pathlib.Path(directory).mkdir(parents=True, exist_ok=True)


def to_tsv(lines, output_filename):
    _create_directories(output_filename)
    label_counts = Counter()
    with open(output_filename, 'w') as f:
        prev = None
        cnt_lines = 0
        for line in lines:
            cnt_lines += 1

            line = line.lower()
            if '\"' in line:
                continue

            if cnt_lines % 10000 == 0:
                print(f"Processed {cnt_lines} lines")

            for tok in tokenize_sentence_into_words(line):
                tok = tok.strip()
                if len(tok) == 0:
                    continue

                if prev == None:
                    prev = tok
                    continue

                if tok in RELEVANT_PUNCT:  # subtask 2 label
                    t2_label = tok
                    label_counts[tok] += 1
                else:
                    t2_label = 0

                s = f'{prev}\t{t2_label}'
                f.write(s + "\n")

                if tok in RELEVANT_PUNCT:  # subtask 2 label
                    prev = None
                else:
                    prev = tok

    print(output_filename)
    for label, cnt in label_counts.most_common():
        print(label, cnt)


def dev_test_train(text_file: str, outpath: str, dev_size=0.2, test_size=0.16):
    with open(text_file, 'r') as f:
        lines = f.readlines()
    random.shuffle(lines)
    train_lines, test_lines = train_test_split(lines, test_size=test_size, random_state=RAND_SEED)
    train_lines, dev_lines = train_test_split(train_lines, test_size=dev_size, random_state=RAND_SEED)

    to_tsv(train_lines, f"{outpath}/train/train.tsv")
    to_tsv(dev_lines, f"{outpath}/dev/dev.tsv")
    to_tsv(test_lines, f"{outpath}/test/test.tsv")

    print(f"Wrote zip intp {outpath} using {text_file} as input corpus")


if __name__ == "__main__":
    print("Reads a corpus text file and converts it to tsv format suitable for training")

    # Input corpus. Each line is a sentence
    input_corpus = '/home/ngocnp/Downloads/vietnamese_data/text-324.9 MB/test.txt'
    outpath = 'data/test/'

    # target zip file to produce (the system expects sepp_nlg_2021_train_dev_data_v5.zip)
    zipfile = 'data-ca.zip'

    dev_test_train(input_corpus, outpath=outpath)
    os.system(f"zip -r {zipfile} {outpath}")