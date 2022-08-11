import random
import os
import shutil
from collections import Counter
from tqdm import tqdm


def fix_labels(path, new_path):
    old2new = {"O": "O", "COMMA": ",", "PERIOD": ".", "QMARK": "?", "EXCLAM": "!", "COLON": ":", "SEMICOLON": ";"}
    new_lines = []
    with open(path) as f:
        lines = f.read().splitlines()
    for line in tqdm(lines):
        parts = line.split()
        if len(parts) != 2:
            raise Exception("Error")
        word, label = parts[0], parts[1]
        label = old2new[label]
        new_lines.append("{} {}".format(word, label))
    with open(new_path, "w") as f:
        f.write("\n".join(new_lines))


def split_into_files(path, out_dir):
    with open(path) as f:
        lines = f.read().splitlines()
    file_lines = []
    count = 0
    for l in tqdm(lines):
        if l.startswith("tt ") or l.startswith("tto "):
            if file_lines:
                with open(f"{out_dir}/{count}.txt", "w") as fout:
                    fout.write("\n".join(file_lines))
                print("export file {}".format(f"{out_dir}/{count}.txt"))
                count += 1
            file_lines = []
        else:
            file_lines.append(l)
    if file_lines:
        with open(f"{out_dir}/{count}.txt", "w") as fout:
            fout.write("\n".join(file_lines))
        count += 1
    print("Number of files split from {}: {}".format(path, count))


def split_based_on_punctuation(path, rm=True):
    def write_file(count, f_lines):
        file_names = ".".join(path.split(".")[:-1]) + f"-{count}" + ".txt"
        with open(file_names, "w") as f:
            f.write("\n".join(f_lines))
        # print("export file {}".format(file_names))

    stop_punc = ["PERIOD", "EXCLAM", "QMARK"]
    with open(path) as f:
        lines = f.read().splitlines()
    if len(lines) > 500:
        num_file = len(lines) // 300
        num_word = len(lines) // num_file
        file_lines = []
        count = 0
        i = 0
        while i < len(lines):
            if not file_lines:
                file_lines.extend(lines[i: i + num_word])
                i += num_word
                if file_lines[-1].split()[-1] in stop_punc:
                    write_file(count, file_lines)
                    count += 1
                    file_lines = []
            else:
                file_lines.append(lines[i])
                if lines[i].split()[-1] in stop_punc:
                    write_file(count, file_lines)
                    count += 1
                    file_lines = []
                i = i + 1
        if file_lines:
            write_file(count, file_lines)
            count += 1
        print("Number of files split from {}: {}".format(path, count))
        if rm:
            os.remove(path)
    else:
        print("File {} does not need to split".format(path))


# folder = "data/novels/train"
# list_path = os.listdir(folder)
# for file in list_path:
#     path = os.path.join(os.path.join(folder, file))
#     split_based_on_punctuation(path)

def move_to_data_folder(folder, out_folder, name):
    old2new = {"O": "O", "COMMA": ",", "PERIOD": ".", "QMARK": "?", "EXCLAM": "!", "COLON": ":", "SEMICOLON": ";"}
    list_files = os.listdir(folder)
    for file in tqdm(list_files):
        with open(os.path.join(folder, file)) as f:
            lines = f.read().splitlines()
        new_lines = []
        for line in lines:
            parts = line.split()
            if len(parts) != 2:
                raise Exception("Error")
            word, label = parts[0], parts[1]
            label = old2new[label]
            new_lines.append("{} {}".format(word, label))
        with open(os.path.join(out_folder, name + file), "w") as f:
            f.write("\n".join(new_lines))


def copy_subfiles(source_dir, dest_dir, num):
    files = os.listdir(source_dir)
    random_ids = random.sample(list(range(0, len(files))), k=num)
    chosen_files = [files[i] for i in random_ids]
    for file in chosen_files:
        shutil.copy(os.path.join(source_dir, file), dest_dir)
    print("Copy random {} files from {} to {}".format(num, source_dir, dest_dir))


def replace_semicolon_by_comma():
    def rep_punc(lines):
        for i in range(len(lines)):
            parts = lines[i].split()
            if len(parts) != 2:
                raise Exception("error in line {}".format(i))
            if parts[1] == ";":
                parts[1] = ","
                lines[i] = " ".join(parts)
        return lines
    list_files = []
    for r, d, f in os.walk("data/vi"):
        list_files.extend([os.path.join(r, file) for file in f if file.endswith(".txt")])
    for file in tqdm(list_files):
        with open(file) as f:
            lines = f.read().splitlines()
        lines = rep_punc(lines)
        new_file = file.replace("data/vi", "data/vi_fix")
        with open(new_file, "w") as f:
            f.write("\n".join(lines))


def count_labels(path):
    list_files = []
    counter = Counter()
    for r, d, f in os.walk(path):
        list_files.extend([os.path.join(r, file) for file in f if file.endswith(".txt")])
    for file in tqdm(list_files):
        with open(file) as f:
            lines = f.read().splitlines()
        list_labels = [l.split()[-1] for l in lines]
        file_label_counter = Counter(list_labels)
        counter.update(file_label_counter)
    for k, v in counter.items():
        print("{} {}".format(k, v))


def join_train_and_test():
    for file in os.listdir("data/vi/test"):
        if file.endswith(".txt"):
            shutil.copyfile(os.path.join("data/vi/test", file), os.path.join("data/vi/train", "test_" + file))

# count_labels("data/vi/valid")


def count_files(path):
    count = 0
    for r, d, f in os.walk(path):
        count += len([os.path.join(path, file) for file in f if file.endswith(".txt")])
    return count

print(count_files("data/vi/train"))
