import random
import os
import shutil
from tqdm import tqdm


# def join(path1, path2, output):
#     with open(path1, encoding="utf-8") as f:
#         lines_1 = f.read()
#     with open(path2,  encoding="utf-8") as f:
#         lines_2 = f.read()
#         if lines_1.endswith("\n"):
#             total = lines_1 + lines_2
#         else:
#             total = lines_1 + "\n" + lines_2
#     with open(output, "w") as f:
#         f.write(total)
#
#
# def count_labels(path):
#     list_labels = {}
#     with open(path, encoding="utf-8") as f:
#         lines = f.read().splitlines()
#     for line in lines:
#         lab = line.split()[1]
#         count = list_labels.get(lab, 0)
#         count = count + 1
#         list_labels[lab] = count
#     print(list_labels)
#
#
# def split_file(path, n_split, path1, path2):
#     with open(path) as f:
#         lines = f.read().splitlines()
#     part1 = lines[:n_split]
#     part2 = lines[n_split:]
#     with open(path1, "w") as f:
#         f.write("\n".join(part1))
#     with open(path2, "w") as f:
#         f.write("\n".join(part2))


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


# print("TRAIN")
# train_files = os.listdir("data/vi/train")
# print(len(train_files), len([f for f in train_files if f.startswith("news_")]), len([f for f in train_files if f.startswith("novels")]))
# print("VALID")
# valid_files = os.listdir("data/vi/valid")
# print(len(valid_files), len([f for f in valid_files if f.startswith("news_")]), len([f for f in valid_files if f.startswith("novels")]))
# print("TEST")
# test_files = os.listdir("data/vi/test")
# print(len(test_files), len([f for f in test_files if f.startswith("news_")]), len([f for f in test_files if f.startswith("novels")]))


copy_subfiles("/home/ngocnp/PycharmProjects/vi_punctuation_prediction/data/vi/valid",
              "/home/ngocnp/PycharmProjects/vi_punctuation_prediction/data/vi/small_valid",
              num=300)
