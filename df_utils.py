from tqdm import tqdm


def join(path1, path2, output):
    with open(path1, encoding="utf-8") as f:
        lines_1 = f.read()
    with open(path2,  encoding="utf-8") as f:
        lines_2 = f.read()
        if lines_1.endswith("\n"):
            total = lines_1 + lines_2
        else:
            total = lines_1 + "\n" + lines_2
    with open(output, "w") as f:
        f.write(total)


def count(path):
    list_labels = {}
    with open(path, encoding="utf-8") as f:
        lines = f.read().splitlines()
    for line in lines:
        lab = line.split()[1]
        count = list_labels.get(lab, 0)
        count = count + 1
        list_labels[lab] = count
    print(list_labels)


def split_file(path, n_split, path1, path2):
    with open(path) as f:
        lines = f.read().splitlines()
    part1 = lines[:n_split]
    part2 = lines[n_split:]
    with open(path1, "w") as f:
        f.write("\n".join(part1))
    with open(path2, "w") as f:
        f.write("\n".join(part2))


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

fix_labels("data/vi/test.txt", "data/vi/punc_label_test.txt")