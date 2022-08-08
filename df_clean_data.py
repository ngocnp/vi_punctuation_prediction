import json
import os
import re
from tqdm import tqdm


def clean_topic_news():
    count = 0
    fout = open("/home/ngocnp/Downloads/vietnamese_data/topic_news.txt", "w")
    list_files = []
    for r, d, f in os.walk("/home/ngocnp/Downloads/vietnamese_data/new topics"):
        list_files.extend([os.path.join(r, file) for file in f if file.endswith(".txt")])
    for file_path in tqdm(list_files):
        with open(file_path, encoding="UTF-16") as f:
            data = f.read().splitlines()
        data = [l for l in data if len(l) > 20]
        fout.write("\n".join(data))
        fout.write("\n")
        count += len(data)
    fout.close()
    print("total lines: ", count)


def get_from_huggingface():
    from datasets import load_dataset
    dataset = load_dataset("uit-nlp/vietnamese_students_feedback")
    print(dataset.data)


def get_vn_fairy_tales():
    with open("/home/ngocnp/Downloads/vietnamese_data/Kho tàng truyện cổ tích Việt Nam - Quyển II - Nguyễn Đổng Chi_djvu.txt") as f:
        lines = f.read().splitlines()
    data = lines[2055:48659]
    all_lines = []
    paragraph = []
    for line in tqdm(data):
        if line == "":
            if paragraph:
                if not any(re.match("(\d+\.)|(Khảo dị)", l) for l in paragraph):
                    para_text = " ".join(paragraph)
                    para_text = re.sub("\s+", " ", para_text)
                    all_lines.append(para_text)
                paragraph = []
        elif "KHO TÀNG TRUYỆN CỔ TÍCH VIỆT NAM" in line:
            continue
        else:
            paragraph.append(line)
    i = 0
    while i < len(all_lines):
        if len(all_lines[i]) < 20 :
            all_lines.pop(i)
        else:
            i = i + 1
    with open("Cotichvn_p1.txt", "w") as f:
        f.write("\n".join(all_lines))


def get_vn_fairy_tale_crawler():
    lines = []
    data = json.load(open("/home/ngocnp/PycharmProjects/crawler_tool/fairy_tales.json"))
    for item in data:
        text = item["text"]
        for l in text:
            l = l.strip()
            if l:
                if "https://" in l or "Đọc " in l:
                    continue
                l = l.replace(" ", " ")
                lines.append(l)
    real_lines = []
    temp_line = []
    for l in lines:
        if re.search("[.:!?]$", l):
            if not temp_line:
                if len(l) > 15:
                    l = l.replace(" :", ":")
                    real_lines.append(l)
            else:
                temp_line.append(l)
                l_str = " ".join(temp_line)
                l_str = re.sub("\s\s+", " ", l_str)
                if len(l) > 15:
                    l_str = l_str.replace(" :", ":")
                    real_lines.append(l_str)
                print(l_str)
            temp_line = []
        else:
            temp_line.append(l)
    if temp_line:
        l_str = " ".join(temp_line)
        l_str = re.sub("\s\s+", " ", l_str)
        if len(l_str) > 15:
            l_str = l_str.replace(" :", ":")
            real_lines.append(l_str)
    with open("fairy_tale_crawler.txt", "w") as f:
        f.write("\n".join(real_lines))


def clean_vn_novel():
    with open("/home/ngocnp/Downloads/vietnamese_data/collections.txt") as f:
        lines = f.read().splitlines()
    real_lines = []
    temp_line = []
    for line in lines:
        if line.endswith(":"):
            temp_line.append(line)
        elif (line.startswith("-") or line.startswith("“")) and temp_line:
            temp_line.append(line)
            real_lines.append(" ".join(temp_line))
            temp_line = []
        else:
            if temp_line:
                real_lines.append(" ".join(temp_line))
            real_lines.append(line)
    i = 0
    while i < len(real_lines):
        if not real_lines[i].strip() or len(real_lines[i]) < 20 or re.search("(^Chương )|(^Tác giả)", real_lines[i]):
            real_lines.pop(i)
        else:
            i = i + 1
    with open("clear_novel.txt", "w") as f:
        f.write("\n".join(real_lines))


with open("/home/ngocnp/Downloads/vietnamese_data/vn_text.txt") as f:
    content1 = f.read()
with open("/home/ngocnp/Downloads/vietnamese_data/topic_news.txt") as f:
    content2 = f.read()
with open("/home/ngocnp/Downloads/vietnamese_data/text-324.9 MB/clean_total.txt") as f:
    content3 = f.read()
if not content1.endswith("\n"):
    content1 = content1 + "\n"
if not content2.endswith("\n"):
    content2 = content2 + "\n"
with open("/home/ngocnp/Downloads/vietnamese_data/vn_nlp.txt", "w") as f:
    f.write(content1 + content2 + content3)