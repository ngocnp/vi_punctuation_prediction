from concurrent.futures import process
from transformers import pipeline
import re
import torch


class PunctuationModel():
    def __init__(self, model, label_dict=None) -> None:
        if torch.cuda.is_available():
            self.pipe = pipeline("ner", model, grouped_entities=False, device=0)
        else:
            self.pipe = pipeline("ner", model, grouped_entities=False)
        self.label_dict = label_dict

    def preprocess(self, text):
        # remove markers except for markers in numbers
        text = re.sub(r"(?<!\d)[.,;:!?](?!\d)", "", text)
        # todo: match acronyms https://stackoverflow.com/questions/35076016/regex-to-match-acronyms
        text = text.split()
        return text

    def restore_punctuation(self, text):
        result = self.predict(self.preprocess(text))
        return self.prediction_to_text(result)

    def overlap_chunks(self, lst, n, stride=0):
        """Yield successive n-sized chunks from lst with stride length of overlap."""
        for i in range(0, len(lst), n - stride):
            yield lst[i:i + n]

    def predict(self, words):
        overlap = 5
        chunk_size = 230
        if len(words) <= chunk_size:
            overlap = 0

        batches = list(self.overlap_chunks(words, chunk_size, overlap))

        # if the last batch is smaller than the overlap, 
        # we can just remove it
        if len(batches[-1]) <= overlap:
            batches.pop()

        tagged_words = []
        for batch in batches:
            # use last batch completely
            if batch == batches[-1]:
                overlap = 0
            text = " ".join(batch)
            result = self.pipe(text)
            assert len(text) == result[-1]["end"], "chunk size too large, text got clipped"

            char_index = 0
            result_index = 0
            for word in batch[:len(batch) - overlap]:
                char_index += len(word) + 1
                # if any subtoken of an word is labled as sentence end
                # we label the whole word as sentence end        
                label = "0"
                while result_index < len(result) and char_index > result[result_index]["end"]:
                    label = result[result_index]['entity']
                    score = result[result_index]['score']
                    result_index += 1
                tagged_words.append([word, label, score])

        assert len(tagged_words) == len(words)
        return tagged_words

    def prediction_to_text(self, prediction):
        result = ""
        if not self.label_dict:
            for word, label, _ in prediction:
                result += word
                if label == "O":
                    result += " "
                if label in ".,?-:;":
                    result += label + " "
        else:
            for word, label, _ in prediction:
                result += word
                real_label = self.label_dict[label]
                if real_label == "O":
                    result += " "
                if real_label in ".,?-:;":
                    result += real_label + " "
        return result.strip()


if __name__ == "__main__":
    import time
    import pandas

    label_dict = {"LABEL_0": "O",
                  "LABEL_1": ".",
                  "LABEL_2": ",",
                  "LABEL_3": "?",
                  "LABEL_4": "!",
                  "LABEL_5": ";",
                  "LABEL_6": ":"}
    model = PunctuationModel(
        model="/home/ngocnp/PycharmProjects/vi_punctuation_prediction/saved_models/train_full_multilingual",
        label_dict=label_dict)
    # data = []
    # with open("test_text.txt") as f:
    #     texts = f.read().splitlines()
    # for text in texts:
    #     s = time.time()
    #     input_text = text.lower()
    #     input_text = re.sub(r"(?<!\d)[.,](?!\d)", "", input_text)
    #     input_text = re.sub(r"[;:!?\"]", "", input_text)
    #
    #     result_text = model.restore_punctuation(input_text)
    #     delta_time = float(time.time() - s)
    #     data.append([input_text, result_text, text, delta_time])
    # df = pandas.DataFrame(data=data, columns=["Input", "Output", "Target", "Time"])
    # df.to_csv("punc_prediction_results.csv")

    text = '''Mạnh mẽ chính xác và yên tĩnh M590 được thiết kế để thể hiện và bạn cũng có thể như vậy hình dáng uốn lượn thuận tay phải cho phép bạn làm việc thoải mái trong hàng giờ liền trong khi thiết kế nhỏ gọn và thời lượng pin lên tới 2 năm cho bạn sự tự do để đem công việc đi bất kỳ đâu'''
    print(model.restore_punctuation(text))

    # clean_text = model.preprocess(text)
    # labled_words = model.predict(clean_text)
    # print(labled_words)
