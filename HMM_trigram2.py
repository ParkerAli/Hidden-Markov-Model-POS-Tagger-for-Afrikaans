import numpy as np
import pandas as pd
from sklearn import metrics
from collections import defaultdict, Counter
from itertools import permutations


class HMM(object):

    def __init__(self, train_filename: str, test_filename: str) -> None:

        self.data, self.tags, self.word_tag_pairs = self.prepare_training(
            pd.read_csv(f"{train_filename}.csv", na_filter=False))  # I used windows so change \\ to /
        self.vocab = self.get_vocab(self.word_tag_pairs)
        self.test = self.prepare_word_tags(pd.read_csv(f"{test_filename}.csv", na_filter=False))

    def prepare_word_tags(self, df):
        word_tag_pairs = []

        for row in df.itertuples(index=False):

            if row[0] != "NA" and row[1] != "NA":
                word_tag_pairs.append((row[0].lower(), row[1][0] if row[0].lower() in self.vocab else "<UNK>"))
            else:
                word_tag_pairs.append(("<s>", "<s>"))
                word_tag_pairs.append(("<s>", "<s>"))

        print(word_tag_pairs)
        return word_tag_pairs

    def prepare_training(self, df) -> list:
        """
        The process function takes in a df with a single word and tag in each row and returns a df with a single sentence (and its tags) contained in each row.
        :param df: Dataframe to be processed
        :return: Processed dataframe
        """
        processed_df = defaultdict(int)
        tags = defaultdict(set)
        tags['<s>'].add(0)
        word_tag_pairs = []

        # counter = Counter()
        # for row in df.itertuples(index=False):
        #     counter.update({row[0].lower() : 1})
        #
        # for key, count in dropwhile(lambda key_count: key_count[1] >= 1, counter.most_common()):
        #     del counter[key]

        for i, row in enumerate(df.itertuples(index=False)):

            if row[0] != "NA" and row[1] != "NA":

                processed_df[(row[0].lower(), row[1])] += 1
                tags[row[1][0]].add(i + 1)
                word_tag_pairs.append((row[0].lower(), row[1][0]))

            else:

                processed_df[("<s>", "<s>")] += 1
                word_tag_pairs.append(("<s>", "<s>"))

                tags['<s>'].add(i + 1)

                processed_df[("<s>", "<s>")] += 1
                word_tag_pairs.append(("<s>", "<s>"))

                tags['<s>'].add(i + 2)

        tags['<s>'].remove(max(tags['<s>']))

        print(tags.keys())
        # print(word_tag_pairs)
        return processed_df, tags, word_tag_pairs

    def get_tags(self, df) -> set:
        return {tup[1] for tup in df}

    def get_vocab(self, df) -> set:
        return {tup[0] for tup in df}

    def calculate_emmision_probability(self, word, tag) -> tuple:  # unigram
        """
        Calculates Pr(word|tag)
        """
        tag_count = len(self.tags[tag])

        if tag_count == 0:
            return 0
        words_tag_count = self.data[(word, tag)]

        return words_tag_count / tag_count

    def calculate_transition_probability(self, tag_1, tag_2) -> int:  # bigram numberator
        """
        Calculates the probability of tag 2 appearing after tag 1.
        Pr(t2|t1)
        """
        tag_1_then_tag_2_count = 0

        for index in self.tags[tag_1]:
            if index + 1 in self.tags[tag_2]:
                tag_1_then_tag_2_count += 1

        return tag_1_then_tag_2_count

    def trigram(self, tag_1, tag_2, tag_3) -> float:  # trigram
        """
        Calculates the probability of tag 3 appearing after tag 2 and tag 1.
        Pr(t3|t2,t1) = C(t1,t2,t3)/C(t1,t2)
        """
        # tag_1_then_tag_2_count = self.calculate_transition_probability(tag_1=tag_1, tag_2= tag_2)
        # calculated = set()
        counter = 0
        for index in self.tags[tag_3]:
            if index - 1 in self.tags[tag_2] and index - 2 in self.tags[
                tag_1]:  # ((id_1:=index - 2),(id_2:=index - 1)) not in calculated and
                # calculated.add((id_1,id_2))
                counter += 1

        return counter + 1  # / (tag_1_then_tag_2_count + len(self.vocab))

    def create_transition_matrix(self) -> np.matrix:
        """
        Creates a matrix of transition probabilities
        """
        tags_list = list(self.tags.keys())
        trigram = defaultdict(np.float32)
        DP_values = {}

        for i, tags_pair in enumerate(permutations(tags_list, 3)):

            if (pair := (tags_pair[0], tags_pair[1])) in DP_values:

                k = DP_values[pair]
            else:

                DP_values[pair] = k = self.calculate_transition_probability(*pair) + len(self.vocab)

            trigram[tags_pair] = self.trigram(*tags_pair) / k

        return trigram, tags_list

    def classification(self, pairs=None):

        if pairs is None:
            pairs = self.word_tag_pairs
        hits = 0
        offset = 0

        actual_list = []
        predicted_list = []

        for actual, predicted in zip(pairs, self.dp_viterbi_algorithm(*self.create_transition_matrix(),
                                                                      (tup[0] for tup in pairs))):
            # print(actual,predicted)
            if actual[0] == "<s>" or actual[1] == "<UNK>":
                offset += 1
            else:
                actual_list.append(actual[1])
                predicted_list.append(predicted[1])
                hits += actual == predicted

        # pd.DataFrame(metrics.classification_report(actual_list, predicted_list, output_dict=True)).transpose().to_csv("cp3.csv")

        return hits / (len(pairs) - offset)

    def dp_viterbi_algorithm(self, transition_matrix, tags_list, words=None):
        if not words:
            words = self.word_tag_pairs

        D = {v: i for i, v in enumerate(tags_list)}

        for i, word in enumerate(words):

            p_max = float('-inf')
            p_max_index = -1

            for pair in permutations(tags_list, 2):
                if i == 1:
                    tag_prob = transition_matrix[("<s>", pair[0], pair[1])]
                elif i == 0:
                    tag_prob = transition_matrix[("<s>", "<s>", pair[0])]
                else:
                    tag_prob = transition_matrix[(tags_list[p_max_index], pair[0], pair[1])]

                emission_probability = self.calculate_emmision_probability(word, pair[0])
                state_prob = emission_probability * tag_prob

                if p_max < state_prob:
                    p_max_index = D[pair[0]]
                    p_max = state_prob

            if word != "<s>" and tags_list[p_max_index] == "<s>":
                yield word, "<UNK>"
            else:
                yield word, tags_list[p_max_index]


if __name__ == "__main__":
    hmm = HMM("AfrikaansPOSData/train", "AfrikaansPOSData/test")

    words = [
        ("laai", "VTHOG"),
        ("die", "LB"),
        ("elektroniese", "ASA"),
        ("aansoekvorm", "NSE"),
        ("af", "UPW"),
        ("of", "KN"),
        ("kry", "VTHOG"),
        ("dit", "PDOENP"),
        ("by", "SVS"),
        ("jou", "PTEB"),
        ("naaste", "AOA"),
        ("said-kantoor", "NSE"),
        (".", "ZE")
    ]

    import time

    t0 = time.time()
    # hmm.create_transition_matrix()
    print(hmm.classification(hmm.test))
    t1 = time.time()

    total = t1 - t0
    print(total, "seconds")