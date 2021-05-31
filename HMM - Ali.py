import numpy as np
import pandas as pd
from sklearn import metrics
from collections import defaultdict, Counter
from itertools import dropwhile


class HMM(object):

    def __init__(self, dirname: str) -> None:

        self.data, self.tags, self.word_tag_pairs = self.prepare_training(
            pd.read_csv(f"{dirname}\\train.csv", na_filter=False))  # I used windows so change \\ to /
        self.vocab = self.get_vocab(self.word_tag_pairs)
        self.test = self.prepare_word_tags(pd.read_csv(f"{dirname}\\test.csv", na_filter=False))

    def prepare_word_tags(self, df):
        word_tag_pairs = []
        for row in df.itertuples(index=False):

            if row[0] != "NA" and row[1] != "NA":
                word_tag_pairs.append((row[0].lower(), row[1] if row[0].lower() in self.vocab else "<UNK>"))
            else:
                word_tag_pairs.append(("<s>", "<s>"))
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

                # if row[0] in counter:

                processed_df[(row[0].lower(), row[1])] += 1
                tags[row[1]].add(i + 1)
                word_tag_pairs.append((row[0].lower(), row[1]))
                # else:
                #     processed_df[(row[0],"<UNK>")] += 1
                #     tags["<UNK>"].add(i+1)
                #     word_tag_pairs.append((row[0],"<UNK>"))
            else:

                processed_df[("<s>", "<s>")] += 1
                word_tag_pairs.append(("<s>", "<s>"))

                tags['<s>'].add(i + 1)

        tags['<s>'].remove(max(tags['<s>']))

        return processed_df, tags, word_tag_pairs

    def get_tags(self, df) -> set:
        return {tup[1] for tup in df}

    def get_vocab(self, df) -> set:
        return {tup[0] for tup in df}

    def calculate_emmision_probability(self, word, tag) -> tuple: #unigram
        """
        Calculates Pr(word|tag)
        """
        tag_count = len(self.tags[tag])

        if tag_count == 0:
            return 0
        words_tag_count = self.data[(word, tag)]

        return words_tag_count / tag_count

    def calculate_transition_probability(self, tag_1, tag_2) -> float: #bigram numberator
        """
        Calculates the probability of tag 2 appearing after tag 1.
        """
        tag_1_count = len(self.tags[tag_1])
        tag_1_then_tag_2_count = 0

        for index in self.tags[tag_1]:
            if index + 1 in self.tags[tag_2]:
                tag_1_then_tag_2_count += 1

        return (tag_1_then_tag_2_count) #/ (tag_1_count)

    

    def trigram(self, tag_1, tag_2, tag_3) -> float: #trigram
        """
        Calculates the probability of tag 3 appearing after tag 2 and tag 1.
        """
        tag_1_then_tag_2_count = self.calculate_transition_probability(tag_1=tag_1, tag_2= tag_2)
        tag_1_then_tag_2_then_tag_3_count = 0

        for index in self.tags[tag_1]:
            if index + 1 in self.tags[tag_2] and index + 2 in self.tags[tag_3]:
                tag_1_then_tag_2_then_tag_3_count += 1

        return (tag_1_then_tag_2_then_tag_3_count + 1) / (tag_1_then_tag_2_count + len(self.vocab))

    def create_transition_matrix(self) -> np.matrix:
        """
        Creates a matrix of transition probabilities
        """
        tags_list = list(self.tags.keys())
        tags_count = len(tags_list)
        tags_transition_matrix = np.zeros((tags_count, tags_count, tags_count), dtype="float64")

        for i, tag_1 in enumerate(tags_list):
            for j, tag_2 in enumerate(tags_list):
                for k, tag_3 in enumerate(tags_list):
                    tags_transition_matrix[i, j, k] = self.trigram(tag_1=tag_1, tag_2=tag_2, tag_3 = tag_3)
        print(pd.DataFrame(tags_transition_matrix).shape())
        return tags_transition_matrix #, tags_list

    def classification(self, pairs=None):

        if pairs is None:
            pairs = self.word_tag_pairs
        hits = 0
        offset = 0

        actual_list = []
        predicted_list = []

        for actual, predicted in zip(pairs, self.dp_viterbi_algorithm(*self.create_transition_matrix(),
                                                                      (tup[0] for tup in pairs))):

            if actual[0] == "<s>" or actual[1] == "<UNK>":
                offset += 1
                continue

            actual_list.append(actual[1])
            predicted_list.append(predicted[1])
            hits += actual == predicted

        pd.DataFrame(metrics.classification_report(actual_list, predicted_list, output_dict=True)).transpose().to_csv("cp2.csv")

        return hits / (len(pairs) - offset)

    # def dp_viterbi_algorithm(self, matrix, tags_list, words=None):
    #     if not words:
    #         words = self.word_tag_pairs
    #
    #     start_index = tags_list.index("<s>")
    #
    #     for i, word in enumerate(words):
    #
    #         p_max = float('-inf')
    #         p_max_index = -1
    #
    #         for j, tag in enumerate(tags_list):
    #             if i == 0:
    #                 tag_prob = matrix[start_index, j]
    #             else:
    #                 tag_prob = matrix[p_max_index, j]
    #
    #             emission_probability = self.calculate_emmision_probability(word, tag)
    #             state_prob = emission_probability * tag_prob
    #
    #             if p_max < state_prob:
    #                 p_max_index = j
    #                 p_max = state_prob
    #
    #         if word != "<s>" and tags_list[p_max_index] == "<s>":
    #             yield word, "<UNK>"
    #         else:
    #             yield word, tags_list[p_max_index]
    def viterbi(self,sent,method='UNK'):
        V = {}
        path = {}
        # init
        V[0,'',''] = 1
        path['',''] = []
        # run
        #sys.stderr.write("entering k loop\n")
        for k in range(1,len(sent)+1):
            temp_path = {}
            word = self.get_word(sent,k-1)
            ## handling unknown words in test set using low freq words in training set
            # if word not in self.words:
            #     print word
            #     if method=='UNK':
            #         word = '<UNK>'
            #     elif method == 'MORPHO':
            #         word = self.subcategorize(word)
            #sys.stderr.write("entering u loop "+str(k)+"\n")
            for u in self.get_possible_tags(k-1):
                #sys.stderr.write("entering v loop "+str(u)+"\n")
                for v in self.get_possible_tags(k):
                    V[k,u,v],prev_w = max([(V[k-1,w,u] * self.Q[w,u,v] * self.E[word,v],w) for w in self.get_possible_tags(k-2)])
                    temp_path[u,v] = path[prev_w,u] + [v]
            path = temp_path
        # last step
        prob,umax,vmax = max([(V[len(sent),u,v] * self.Q[u,v,''],u,v) for u in self.tags for v in self.tags])
        return path[umax,vmax]




    # def dp_viterbi_algorithm(self, matrix, tags_list, words=None):
    #     if not words:
    #         words = self.word_tag_pairs
    #
    #     start_index = tags_list.index("<s>")
    #
    #     for i, word in enumerate(words):
    #
    #         p_max = float('-inf')
    #         p_max_index = -1
    #
    #         for j, tag in enumerate(tags_list):
    #             for k, tag_2 in enumerate(tags_list):
    #                 if i == 0:
    #                     tag_prob = matrix[start_index, j,k]
    #                 else:
    #                     tag_prob = matrix[p_max_index, j,k]
    #
    #                 emission_probability = self.calculate_emmision_probability(word, tag)
    #                 state_prob = emission_probability * tag_prob
    #
    #                 if p_max < state_prob:
    #                     p_max_index = k
    #                     p_max = state_prob
    #
    #         if word != "<s>" and tags_list[p_max_index] == "<s>":
    #             yield word, "<UNK>"
    #         else:
    #             yield word, tags_list[p_max_index]


if __name__ == "__main__":
    hmm = HMM("AfrikaansPOSData")

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
    hmm.create_transition_matrix()
    # print(hmm.classification(hmm.test))
    t1 = time.time()

    total = t1 - t0
    print(total, "seconds")