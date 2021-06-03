import numpy as np
import pandas as pd
from sklearn import metrics
from collections import defaultdict, Counter
from itertools import permutations, dropwhile

class HMM(object):

    def __init__(self, dirname: str) -> None:

        self.unigram_c, self.tags, self.val, self.vocab = self.preprocess(
            pd.read_csv(f"{dirname}\\train.csv", na_filter=False))  # I used windows so change \\ to /
        self.test = self.test_preprocess(pd.read_csv(f"{dirname}\\test.csv", na_filter=False))

        
        self.unigram = self.unigram_probability()
        self.bigram = self.bigram_probability()
        self.trigram = self.trigram_probability()

    def preprocess(self, df):
        # preparing training and validation indices
        np.random.seed(69)  # set random seed for sampling
        num_sent = len(df.loc[df['Token'] == "NA"])  # number of sentences
        train_indices = np.random.choice(a=num_sent, size=int(num_sent * 0.7), replace=False)  # list of training sents
        print(len(train_indices))
        print(num_sent)

        # training vars
        unigram_c = defaultdict(int)  # the unigram counts
        train_tags = defaultdict(set)  # tags in the training set
        train_tags['<s>'].add(0)  # add first <s> token because no NA value start the dataset
        train_tags['<s>'].add(1)  # add second <s> token because no NA value start the dataset
        # train_word_tag_pairs = []  # stores word-tag pairs for training set
        # validation var
        val_word_tag_pairs = []  # stores word-tag pairs for validation set
        vocabulary = set()
        # populates training set
        sent_counter = 0  # counts num sentences
        for i, row in enumerate(df.itertuples(index=False)):
            vocabulary.add(row[0])
            if sent_counter in train_indices:
                if row[0] != "NA" and row[1] != "NA":
                    unigram_c[(row[0].lower(), row[1])] += 1
                    train_tags[row[1]].add(i + 1)
                    # train_word_tag_pairs.append((row[0].lower(), row[1]))
                else:
                    unigram_c[("<s>", "<s>")] += 2
                    # train_word_tag_pairs.append(("<s>", "<s>"))
                    # train_word_tag_pairs.append(("<s>", "<s>"))
                    train_tags['<s>'].add(i + 1)
                    train_tags['<s>'].add(i + 2)
                    sent_counter += 1
            else:
                if row[0] == "NA" and row[1] == "NA":
                    sent_counter += 1

        # initialises vocabulary from the word-tag pairs in the training set
        # vocabulary = {tup[0] for tup in train_word_tag_pairs}

        # populates validation set
        sent_counter = 0  # reset counter for next loop
        for row in df.itertuples(index=False):
            if sent_counter not in train_indices:
                if row[0] != "NA" and row[1] != "NA":
                    val_word_tag_pairs.append((row[0].lower(), row[1] if row[0].lower() in vocabulary else "<UNK>"))
                else:
                    val_word_tag_pairs.append(("<s>", "<s>"))
                    val_word_tag_pairs.append(("<s>", "<s>"))
                    sent_counter += 1
            else:
                if row[0] == "NA" and row[1] == "NA":
                    sent_counter += 1

        # the last index in the dataset is NA and we need to remove unnecessary <s> token
        if (num_sent - 1 in train_tags):
            train_tags['<s>'].remove(max(train_tags['<s>']))
            train_tags['<s>'].remove(max(train_tags['<s>']))
        # print(len(train_word_tag_pairs))
        # print(len(val_word_tag_pairs))

        return unigram_c, train_tags, val_word_tag_pairs, vocabulary

    def test_preprocess(self, df):
        word_tag_pairs = []
        for row in df.itertuples(index=False):
            if row[0] != "NA" and row[1] != "NA":
                word_tag_pairs.append((row[0].lower(), row[1] if row[0].lower() in self.vocab else "<UNK>"))
            else:
                word_tag_pairs.append(("<s>", "<s>"))
                word_tag_pairs.append(("<s>", "<s>"))
        return word_tag_pairs

    def _emission_probability(self, word, tag) -> tuple:  # unigram
        """
        Calculates Pr(word|tag)
        """
        tag_count = len(self.tags[tag])

        if tag_count == 0:
            return 0
        words_tag_count = self.unigram_c[(word, tag)]

        return words_tag_count / tag_count


    def bigram_counter(self, tag_1, tag_2) -> int:  # bigram numberator
        """
        Calculates the probability of tag 2 appearing after tag 1.
        Pr(t2|t1)
        """
        tag_1_then_tag_2_count = 0

        for index in self.tags[tag_1]:
            if index + 1 in self.tags[tag_2]:
                tag_1_then_tag_2_count += 1

        return tag_1_then_tag_2_count

    def trigram_counter(self, tag_1, tag_2, tag_3) -> float:  # trigram
        """
        Calculates the probability of tag 3 appearing after tag 2 and tag 1.
        Pr(t3|t2,t1) = C(t1,t2,t3)/C(t1,t2)
        """

        counter = 0
        for index in self.tags[tag_3]:
            if index - 1 in self.tags[tag_2] and index - 2 in self.tags[tag_1]:  # ((id_1:=index - 2),(id_2:=index - 1)) not in calculated and
                counter += 1

        return counter + 1

    def unigram_probability(self):
        D = defaultdict(int)

        for word_tag in self.unigram_c:
            D[word_tag] = self._emission_probability(*word_tag)
        return D

    def bigram_probability(self) -> defaultdict:
        # tags_list = list(
        bigram = defaultdict(np.float32)
        DP_values = {}
        
        for tags_pair in permutations(self.tags.keys(),2):

            if (pair := (tags_pair[0])) in DP_values:

                k = DP_values[pair]
            else:
                # print(pair)
                DP_values[pair] = k = self.unigram_c[tags_pair] + len(self.vocab)

            bigram[tags_pair] = self.bigram_counter(*tags_pair) / k


        return bigram

    def trigram_probability(self) -> defaultdict:
        """
        Creates a matrix of transition probabilities
        """
        tags_list = list(self.tags.keys())
        trigram = defaultdict(np.float32)
        DP_values = {}
        
        for tags_pair in permutations(tags_list,3):

            if (pair := (tags_pair[0], tags_pair[1])) in DP_values:

                k = DP_values[pair]
            else:

                DP_values[pair] = k = self.bigram_counter(*pair) + len(self.vocab)

            trigram[tags_pair] = self.trigram_counter(*tags_pair) / k
            # yield self.trigram(*tags_pair) / k
        # print(hmm.classification())
        # self.
        return trigram


    def classification(self, pairs=None):

        if pairs is None:
            pairs = self.test
        hits = 0
        offset = 0

        actual_list = []
        predicted_list = []
        


        for i,actual, predicted in zip(range(1,len(pairs)-1),pairs, self.dp_viterbi_algorithm(self.trigram,
                                                                      (tup[0] for tup in pairs))):
            # print(actual,predicted)
            if i % 50 == 0:
                print(i,"PRECISION",hits / (i - offset))

            if actual[0] == "<s>":
                offset += 1
            else:
                actual_list.append(actual[1])
                predicted_list.append(predicted[1])
                hits += actual == predicted

        # pd.DataFrame(metrics.classification_report(actual_list, predicted_list, output_dict=True)).transpose().to_csv("trigram_cp.csv")

        return hits / (len(pairs) - offset)

    def interpolation(self,tag_1,tag_2,tag_3=None):
        # l1 = 1/3
        # l2 = 1/3
        # l3 = 1/3
        
        # uni = self.unigram
        pass

    def dp_viterbi_algorithm(self, transition_matrix, sentence):
        if not sentence:
            Exception("Sentence not found.")
        tags_list = list(self.tags.keys())
        D = {v: i for i, v in enumerate(tags_list)}

        for i, word in enumerate(sentence):

            p_max = float('-inf')
            p_max_index = 0

            for pair in permutations(tags_list, 2):
                # remove this bc Ali says so :)
                if i == 1:
                    tag_prob = transition_matrix[("<s>", pair[0], pair[1])]
                elif i == 0:
                    tag_prob = transition_matrix[("<s>", "<s>", pair[0])]
                else:
                    tag_prob = transition_matrix[(tags_list[p_max_index], pair[0], pair[1])]

                emission_probability = self.unigram[(word, pair[0])]
                state_prob = emission_probability * tag_prob

                if p_max < state_prob:
                    p_max_index = D[pair[0]]
                    p_max = state_prob
            
            yield word, tags_list[p_max_index]

            # else:
            # # if word != "<s>" and tags_list[p_max_index] == "<s>":
            #     yield word, "<UNK>"
            # # else:


if __name__ == "__main__":
    hmm = HMM("AfrikaansPOSData")

    sentence = [
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
    print(hmm.classification(hmm.val))
    t1 = time.time()

    total = t1 - t0
    print(total, "seconds")