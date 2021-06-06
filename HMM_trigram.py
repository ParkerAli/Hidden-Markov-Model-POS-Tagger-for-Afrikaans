import numpy as np
import pandas as pd
from sklearn import metrics
from collections import defaultdict, Counter, deque
from itertools import permutations, dropwhile
import unicodedata

import string

import os.path

import pickle

# puntuasie
PUNCTUATION = {'.', ',', ';', ':', "'", '"', '$', '#', '@', '!', '?', '/', '*', '&', '^', '-', '+', '(', ')', '[', ']',
               '{', '}', '\\'}

class HMM(object):

    def __init__(self, dirname: str) -> None:

        self.unigram_c, self.tags, self.train,self.val, self.vocab = self.preprocess(
            pd.read_csv(f"{dirname}\\train.csv", na_filter=False))  # I used windows so change \\ to /
        self.test = self.test_preprocess(pd.read_csv(f"{dirname}\\test.csv", na_filter=False))

        
        self.unigram = self.unigram_probability()
        self.bigram = self.bigram_probability()

        # We store the trigram probabilities using pickle to save runtime
        if not os.path.exists("trigram.pkl"):
            with open("trigram.pkl","wb") as f:
                self.trigram = self.trigram_probability()
                pickle.dump(self.trigram,f)

        else:
             with open("trigram.pkl","rb") as f:
                self.trigram = pickle.load(f)


    def preprocess(self, df):
        """
        Splits the training data and prepares the training and validation sets
        :param df: The training dataframe
        :return: The unigram counts, the tags in the training data, word-tag pairs for the training data, the word-tag
        pairs for the validation data and the vocabulary
        """
        # preparing training and validation indices
        df = df[~df["Token"].isin(PUNCTUATION)]

        np.random.seed(69)  # set random seed for sampling
        num_sent = len(df.loc[df['Token'] == "NA"])  # number of sentences
        train_indices = np.random.choice(a=num_sent, size=int(num_sent * 0.7), replace=False)  # list of training sents
        print(len(train_indices))
        print(num_sent)

        # training vars
        unigram_c = defaultdict(int)  # the unigram counts
        train_tags = defaultdict(set)  # tags in the training set
        train_tags['<s>'].add(0)  # add first <s> token because no NA value start the dataset
        # train_tags['<s>'].add(1)  # add second <s> token because no NA value start the dataset
        train_word_tag_pairs = []  # stores word-tag pairs for training set
        # validation var
        val_word_tag_pairs = []  # stores word-tag pairs for validation set
        vocabulary = set()
        # populates training set
        sent_counter = 0  # counts num sentences

        counter = Counter()
        for row in df.itertuples(index=False):
            counter.update({row[0].lower() : 1})

        for key, count in dropwhile(lambda key_count: key_count[1] > 1, counter.most_common()):
            del counter[key]


        for i, row in enumerate(df.itertuples(index=False)):
            
            if sent_counter in train_indices:
                
                if row[0] != "NA":
                    word = row[0].lower()
                    vocabulary.add(word)
                    if word in counter:

                        # vocabulary.add(word)
                        unigram_c[(word, row[1])] += 1
                        train_tags[row[1]].add(i)
                        train_word_tag_pairs.append((word, row[1]))
                    else:
                        # vocabulary.add(word)
                        unigram_c[(word, "<UNK>")] += 1
                        train_tags["<UNK>"].add(i)
                        train_word_tag_pairs.append((word, "<UNK>"))

                else:
                    # unigram_c[("</s>", "</s>")] += 1
                    # unigram_c[("<s>", "<s>")] += 1
                    train_word_tag_pairs.append(("</s>", "</s>"))
                    train_word_tag_pairs.append(("<s>", "<s>"))

                    train_tags['<s>'].add(i + 2)
                    train_tags['</s>'].add(i + 1)
                    sent_counter += 1
            else:
                if row[0] == "NA" and row[1] == "NA":
                    sent_counter += 1


        
        train_tags['<s>'].remove(max(train_tags['<s>']))
        # initialises vocabulary from the word-tag pairs in the training set
        # vocabulary = {tup[0] for tup in train_word_tag_pairs}

        # populates validation set


        sent_counter = 0  # reset counter for next loop
        for row in df.itertuples(index=False):
            if sent_counter not in train_indices:
                # word = row[0].lower()
                if row[0] != "NA":
                    val_word_tag_pairs.append((row[0].lower(), row[1] if row[0].lower() in vocabulary else "<UNK>"))
                else:
                    val_word_tag_pairs.append(("</s>", "</s>"))
                    val_word_tag_pairs.append(("<s>", "<s>"))

                    sent_counter += 1
            else:
                if row[0] == "NA" and row[1] == "NA":
                    sent_counter += 1

        val_word_tag_pairs.pop()
        # the last index in the dataset is NA and we need to remove unnecessary <s> token

            # train_tags['<s>'].remove(max(train_tags['<s>']))
        print(len(vocabulary))
        print(len(val_word_tag_pairs))

        return unigram_c, train_tags,train_word_tag_pairs, val_word_tag_pairs, vocabulary

    def test_preprocess(self, df):
        """
        Transforms the test data into the correct format for tagging
        :param df: Test dataframe
        :return: word-tag
        """
        word_tag_pairs = [("<s>", "<s>")]
        df = df[~df["Token"].isin(PUNCTUATION)]

        for row in df.itertuples(index=False):
            if row[0] != "NA":
                word_tag_pairs.append((row[0].lower(), row[1] if row[0].lower() in self.vocab else "<UNK>" ))
            else:
                word_tag_pairs.append(("</s>", "</s>"))
                word_tag_pairs.append(("<s>", "<s>"))
        
        word_tag_pairs.pop()
        return word_tag_pairs

    def _emission_probability(self, word, tag) -> tuple:  # unigram
        """
        Calculates Pr(word|tag)
        :param word: a word
        :param tag: a tag
        :return: Pr(word|tag)
        """
        tag_count = len(self.tags[tag])

        if tag_count == 0:
            return 0
        words_tag_count = self.unigram_c[(word, tag)]
        return words_tag_count / tag_count


    def bigram_counter(self, tag_1, tag_2) -> int:  # bigram numberator
        """
        Calculates the bigram count for 2 tags
        :param tag_1: The first tag
        :param tag_2: The second tag
        :return: bigram count for the two tags
        """
        tag_1_then_tag_2_count = 0

        for index in self.tags[tag_1]:
            if index + 1 in self.tags[tag_2]:
                tag_1_then_tag_2_count += 1

        return tag_1_then_tag_2_count

    def trigram_counter(self, tag_1, tag_2, tag_3) -> float:  # trigram
        """
        Calculates the trigram count for 3 tags
        :param tag_1: the first tag
        :param tag_2: the second tag
        :param tag_3: the third tag
        :return:
        """
        counter = 0
        for index in self.tags[tag_3]:
            if index - 1 in self.tags[tag_2] and index - 2 in self.tags[tag_1]:
                counter += 1

        return counter + 1

    def unigram_probability(self):
        """
        Calculates the matrix of unigram (emission) probabilities for the training data using Pr(tag) = Pr(word|tag)
        :return: unigram probability matrix
        """
        D = defaultdict(int)

        for word_tag in self.unigram_c:
            D[word_tag] = self._emission_probability(*word_tag)
        return D

    def bigram_probability(self) -> defaultdict:
        """
        Calculates the matrix of bigram probabilities for the training data using Pr(tag2|tag1) = C(tag1,tag2)/C(tag1)
        :return: bigram probability matrix
        """
        bigram = defaultdict(np.float64)
        DP_values = {}
                
        for tags_pair in permutations(self.tags.keys(),2):
            # print(tags_pair)
            if (tag := tags_pair[0]) in DP_values:

                k = DP_values[tag]
            else:
                DP_values[tag] = k = len(self.tags[tags_pair[0]]) + len(self.vocab)

            bigram[tags_pair] = (self.bigram_counter(*tags_pair)+1) / k


        return bigram

    def trigram_probability(self) -> defaultdict:
        """
        Calculates the matrix of trigram probabilities for the training data using
        Pr(tag3|tag2,tag1) = C(tag1,tag2,tag3)/C(tag1,tag2)

        :return: trigram probability matrix
        """
        trigram = defaultdict(np.float64)
        DP_values = {}
        # import time

        # t0 = time.time()

        for tags_pair in permutations(self.tags.keys(),3):

            if (pair := (tags_pair[0], tags_pair[1])) not in DP_values:
            #     k = DP_values[pair]
            # else:

                DP_values[pair] = self.bigram_counter(*pair) + len(self.vocab)

            trigram[tags_pair] = self.trigram_counter(*tags_pair) / DP_values[pair]

        # t1 = time.time()

        # print(t1 - t0,"seconds")
        return trigram


    def classification(self, pairs=None,filename="test"):
        """
        Tags and classifies the parameterised data using the Viterbi function, writes the output to a csv file,
        and produces a classification report
        :param pairs: word tag pairs for the training data
        :param filename: name of the file to write the tagging results to
        :return: tagging precision
        """
        if pairs is None:
            pairs = self.test
        hits = 0
        # offset = 0

        with open(f"{filename}.csv","w") as output:
            print("Token","Actual Tag","Predicted Tag",sep=",",file=output)
            total = 1
            loc = 0
            sentence = deque()
            c=0
            for word,tag in pairs:
                sentence.append((word,tag))
                if tag == "</s>":
                    sentence.popleft()
                    sentence.pop()
                    for actual,predicted in zip(sentence,self.dp_viterbi_algorithm(sentence)):
                        print(actual[0],actual[1],predicted[1],sep=",",file=output)
                        if actual[1] == predicted[1]:
                            loc += 1
                        if predicted[1][0] == predicted[1][0]:
                            c +=1

                    # print("\nPRECISION:",loc/len(sentence))
                    total += len(sentence)
                    hits += loc
                    loc = 0
                    sentence = deque()


        print(c/total)


        # pd.DataFrame(metrics.classification_report(actual_list, predicted_list, output_dict=True)).transpose().to_csv("trigram_cp.csv")

        return hits / total

    def interpolation(self,word,l1,l2,tag1,tag2,tag3):
        """
        Linearly interpolates predictions from on the unigram, bigram and trigram probability matrices by assigning
        weights to the probability from each respective probability matrix
        :param word:
        :param l1: weight for the unigram probability
        :param l2: weight for the bigram probability
        :param tag1: the first tag
        :param tag2: the second tag
        :param tag3: the third tag
        :return: linear interpolation of the unigram, bigram and trigram probabilities
        """
        l3 = 1-l1-l2

        # print(word,tag3,self.unigram[(word,tag3)])
        unigram_p = self.unigram[(word,tag3)]  # unigram prediction
        bigram_p = self.bigram[(tag2, tag3)]  # bigram prediction
        trigram_p = self.trigram[(tag1, tag2, tag3)]  # trigram prediction
        return l1*unigram_p + l2*bigram_p + l3*trigram_p

    def dp_viterbi_algorithm(self, sentence):
        """
        Produces a tag prediction for a word by implementing the Viterbi algorithm using the linearly interpolated
        probabilities from the interpolation() function.
        :param sentence: sentence to be tagged
        :return: words and their corresponding tags
        """
        if not sentence:
            Exception("Sentence not found.")
        tags_list = self.tags.keys() #- set(["</s>"])

        l1 = 0
        l2 = 0.6

        p_max_tag = "<UNK>"
        for i,(word,_) in enumerate(sentence):

            p_max = float('-inf')
            if word not in self.vocab:
                p_max_tag = "<UNK>"
            for tag_1,tag_2 in permutations(tags_list, 2):

                if i > 1:
                    tag_prob = self.interpolation(word,l1,l2,p_max_tag, tag_1,tag_2)
                elif i == 1:
                    tag_prob = self.interpolation(word,l1,l2,"<s>", tag_1,tag_2)
                elif i == 0:
                    tag_prob = self.interpolation(word,l1,l2,"</s>", "<s>",tag_1)


                emission_probability = self.unigram[(word, tag_1)]
                state_prob = emission_probability * tag_prob
                if p_max < state_prob:
                    p_max_tag = tag_1
                    p_max = state_prob

            yield word, p_max_tag


    def dp_viterbi_algorithm_bigram(self, transition_matrix, sentence=None):
        """
        Produces a tag prediction by implementing the Viterbi algorithm using the bigram probabilities
        :param transition_matrix: the matrix of bigram probabilities
        :param sentence: sentence to be tagged
        :return: words and their corresponding tags
        """
        if not sentence:
            Exception("Sentence not found.")

        tags_list = self.tags.keys() - set(["</s>"])

        p_max_tag = "<s>"
        for i,(word,_) in enumerate(sentence):

            p_max = float('-inf')
            if word not in self.vocab:
                
                for tag in tags_list:
                    
                    p_max_running = self.bigram[(tag,p_max_tag)]#self.interpolation(l1,l2,tag,prior_tags[-1] ,prior_tags[-2])
                    if p_max_running> p_max:
                        p_max_tag = tag
                        p_max = p_max_running 

                yield word, "<UNK>"
            else:
                for tag_1 in tags_list:

                    tag_prob = transition_matrix[(p_max_tag, tag_1)]
                    emission_probability = self.unigram[(word, tag_1)]
                    state_prob = emission_probability * tag_prob
                    if p_max < state_prob:
                        p_max_tag = tag_1
                        p_max = state_prob
            
                yield word, p_max_tag



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
    print(hmm.classification(hmm.train,"training"))

    print(f"Training runtime = {time.time()-t0} seconds")

    # t0 = time.time()

    # print(hmm.classification(hmm.test,"testing"))
    # print(f"Testing runtime = {time.time()-t0} seconds")

    # t0 = time.time()

    # print(hmm.classification(hmm.val,"validation"))
    # print(f"Validation runtime = {time.time()-t0} seconds")

