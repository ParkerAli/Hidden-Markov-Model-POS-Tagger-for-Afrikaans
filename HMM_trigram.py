import numpy as np
import pandas as pd
from sklearn import metrics
from collections import defaultdict, Counter, deque
from itertools import permutations, dropwhile
import time
import os.path
import pickle
from joblib import Parallel, delayed
import warnings
warnings.filterwarnings('ignore')

# Punktuasie
PUNCTUATION = {'.', ',', ';', ':', "'", '"', '$', '#', '@', '!', '?', '/', '*', '&', '^', '-', '+', '(', ')', '[', ']',
               '{', '}', '\\'}


class HMM(object):

    def __init__(self, dirname: str, baseline=True) -> None:

        self.unigram_c, self.tags, self.train,self.val, self.vocab = self.preprocess(
            pd.read_csv(f"{dirname}\\train.csv", na_filter=False), 0.7)  # I used windows so change \\ to /
        self.test = self.test_preprocess(pd.read_csv(f"{dirname}\\test.csv", na_filter=False))
        self.baseline = baseline
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


    # def train(self):


    def preprocess(self, df, training_proportion):
        """
        Splits the training data and prepares the training and validation sets
        :param df: The training dataframe
        :param training_proportion: The training training set proportions
        :return: The unigram counts, the tags in the training data, word-tag pairs for the training data, the word-tag
        pairs for the validation data and the vocabulary
        """
        # preparing training and validation indices
        df = df[~df["Token"].isin(PUNCTUATION)]

        np.random.seed(69)  # set random seed for sampling
        num_sent = len(df.loc[df['Token'] == "NA"])  # number of sentences
        train_indices = np.random.choice(a=num_sent, size=int(num_sent * training_proportion), replace=False)  # list of training sents


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
                        unigram_c[(word, row[1])] += 1
                        train_tags[row[1]].add(i)
                        train_word_tag_pairs.append((word, row[1]))
                    else:
                        unigram_c[(word, "<UNK>")] += 1
                        train_tags["<UNK>"].add(i)
                        train_word_tag_pairs.append((word, "<UNK>"))

                else:
                    train_word_tag_pairs.append(("</s>", "</s>"))
                    train_word_tag_pairs.append(("<s>", "<s>"))

                    train_tags['<s>'].add(i + 2)
                    train_tags['</s>'].add(i + 1)
                    sent_counter += 1
            else:
                if row[0] == "NA" and row[1] == "NA":
                    sent_counter += 1

        train_tags['<s>'].remove(max(train_tags['<s>']))

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

        train_tags['<s>'].remove(max(train_tags['<s>']))
        print(f"The number of sentences in the training set is {len(train_indices)}")
        print(f"The number of sentences in the validation set is {num_sent-len(train_indices)}")
        print(f"The vocabulary size is {len(vocabulary)}")

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

    def _emission_probability(self, word, tag) -> float:
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


    def bigram_counter(self, tag_1, tag_2) -> int:
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
        unigram = defaultdict(int)

        for word_tag in self.unigram_c:
            unigram[word_tag] = self._emission_probability(*word_tag)
        return unigram

    def bigram_probability(self) -> defaultdict:
        """
        Calculates the matrix of bigram probabilities for the training data using Pr(tag2|tag1) = C(tag1,tag2)/C(tag1)
        :return: bigram probability matrix
        """
        bigram = defaultdict(np.float64)
        tags_list = list(self.tags.keys() - {"<\s>"})
        for i, tag_1 in enumerate(tags_list):
            for j, tag_2 in enumerate(tags_list):
                bigram[(tag_1, tag_2)] = (self.bigram_counter(tag_1,tag_2)+1) / (len(self.tags[tag_1]) + len(self.vocab))
        return bigram

    def trigram_probability(self) -> defaultdict:
        """
        Calculates the matrix of trigram probabilities for the training data using
        Pr(tag3|tag2,tag1) = C(tag1,tag2,tag3)/C(tag1,tag2)

        :return: trigram probability matrix
        """
        trigram = defaultdict(np.float64)
        DP_values = {}
        for tags_pair in permutations(self.tags.keys(),3):

            if (pair := (tags_pair[0], tags_pair[1])) not in DP_values:
                DP_values[pair] = self.bigram_counter(*pair) + len(self.vocab)

            trigram[tags_pair] = self.trigram_counter(*tags_pair) / DP_values[pair]
        return trigram


    # def classification(self, pairs=None,filename="test", l1=0,l2=1):
    #     """
    #     Tags and classifies the parameterised data using the Viterbi function, writes the output to a csv file,
    #     and produces a classification report
    #     :param pairs: word tag pairs for the training data
    #     :param filename: name of the file to write the tagging results to
    #     :param l1: weight for the unigram probability
    #     :param l2: weight for the bigram probability
    #     :return: tagging precision
    #     """
    #     if pairs is None:
    #         pairs = self.test
    #     hits = 0
    #     tags_list = sorted(list(self.tags.keys() - {"</s>", "<s>"}))
    #
    #     token_list = []
    #     actual_list = []
    #     predicted_list = []
    #     with open(f"{filename}.csv","w") as output:
    #         print("Token", "Actual Tag", "Predicted Tag",sep=",", file=output)
    #         total = 1
    #         loc = 0
    #         sentence = deque()
    #         c = 0
    #         for word,tag in pairs:
    #             sentence.append((word,tag))
    #             if tag == "</s>":
    #                 sentence.popleft()
    #                 sentence.pop()
    #                 if self.baseline:
    #                     for actual, predicted in zip(sentence, self.dp_viterbi_algorithm_bigram(sentence, tags_list)):
    #                         print(actual[0],actual[1],predicted[1],sep=",",file=output)
    #                         if actual[1] == predicted[1]:
    #                             loc += 1
    #                         if predicted[1][0] == predicted[1][0]:
    #                             c +=1
    #                 else:
    #                     for actual,predicted in zip(sentence,self.dp_viterbi_algorithm(sentence, l1,l2, tags_list)):
    #                         print(actual[0], actual[1], predicted[1], sep=",", file=output)
    #                         if actual[1] == predicted[1]:
    #                             loc += 1
    #                         if predicted[1][0] == predicted[1][0]:
    #                             c += 1
    #
    #                 # print("\nPRECISION:",loc/len(sentence))
    #                 total += len(sentence)
    #                 hits += loc
    #                 loc = 0
    #                 sentence.clear()
    #
    #     print(c/total)
    #     # pd.DataFrame(metrics.classification_report(actual_list, predicted_list, output_dict=True)).transpose().to_csv("trigram_cp.csv")
    #
    #     return hits / total

    def classification(self, pairs=None, filename="test", l1=0, l2=1):
        """
        Tags and classifies the parameterised data using the Viterbi function, writes the output to a csv file,
        and produces a classification report
        :param pairs: word tag pairs for the training data
        :param filename: name of the file to write the tagging results to
        :param l1: weight for the unigram probability
        :param l2: weight for the bigram probability
        :return: tagging precision
        """
        if pairs is None:
            pairs = self.test
        tags_list = sorted(list(self.tags.keys() - {"</s>", "<s>"}))

        token_list = []
        actual_list = []
        predicted_list = []

        hits = 0
        total = 0
        hits_less_tags = 0

        sentence = deque()
        for word, tag in pairs:
            sentence.append((word, tag))
            if tag == "</s>":
                sentence.popleft()
                sentence.pop()
                if self.baseline:
                    for actual, predicted in zip(sentence, self.dp_viterbi_algorithm_bigram(sentence, tags_list)):
                        token_list.append(actual[0])
                        actual_list.append(actual[1])
                        predicted_list.append(predicted[1])
                        if (actual[1] == predicted[1]):
                            hits+=1
                        if (actual[1][0] == predicted[1][0]):
                            hits_less_tags+=1
                else:
                    for actual, predicted in zip(sentence, self.dp_viterbi_algorithm(sentence, l1, l2, tags_list)):
                        token_list.append(actual[0])
                        actual_list.append(actual[1])
                        predicted_list.append(predicted[1])
                        if (actual[1] == predicted[1]):
                            hits+=1
                        if (actual[1][0] == predicted[1][0]):
                            hits_less_tags+=1

                total += len(sentence)

                sentence.clear()

        print(f"Precision: {hits / total}")
        print(f"Reduced (14) tag precision: {hits_less_tags / total}\n")

        pd.DataFrame({"Tokens": token_list, "Actual": actual_list, "Predicted": predicted_list}).to_csv(f"{filename}.csv", index=False)

        # Provides classification reports. Commented out to prevent significantly slower runtime
        # pd.DataFrame(metrics.classification_report(actual_list, predicted_list, output_dict=True)).transpose().to_csv(f"{filename}_CP.csv")
        # pd.DataFrame(metrics.classification_report(actual_list, predicted_list, output_dict=True)).transpose().to_csv(f"{filename}_reduced_tags_CP.csv")

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
        unigram_p = self.unigram[(word,tag3)]  # unigram prediction
        bigram_p = self.bigram[(tag2, tag3)]  # bigram prediction
        trigram_p = self.trigram[(tag1, tag2, tag3)]  # trigram prediction
        return l1*unigram_p + l2*bigram_p + l3*trigram_p


    def interpolation_gridsearch(self):
        """
        Iterates the classification function over a grid of parameter values for lambda_1 and lambda_2 and returns the
        values which maximise classification accuracy
        :return: optimal values for lambda_1 and lambda_2
        """
        # parameter grid
        l1_grid = np.arange(0, 1.1, 0.1)
        l2_grid = np.arange(0, 1.1, 0.1)
        l1_max = l2_max = cmax = 0

        run=1
        for i in l1_grid:
            for j in l2_grid:
                print(f"Run {run}")
                run += 1
                if i+j <= 1:
                    current = self.classification(self.val, l1=i, l2=j)
                    if current > cmax:
                        l1_max = i
                        l2_max = j
                        cmax = current
                        print(f"New Max Precision = {cmax}, l1 =  {l1_max}, l2 = , {l2_max}\n")
        return l1_max, l2_max

    # def interpolation_gridsearch(self):
    #     """
    #     Iterates the classification function over a grid of parameter values for lambda_1 and lambda_2 and returns the
    #     values which maximise classification accuracy
    #     :return: optimal values for lambda_1 and lambda_2
    #     """
    #     # parameter grid
    #     l1_grid = range(0, 11, 1)
    #     l2_grid = range(0, 11, 1)
    #     # l1_max = l2_max = cmax = 0
    #     # run=1
    #     def loops(l1, l2):
    #         run = 1
    #         l1_max = l2_max = cmax = 0
    #         for i in l1:
    #             for j in l2:
    #                 a=i/10
    #                 b=j/10
    #                 print(f"Run {run}")
    #                 run += 1
    #                 if a+b < 1:
    #                     current = self.classification(self.val, l1=a, l2=b)
    #                     if current > cmax:
    #                         l1_max = i
    #                         l2_max = b
    #                         cmax = current
    #                         print(f"New Max Precision = {cmax}, l1 =  {l1_max}, l2 = , {l2_max}\n")
    #         return l1_max, l2_max
    #
    #     Parallel(n_jobs=-1, verbose=32)(delayed(loops)(l1_grid, l2_grid) for l1_grid in l1_grid for l2_grid in l2_grid)
    #     # return l1_max, l2_max


    def dp_viterbi_algorithm(self, sentence, l1,l2, tags_list):
        """
        Produces a tag prediction for a word by implementing the Viterbi algorithm using the linearly interpolated
        probabilities from the interpolation() function.
        :param sentence: sentence to be tagged
        :return: words and their corresponding tags
        """
        l1 = l1
        l2 = l2

        p_max_tag = "<UNK>"
        for i,(word,_) in enumerate(sentence):

            p_max = float('-inf')
            if word not in self.vocab:
                p_max_tag = "<UNK>"
            else:
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

    def dp_viterbi_algorithm_bigram(self, words, tags_list):

        for i, (word,_) in enumerate(words):
            if word not in self.vocab:
                p_max_tag = "<UNK>"

            else:
                p_max = float('-inf')
                p_max_tag = "<s>"
                for j, tag in enumerate(tags_list):
                    if i == 0:
                        tag_prob = self.bigram[("<s>", tag)]
                    else:
                        tag_prob = self.bigram[(p_max_tag, tag)]
                    emission_probability = self.unigram[(word, tag)]
                    state_prob = emission_probability * tag_prob

                    if p_max < state_prob:
                        p_max_tag = tag
                        p_max = state_prob
            yield word, p_max_tag


if __name__ == "__main__":
    ## Data Processing
    # If trigram.pkl already in directory this will run significantly faster
    t0 = time.time()
    hmm = HMM("AfrikaansPOSData", baseline=True)

    ## Training

    ## Baseline Model
    # hmm.classification(hmm.val, filename="val_base.csv")
    # hmm.classification(hmm.test, filename="test_base.csv")

    ## Linear Interpolation Gridesearch
    hmm.baseline = False
    hmm.interpolation_gridsearch()
    # print(hmm.classification(hmm.val, "test", hmm.interpolation_gridsearch()))

    ## Linearly interpolated uni/bi/trigram model
    # print(f"Data Processing and Training:")
    # print(hmm.classification(hmm.val,"test"))
    # hmm.baseline = False
    # hmm.classification(hmm.val, filename="val_improved.csv")
    # hmm.classification(hmm.test, filename="test_improved.csv")



    print(f"Training runtime = {time.time()-t0} seconds")


