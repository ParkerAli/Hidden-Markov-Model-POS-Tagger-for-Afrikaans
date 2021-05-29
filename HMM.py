import numpy as np
import pandas as pd
from collections import defaultdict
from queue import PriorityQueue

class HMM(object):

    def __init__(self,directory_name : str) -> None:
        self.training_set = self.process(pd.read_csv(f"{directory_name}\\train copy.csv",na_filter=False)) # I used windows so change \\ to /


    def process(self,df) -> list:
        """
        The process function takes in a df with a single word and tag in each row and returns a df with a single sentence (and its tags) contained in each row.
        :param df: Dataframe to be processed
        :return: Processed dataframe
        """
        processed_df = defaultdict(int)
        self.tags = defaultdict(set)

        self.tags['<s>'].add(0)

        for i,row in enumerate(df.itertuples(index=True)):

            if  row[0] != "NA" and row[1] != "NA":
                processed_df[row] += 1
                self.tags[row[1]].add(i+1)
            else:
                self.tags['<s>'].add(i+1) # Begin sentence tokens
        
        self.tags['<s>'].remove(max(self.tags['<s>']))

        return processed_df


    def get_tags(self,df) -> set:
        return {tup[1] for tup in df}
    
    def get_vocab(self,df) -> set:
        return {tup[0] for tup in df}

    def calculate_emmision_probability(self,word,tag,bag) -> tuple:
        """
        Calculates Pr(word|tag)
        """

        tag_pairs = [pair for pair in bag if tag == pair[1]]

        tag_count = sum(bag[pair] for pair in tag_pairs)
        words_tag_count = sum(bag[pair] for pair in tag_pairs if word == pair[0])
        return words_tag_count / tag_count

    def calculate_transition_probability(self,tag_1,tag_2,bag) -> float:
        """
        Calculates the probability of tag 2 appearing after tag 1.
        """
        tag_1_count = sum(bag[pair] for pair in bag if pair[1]==tag_1)
        tag_1_then_tag_2_count = 0
        
        for index in self.tags[tag_1]:
            if index+1 in self.tags[tag_2]:
                tag_1_then_tag_2_count+=1

        return tag_1_then_tag_2_count / tag_1_count

    def create_transition_matrix(self) -> np.matrix:
        """
        Creates a matrix of transition probabilities
        """
        tags_list = list(self.get_tags(self.training_set))
        tags_count = len(tags_list)

        tags_transition_matrix = np.zeros((tags_count,tags_count),dtype="float64")

        for i,tag_1 in enumerate(tags_list):
            for j,tag_2 in enumerate(tags_list):
                tags_transition_matrix[i,j] = self.calculate_transition_probability(tag_1=tag_1,tag_2=tag_2,bag=self.training_set)

        return tags_transition_matrix,tags_list


    def dp_viterbi_algorithm(self,words,bag,matrix,tags_list):
        pass
        # states = []

        # for i,word in enumerate(words):
        #     p_max = float('-inf')
        #     p_max_index=-1
        #     p =[]
        #     for tag in tags_list:
        #         if i == 0:
        #             tag_prob = matrix['<s>',tag]
        #         else:
        #             tag_prob = matrix[states[-1],tag]
        #         emission_probability = self.calculate_emmision_probability(word,tag,bag)
        #         state_prob = emission_probability*tag_prob
        #         p.append(state_prob)
        #         if p_max > state_prob:

        #         p_max = max(state_prob,p_max)

if __name__ == "__main__":

    hmm = HMM("AfrikaansPOSData")

    # hmm.create_transition_matrix()


