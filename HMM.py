import numpy as np
import pandas as pd
from collections import defaultdict

class HMM(object):

    def __init__(self,directory_name : str) -> None:
        self.training_set = self.process(pd.read_csv(f"{directory_name}\\train copy.csv",na_filter=True)) # I used windows so change \\ to /


    def process(self,df) -> list:
        """
        The process function takes in a df with a single word and tag in each row and returns a df with a single sentence (and its tags) contained in each row.
        :param df: Dataframe to be processed
        :return: Processed dataframe
        """
        processed_df = defaultdict(int)
        self.tags = defaultdict(set)

        for i,row in enumerate(df.itertuples(index=False)):

            if not pd.isna(row[1]):                
                processed_df[row] += 1
                self.tags[row[1]].add(i)

        return processed_df


    def get_tags(self,df) -> set:
        return {tup[1] for tup in df}
    
    def get_vocab(self,df) -> set:
        return {tup[0] for tup in df}


    def calculate_emmision_probability(self,word,tag,bag) -> float:
        """
        Calculates Pr(word|tag)
        """

        tag_pairs = [pair for pair in bag if tag == pair[1]]

        tag_count = sum(bag[pair] for pair in tag_pairs)
        words_tag_count = sum(bag[pair] for pair in tag_pairs if word == pair[0])
        return words_tag_count / tag_count

    def calculate_transition_probability(self,tag_1,tag_2,bag) -> float:
        """
        Calculates the probability of tag 2 appearing after tag 2.
        """
        tag_1_count = sum(bag[pair] for pair in bag if pair[1]==tag_1)
        tag_1_then_tag_2_count = 0
        
        for v in self.tags[tag_1]:
            if v+1 in self.tags[tag_2]:
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


        return tags_transition_matrix


if __name__ == "__main__":

    hmm = HMM("AfrikaansPOSData")

    hmm.create_transition_matrix()


