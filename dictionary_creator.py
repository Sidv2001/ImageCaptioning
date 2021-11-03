import os
import numpy as np 
import torch
import json
import re

class DictionaryEmbedding:
    """
    This class is used to create a dictionary of words which is then associated to 
    its embedding, which is found in the glove models we load in. 
    This class can take multiple files to create a dictionary with as well as different embeddings.

    If the word does not exist in the embedding dataset, we create a new random vector for the word. 
    At the end we add a start, end and unknown token to the embedding, for training and inference purposes. 

    Requirements:
    Every word in the embedding dataset should have a vector of the same length as it embedding.
    You must define a subclass where you write code to explain how to load your datasets into the 
    required format.

    The loaded dataset must be a list of sentences, not necesarily unique.
    The loaded embeddings should be a dictiorary where the key is a word and the value is a numpy
    array representing the vector embeddings.


    """

    def __init__(self, embedding_rel_dir, dict_output_name="prag_dictionary.json", embed_output_name="prag_embed_matrix.pt", dataset_paths_rel_dir=[]):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        total_data = []
        for data_rel_dir in dataset_paths_rel_dir:
            dataset_path = os.path.join(script_dir, data_rel_dir)
            dataset = self.load_dataset(dataset_path)
            total_data += dataset
        self.dataset = total_data
        embedding_path = os.path.join(script_dir, embedding_rel_dir)
        self.embedding, self.vector_size = self.load_embedding(embedding_path)
        self.vocab = self.dataset_to_list_of_word()
        self.dictionary, self.embed_matrix = self.create_dictionary()
        self.output_dict_embed(dict_output_name, embed_output_name)

    def dataset_to_list_of_word(self):
        words = set()
        for sentence in self.dataset:
            for word in re.findall(r"[\w']+|[.,!?;]", sentence):
                words.add(word)
        return list(words)
    
    def create_dictionary(self):
        length = len(self.vocab)
        embed_matrix = []
        dictionary = {}
        idx = 0
        for word in self.vocab:
            try:
                vect_embed = self.embedding[word.lower()]
                embed_matrix.append(vect_embed)
                dictionary[word.lower()] = idx
                idx += 1
            except KeyError:
                print (word)
        dictionary["unknown_word"] = idx 
        dictionary["start_sentence"] = idx+1 #start
        dictionary["end_sentence"] = idx+2 #end
        for i in range(3):
            embed_matrix.append(np.random.rand(self.vector_size))
        
        return dictionary, np.array(embed_matrix)

    
    def output_dict_embed(self, dict_output_name, embed_output_name):
        embed_output = torch.tensor(self.embed_matrix)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        dict_file_path = os.path.join(script_dir, dict_output_name)
        f = open(dict_file_path, "w")
        f.truncate(0)
        jsondict = json.dumps(self.dictionary)
        f.write(jsondict)
        f.close()
        torch.save(embed_output, embed_output_name)

    
    def load_dataset(self, data):
        raise NotImplementedError()
    
    def load_embedding(self, data):
        raise NotImplementedError()
    

class Sid_NLVR_Glove_Dictionary(DictionaryEmbedding):
    """
    This Class is for unloading Sidharth Vasudev's custom nlvr dataset 
    made by dataset_maker, as well as loading in the glove models
    """

    def load_dataset(self, dataset_path):
        dataset = [json.loads(l) for l in open(dataset_path).readlines()]
        def extract_sentence(dictionary):
            return dictionary["sentence"]
        list_of_sentences = list(map(extract_sentence, dataset))
        return list_of_sentences
    
    def load_embedding(self, embedding_path):
        words = {}
        with open(embedding_path) as f:
            for l in f.readlines():
                line = l.split()
                word = line[0]
                vect = np.array(line[1:]).astype(np.float64)
                words[word] = vect
        return words, len(vect)

if __name__ == "__main__":
    Sid_NLVR_Glove_Dictionary("glove/glove.42B.300d.txt", dataset_paths_rel_dir=["train/speaker_train.json", "train/speaker_val.json"])



