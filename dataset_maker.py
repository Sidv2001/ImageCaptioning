import json, argparse
from typing import List
import os
from collections import defaultdict
import itertools
import json
import random

class Pragmatics_Dataset_Maker:
    """
    This class works towards generating datasets that can be used for Pragmatics. 
    For this we need to create 2 different datasets.
    Speaker Dataset: 
    This dataset has every single sentence image pair where the sentence actually 
    describes the image correctly. 

    Listener / Pragmatics Dataset: 
    This dataset contains all possible events containing the following: 
    A set of n images (n defined by the user), and a sentence, which:
    Correctly describes one of these n images
    Does not describe any of the other images
    We have an index as to which image it correctly describes
    """
    def __init__(self, nlvr_train_rel_dir, nlvr_test_rel_dir, val_split=0.2, n_listen=2):
        """
        Creates 4 lists of dictionaries (speaker train and test, 
        listener/pragmatics train and test) where each dictionary is
        one example in the dataset
        Parameters:
        nlvr_train_rel_dir: The relative directory of the nlvr_train_file
        nlvr_test_rel_dir: The relative directory of the nlvr_test_file
        val_split: Percentage of the training data to use for the validation set
        n_listen: The number of images in each example of the listener/pragmatics datasets 
        """
        script_dir = os.path.dirname(os.path.abspath(__file__))

        train_file_path = os.path.join(script_dir, nlvr_train_rel_dir)
        self.nlvr_train = [json.loads(l) for l in open(train_file_path).readlines()]

        test_file_path = os.path.join(script_dir, nlvr_test_rel_dir)
        self.nlvr_test = [json.loads(l) for l in open(test_file_path).readlines()]
        self.val_split = val_split
        self.n_listen = n_listen
        self.generate_speaker_dataset()
        self.generate_list_prag_dataset()
    
    @staticmethod   
    def val_split(nlvr_file, val_split):
        """
        Splits a list (dataset) into two parts:
        The first part has 1 - val_split of the original file and the 
        second has val_split of the original file.
        """
        n = len(nlvr_file)
        split = int((1 - val_split) * n)
        random.shuffle(nlvr_file)
        train = nlvr_file[:split]
        val = nlvr_file[split:]
        return train, val


    def generate_speaker_dataset(self):
        """
        Generates the speaker dataset:
        Looks through every element of the nlvr file and checks if the label is true 
        (i.e. if the sentence correctly describes the image).
        It then adds the relevant parts of the example (Image directory + identifier, sentence)
        to the new dataset
        """

        def helper(nlvr_file):
            speaker_dataset = []
            for example in nlvr_file:
                if example["label"] == "true":
                    keys_to_extract = ["directory", "identifier", "sentence"]
                    speaker_example = {key: example[key] for key in keys_to_extract}
                    speaker_dataset.append(speaker_example)
            return speaker_dataset

        speaker_train = helper(self.nlvr_train)
        self.speaker_test = helper(self.nlvr_test)
        self.speaker_train, self.speaker_val = Pragmatics_Dataset_Maker.val_split(speaker_train, self.val_split) 

    
    def generate_list_prag_dataset(self):
        """
        Generates the dataset used for the listener model and the pragmatics part of the model.

        This functions collects all examples that are under the same identifier (i.e. have the same setnence)
        and seperates them into true and false. 
        It then generates all possible combinations of 1 true and n-1 (where n is n_listen) 
        false examples into one new example (i.e. each example will contain n_listen different examples)
        _________________________________________________________________________________________________
        IMP:
        The final image in the list is the image that is correctly defined.
        """
        def helper(nlvr_file, n_listen):
            example_sets = defaultdict(list)
            for example in nlvr_file:
                keys_to_extract = ["directory", "identifier", "sentence", "label"]
                list_prag_example = {key: example[key] for key in keys_to_extract}
                example_sets[example['identifier'].split('-')[0]].append(list_prag_example)
            list_prag = []
            a = 0
            for _, sets in example_sets.items():
                sentence = sets[0]["sentence"]
                true = [(example["identifier"], example["directory"]) for example in sets if example["label"] == "true"]
                false = [(example["identifier"], example["directory"]) for example in sets if example["label"] == "false"]
                combos = list(itertools.combinations(false, n_listen - 1))
                for true_ex in true:
                    for false_combo in combos:
                        res = list(false_combo)
                        res.append(true_ex)
                        final_res = {"sentence": sentence, "images": res}
                        list_prag.append(final_res)
                        a += 1
            return list_prag
        
        list_prag_train = helper(self.nlvr_train, self.n_listen)
        self.list_prag_test = helper(self.nlvr_test, self.n_listen)
        self.list_prag_train, self.list_prag_val = Pragmatics_Dataset_Maker.val_split(list_prag_train, self.val_split)

    def convert_to_json(self):

        def helper(nlvr_file, rel_out_file):
            print("converting_files")
            script_dir = os.path.dirname(os.path.abspath(__file__))
            train_file_path = os.path.join(script_dir, rel_out_file)
            f = open(train_file_path, "w")
            f.truncate(0)
            for example in nlvr_file:
                jsonString = json.dumps(example)
                f.write(jsonString + "\n")
            f.close()
        
        helper(self.speaker_train, "train/speaker_train.json")
        helper(self.speaker_val, "train/speaker_val.json")
        helper(self.speaker_test, "test/speaker_test.json")
        helper(self.list_prag_train, "train/list_prag_train.json")
        helper(self.list_prag_val, "train/list_prag_val.json")
        helper(self.list_prag_test, "test/list_prag_test.json")


if __name__ == "__main__":
    datasets = Pragmatics_Dataset_Maker("train/train.json", "test/test.json")
    datasets.convert_to_json()



