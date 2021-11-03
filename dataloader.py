import json, argparse
from typing import List
import os
from collections import defaultdict
import json
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import re
from PIL import Image
import torchvision


class Pragmatics_Dataloader(Dataset):
    """
    Dataloader for the Pragmatics Model.
    Loads in the dictionary and dataset, converts the sentences into the tokens they represent in the embedding
    Defined classes for the listener and speaker models that return the specific stuff respective to their classes
    """

    def __init__(self, dictionary_dir, dataset_dir, dataset_type):
        super().__init__()
        script_dir = os.path.dirname(os.path.abspath(__file__))
        dataset_path = os.path.join(script_dir, dataset_dir)
        dictionary_path = os.path.join(script_dir, dictionary_dir)
        self.dataset_type = dataset_type
        f_dict = open(dictionary_path).readlines()
        self.dictionary = json.loads(f_dict[0])
        self.dataset = [json.loads(l) for l in open(dataset_path).readlines()]
        self.sentence_tokenization()
    
    def sentence_tokenization(self):
        for j in range(len(self.dataset)):
            example = self.dataset[j]
            sentence = example["sentence"]
            sentence_list = re.findall(r"[\w']+|[.,!?;]", sentence)
            length = len(sentence_list)
            token_list = torch.zeros(length + 2, dtype=torch.long)
            for i in range(length):
                try:
                    word = sentence_list[i]
                    idx = self.dictionary[word]
                    token_list[i+1] = idx
                except KeyError:
                    idx = self.dictionary["unknown_word"]
                    token_list[i+1] = idx
            self.dataset[j]["token_sentence"] = token_list
    
    def getImage(self, img_dir, img_idt):
        """ generator tensor for referent image """
        script_dir = os.path.dirname(os.path.abspath(__file__))
        img_path = script_dir + "\\" + self.dataset_type + "\\images" + "\\" + img_dir + "\\" + self.dataset_type + "-" + img_idt + ".png"
        img = Image.open(img_path)
        transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), lambda x: x[:3]])
        t_img = transforms(img)
        return t_img

    def __len__(self):
        return len(self.dataset)
    
class Speaker_Dataset(Pragmatics_Dataloader):
    """
    Dataloader for the speaker part of the model
    Returns: 
    The tokenized sentence (Convert each word/punctuation in the sentence into a token)
    The image it describes
    """
    def __getitem__(self, index):
        example = self.dataset[index]
        example_directory = example["directory"]
        example_identifier = example["identifier"]
        choice = np.random.choice(6)
        image = self.getImage(example_directory, str(example_identifier) + "-" + str(choice))
        return (
            example["token_sentence"],
            image
        )
    
class Listener_Dataset(Pragmatics_Dataloader):
    """
    Dataloader for the listener part of the model
    Returns: 
    The tokenized sentence (Convert each word/punctuation in the sentence into a token)
    A list of images to run through and see which it would describe
    An index refering to which image is the one it actually describes (only one in the list) (can be -1)
    """
    def __getitem__(self, index):
        example = self.dataset[index]
        image_paths = example["images"]
        images = []
        for img_idx in image_paths:
            img_identifier = img_idx[0]
            img_directory = img_idx[1]
            choice = np.random.choice(6)
            image = self.getImage(img_directory, img_identifier + "-" + str(choice))
            images.append(image.expand(1, -1, -1, -1))
            correct = -1
        return (
            example["token_sentence"],
            torch.cat(images, 0),
            correct
        )
