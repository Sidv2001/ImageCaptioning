import dataloader
import os
import json
import torchvision
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import global_config
import matplotlib
import matplotlib.pyplot as plt
import nltk
import importlib
from dataclasses import dataclass
import time
import pickle
import torchvision.models as models
import torchvision.transforms as transforms



matplotlib.use("Agg")

plt.style.use("ggplot")

@dataclass
class AvgResult:
    """Averaged result of loss and IPS objectives
    """

    sum_loss: float
    num_samples: int
    name: str

    def _avg(self, val):
        return val / self.num_samples

    def avg_loss(self):
        return self._avg(self.sum_loss)

    def to_dict(self):
        """Convert the object to a dictionary

        Returns:
            dict: Averaged loss and IPS objectives in dictionary format
        """
        return {
            "avg_loss": self.avg_loss(),
        }

    @property
    def log_str(self):
        return (
            f"{self.name} Loss {self.avg_loss():.4f} "

        )


class BestManager:
    def __init__(self):
        self.best_epoch = -1
        self.best_score = None
        self.best_model = None

    def is_best(self, score):
        return self.best_score is None or score < self.best_score

    def store_if_best(self, *, score, network, epoch, msg):
        if self.is_best(score):
            self.best_epoch = epoch
            self.best_score = score
            self.best_model = network.state_dict()
            print(msg)

class Listener_Model(torch.nn.Module):
    def __init__(self, lstm, embedding, post_lstm, device):
        super(Listener_Model, self).__init__()
        self.device = device
        output_size = lstm["input_size"] - 300
        self.listener_ref = (self.construct_resnet(output_size)).to(self.device)
        self.listener_lstm = (self.construct_lstm(lstm)).to(self.device)
        self.embedding = (self.construct_embedding(embedding)).to(self.device)
        self.embedding.requires_grad_ = False
        self.post_lstm = self.construct_sequential(post_lstm)
    
    def forward(self, example):
        tkn_sentence, sentence_imgs, correct = example
        imgs = torch.squeeze(sentence_imgs)
        sentence = torch.squeeze(tkn_sentence)
        ref = self.listener_ref(imgs.to(self.device))
        embed_sentence = self.embedding(sentence.to(self.device))
        length = embed_sentence.shape[0]
        num_imgs = ref.shape[0]
        lstm_ref = torch.unsqueeze(ref, 1).expand(-1, length, -1)
        lstm_sent = torch.unsqueeze(embed_sentence, 0).expand(num_imgs, -1, -1)
        res = torch.cat([lstm_sent, lstm_ref], dim=2)
        lstm_out, (_, _) = self.listener_lstm(res)
        fin_lstm = torch.sum(lstm_out, 1)
        fin = self.post_lstm(fin_lstm)
        return fin


    def construct_resnet(self, output_size):
        model =  models.resnet18(pretrained=True)
        model.fc = nn.Linear(in_features=512, out_features=output_size)
        return model




    def construct_layer(self, obj):
        """
        Constructs a layer of a sequential model, that is either a convolutional layer, 
        a linear layer, or a flatten function (which just flattens out the vector for our use)
        
        Args:
            obj (Dictionary): Parameters for for the layer (name of function, and parameters for that layer)
        """
        if obj["model"] == "conv2d":
            in_channels = obj["in_channels"]
            out_channels = obj["out_channels"]
            kernel = obj["kernel_size"]
            stride = obj["stride"]
            return nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel, stride=stride)
        
        elif obj["model"] == "linear":
            in_features = obj["in_features"]
            out_features = obj["out_features"]
            return nn.Linear(in_features=in_features, out_features=out_features)

        elif obj["model"] == "flatten":
            return Flatten()
        
        else:
            raise NotImplementedError()
    
    def construct_sequential(self, lst):
        """
        Constructs a  sequential model, by taking a lst of layer parameters and running the 
        construct layer function on the layers. 
        
        Args:
            lst (List of Dictionaries): List of Parameters for for the layer (name of function, and parameters for that layer)
        """
        res = []
        for obj in lst:
            layer = self.construct_layer(obj)
            res += [layer]
            res += [nn.ReLU()]
        return nn.Sequential(*res)

    def construct_lstm(self, obj):
        """
        Constructs and LSTM, given the the parameters for the LSTM (input size, hidden size, num_layers and drop prob)
        Arguments for the LSTM are provided in a dictionary so as to allow more parameters to be modififed with lesser work.

        Args:
            obj (Dictionary): Parameters for for the LSTM (input size, hidden size, num_layers and drop prob, and batch first)
        """
        input_size = obj["input_size"]
        hidden_size = obj["hidden_size"]
        num_layers = obj["num_layers"]
        drop_prob = obj["drop_prob"]
        if obj["batch_first"] == "True":
            batch_first = True
        else:
            batch_first = False

        return nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            dropout=drop_prob, 
            batch_first=batch_first
            )
    
    def construct_embedding(self, obj):
        """
        Constructs the embedding layer for the model, by generating an NN.Embedding of the appropriate size
        and then loading in the embedding matrix which has been pre caclulated. 
        
        Args:
            obj (Dictionary): Parameters for the embedding: Token size, Vocab Size and the path to the embedding matrix
        """

        token_size = obj["token_size"]
        vocab_length = obj["vocab_size"]
        self.vocab_size = vocab_length
        embedding = nn.Embedding(vocab_length, token_size)
        weights = torch.load(obj["embedding"])
        embedding.load_state_dict({"weight": weights})
        return embedding



class Listener:
    """listener Training Model. It contains methods for optimizing and testing a an image captioning model 

    Args:
        param_args (global_config.ParamArgs): Parameters for training
    """

    def __init__(self, param_args):

        train_dir = param_args.train_dataset
        val_dir = param_args.val_dataset
        test_dir = param_args.train_dataset
        dictionary_dir = param_args.dictionary

        train_data = dataloader.Listener_Dataset(dictionary_dir, train_dir, "train")
        val_data = dataloader.Listener_Dataset(dictionary_dir, val_dir, "train")
        test_data = dataloader.Listener_Dataset(dictionary_dir, test_dir, "test") 

        self.train_loader = torch.utils.data.DataLoader(train_data, batch_size=1,
                        shuffle=True, num_workers=6)
        self.val_loader = torch.utils.data.DataLoader(val_data, batch_size=1,
                            shuffle=False, num_workers=6)
        self.test_loader = torch.utils.data.DataLoader(test_data, batch_size=1,
                            shuffle=False, num_workers=6)
        self.batch_size = param_args.batch_size
        self.device = param_args.device
        self.epochs = param_args.epochs
        self.lr = param_args.lr
        self.weight_decay = param_args.weight_decay
        self.step_size = param_args.step_size
        self.network = Listener_Model(param_args.listener_lstm, param_args.embedding, param_args.post_lstm, self.device).to(self.device)
        self.dictionary = self.load_dictionary(param_args.dictionary)
        self.optimizer = self.construct_optimizer(param_args.optimizer, param_args.lr, param_args.weight_decay)
        self.loss_fn =  nn.CrossEntropyLoss()
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=param_args.step_size, gamma=param_args.gamma)


    def load_dictionary(self, dictionary_dir):
        """
        Loads in the dictionary which contains the conversion from word to token

        Args:
            dictionary_dir (string): Path to the json file containing the pre-compiled dictionary
        """
        script_dir = os.path.dirname(os.path.abspath(__file__))
        dictionary_path = os.path.join(script_dir, dictionary_dir)
        f_dict = open(dictionary_path).readlines()
        dictionary = json.loads(f_dict[0])
        return dictionary

    def construct_optimizer(self, optim_type, learning_rate, L2):
        """
        Constructs the optimizer for the model, and lists out which of the parameters are
        updated based on the gradients calculated. (This is Important in the Pragmatic_listener model)
        """
        if optim_type == "sgd":
            return torch.optim.SGD(self.network.parameters(), lr=learning_rate, weight_decay=L2)
        if optim_type == "adam":
            return torch.optim.Adam(self.network.parameters(), lr=learning_rate, weight_decay=L2)

    def untokenize_sentence(self, tkn_sentence):
        key_list = list(self.dictionary.keys())
        val_list = list(self.dictionary.values())
        sentence = []
        for word in tkn_sentence:
            position = val_list.index(word)
            sentence.append(key_list[position])
        return sentence


    def process_batch(self, example):
        """Compute loss and Bleu Score for a batch of image captioning samples.

        Gradient will be saved in self.network, if torch.set_grad

        Args:
            Example: A tuple containing the following:
            Image (torch.FloatTensor) : List of dense matrices with each matrix being an image
            Token_Sentence (torch.Long): A tokenized sentence that is the caption for that image. 

        Returns:
            tuple : batch_loss
        """
        tkn_sentence, sentence_img, correct = example

        predicted = self.network(example).reshape(1, -1)
        target = torch.tensor(correct)
        batch_loss = self.loss_fn(predicted, target.to(self.device))
        
        return batch_loss


    def train(self, loader):
        """Train the policy network on self.train_loader

        Returns:
            AvgResult: Avg loss and Bleu_Score for train set
        """
        loader = self.train_loader

        self.network.train()

        sum_loss = 0.0
        num_samples = 0
        batch = 0
        batch_loss = 0
        for batch_idx, d in enumerate(loader):
            self.optimizer.zero_grad()
            loss_val = self.process_batch(d)
            batch_loss += loss_val
            batch += 1
            if batch >= self.batch_size:
                batch_loss.backward()
                self.optimizer.step()
                batch_loss = 0
                batch = 0
            sum_loss += float(loss_val.clone().detach())
            num_samples += 1

        return AvgResult(sum_loss, num_samples, "Train")

    def test(self, loader):
        """Evaluate the policy network on loader

        Args:
            loader (PyTorch DataLoader) : Dataloader of image captioning

        Returns:
            AvgResult: Avg loss and IPS on this loader
        """
        loader = self.train_loader

        self.network.eval()

        sum_loss = 0.0
        num_samples = 0
        for batch_idx, d in enumerate(loader):
            with torch.set_grad_enabled(False):
                loss_val = self.process_batch(d)
                sum_loss += float(loss_val.clone().detach())
                num_samples += 1

        return AvgResult(sum_loss, num_samples, "Val")

    def optimize(self):
        """Perform train+val loop

        Returns:
            tuple: (train_results, val_results, best_val_softmax_ips_model)
                   train_results and val_results are lists of AvgResult.
                   best_val_softmax_ips_model is the policy network that achieves best softmax IPS objective on val set.
        """
        train_results = []
        val_results = []

        best_manager = BestManager()

        for epoch in range(self.epochs):
            start = time.time()

            train_result = self.train(self.train_loader)
            train_results.append(train_result)

            val_result = self.test(self.val_loader)
            val_results.append(val_result)

            self.scheduler.step()
            end = time.time()

            # Log what's happening
            print(
                f"Epoch {epoch}: \n"
                f"{train_result.log_str}\n{val_result.log_str}\n"
                f"Elapsed Time {end-start:.2f} sec."
            )

            # Save the state_dict of the model that achieves best validation
            # set average softmax IPS objective
            best_manager.store_if_best(
                score=val_result.avg_loss(),
                epoch=epoch, network=self.network, msg=(
                    f"Best validation set loss at epoch "
                    f"{epoch}: {val_result.avg_loss():.4f}"
                )
            )
            print("")

        print(
            f"Use the model checkpoint at epoch {best_manager.best_epoch} on test set evaluation."
        )
        self.network.load_state_dict(best_manager.best_model)
        return train_results, val_results, best_manager.best_model


def load_or_create_pickle_file(create_func, save_dir):
    """Create a pickle file using create_func() or load from existing location

    Args:
        create_func (func): Call create_func() to create the object to be saved
        save_dir (str): Where the pickle file is or will be saved

    Returns:
        object: The object loaded from save_dir or from create_func()
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    pickle_path = os.path.join(save_dir, "result.pickle")
    if os.path.exists(pickle_path):
        print(f"Pickle File already exists. Loading from {pickle_path}")
        with open(pickle_path, "rb") as picklefile:
            result = pickle.load(picklefile)
    else:
        print(f"Creating pickle file at {pickle_path}")
        result = create_func()
        with open(pickle_path, "wb") as picklefile:
            pickle.dump(result, picklefile)
    return result


def main():
    """Start Image Captioning learning experiment
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_name",
        default="basic",
        help="The config file name should correspond to the configs/[config_name].py",
    )
    config_args = parser.parse_args()

    param_args = global_config.get_param_args(config_args.config_name)

    # Set manual seed if seed is not None
    if param_args.seed != None:
        torch.manual_seed(param_args.seed)
        if param_args.device == "cuda":
            torch.cuda.manual_seed(param_args.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    
    listener = Listener(param_args)

    (
        train_results,
        val_results,
        best_val_softmax_ips_model,
    ) = listener.optimize()  # Perform train + val loops for param_args.epochs

    test_results = listener.test(
        listener.test_loader
    )  # This performs testing using the model that achieve the best valiation set softmax IPS Objective

    # Now log all the results
    out_dir = prepare_output_folder(config_args.config_name)

    print(
        f"Test Loss {test_results.avg_loss():.4f} "
    )
    with open(os.path.join(out_dir, f"test_result.json"), "w+") as file:
        file.write(json.dumps(test_results.to_dict()))

    print(
        f"Save the model that achieves best loss on validation set to {os.path.join(out_dir, 'best_val_loss_model.pt')}"
    )
    torch.save(
        best_val_softmax_ips_model,
        os.path.join(out_dir, "best_val_loss_model.pt"),
    )

    print(
        f"Copy the parameter dict to output folder at {os.path.join(out_dir, 'parameter_dict.json')}"
    )
    with open(os.path.join(out_dir, "parameter_dict.json"), "w+") as file:
        file.write(json.dumps(param_args.__dict__))

    print(f"Saving the results on train/val/test sets in {out_dir}")
    plot_names = ["avg_loss"]
    for results, phase_name in [
        (train_results, "train"),
        (val_results, "val"),
    ]:
        # First save everything in a json file
        with open(os.path.join(out_dir, f"{phase_name}_result.json"), "w+") as file:
            file.write(json.dumps([res.to_dict() for res in results]))

        for plot_name in plot_names:
            curve = [getattr(res, plot_name)() for res in results]
            if type(curve) == list:
                # Plot the curve
                img_path = os.path.join(out_dir, f"{phase_name}_{plot_name}.png")
                plt.figure(figsize=(10, 10))
                axes = plt.gca()
                plt.title(f"{phase_name} {plot_name} plot", fontsize=12)
                plt.xlabel("Epoch", fontsize=12)
                plt.ylabel(f"{plot_name}", fontsize=12)

                plt.plot(range(len(curve)), curve, linestyle="-", label=plot_name)
                plt.legend()
                plt.tight_layout()
                plt.savefig(img_path)
                plt.close("all")


def prepare_output_folder(config_name):
    """Create the output folder if not exist

    Args:
        config_name (str): The configuration name corresponding to configs/[config_name].py

    Returns:
        str: out_dir
             The output directory
    """
    print(f"Using the config file: {os.path.join('configs', config_name)}.py")
    out_dir = os.path.join("outputs", config_name)
    print(f"Output folder is: {out_dir}")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    return out_dir


if __name__ == "__main__":
    main()

