import importlib
from dataclasses import dataclass
import json
import os
from torch._C import ParameterDict

@dataclass
class ParamArgs:
    """Parameters for learning experiment
    """
    speaker_name: str = "Speaker_Test"
    listener_name: str = "Listener_Test"
    seed: int = None # Random seed for training
    batch_size: int = 1
    device: str = "cuda"
    epochs: int = 100
    optimizer: str = "adam"
    lr: float = 0.0001
    weight_decay: float = 1e-5
    step_size: int = 60
    gamma: float = 0.1
    image_batch_size : int = 1
    dictionary: str = "prag_dictionary.json"
    train_dataset: str = "train/speaker_train.json"
    val_dataset: str = "train/speaker_val.json"
    test_dataset: str = "test/speaker_test.json"

    def __init__(self, **kwargs):
        super().__init__()
        self.referent = kwargs["referent"]
        self.speaker_lstm = kwargs["speaker_lstm"]
        self.embedding = kwargs["embedding"]
        self.listener_lstm = kwargs["listener_lstm"]
        self.post_lstm = kwargs["post_lstm"]

        



def get_param_args(config_name):
    """Prepare ParamArgs object that holds experiment configuration details

    Args:
        config_name (str): Correspond to configs/[config_name].py file.

    Returns:
        ParamArgs : All hyperparameters for training/val/testing
    """
    script_dir = os.path.dirname(os.path.normpath(__file__))
    path = os.path.normpath(os.path.join(script_dir, "configs", config_name + ".json"))
    parameter_dict = json.load(open(path))
    param_args = ParamArgs(**parameter_dict)

    return param_args