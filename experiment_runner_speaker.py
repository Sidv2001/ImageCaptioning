import os
import json

Learning_Rates = [0.0001, 0.001, 0.00001]

lstm_input_sizes = [400, 500, 600] #Name the policy you want to use from policy net, add new policies to policy net if you wish to customize

optimizer = ["adam"]

epochs = [100]

weight_decay = [1e-5, 1e-4, 1e-3] #Optimizer weight decay for each step (weight decay is your L2 regularizer)

step_size = [60] #Learning Rate Scheduler

gammas = [0.1] #Learning rate Scheduler

#TO run this script add to the list of parameters you want. All the possible parameters are given a list here
#Add items to it to create your grid. This code will create the config files and the 
# EVERY LIST SHOULD HAVE AT LEAST ONE VALUE

for lr in Learning_Rates:
    for lstm_input_size in lstm_input_sizes:
        for optim in optimizer:
            for epoch in epochs:
                for weight in weight_decay:
                    for step in step_size:
                        for gamma in gammas:
                                parameter_dict = {
                                "referent": [
                                        {"model": "conv2d", "in_channels": 3, "out_channels": 6, "kernel_size": 5, "stride": 2}, 
                                        {"model": "conv2d", "in_channels": 6, "out_channels": 3, "kernel_size": 5, "stride": 2},
                                        {"model": "conv2d", "in_channels": 3, "out_channels": 1, "kernel_size": 5, "stride": 2},
                                        {"model": "conv2d", "in_channels": 1, "out_channels": 1, "kernel_size": 5, "stride": 1},
                                        {"model": "flatten"},
                                        {"model": "linear", "in_features": 215, "out_features": 50}
                                ],

                                "speaker_lstm": {
                                    "input_size": lstm_input_size,
                                    "hidden_size": 237,
                                    "num_layers": 1,
                                    "drop_prob": 0,
                                    "batch_first": "True"
                                },

                                "embedding" : {
                                    "token_size": 300,
                                    "vocab_size": 237,
                                    "embedding": "prag_embed_matrix.pt"
                                },
                                
                                "listener_lstm" : {
                                    "input_size": 350,
                                    "hidden_size": 150,
                                    "num_layers": 1,
                                    "drop_prob": 0,
                                    "batch_first": "True"
                                }, 

                                "post_lstm" : [ 
                                    {"model": "linear", "in_features": 150, "out_features": 75},
                                    {"model": "linear", "in_features": 75, "out_features": 25}, 
                                    {"model": "linear", "in_features": 25, "out_features": 1}
                                ],

                                'optimizer' : optim,
                                'epochs' : epoch,
                                'lr' : lr,
                                'weight_decay' : weight,
                                'step_size' : step,
                                'gamma' : gamma,

                                }
                                script_dir = os.path.dirname(os.path.abspath(__file__))
                                name = "_".join(["speaker", str(lr), str(lstm_input_size), str(weight)]) #Add whichever lists of yours that have more than one parameter
                                cfg_file_name = os.path.join(script_dir, "configs", name + ".json")
                                f = open(cfg_file_name, "w")
                                jsonString = json.dumps(parameter_dict)
                                f.write(jsonString)
                                f.close()
                                f = open(name + ".sub", "w")
                                f.write("#!/bin/bash" + "\n")
                                f.write("#SBATCH -J '" + name + "'                         # Job name" + "\n")
                                f.write("#SBATCH -o '" + name + "_%j.out'                  # output file (%j expands to jobID)" + "\n")
                                f.write("#SBATCH -e '" + name + "_%j.err'                  # error log file (%j expands to jobID)" + "\n")
                                f.write("#SBATCH -N 1                              # Total number of nodes requested" + "\n")
                                f.write("#SBATCH -n 8                                 # Total number of cores requested" + "\n")
                                f.write("#SBATCH --get-user-env                       # retrieve the users login environment" + "\n")
                                f.write("#SBATCH --mem=6000                           # server memory requested (per node)" + "\n")
                                f.write("#SBATCH -t 12:00:00                           # Time limit (hh:mm:ss)" + "\n")
                                f.write("#SBATCH --partition=default_partition       # Request partition" + "\n")
                                f.write("#SBATCH --gres=gpu:1080ti:1                  # Type/number of GPUs needed" + "\n")
                                f.write("python speaker_model_final.py --config_name '" + name + "'")
                                f.close()
                                os.system("sbatch --requeue '" + name + ".sub'")
