import torch
import os
import argparse
from model.lstm import LSTM
import numpy as np
import pandas as pd
import json

from predict import predict

parser = argparse.ArgumentParser(description="main.py")

parser.add_argument("--swa",
                    type=str,
                    default="False",
                    choices=["True", "False"])
parser.add_argument("--trained_model_dir", type=str, default="trained_model")

args = parser.parse_args()
args.swa = True if args.swa == "True" else False
args.label_dic = {0: "hi", 1: "after", 2: "back"}

model_name = "swa_model.pt" if args.swa else "model.pt"
model_arguments = json.load(
    open(os.path.join(args.trained_model_dir, 'config.json')))

max_length = model_arguments["input_size"]

model = LSTM(model_arguments["input_size"], model_arguments["hidden_size"],
             model_arguments["num_layers"], model_arguments["output_size"])

model = torch.optim.swa_utils.AveragedModel(model) if args.swa else model

model.load_state_dict(
    torch.load(os.path.join(args.trained_model_dir, model_name)))

files = os.listdir("test_data")
files.sort()

accuracy = 0
counter = 0
for loc in files:
    if ".csv" in loc:
        print(loc, end=" ")
        gesture = np.array(pd.read_csv(os.path.join("test_data", loc)))
        output = predict(model, gesture, max_length, args)

        loc = loc.split(".")[0]
        loc = loc[:-1]
        if output == loc:
            accuracy += 1
        counter += 1

print(accuracy, counter)
print(accuracy / counter)
print(f"Accuracy: {accuracy}/{counter}")