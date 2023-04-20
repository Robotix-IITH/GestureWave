import torch
import numpy as np
import pandas as pd
import argparse
import json
import os

from model.lstm import LSTM


def predict(model, gesture, max_length, args):
    model.eval()
    with torch.no_grad():
        gesture = np.vstack([
            gesture,
            np.zeros([max_length - gesture.shape[0], gesture.shape[1]])
        ])
        gesture = torch.from_numpy(gesture)
        gesture = gesture.unsqueeze(dim=0)
        shp = gesture.shape
        gesture = torch.reshape(gesture,
                                [shp[0], shp[2], shp[1]]).type(torch.float32)

        output = model(gesture)
        output = torch.softmax(output, dim=1)

        output = args.label_dic[torch.argmax(output, dim=1).item()]

    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="main.py")

    parser.add_argument("--swa",
                        type=str,
                        default="False",
                        choices=["True", "False"])
    parser.add_argument("--trained_model_dir",
                        type=str,
                        default="trained_model")
    parser.add_argument("--gesture_loc", type=str, required=True)

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

    gesture = np.array(pd.read_csv(args.gesture_loc))

    predict(model, gesture, max_length, args)
