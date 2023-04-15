import numpy as np
from model.lstm import LSTM
import torch
import torch.nn as nn
import os

from train import train

import json
import argparse


def main(args):
    train_loader = torch.load(os.path.join(args.data_dir, "train_loader.pth"))
    val_loader = torch.load(os.path.join(args.data_dir, "val_loader.pth"))

    data_arguments = json.load(open(os.path.join(args.data_dir,
                                                 'config.json')))

    args.input_size = data_arguments["max_length"]
    args.hidden_size = data_arguments["input_size"]

    model = LSTM(args.input_size, args.hidden_size, args.num_layers,
                 args.output_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                           T_max=args.epochs)

    model_credentials = (model, criterion, optimizer, scheduler)
    dataloader = (train_loader, val_loader)

    train(model_credentials, dataloader, args)

    with open(os.path.join(args.model_save_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="main.py")

    parser.add_argument("--data_dir", type=str, default="dataloader")
    parser.add_argument("--model_save_dir", type=str, default="trained_model")

    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--evaluation_period", type=int, default=1)

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)

    parser.add_argument("--swa",
                        type=str,
                        default="True",
                        choices=["True", "False"])
    parser.add_argument("--swa_start", type=int, default=30)
    parser.add_argument("--swa_lr", type=float, default=1e-4)

    parser.add_argument("--noise",
                        type=str,
                        default="True",
                        choices=["True", "False"])
    parser.add_argument("--noise_scale", type=float, default=1e-1)
    parser.add_argument("--random_seed", type=int, default=4)

    args = parser.parse_args()

    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)

    args.output_size = 3
    args.swa = True if args.swa == "True" else False
    args.noise = True if args.noise == "True" else False

    main(args)