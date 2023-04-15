import os
import numpy as np
import pandas as pd
import torch
import argparse

from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

import json


def dataloader_gen(data, target, normalisation, batch_size, test_ratio):
    data = torch.reshape(torch.from_numpy(data),
                         [data.shape[0], data.shape[2], data.shape[1]]).type(
                             torch.float32)

    target = torch.from_numpy(target).type(torch.long)

    print(f"Data Shape: {data.shape}| Target Shape: {target.shape}")

    if normalisation:
        data = torch.nan_to_num(data -
                                torch.mean(data, dim=2).unsqueeze(dim=2) /
                                torch.std(data, dim=2).unsqueeze(dim=2))

    train_X, test_X, train_y, test_y = train_test_split(data,
                                                        target,
                                                        test_size=test_ratio,
                                                        shuffle=True)

    train_dataset = TensorDataset(train_X, train_y)
    val_dataset = TensorDataset(test_X, test_y)

    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


def data_reader(data_dir):

    actions = ["hi", "after", "back"]
    data = []
    max_lengths = [0, 0, 0]

    for idx, action in enumerate(actions):
        data.append([])

        for file in os.listdir(os.path.join(data_dir, action)):
            if ".csv" in file:
                gesture = pd.read_csv(os.path.join(data_dir, action, file))
                data[idx].append(np.array(gesture))

                if len(gesture) > max_lengths[idx]:
                    max_lengths[idx] = len(gesture)

    args.max_length = max(max_lengths)
    args.input_size = gesture.shape[1]
    # padding
    for i in range(len(actions)):
        for idx, gesture in enumerate(data[i]):
            data[i][idx] = np.vstack([
                gesture,
                np.zeros(
                    [args.max_length - gesture.shape[0], gesture.shape[1]])
            ])
        data[i] = np.array(data[i])

    target = np.hstack([[0] * len(data[0]), [1] * len(data[1]),
                        [2] * len(data[2])])
    data = np.vstack([data[0], data[1], data[2]])

    print(f"Data Shape: {data.shape}| Target Shape: {target.shape}")
    return data, target


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="dataloader.py")

    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--normalisation", type=bool, default=True)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--data_save_dir", type=str, default="dataloader")

    args = parser.parse_args()

    data, target = data_reader(args.data_dir)
    train_loader, val_loader = dataloader_gen(data, target, args.normalisation,
                                              args.batch_size, args.test_ratio)

    if not os.path.isdir(args.data_save_dir):
        os.mkdir(args.data_save_dir)

    torch.save(train_loader,
               os.path.join(args.data_save_dir, "train_loader.pth"))
    torch.save(val_loader, os.path.join(args.data_save_dir, "val_loader.pth"))

    with open(os.path.join(args.data_save_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f)
