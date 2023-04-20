import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from dataloader import data_reader
from sklearn.svm import SVC

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import accuracy_score, f1_score
from sklearn.manifold import TSNE

import argparse
import os


def svc_model(data, target, args):
    X_train, X_val, y_train, y_val = train_test_split(data,
                                                      target,
                                                      test_size=args.val_size)

    clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
    clf.fit(X_train, y_train)

    val_predictions = clf.predict(X_val)
    print(f"Validation Accuracy: {accuracy_score(y_val, val_predictions)}")

    return clf


def test_data_reader(data_dir, max_length):
    labels = {"hi": 0, "after": 1, "back": 2}

    files = os.listdir(data_dir)
    files.sort()

    data = []
    target = []
    for loc in files:
        if ".csv" in loc:
            gesture = np.array(pd.read_csv(os.path.join("test_data", loc)))
            gesture = np.vstack([
                gesture,
                np.zeros([max_length - gesture.shape[0], gesture.shape[1]])
            ]).reshape(1, -1)

            data.append(gesture)

            loc = loc.split(".")[0]
            loc = loc[:-1]
            target.append(labels[loc])

    return np.vstack(data), np.array(target)


def testing(model, test_data, test_target):
    test_predictions = model.predict(test_data)
    print(
        f"Test Accuracy: {accuracy_score(test_target, test_predictions)}| F1 score: {f1_score(test_target, test_predictions, average='macro')}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="svm.py")

    parser.add_argument("--val_size", type=float, default=0.1)
    parser.add_argument("--random_seed", type=int, default=4)
    parser.add_argument("--tsne_dim", type=int, default=100)
    parser.add_argument("--test_data_dir", type=str, default="test_data")

    args = parser.parse_args()

    np.random.seed(args.random_seed)

    data, target, max_length = data_reader("data", args)
    data = data.reshape(data.shape[0], -1)

    test_data, test_target = test_data_reader(args.test_data_dir, max_length)

    print("SVM on Original Data")
    trained_svc = svc_model(data, target, args)
    testing(trained_svc, test_data, test_target)

    data_embed = TSNE(n_components=100, method="exact").fit_transform(
        np.vstack([data, test_data]))
    data_embed, test_data_embed = data_embed[:data.shape[0]], data_embed[
        data.shape[0]:]

    print("SVM on TSNE Data")
    trained_svc_tsne = svc_model(data_embed, target, args)
    testing(trained_svc_tsne, test_data_embed, test_target)