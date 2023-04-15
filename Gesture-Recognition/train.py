import torch
import numpy as np
import os

from sklearn.metrics import accuracy_score
    

def train(model_credentials, dataloader, args):
    model, criterion, optimizer, scheduler = model_credentials
    train_loader, val_loader = dataloader

    if args.swa:
        swa_model = torch.optim.swa_utils.AveragedModel(model)
        swa_start = args.swa_start
        swa_scheduler = torch.optim.swa_utils.SWALR(optimizer,
                                                    swa_lr=args.swa_lr)

    best_val_loss = np.inf

    for epoch in range(1, args.epochs + 1):
        model.train()

        train_loss = 0.0
        train_accuracy = 0

        for idx, (inputs, labels) in enumerate(train_loader):
            if args.noise:
                inputs += torch.from_numpy(
                    np.random.normal(loc=0,
                                     scale=args.noise_scale,
                                     size=inputs.shape)).to(torch.float32)

            optimizer.zero_grad()
            outputs = model(inputs)
            outputs = torch.softmax(outputs, dim=1)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            predict_labels = torch.argmax(outputs, dim=1)
            train_accuracy += accuracy_score(predict_labels, labels)
            train_loss += loss.item()

        if args.swa:
            if epoch > swa_start:
                swa_model.update_parameters(model)
                swa_scheduler.step()
            else:
                scheduler.step()
        else:
            scheduler.step()

        train_accuracy = train_accuracy / len(train_loader)
        train_loss = train_loss / len(train_loader)

        print(
            f"Epoch: {epoch}/{args.epochs}| Training loss: {train_loss:.4f}| Training Accuracy: {train_accuracy:.4f}"
        )

        if epoch % args.evaluation_period == 0:
            val_loss, val_accuracy = model_eval(model, val_loader, criterion)
            print(
                f"Validation loss: {val_loss:.4f}| Validation Accuracy: {val_accuracy:.4f}"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                print(f"Saving model at epoch {epoch}")
                torch.save(model.state_dict(),
                           os.path.join(args.model_save_dir, "model.pt"))

    if args.swa:
        torch.optim.swa_utils.update_bn(train_loader, swa_model)

        val_loss, val_accuracy = model_eval(swa_model, val_loader, criterion)
        print(
            f"SWA Validation Loss: {val_loss}| SWA Validation Accuracy {val_accuracy}"
        )

        print(f"Saving SWA model")
        torch.save(swa_model.state_dict(),
                   os.path.join(args.model_save_dir, "swa_model.pt"))


def model_eval(model, val_loader, criterion):
    model.eval()
    val_loss = 0.0
    val_accuracy = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)
            outputs = torch.softmax(outputs, dim=1)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            predict_labels = torch.argmax(outputs, dim=1)
            val_accuracy += accuracy_score(predict_labels, labels)

    val_accuracy = val_accuracy / len(val_loader)
    val_loss = val_loss / len(val_loader)

    return val_loss, val_accuracy
