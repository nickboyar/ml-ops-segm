import os
import shutil
import subprocess
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import torch
from model import Unet
from omegaconf import DictConfig
from skimage.transform import resize
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm


BASE_PATH = str(Path(__file__).parent.parent / "configs")


def train_model(model, train_loader, optimizer, loss_fn):
    model.train()

    train_loss = 0
    total = 0
    correct = 0

    for x, y in tqdm(train_loader, desc="Train"):
        bs = y.size(0)
        inp, targ = x, y.squeeze(1)
        optimizer.zero_grad()
        output = model(inp)
        loss = loss_fn(output.reshape(bs, 1, -1).squeeze(), targ.reshape(bs, -1))
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
        _, y_pred = output.max(dim=1)
        total += targ.size(0) * targ.size(1) * targ.size(2)
        correct += (targ == y_pred).sum().item()

    train_loss /= len(train_loader)
    accuracy = correct / total

    return train_loss, accuracy


@torch.inference_mode()
def evaluate_model(model, loader, loss_fn):
    model.eval()

    total_loss = 0
    total = 0
    correct = 0

    for x, y in tqdm(loader, desc="Evaluation"):
        bs = y.size(0)
        inp, targ = x, y.squeeze(1)
        output = model(inp)
        loss = loss_fn(output.reshape(bs, 1, -1).squeeze(), targ.reshape(bs, -1))
        total_loss += loss.item()
        _, y_pred = output.max(dim=1)
        total += targ.size(0) * targ.size(1) * targ.size(2)
        correct += (targ == y_pred).sum().item()

    total_loss /= len(loader)
    accuracy = correct / total

    return total_loss, accuracy


def bce_loss(y_real, y_pred):
    loss = y_pred - y_real * y_pred + torch.log(1 + torch.exp(-y_pred))
    return torch.mean(loss)


@hydra.main(
    config_path=BASE_PATH,
    config_name="main",
    version_base="1.2",
)
def train(cfg: DictConfig):

    os.system(cfg.dvc.pull)
    base_workdir = Path(__file__).parent.parent
    train_path = base_workdir / cfg.data_path.train_data
    list_of_train = sorted(train_path.glob("*"))
    masks = []
    images = []
    for i in range(len(list_of_train)):
        mask = plt.imread(list_of_train[i] / "mask.jpg")
        image = plt.imread(list_of_train[i] / "slice.jpg")
        masks.append(mask[:, :, 0])
        images.append(image)

    size = (cfg.resize.size, cfg.resize.size)
    X_data = [
        resize(
            x,
            size,
            mode=cfg.resize.mode,
            anti_aliasing=True,
        )
        for x in images
    ]
    Y_data = [
        resize(y, size, mode=cfg.resize.mode, anti_aliasing=False) > cfg.resize.threshold
        for y in masks
    ]

    X_data = np.array(X_data, np.float32)
    Y_data = np.array(Y_data, np.float32)

    np.random.seed(cfg.seed)
    ix = np.random.choice(len(X_data), len(X_data), False)
    tr, val = np.split(ix, [cfg.train_prep.num_train])

    train_loader = DataLoader(
        list(zip(np.rollaxis(X_data[tr], 3, 1), Y_data[tr, np.newaxis])),
        batch_size=cfg.train_prep.batch_size,
        shuffle=True,
    )
    valid_loader = DataLoader(
        list(zip(np.rollaxis(X_data[val], 3, 1), Y_data[val, np.newaxis])),
        batch_size=cfg.train_prep.batch_size,
        shuffle=True,
    )

    loss_fn = bce_loss

    model = Unet()
    optimizer = Adam(model.parameters(), lr=cfg.train_prep.lr)

    train_loss_history, valid_loss_history = [], []
    train_accuracy_history, valid_accuracy_history = [], []

    mlflow.set_tracking_uri(uri=cfg.ml_flow.server)

    mlflow.set_experiment(cfg.ml_flow.experiment_name)

    git_id = subprocess.run(
        ["git", "log", '--format="%H"', "-n", "1"], stdout=subprocess.PIPE
    )

    params = {
        "lr": cfg.train_prep.lr,
        "epoches": cfg.train_prep.epoches,
        "batch_size": cfg.train_prep.batch_size,
        "git commit id": git_id.stdout.decode("utf-8"),
    }

    for epoch in range(cfg.train_prep.epoches):
        train_loss, train_accuracy = train_model(model, train_loader, optimizer, loss_fn)
        valid_loss, valid_accuracy = evaluate_model(model, valid_loader, loss_fn)

        train_loss_history.append(train_loss)
        valid_loss_history.append(valid_loss)

        train_accuracy_history.append(train_accuracy)
        valid_accuracy_history.append(valid_accuracy)

    with mlflow.start_run():
        mlflow.log_params(params)
        for idx in range(len(train_loss_history)):
            mlflow.log_metrics(
                {
                    "train_accuracy_history": train_accuracy_history[idx],
                    "valid_accuracy_history": valid_accuracy_history[idx],
                    "train_loss": train_loss_history[idx],
                    "valid_loss": valid_loss_history[idx],
                },
                step=idx + 1,
            )

    onnx_save_path = base_workdir / cfg.data_path.onnx_save_path

    if os.path.exists(onnx_save_path):
        shutil.rmtree(onnx_save_path)

    os.mkdir(onnx_save_path)

    torch.onnx.export(
        model,
        torch.randn(1, 3, 256, 256, requires_grad=True),
        onnx_save_path / cfg.data_path.name_model,
        export_params=True,
        opset_version=cfg.train_prep.opset,
        do_constant_folding=True,
        input_names=["modelInput"],
        output_names=["modelOutput"],
    )

    os.system(cfg.dvc.add_model)
    os.system(cfg.dvc.push)


if __name__ == "__main__":
    train()
