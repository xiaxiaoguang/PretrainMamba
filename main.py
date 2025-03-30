import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR
import argparse
import sys
import toml
import os
import numpy as np
import datetime
import shutil
import time

from dataprovider.data_factory import data_provider
from dataprovider.data_factory import load_adj
from dataprovider.dataArgs import load_arg

from matplotlib import pyplot as plt
import pandas as pd

from model import ModelArgs,Mamba,MambaPre

config = toml.load("./config.toml")

parser = argparse.ArgumentParser()

parser.add_argument("--model_path", default=None,type=str)
parser.add_argument("--frozen_type", default=None, type=str)
parser.add_argument("--pred_len", default=-1, type=int)
parser.add_argument("--pre_layer", default=-1,type=int)
parser.add_argument("--dataset_name", default="ETTh1", type=str)
parser.add_argument("--device", default=0, type=int)
parser.add_argument("--mod", default="pretrain", type=str)

args = parser.parse_args()


def masked_mse(preds, labels, null_val=0.0):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = labels != null_val
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = (preds - labels) ** 2
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_mae(preds, labels, null_val=0.0):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = labels != null_val
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds - labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_mape(preds, labels, null_val=0.0):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = labels != null_val
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds - labels) / labels
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_rmse(preds, labels, null_val=0.0):
    return torch.sqrt(masked_mse(preds=preds, labels=labels, null_val=null_val))


def repeat_and_add_noise(input_tensor, repeat_times=3, noise_std=0.01):
    """
    Repeat each time step in the input tensor and add Gaussian noise.

    Args:
        input_tensor (torch.Tensor): The input tensor of shape (B, T, F).
        repeat_times (int): The number of times each time step is repeated.
        noise_std (float): The standard deviation of the Gaussian noise to be added.

    Returns:
        torch.Tensor: The output tensor with shape (B, 3T, F).
    """
    B, T, F = input_tensor.shape
    repeated_tensor = input_tensor.unsqueeze(2).repeat(1, 1, repeat_times, 1)
    repeated_tensor = repeated_tensor.view(B, T * repeat_times, F)
    noise = torch.randn_like(repeated_tensor) * noise_std
    noisy_tensor = repeated_tensor + noise
    return noisy_tensor


def info_nce_loss(anchor, positive_samples, negative_samples, temperature=0.1):
    """
    Compute the InfoNCE loss with multiple positives and negatives.

    Args:
        anchor (torch.Tensor): The anchor sample of shape (batch_size, embedding_dim).
        positive_samples (torch.Tensor): The positive samples of shape (batch_size, num_positives, embedding_dim).
        negative_samples (torch.Tensor): The negative samples of shape (batch_size, num_negatives, embedding_dim).
        temperature (float): The temperature parameter for scaling the logits.

    Returns:
        torch.Tensor: The computed InfoNCE loss.
    """
    pos_sample_num = positive_samples.shape[1]
    anchor = F.normalize(anchor, dim=-1)
    positive_samples = F.normalize(positive_samples, dim=-1)
    negative_samples = F.normalize(negative_samples, dim=-1)
    pos_logits = torch.einsum("bd,bpd->bp", anchor, positive_samples) / temperature
    pos_logits = pos_logits.view(-1, 1)  # (batch_size * num_positives, 1)
    neg_logits = torch.einsum("bd,bnd->bn", anchor, negative_samples) / temperature
    neg_logits = neg_logits.view(
        -1, 1, negative_samples.size(1)
    )  # (batch_size, num_negatives)
    neg_logits = neg_logits.repeat(1, pos_sample_num, 1)
    neg_logits = neg_logits.view(-1, negative_samples.size(1))
    logits = torch.concat(
        [pos_logits, neg_logits], dim=1
    )  # (batch_size * num_positives, 1 + num_negatives)
    labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
    loss = F.cross_entropy(logits, labels, reduction="mean")
    return loss


def cal_temporal_memory_loss(o_x, o_a, repeat_time=3):
    length = o_x.shape[1]
    losses = []

    for i in range(length - 1):
        anchor = o_x[:, i, ...]
        pos = o_a[:, i * repeat_time : repeat_time * (i + 1), ...]
        neg = torch.concat(
            [
                o_a[:, repeat_time * (i + 1) : repeat_time * (i + 2)],
                o_x[:, i + 1 : i + 2, ...],
            ],
            dim=-2,
        )
        losses.append(
            info_nce_loss(anchor=anchor, positive_samples=pos, negative_samples=neg)
        )

    return torch.mean(torch.stack(losses))


def inference_main():
    global config
    config = config["inference"]
    device = f"cuda:{args.device}"
    data_args = load_arg(name=args.dataset_name)

    if args.pre_layer != -1:
        config["model"]["pre_layer"] = [args.pre_layer]
    if args.pred_len != -1:
        data_args.pred_len = args.pred_len
    if args.frozen_type != None:
        config["model"]["frozentype"] = args.frozen_type
    if args.model_path != None:
        config['model']['load_path'] = args.model_path
    

    data_args.device = device

    train_dataset, train_dataloader = data_provider(data_args, "train")
    val_dataset, val_dataloader = data_provider(data_args, "val")
    test_dataset, test_dataloader = data_provider(data_args, "test")


    feat_num = next(iter(train_dataloader))[0].shape[-1]
    seq_len = next(iter(train_dataloader))[0].shape[-2]
    pred_len= next(iter(train_dataloader))[1].shape[-2]
    label_num = next(iter(train_dataloader))[1].shape[-1]
    
    model_args=ModelArgs( 
        d_conv =config['model']['d_conv'],
        e_fact = config['model']['e_fact'],
        seq_len=data_args.seq_len,
        pred_len=data_args.pred_len,
    )
    model = Mamba(model_args)

    print(f"input dim {feat_num}, pred dim {label_num}, seq len {seq_len}, pred len {pred_len}")

    if config["model"]["pre_layer"][0] != -1:
        model.load_params(
            config["model"]["pre_layer"],
            config["model"]["load_path"],
            config["model"]["frozentype"],
        )

    model = model.to(device)

    opti = torch.optim.Adam(
        model.parameters(),
        lr=config["train"]["lr"],
        weight_decay=config["train"]["weight_decay_rate"],
        amsgrad=False,
    )
    scheduler = MultiStepLR(opti,milestones=config['train']['milestones'] ,gamma=config['train']['gamma'])


    save_path = os.path.join(
        os.path.dirname(config["model"]["load_path"]),
        "&".join(np.array(config["model"]["pre_layer"]).astype(str)),
        config["model"]["frozentype"],
        str(data_args.pred_len),
        f"{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}",
    )

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    shutil.copy("./config.toml",save_path)
    shutil.copy("./model.py",save_path)

    train_loss = []
    val_loss = []

    best_loss = np.inf
    now_time = time.time()
    for epoch in range(config["train"]["epoch"]):
        print(f"----{epoch}----")
        model.train()
        epoch_loss = []
        for x, y in train_dataloader:
            batch_size, stock_num, time_step, feature_num = x.shape
            x = x.reshape(batch_size * stock_num, time_step, feature_num).to(
                torch.float32
            )
            batch_size, stock_num, time_step, feature_num = y.shape
            y = y.reshape(batch_size * stock_num, time_step, feature_num).to(
                torch.float32
            )

            x, y = x.to(device), y.to(device)
            o=model(x)
            loss = masked_mae(o, y)
            opti.zero_grad()
            loss.backward()

            opti.step()
            epoch_loss.append(loss.detach().cpu().numpy())
        
        scheduler.step()
        print(np.mean(epoch_loss),time.time()- now_time)
        
        train_loss.append(np.mean(epoch_loss))

        epoch_loss = []

        model.eval()
        for x, y in val_dataloader:
            batch_size, stock_num, time_step, feature_num = x.shape
            x = x.reshape(batch_size * stock_num, time_step, feature_num).to(
                torch.float32
            )
            batch_size, stock_num, time_step, feature_num = y.shape
            y = y.reshape(batch_size * stock_num, time_step, feature_num).to(
                torch.float32
            )

            x, y = x.to(device), y.to(device)

            o=model(x)

            loss = masked_mae(o, y)

            epoch_loss.append(loss.detach().cpu().numpy())

            if loss.detach().cpu().numpy() < best_loss:
                best_loss = loss.detach().cpu().numpy()
                torch.save(model, os.path.join(save_path, "best_model.pt"))

        with open(os.path.join(save_path, "loss.txt"),'a') as file:
            print(train_loss[-1],np.mean(epoch_loss),file=file)

        print(np.mean(epoch_loss))
        val_loss.append(np.mean(epoch_loss))
        
        if epoch % 10 == 0:
            plt.clf()
            plt.plot(train_loss)
            plt.savefig(os.path.join(save_path, "pretrain_train_loss.png"))

            plt.clf()
            plt.plot(val_loss)
            plt.savefig(os.path.join(save_path, "pretrain_val_loss.png"))

        torch.save(model, os.path.join(save_path, f"pretrained_block.pt"))
    
        test_loss = []
        test_loss_2 = []
        for x, y in test_dataloader:
            batch_size, stock_num, time_step, feature_num = x.shape
            x = x.reshape(batch_size * stock_num, time_step, feature_num).to(
                torch.float32
            )
            batch_size, stock_num, time_step, feature_num = y.shape
            y = y.reshape(batch_size * stock_num, time_step, feature_num).to(
                torch.float32
            )

            x, y = x.to(device), y.to(device)
            o=model(x)
            maeloss = masked_mae(o, y)
            mseloss = masked_mse(o, y)
            test_loss.append(maeloss.detach().cpu().numpy())
            test_loss_2.append(mseloss.detach().cpu().numpy())
        
        with open(os.path.join(save_path, "test_loss.txt"),'a') as file:
            print(np.mean(test_loss),np.mean(test_loss_2),file=file)
            
        print(np.mean(test_loss),np.mean(test_loss_2))



def pretrain_main():
    global config
    config = config["pretrain"]

    device = f"cuda:{args.device}"
    data_args = load_arg(name=args.dataset_name)
    data_args.device = device

    train_dataset, train_dataloader = data_provider(data_args, "train")
    val_dataset, val_dataloader = data_provider(data_args, "val")
    test_dataset, test_dataloader = data_provider(data_args, "test")

    feat_num = next(iter(train_dataloader))[0].shape[-1]
    label_num = next(iter(train_dataloader))[1].shape[-1]

    print(f"input dim {feat_num}, pred dim {label_num}")

    model_args = ModelArgs(
        d_model=config["model"]["d_model"],
        n_layer=config["model"]["n_layer"],
        e_fact=config["model"]["e_fact"],
        input_dim=feat_num,
        output_dim=label_num,
    )

    model = MambaPre(model_args).to(device)

    opti = torch.optim.Adam(
        model.parameters(),
        lr=config["train"]["lr"],
        weight_decay=config["train"]["weight_decay_rate"],
        amsgrad=False,
    )


    save_path = f"./result/{args.dataset_name}/{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    pretrain_loss = []
    pretrain_val_loss = []

    for epoch in range(config["train"]["epoch"]):
        print(f"----{epoch}----")
        model.train()
        epoch_loss = []
        i = 0
        for x, y in train_dataloader:
            batch_size, stock_num, time_step, feature_num = x.shape
            x = x.reshape(batch_size * stock_num, time_step, feature_num).to(
                torch.float32
            )
            batch_size, stock_num, time_step, feature_num = y.shape
            y = y.reshape(batch_size * stock_num, time_step, feature_num).to(
                torch.float32
            )
            x_repeat = repeat_and_add_noise(x, repeat_times=3, noise_std=0.001)
            x, x_repeat = x.to(device), x_repeat.to(device)

            o_x = model(x)
            o_a = model(x_repeat)

            loss = cal_temporal_memory_loss(o_x=o_x, o_a=o_a)

            opti.zero_grad()
            loss.backward()
            opti.step()

            epoch_loss.append(loss.detach().cpu().numpy())

        print(np.mean(epoch_loss))

        pretrain_loss.append(np.mean(epoch_loss))

        epoch_loss = []
        model.eval()
        for x, y in val_dataloader:
            batch_size, stock_num, time_step, feature_num = x.shape
            x = x.reshape(batch_size * stock_num, time_step, feature_num).to(
                torch.float32
            )
            batch_size, stock_num, time_step, feature_num = y.shape
            y = y.reshape(batch_size * stock_num, time_step, feature_num).to(
                torch.float32
            )
            x_repeat = repeat_and_add_noise(x, repeat_times=3, noise_std=0.001)
            x, x_repeat = x.to(device), x_repeat.to(device)

            o_x = model(x)
            o_a = model(x_repeat)

            loss = cal_temporal_memory_loss(o_x=o_x, o_a=o_a)
            epoch_loss.append(loss.detach().cpu().numpy())

        print(np.mean(epoch_loss))

        pretrain_val_loss.append(np.mean(epoch_loss))
        plt.clf()
        plt.plot(pretrain_loss)
        plt.savefig(os.path.join(save_path, "pretrain_train_loss.png"))

        plt.clf()
        plt.plot(pretrain_val_loss)
        plt.savefig(os.path.join(save_path, "pretrain_val_loss.png"))

        torch.save(
            model.state_dict(), os.path.join(save_path, f"pretrained_block_{epoch}.pt")
        )


def test_main():
    global config
    config = config["test"]

    device = f"cuda:{args.device}"
    data_args = load_arg(name=args.dataset_name)
    data_args.device = device

    test_dataset, test_dataloader = data_provider(data_args, "test")
    model = torch.load(config["load_path"]).to(device)
    pred = []
    label = []

    with torch.no_grad():
        model.eval()
        for x, y in test_dataloader:
            batch_size, stock_num, time_step, feature_num = x.shape
            x = x.reshape(batch_size * stock_num, time_step, feature_num).to(
                torch.float32
            )

            x = x.to(device)

            o = model(x).reshape(y.shape)

            y = y.cpu().detach().numpy()
            o = o.cpu().detach().numpy()                
            y = y.reshape(-1)
            o = o.reshape(-1)
            pred.append(o)
            label.append(y)

    preds = np.concatenate(pred)
    labels = np.concatenate(label)
    print(preds.shape, labels.shape)

    columns = ["MAE", "MSE", "RMSE", "MAPE"]
    preds = torch.from_numpy(preds)
    targets = torch.from_numpy(labels)
    mae = masked_mae(preds, targets).item()
    mse = masked_mse(preds, targets).item()
    rmse = masked_rmse(preds, targets).item()
    mape = masked_mape(preds, targets).item()
    row = [[mae, mse, rmse, mape]]

    save_path = os.path.dirname(config["load_path"])
    results = pd.DataFrame(row, columns=columns)
    results.to_csv(os.path.join(save_path, "metric.csv"))


def test_all_main():
    global config
    config = config["test_all"]

    device = f"cuda:{args.device}"
    data_args = load_arg(name=args.dataset_name)
    data_args.device = device

    test_dataset, test_dataloader = data_provider(data_args, "test")

    folders = os.listdir(config["load_folder"])
    folders = [
        entry
        for entry in folders
        if os.path.isdir(os.path.join(config["load_folder"], entry))
    ]

    dfs = []

    for fold_name in folders:

        subfolder_path = os.path.join(config["load_folder"], fold_name)

        subfolders = os.listdir(subfolder_path)

        subfolders = [
            entry
            for entry in subfolders
            if os.path.isdir(os.path.join(subfolder_path, entry))
        ]

        for subfold_name in subfolders:

            subsubfolder_path = os.path.join(subfolder_path, subfold_name)

            subsubfolders = os.listdir(subsubfolder_path)

            subsubfolders = [
                entry
                for entry in subsubfolders
                if os.path.isdir(os.path.join(subsubfolder_path, entry))
            ]

            for subsubfold_name in subsubfolders:
                model_path = os.path.join(
                    subsubfolder_path, subsubfold_name, "best_model.pt"
                )
                print(model_path)

                if not os.path.exists(model_path):
                    continue

                model = torch.load(model_path).to(device)

                pred = []
                label = []

                with torch.no_grad():
                    model.eval()
                    for x, y in test_dataloader:
                        batch_size, stock_num, time_step, feature_num = x.shape
                        x = x.reshape(
                            batch_size * stock_num, time_step, feature_num
                        ).to(torch.float32)
                        y = y.to(torch.float32)

                        x = x.to(device)

                        o = model(x,x,x,x).reshape(y.shape)

                        y = y.cpu().detach().numpy()
                        o = o.cpu().detach().numpy()

                        o = o.reshape(-1)
                        y = y.reshape(-1)

                        pred.append(o)
                        label.append(y)

                preds = np.concatenate(pred)
                labels = np.concatenate(label)

                columns = ["MAE", "MSE", "RMSE", "MAPE"]
                preds = torch.from_numpy(preds)
                targets = torch.from_numpy(labels)
                mae = masked_mae(preds, targets).item()
                mse = masked_mse(preds, targets).item()
                rmse = masked_rmse(preds, targets).item()
                mape = masked_mape(preds, targets).item()
                row = [[mae, mse, rmse, mape]]

                dfs.append(
                    [fold_name, subfold_name, subsubfold_name, mae, mse, rmse, mape]
                )

    df = pd.DataFrame(
        dfs,
        columns=["pre_layer_idx", "frozen_type", "times", "MAE", "MSE", "RMSE", "MAPE"],
    )
    df.to_csv(os.path.join(config["load_folder"], "all_metrix.csv"))


if __name__ == "__main__":
    if args.mod == "pretrain":
        pretrain_main()
    elif args.mod == "inference":
        inference_main()
    elif args.mod == "test":
        test_main()
    elif args.mod == "testall":
        test_all_main()
    else:
        raise ValueError("Unknown mod.")
