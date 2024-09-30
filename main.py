import torch
import torch.nn.functional as F
import argparse
import sys
import toml
import os
import numpy as np
import datetime
from typing import List

from data.data_provider import data_provider

from matplotlib import pyplot as plt    
import pandas as pd

from model import Mamba,ModelArgs,MambaPretrainBlock
from dataArgs import load_arg

config=toml.load('./config.toml')

parser = argparse.ArgumentParser()

parser.add_argument("--dataset_name",default='ETTh1',type=str)

parser.add_argument("--device",default=0,type=int)

parser.add_argument("--mod",default='pretrain',type=str)

args = parser.parse_args()

def masked_mae(preds, labels, null_val=0.0):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /=  torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds-labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)

def repeat_and_add_noise(input_tensor, repeat_times=3, noise_std=[0.01,0.02]):
    B, T, F = input_tensor.shape

    repeated_tensor = input_tensor.unsqueeze(2).repeat(1, 1, repeat_times, 1)
    repeated_tensor = repeated_tensor.view(B, T * repeat_times, F)
    
    if isinstance(noise_std,List):
        assert len(noise_std) == repeat_times-1,"length of noise std list must equal to repeat_times - 1."
        noise_list=[]
        noise_list.append(torch.zeros_like(input_tensor))
        for n in noise_std:
            noise_list.append(torch.rand_like(input_tensor)*n)
        noise = torch.stack(noise_list,dim=-2)
        noise= noise.view(B, T * repeat_times, F)
    else:
        raise ValueError("noise std must be list.")

            
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
    pos_sample_num=positive_samples.shape[1]
    # Normalize embeddings
    anchor = F.normalize(anchor, dim=-1)
    positive_samples = F.normalize(positive_samples, dim=-1)
    negative_samples = F.normalize(negative_samples, dim=-1)

    # Compute positive logits
    pos_logits = torch.einsum('bd,bpd->bp', anchor, positive_samples) / temperature
    pos_logits = pos_logits.view(-1, 1)  # (batch_size * num_positives, 1)
    # print(pos_logits.shape)
    
    # Compute negative logits
    neg_logits = torch.einsum('bd,bnd->bn', anchor, negative_samples) / temperature
    neg_logits = neg_logits.view(-1,1, negative_samples.size(1))  # (batch_size, num_negatives)
    neg_logits = neg_logits.repeat(1,pos_sample_num,1)
    # print(neg_logits.shape)
    neg_logits = neg_logits.view(-1,negative_samples.size(1))
    
    # Combine positive and negative logits
    logits = torch.concat([pos_logits, neg_logits], dim=1)  # (batch_size * num_positives, 1 + num_negatives)
    
    # Create labels for positive samples
    labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
    
    # Compute InfoNCE loss
    loss = F.cross_entropy(logits, labels, reduction='mean')
    
    # print('infonce',loss)

    return loss

def cal_temporal_memory_loss(o_x,o_a,repeat_time=3):
    length=o_x.shape[1]
    losses=[]

    for i in range(length-1):
        anchor=o_x[:,i,...]
        pos=o_a[:,i*repeat_time:repeat_time*(i+1),...]
        neg=o_x[:,i+1:i+2,...]
        losses.append(info_nce_loss(anchor=anchor,positive_samples=pos,negative_samples=neg))

        anchor=o_a[:,i*repeat_time,...]
        pos=o_a[:,(i*repeat_time+1):repeat_time*(i+1),...]
        neg=o_a[:,repeat_time*(i+1):repeat_time*(i+2)]
        losses.append(info_nce_loss(anchor=anchor,positive_samples=pos,negative_samples=neg))
    
    return torch.mean(torch.stack(losses))

def inference_main():
    global config
    config=config['inference']

    device=f'cuda:{args.device}'
    data_args=load_arg(name=args.dataset_name)
    data_args.device=device

    train_dataset, train_dataloader = data_provider(data_args, 'train')
    val_dataset, val_dataloader = data_provider(data_args, 'val')
    test_dataset, test_dataloader = data_provider(data_args, 'test')

    feat_num=next(iter(train_dataloader))[0].shape[-1]
    label_num=next(iter(train_dataloader))[1].shape[-1]

    model_args=ModelArgs( 
        d_model=config['model']['d_model'],
        n_layer=config['model']['n_layer'],
        input_dim=feat_num,
        output_dim=label_num
    )

    print(f'input dim {feat_num}, pred dim {label_num}')
    
    model = Mamba(model_args)
    if config['model']['pre_layer'][0]!=-1:
        model.load_params(config['model']['pre_layer'],config['model']['load_path'],config['model']['frozentype'])
    model=model.to(device)

    opti=torch.optim.Adam(model.parameters(),lr=config['train']['lr'], weight_decay=config['train']['weight_decay_rate'], amsgrad=False)
    save_path=os.path.join(os.path.dirname(config['model']['load_path']),"&".join(np.array(config['model']['pre_layer']).astype(str)),config['model']['frozentype'],f"{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}")

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    train_loss=[]
    val_loss=[]

    best_loss=np.inf

    for epoch in range(config['train']['epoch']):
        print(f'----{epoch}----')
        model.train()
        epoch_loss=[]
        for x,y in train_dataloader:
            batch_size,stock_num,time_step,feature_num=x.shape
            x=x.reshape(batch_size*stock_num,time_step,feature_num).to(torch.float32)
            batch_size,stock_num,time_step,feature_num=y.shape
            y=y.reshape(batch_size*stock_num,time_step,feature_num).to(torch.float32)

            x,y=x.to(device),y.to(device)

            o=model(x)

            loss=masked_mae(o,y)

            opti.zero_grad()
            loss.backward()

            opti.step()

            epoch_loss.append(loss.detach().cpu().numpy())

        
        train_loss.append(np.mean(epoch_loss))
        
        epoch_loss=[]
        model.eval()
        for x,y in val_dataloader:
            batch_size,stock_num,time_step,feature_num=x.shape
            x=x.reshape(batch_size*stock_num,time_step,feature_num).to(torch.float32)
            batch_size,stock_num,time_step,feature_num=y.shape
            y=y.reshape(batch_size*stock_num,time_step,feature_num).to(torch.float32)
            
            x,y=x.to(device),y.to(device)

            o=model(x)

            loss=masked_mae(o,y)

            epoch_loss.append(loss.detach().cpu().numpy())

            if loss.detach().cpu().numpy()<best_loss:
                best_loss=loss.detach().cpu().numpy()
                torch.save(model,os.path.join(save_path,'best_model.pt'))      
        
        val_loss.append(np.mean(epoch_loss))

        plt.clf()
        plt.plot(train_loss)
        plt.savefig(os.path.join(save_path,'inference_train_loss.png'))

        plt.clf()
        plt.plot(val_loss)
        plt.savefig(os.path.join(save_path,'inference_val_loss.png'))

        torch.save(model,os.path.join(save_path,f'inference_mamba_{epoch}.pt'))

def pretrain_main():
    global config
    config=config['pretrain']

    device=f'cuda:{args.device}'
    data_args=load_arg(name=args.dataset_name)
    data_args.device=device

    train_dataset, train_dataloader = data_provider(data_args, 'train')
    val_dataset, val_dataloader = data_provider(data_args, 'val')

    feat_num=next(iter(train_dataloader))[0].shape[-1]
    label_num=next(iter(train_dataloader))[1].shape[-1]

    print(f'input dim {feat_num}, pred dim {label_num}')

    model_args=ModelArgs( 
        d_model=config['model']['d_model'],
        n_layer=config['model']['n_layer'],
        input_dim=feat_num,
        output_dim=label_num
    )
    
    model = MambaPretrainBlock(model_args).to(device)

    opti=torch.optim.Adam(model.parameters(),lr=config['train']['lr'], weight_decay=config['train']['weight_decay_rate'], amsgrad=False)

    save_path=f"./result/{args.dataset_name}/{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}-dmodel{config['model']['d_model']}-dstate{model_args.d_state}/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    pretrain_loss=[]
    pretrain_val_loss=[]

    for epoch in range(config['train']['epoch']):
        print(f'----{epoch}----')
        model.train()
        epoch_loss=[]
        i=0
        for x,y in train_dataloader:
            batch_size,stock_num,time_step,feature_num=x.shape
            x=x.reshape(batch_size*stock_num,time_step,feature_num).to(torch.float32)
            batch_size,stock_num,time_step,feature_num=y.shape
            y=y.reshape(batch_size*stock_num,time_step,feature_num).to(torch.float32)
            x_repeat=repeat_and_add_noise(x,repeat_times=3,noise_std=[0.001,0.01])
            x,x_repeat=x.to(device),x_repeat.to(device)

            o_x=model(x)
            o_a=model(x_repeat)

            loss=cal_temporal_memory_loss(o_x=o_x,o_a=o_a)

            opti.zero_grad()
            loss.backward()
            opti.step()

            epoch_loss.append(loss.detach().cpu().numpy())
        
        
        pretrain_loss.append(np.mean(epoch_loss))
        
        epoch_loss=[]
        model.eval()
        for x,y in val_dataloader:
            batch_size,stock_num,time_step,feature_num=x.shape
            x=x.reshape(batch_size*stock_num,time_step,feature_num).to(torch.float32)
            batch_size,stock_num,time_step,feature_num=y.shape
            y=y.reshape(batch_size*stock_num,time_step,feature_num).to(torch.float32)
            x_repeat=repeat_and_add_noise(x,repeat_times=3,noise_std=[0.001,0.01])
            x,x_repeat=x.to(device),x_repeat.to(device)

            o_x=model(x)
            o_a=model(x_repeat)

            loss=cal_temporal_memory_loss(o_x=o_x,o_a=o_a)
            epoch_loss.append(loss.detach().cpu().numpy())
        
        
        pretrain_val_loss.append(np.mean(epoch_loss))
        plt.clf()
        plt.plot(pretrain_loss)
        plt.savefig(os.path.join(save_path,'pretrain_train_loss.png'))

        plt.clf()
        plt.plot(pretrain_val_loss)
        plt.savefig(os.path.join(save_path,'pretrain_val_loss.png'))

        torch.save(model.state_dict(),os.path.join(save_path,f'pretrained_block_{epoch}.pt'))

if __name__=='__main__':
    if args.mod=='pretrain':
        pretrain_main()
    elif args.mod=='inference':
        inference_main()
    else:
        raise ValueError('Unknown mod.')
    