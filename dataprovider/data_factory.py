import sys
from data_loader import Dataset_Traffic, Dataset_KDDcup, Dataset_Finance, Dataset_Solar,Dataset_ETT_minute,Dataset_ETT_hour,\
                        Dataset_Custom, Dataset_ETT_hour_IM, Dataset_ETT_minute_IM,ImputationDataset

from torch.utils.data import DataLoader
import pandas as pd

data_dict = {
    'PEMS-BAY': Dataset_Traffic,
    'METR-LA': Dataset_Traffic,
    'cloud_cover': Dataset_Traffic,
    'weather': Dataset_Custom,
    'humidity': Dataset_Traffic,
    'temperature': Dataset_Traffic,
    'kddcup': Dataset_KDDcup,
    'solar-energy': Dataset_Solar,
    'acl18': Dataset_Finance,
    'kdd17': Dataset_Finance,
    'traffic': Dataset_Solar,
    'electricity':Dataset_Solar,
    'exchange_rate':Dataset_Solar,

    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,

    'ETTh1_' : Dataset_ETT_hour_IM,
    'ETTh2_' : Dataset_ETT_hour_IM,
    'ETTm1_': Dataset_ETT_minute_IM,
    'ETTm2_': Dataset_ETT_minute_IM,
    'beijingair_': ImputationDataset,
    'italyair_' : ImputationDataset,
    'physionet2012_': ImputationDataset,
    'physionet2019_': ImputationDataset,
}


def data_provider(args, flag):
    Data = data_dict[args.data]

    if flag == 'test' or flag=='val':
        shuffle_flag = False
        drop_last = False
        batch_size = args.batch_size
    # elif flag == 'pred':
    #     shuffle_flag = False
    #     drop_last = False
    #     batch_size = 1
    #     freq = args.freq
    #     Data = Dataset_Pred
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size

    if args.data[-1] == '_':
        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            imputation_rate = args.imputation_rate,
            size=[args.seq_len],
            device=args.device
        )
        
    elif args.data == 'kdd17' or args.data == 'acl18':
        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.pred_len],
            data_name=args.data,
            stock_num=args.num_nodes,
        )
    else:
        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.pred_len],
            device=args.device
        )

    print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)
    return data_set, data_loader

def load_adj(data, adj_path):
    raw_adj = pd.read_pickle(adj_path)
    if data == 'METR-LA' or data == 'PEMS-BAY':
        adj = raw_adj[2]
    else:
        adj = raw_adj
    return adj
