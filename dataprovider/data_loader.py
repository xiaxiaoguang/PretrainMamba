import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import time
import warnings
import torch

warnings.filterwarnings('ignore')

def load_graph_data(root_path='./dataset/', data_path='traffic/PEMS-BAY/', data_set='traffic'):
    graph_data = pd.read_pickle(os.path.join(root_path, data_path + 'adj.pkl'))
    if data_set == 'traffic':
        return graph_data[2]

class StandardScaler:
    """
    Standard the input
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean
    
class Dataset_Traffic(Dataset):
    def __init__(self, root_path='./dataset/', flag='train', size=None,
                 data_path='traffic/PEMS-BAY/',
                 scale=True, device=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 12
            self.pred_len = 12
        else:
            self.seq_len = size[0]
            self.pred_len = size[1]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.scale = scale
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        data = {}
        for category in ['train', 'val', 'test']:
            cat_data = pd.read_pickle(os.path.join(self.root_path, self.data_path, category + '.pkl'))
            data['x_' + category] = cat_data['x']
            data['y_' + category] = cat_data['y']

        if self.scale:
            scaler = StandardScaler(mean=data['x_train'][..., 0].mean(), std=data['x_train'][..., 0].std())
            # Data format
            for category in ['train', 'val', 'test']:
                data['x_' + category][..., 0] = scaler.transform(data['x_' + category][..., 0])
                data['y_' + category][..., 0] = scaler.transform(data['y_' + category][..., 0])
                self.scaler = scaler

        type_map = {0:'train', 1:'val', 2:'test'}
        self.data_x = data['x_' + type_map[self.set_type]].transpose((0,2,1,3))
        self.data_y = data['y_' + type_map[self.set_type]][:,:,:,:1].transpose((0,2,1,3))

    def __getitem__(self, index):
        return self.data_x[index], self.data_y[index]

    def __len__(self):
        return len(self.data_x)
    
    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

def time2obj(time_sj):
    data_sj = time.strptime(time_sj, "%H:%M")
    return data_sj

def func_add_t(x):
    time_strip = 600
    time_obj = time2obj(x)
    time_e = ((
        (time_obj.tm_sec + time_obj.tm_min * 60 + time_obj.tm_hour * 3600)) //
              time_strip) % 288
    return time_e

def time2obj(time_sj):
    data_sj = time.strptime(time_sj, "%H:%M")
    return data_sj

def func_add_t(x):
    time_strip = 600
    time_obj = time2obj(x)
    time_e = ((
        (time_obj.tm_sec + time_obj.tm_min * 60 + time_obj.tm_hour * 3600)) //
              time_strip) % 288
    return time_e

class Dataset_KDDcup(Dataset):
    def __init__(
        self,
        data_path='electricity/kddcup/wtbdata_245days.csv',
        root_path='./dataset/',
        location_path='./electricity/kddcup/sdwpf_turb_location.CSV',
        flag='train',
        size=None,
        scale=True,
        day_len=24 * 6,
        Turbins=134,
        train_days=155,  # 155 days
        val_days=30,  # 30 days
        test_days=60,  # 60 days
        total_days=245,  # 245 days
        device=None
    ):
        super().__init__()

        # initialization
        self.unit_size = day_len
        if size != None:
            self.input_len = size[0]
            self.output_len = size[1]
        else:
            self.input_len = 144
            self.output_len = 144
        self.start_col = 0
        self.scale = scale
        self.Turbins = Turbins

        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.flag = flag
        self.data_path = data_path
        self.root_path = root_path
        self.location_path = location_path

        self.total_size = total_days * self.unit_size
        self.train_size = train_days * self.unit_size
        self.val_size = val_days * self.unit_size
        self.test_size = test_days * self.unit_size
        self.__read_location__()
        self.__read_data__()


    def __read_data__(self):
        #read wind power data
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        df_data, raw_df_data = self.data_preprocess(df_raw)
        self.df_data = df_data
        self.raw_df_data = raw_df_data

        attr_data, target_data = self.build_data(df_data)
        print(f"attr_data_shape: {attr_data.shape}")
        print(f"target_data_shape: {target_data.shape}")
        self.attr_data = attr_data
        self.target_data = target_data

    def __read_location__(self):
        #read location data
        df_location = pd.read_csv(os.path.join(self.root_path, self.location_path))
        location = df_location.values[:,1:]
        mean = np.mean(location, axis=0, keepdims=True)
        std = np.std(location, axis=0, keepdims=True)
        #print("location mean:{}, std:{}".format(mean.shape,std.shape))
        location = (location-mean)/std
        self.location = location #(134,2)

    def data_preprocess(self, df_data):
        """
        1. 增加time feature
        2. 将nan 置 0
        3. 将prtv和patv小于0置0
        :param df_data:
        :return:
        """
        feature_name = [
            n for n in df_data.columns
            if 'Day' not in n and 'Tmstamp' not in n and "TurbID" not in n
        ]
        #print("feature name:{}".format(feature_name)) #10 columns

        new_df_data = df_data[feature_name]  #10 columns
        # add time attr
        t = df_data['Tmstamp'].apply(func_add_t) #0-143
        new_df_data.insert(0, 'time', t)

        month = (df_data['Day'].apply(lambda x: x // 31)) / 11.0 - 0.5 #0-11 数据集小于1年
        weekday = (df_data['Day'].apply(lambda x: x % 7)) / 6.0 - 0.5 #0-6
        day = (df_data['Day'].apply(lambda x: x % 31)) / 30.0 - 0.5 #0-30
        hour = (new_df_data['time'].apply(lambda x: x//6)) / 23.0 - 0.5 #0-23
        minute = new_df_data['time'].apply(lambda x: x % 6) / 5.0 - 0.5#0-5
        new_df_data.insert(0, 'minute', minute)
        new_df_data.insert(0, 'hour', hour)
        new_df_data.insert(0, 'weekday', weekday)
        new_df_data.insert(0, 'day', day)
        new_df_data.insert(0,'month',month)

        new_df_data.drop(columns='time',inplace=True)
        #print("new columns:{}".format(new_df_data.columns)) #['month', 'day', 'weekday', 'hour', 'minute', 'Wspd', 'Wdir', 'Etmp',
        #'Itmp', 'Ndir', 'Pab1', 'Pab2', 'Pab3', 'Prtv', 'Patv']
        #new_df_data.to_csv('./output/new_df_data.csv')

        pd.set_option('mode.chained_assignment', None)
        raw_df_data = new_df_data.copy(deep=True)
        # new_df_data = new_df_data.replace(
        #     to_replace=np.nan, value=0, inplace=False)
        new_df_data.fillna(method='ffill', axis=1, inplace=True, limit=100)

        raw_df_data.loc[ \
            ((raw_df_data['Patv'] <= 0) & (raw_df_data['Wspd'] > 2.5)) | \
            ((raw_df_data['Pab1'] > 89) | (raw_df_data['Pab2'] > 89) | (raw_df_data['Pab3'] > 89)) | \
            ((raw_df_data['Wdir'] < -180) | (raw_df_data['Wdir'] > 180) | (raw_df_data['Ndir'] < -720) |
            (raw_df_data['Ndir'] > 720)),
            ('Patv')] = np.nan

        return new_df_data, raw_df_data

    def build_data(self, df_data):
        cols_data = df_data.columns[self.start_col:]
        df_data = df_data[cols_data]
        raw_df_data = self.raw_df_data[cols_data]

        raw_data = raw_df_data.values
        raw_data = np.reshape(
            raw_data, [self.Turbins, self.total_size, len(cols_data)])

        data = df_data.values #[4727520,12]
        data = np.reshape(data,
                   [self.Turbins, self.total_size, len(cols_data)]) #[134,35280,12]


        border1s = [
            0,
            self.train_size - self.input_len,
            self.train_size + self.val_size - self.input_len
        ]
        border2s = [
            self.train_size,
            self.train_size + self.val_size,
            self.train_size + self.val_size + self.test_size
        ]
        #print("border1s: ", border1s) #[0,30672,32976]
        #print("border2s: ",border2s) #[30816,33120,35280]

        if self.scale:
            self.data_mean = np.nanmean(
                    data[:, border1s[0]:border2s[0], -10:],
                    axis=1,
                    keepdims=True)
            self.data_scale = np.nanstd(
                    data[:, border1s[0]:border2s[0], -10:],
                    axis=1,
                    keepdims=True)
            self.data_mean = np.around(self.data_mean,4)
            self.data_scale = np.around(self.data_scale,4)
            #print("mean:{}, std:{}".format(self.data_mean[10,0,-1],self.data_scale[10,0,-1]))

            #print("mean shape:{}".format(self.data_mean.shape)) #(134,1,10)
            #print("std shape:{}".format(self.data_scale.shape)) #(134,1,10)

            data[:, :, -10:] = (data[:, :, -10:] - self.data_mean) / self.data_scale  # [134,35280,12]
        
        target_data = data[:,:,-1:] #[134,35280,10]
        location = np.expand_dims(self.location, 1).repeat(data.shape[1], axis=1)  # 复制矩阵 [134,35280,2]
        # attr_data = np.concatenate((location,data),axis=-1) #[134,35280,12] !!!!!!!!!!!!!!!!!!!!!
        attr_data = data #!!!!!!!!!!!!!

        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        #print("border1:{}, border2:{}".format(border1,border2))

        self.raw_df = []
        for turb_id in range(self.Turbins):
            self.raw_df.append(
                pd.DataFrame(
                    data=raw_data[turb_id, border1 + self.input_len:border2],
                    columns=cols_data))
        print("raw df shape:{}".format(self.raw_df[0].shape))
        # 返回train,valid,test对应的数据
        attr_data = attr_data[:, border1:border2, -1:] #!!!!!!!!!!!!!!!
        target_data = target_data[:, border1:border2, :]

        return attr_data, target_data
    
    def get_raw_df(self):
        return self.raw_df

    def __len__(self):
        return self.attr_data.shape[1] - self.input_len - self.output_len + 1

    def select_item(self,index):
        # Sliding window with the size of input_len + output_len
        s_begin = index
        s_end = s_begin + self.input_len
        r_begin = s_end
        r_end = r_begin + self.output_len
        # print("s_begin: {}, s_end: {}, r_begin: {}, r_end: {}".format(s_begin,s_end,r_begin,r_end))
        # origin scale data
        x = self.attr_data[:, s_begin:s_end, :]  # [134,144,10]
        y = self.target_data[:, r_begin:r_end, :]  # [134,288,2]

        return x, y

    def __getitem__(self, index):
        # Sliding window with the size of input_len + output_len
        x, y = self.select_item(index)
        return x, y
    
    def inverse_transform(self, data):
        device = data.device
        mean = torch.from_numpy(self.data_mean[:, :, -1:]).to(device) # [node, 1, 1]
        std = torch.from_numpy(self.data_scale[:, :, -1:]).to(device)
        return ((data * std) + mean)
    

class Dataset_Finance(Dataset):
    def __init__(self, root_path, flag='train', size=None, data_name='acl18',
                 data_path='finance/acl18/process_data.pkl', stock_num=88,
                 scale=True, device=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size != None:
            self.seq_len = size[0]
            self.pred_len = size[1]
        else:
            self.seq_len = 20
            self.pred_len = 10
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        if data_name == 'acl18':
            self.train_from = pd.to_datetime('2014-01-01')
            self.val_from = pd.to_datetime('2015-08-01')
            self.test_from = pd.to_datetime('2015-10-01')
            self.test_end = pd.to_datetime('2016-01-01')
        elif data_name == 'kdd17':
            self.train_from = pd.to_datetime('2007-01-01')
            self.val_from = pd.to_datetime('2015-01-01')
            self.test_from = pd.to_datetime('2016-01-01')
            self.test_end = pd.to_datetime('2017-01-01')
        else:
            print('Wrong Dataset Name!')
        self.scale = scale
        self.scaler = None
        self.stock_num = stock_num

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        ohlcv_data = pd.read_pickle(os.path.join(self.root_path, self.data_path))
        ohlcv_data['gt1'] = self.delay(ohlcv_data.Open, -2) / self.delay(ohlcv_data.Open, -1) - 1
        ohlcv_data.Volume = ohlcv_data.Volume.astype(float)

        data = pd.DataFrame()
        data['rel_close'] = ohlcv_data.Close / self.delay(ohlcv_data.Close, 1) - 1
        data['rel_open'] = ohlcv_data.Open / ohlcv_data.Close - 1
        data['rel_high'] = ohlcv_data.High / ohlcv_data.Close - 1
        data['rel_low'] = ohlcv_data.Low / ohlcv_data.Close - 1
        data['volume'] = ohlcv_data.Volume / (self.ts_mean(ohlcv_data.Volume, 42) + 1e-8) - 1
        data['gt'] = ohlcv_data.gt1
        
        data = data.replace([np.inf, -np.inf], value=np.nan)
        data = data.reset_index().set_index(['Date', 'code']).sort_index()
    
        if self.scale:
            feature_cols = data.columns.drop('gt')
            data_dropna = data.loc[:self.val_from-pd.Timedelta(days=1), feature_cols].dropna()
            mean, std = data_dropna.mean(), data_dropna.std()
            data[feature_cols] = (data[feature_cols] - mean) / std
            data = self.replace_dtypes(data)
            self.mean = mean
            self.std = std

        data = data.fillna(0)
                
        #dataset split
        if self.set_type == 0:
            self.data = data.loc[self.train_from:self.val_from-pd.Timedelta(days=1)]
        elif self.set_type == 1:
            self.data = data.loc[self.val_from-pd.Timedelta(days=self.seq_len+self.pred_len-1)
                                 :self.test_from-pd.Timedelta(days=1)]
        else:
            self.data = data.loc[self.test_from-pd.Timedelta(days=self.seq_len+self.pred_len-1)
                                 :self.test_end-pd.Timedelta(days=1)]

        self.Ts = self.data.index.unique(0)

    def __getitem__(self, index):
        s_begin = self.Ts[index]
        s_end = self.Ts[index + self.seq_len - 1] 
        r_begin = self.Ts[index + self.seq_len - 1] 
        r_end = self.Ts[index + self.seq_len + self.pred_len - 2] 
        x = self.data.drop(columns='gt').loc[s_begin:s_end].fillna(0.).values
        y = self.data['gt'].loc[r_begin:r_end].fillna(0.).values
        x = x.reshape((self.seq_len, self.stock_num, -1)).transpose((1,0,2))
        y = y.reshape((self.pred_len, self.stock_num, -1)).transpose((1,0,2))
        return x, y

    def __len__(self):
        return len(self.Ts) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data): 
        return self.scaler.inverse_transform(data)
    
    def delay(self, series, d, level=1):
        d = int(d)
        return series.groupby(level=level, group_keys=False).shift(d)
    
    def ts_mean(self, series, d):
        d = int(d)
        return series.groupby(level=1, group_keys=False).rolling(d).mean().droplevel(0).sort_index()
    
    def replace_dtypes(self, data, rules={'float64': 'float32', 'int32': 'int64'}):
        for k, v in rules.items():
            cols = data.dtypes[data.dtypes == k].index
            data = data.astype({t: v for t in cols}, copy=False)
        return data



# class MinMaxStandardScaler:
#     """
#     Standard the input
#     """

#     def __init__(self, max, min):
#         self.max = max
#         self.min = min

#     def transform(self, data):
#         return (data - self.min)/(self.max - self.min)

#     def inverse_transform(self, data):
#         return (data * (self.max - self.min)) + self.min
    
# class t_StandardScaler:
#     """
#     Standard the input
#     """

#     def __init__(self, mean, std):
#         self.mean = mean
#         self.std = std

#     def transform(self, data):
#         return (data - self.mean)/self.std

#     def inverse_transform(self, data):
#         return (data * self.std) + self.mean
    
class Dataset_Solar(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 data_path='electricity/solar-energy/solar_AL.txt',
                 scale=True,device='cuda:0'):
        # size [seq_len, label_len, pred_len]
        # info
        if size != None:
            self.seq_len = size[0]
            self.pred_len = size[1]
        else:
            self.seq_len = 24
            self.pred_len = 24
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.scale = scale
        self.device = device

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        df_raw = []
        with open(os.path.join(self.root_path, self.data_path), "r", encoding='utf-8') as f:
            for line in f.readlines():
                line = line.strip('\n').split(',')
                data_line = np.stack([float(i) for i in line])
                df_raw.append(data_line)
        df_raw = np.stack(df_raw, 0)
        df_raw = pd.DataFrame(df_raw)

        num_train = int(len(df_raw) * 0.6)
        num_test = int(len(df_raw) * 0.2)
        num_valid = int(len(df_raw) * 0.2)
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_valid, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        df_data = df_raw.values

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            # self.scaler = StandardScaler(mean=train_data[..., 0].mean(), std=train_data[..., 0].std())
            self.scaler = StandardScaler(mean=train_data.mean(axis=0), std=train_data.std(axis=0))
            data = self.scaler.transform(df_data)
        else:
            data = df_data

        self.data_x = torch.tensor(data[border1:border2]).to(self.device)
        self.data_y = torch.tensor(data[border1:border2]).to(self.device)

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end
        r_end = r_begin + self.pred_len

        x = self.data_x[s_begin:s_end]
        y = self.data_y[r_begin:r_end]

        # x = x.reshape(x.shape[0], x.shape[1], 1)
        # y = y.reshape(y.shape[0], y.shape[1], 1)

        # x = x.transpose((2,0,1))
        # y = y.transpose((2,0,1))

        x = x.unsqueeze(0)
        y = y.unsqueeze(0)
        return x, y

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        device = data.device
        mean = torch.from_numpy(self.scaler.mean).unsqueeze(0).unsqueeze(0).unsqueeze(0).to(device)
        std = torch.from_numpy(self.scaler.std).unsqueeze(0).unsqueeze(0).unsqueeze(0).to(device)
        return (data * std) + mean

from timefeatures import time_features

class Dataset_ETT_hour_IM(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='M', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', seasonal_patterns=None,
                 device='cuda:0', imputation_rate=0.1,seed=42):
        if size is None:
            self.seq_len = 24 * 4 * 4
        else:
            self.seq_len = size[0]
        
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        
        self.seed = seed
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.device = device
        self.imputation_rate = imputation_rate

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features in ['M', 'MS']:
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        from sklearn.preprocessing import StandardScaler        
        self.scaler = StandardScaler()
        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data_all = self.scaler.transform(df_data.values)
        else:
            data_all = df_data.values

        data_complete = data_all.copy()
        data_incomplete = data_all.copy()

        mask = np.ones_like(data_incomplete)
        np.random.seed(self.seed)
        num_missing = int(self.imputation_rate * data_incomplete.size)
        missing_indices = np.random.choice(data_incomplete.size, num_missing, replace=False)
        data_incomplete.ravel()[missing_indices] = np.nan
        mask.ravel()[missing_indices] = 0
        inds = np.isnan(data_incomplete)
        data_incomplete[inds] = 0

        data_complete = data_complete[border1:border2]
        data_incomplete = data_incomplete[border1:border2]
        mask = mask[border1:border2]

        self.data_complete = torch.tensor(data_complete).to(self.device)
        self.data_incomplete = torch.tensor(data_incomplete).to(self.device)
        self.mask = torch.tensor(mask).to(self.device)

    def __getitem__(self, index):
        window_length = self.seq_len
        incomplete_window = self.data_incomplete[index:index + window_length].unsqueeze(0)
        # mask_window = self.mask[index:index + window_length].unsqueeze(0)
        complete_window = self.data_complete[index:index + window_length].unsqueeze(0)
        return incomplete_window, complete_window

    def __len__(self):
        window_length = self.seq_len
        return len(self.data_complete) - window_length + 1

    def inverse_transform(self, data):
        B, N, T, F = data.shape
        data = data.squeeze(1)
        data = data.reshape((B * T, F)).to('cpu')
        data = self.scaler.inverse_transform(data)
        return data.reshape((B, 1, T, F))


class Dataset_ETT_hour(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='M', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', seasonal_patterns=None, 
                 device='cuda:0'):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.pred_len = size[1]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.device = device

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        from sklearn.preprocessing import StandardScaler
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            print(train_data.values.shape)
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], axis=1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = torch.tensor(data[border1:border2]).to(self.device)
        self.data_y = torch.tensor(data[border1:border2]).to(self.device)
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end
        r_end = r_begin + self.pred_len
        seq_x = self.data_x[s_begin:s_end].unsqueeze(0)
        seq_y = self.data_y[r_begin:r_end].unsqueeze(0)
        return seq_x, seq_y

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        B, N, T, F = data.shape
        data = data.squeeze(1)
        data = data.reshape((B*T,F)).to('cpu')
        data = self.scaler.inverse_transform(data)
        return data.reshape((B,1,T,F))

class Dataset_ETT_minute_IM(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='M', data_path='ETTm1.csv',
                 target='OT', scale=True, timeenc=0, freq='t', device='cuda:0', imputation_rate=0.1, seed=42):
        # Size: [seq_len]
        if size is None:
            self.seq_len = 24 * 4 * 4
        else:
            self.seq_len = size[0]
        
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.device = device
        self.imputation_rate = imputation_rate
        self.seed = seed

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        from sklearn.preprocessing import StandardScaler
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        border1s = [0, 12 * 30 * 24 * 4 - self.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
        border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features in ['M', 'MS']:
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data_all = self.scaler.transform(df_data.values)
        else:
            data_all = df_data.values

        data_complete = data_all.copy()
        data_incomplete = data_all.copy()
        mask = np.ones_like(data_incomplete)
        np.random.seed(self.seed)
        num_missing = int(self.imputation_rate * data_incomplete.size)
        missing_indices = np.random.choice(data_incomplete.size, num_missing, replace=False)
        data_incomplete.ravel()[missing_indices] = np.nan
        mask.ravel()[missing_indices] = 0
        data_incomplete[np.isnan(data_incomplete)] = 0

        data_complete = data_complete[border1:border2]
        data_incomplete = data_incomplete[border1:border2]
        mask = mask[border1:border2]

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday())
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute // 15)
            data_stamp = df_stamp.drop(['date'], axis=1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)
        
        self.data_complete = torch.tensor(data_complete).to(self.device)
        self.data_incomplete = torch.tensor(data_incomplete).to(self.device)
        self.mask = torch.tensor(mask).to(self.device)
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        window_length = self.seq_len
        incomplete_window = self.data_incomplete[index:index + window_length].unsqueeze(0)
        complete_window = self.data_complete[index:index + window_length].unsqueeze(0)
        return incomplete_window, complete_window

    def __len__(self):
        return len(self.data_complete) - self.seq_len + 1

    def inverse_transform(self, data):
        B, N, T, F = data.shape
        data = data.squeeze(1)
        data = data.reshape((B * T, F)).to('cpu')
        data = self.scaler.inverse_transform(data)
        return data.reshape((B, 1, T, F))

class Dataset_ETT_minute(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='M', data_path='ETTm1.csv',
                 target='OT', scale=True, timeenc=0, freq='t', seasonal_patterns=None,device='cuda:0'):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.pred_len = size[1]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.device = device

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        from sklearn.preprocessing import StandardScaler
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 * 4 - self.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
        border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], axis=1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = torch.tensor(data[border1:border2]).to(self.device)
        self.data_y = torch.tensor(data[border1:border2]).to(self.device)
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end
        r_end = r_begin + self.pred_len
        seq_x = self.data_x[s_begin:s_end].unsqueeze(0)
        seq_y = self.data_y[r_begin:r_end].unsqueeze(0)

        return seq_x, seq_y
    
    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        B, N, T, F = data.shape
        data = data.squeeze(1)
        data = data.reshape((B*T,F)).to('cpu')
        data = self.scaler.inverse_transform(data)
        return data.reshape((B,1,T,F))

class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='M', data_path='weather.csv',
                 target='OT', scale=True, timeenc=1, freq='h', 
                 seasonal_patterns=None, device=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.pred_len = size[1]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}

        self.set_type = type_map[flag]
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.device = device
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        from sklearn.preprocessing import StandardScaler
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]

        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test

        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = torch.tensor(data[border1:border2]).to(self.device)
        self.data_y = torch.tensor(data[border1:border2]).to(self.device)
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end
        r_end = r_begin + self.pred_len

        seq_x = self.data_x[s_begin:s_end].unsqueeze(0)
        seq_y = self.data_y[r_begin:r_end].unsqueeze(0)

        return seq_x, seq_y

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):

        return self.scaler.inverse_transform(data)


class ImputationDataset(Dataset):
    def __init__(self, root_path, flag='train', data_path=None, size=None,
                  device='cuda:0', imputation_rate=0, seed=42):
        
        self.seq_len = size[0] if size is not None else 96
        self.device = device
        self.seed = seed
        self.flag = flag
        self.root_path = root_path+flag+'.h5'
        
        self.__read_data__()
    
    def __read_data__(self):
        np.random.seed(self.seed)
        import h5py
        with h5py.File(self.root_path, "r") as f:
            print("Keys in HDF5 file:", list(f.keys()))
            Train_X = np.array(f["X"], dtype=np.float32)
            Train_X_ORI = np.array(f["X_ori"], dtype=np.float32)

        self.data_complete = torch.tensor(Train_X_ORI).to(self.device)
        self.data_incomplete = torch.tensor(Train_X).to(self.device)
        self.data_incomplete[self.data_incomplete.isnan()] = 0
        self.mask = (self.data_incomplete != 0).float()
    
    def __getitem__(self, index):
        incomplete_window = self.data_incomplete[index].unsqueeze(0)
        complete_window = self.data_complete[index].unsqueeze(0)
        mask = self.mask[index].unsqueeze(0)
        return incomplete_window, complete_window, mask

    def __len__(self):
        return len(self.data_complete)
