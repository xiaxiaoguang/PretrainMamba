class PEMSBayArgs: 
    def __init__(self):
        self.device ='cuda:1'
        self.data ='PEMS-BAY' 
        self.root_path ='dataset/'
        self.data_path ='traffic/PEMS-BAY/'

        self.adj_data ='dataset/traffic/PEMS-BAY/adj.pkl'

        self.seq_len =12 
        self.pred_len =12 

        self.batch_size =64
        self.num_nodes  =325 

        self.num_workers =0

class METRLAArgs:
    def __init__(self):
        self.device ='cuda:1'
        self.data ='METR-LA' 
        self.root_path ='dataset/'
        self.data_path ='traffic/METR-LA/'

        self.adj_data ='dataset/traffic/METR-LA/adj.pkl'

        self.seq_len =12 
        self.pred_len =12 

        self.batch_size =64
        self.num_nodes = 207 

        self.num_workers =0

class TrafficArgs:
    def __init__(self):
        self.device ='cuda:1'
        self.data ='traffic' 
        self.root_path ='dataset/'
        self.data_path ='traffic/traffic/traffic.txt'

        self.adj_data ='dataset/traffic/traffic/adj.pkl'

        self.seq_len =96
        self.pred_len =96
        
        self.num_nodes = 1
        self.batch_size = 64

        self.num_workers =0

class ElectricityArgs:
    def __init__(self):
        self.device ='cuda:1'
        self.data ='electricity' 
        self.root_path ='dataset/'
        self.data_path ='electricity/electricity/electricity.txt'

        self.adj_data ='dataset/electricity/electricity/adj.pkl'

        self.seq_len =96
        self.pred_len = 96
        
        self.num_nodes = 1
        self.batch_size =128

        self.num_workers =0

class ETTh1Args:
    def __init__(self):
        self.device ='cuda:1'
        self.data ='ETTh1' 
        self.root_path ='dataset/'
        self.data_path ='electricity/ETT/ETTh1.csv'

        self.adj_data ='dataset/electricity/ETT/adj.pkl'

        self.seq_len =96
        self.pred_len =192
        
        self.num_nodes = 1
        self.batch_size =512

        self.num_workers =0

class ETTh2Args:
    def __init__(self):
        self.device ='cuda:1'
        self.data ='ETTh2' 
        self.root_path ='dataset/'
        self.data_path ='electricity/ETT/ETTh2.csv'

        self.adj_data ='dataset/electricity/ETT/adj.pkl'

        self.seq_len =96
        self.pred_len =720
        
        self.num_nodes = 1
        self.batch_size =512

        self.num_workers =0

class ETTm1Args:
    def __init__(self):
        self.device ='cuda:1'
        self.data ='ETTm1' 
        self.root_path ='dataset/'
        self.data_path ='electricity/ETT/ETTm1.csv'

        self.adj_data ='dataset/electricity/ETT/adj.pkl'

        self.seq_len =96
        self.pred_len =96
        
        self.num_nodes = 1
        self.batch_size =512

        self.num_workers =0

class ETTm2Args:
    def __init__(self):
        self.device ='cuda:1'
        self.data ='ETTm2' 
        self.root_path ='dataset/'
        self.data_path ='electricity/ETT/ETTm2.csv'

        self.adj_data ='dataset/electricity/ETT/adj.pkl'

        self.seq_len =96
        self.pred_len =96
        
        self.num_nodes = 1
        self.batch_size =512

        self.num_workers =0

def load_arg(name):
    if name=='pems':
        return PEMSBayArgs()
    elif name=='metr':
        return METRLAArgs()
    elif name=='traffic':
        return TrafficArgs()
    elif name=='electricity':
        return ElectricityArgs()
    elif name=='ETTh1':
        return ETTh1Args()
    elif name=='ETTh2':
        return ETTh2Args()
    elif name=='ETTm1':
        return ETTm1Args()
    elif name=='ETTm2':
        return ETTm2Args()
    else:
        raise NotImplementedError(f'dataset {name} not exist.')