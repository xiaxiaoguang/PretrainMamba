class TrafficArgs:
    def __init__(self):
        self.device ='cuda:1'
        self.data ='traffic' 
        self.root_path ='./dataset/'
        self.data_path ='traffic.txt'

        self.adj_data =''

        self.seq_len =96
        self.pred_len =96
        
        self.num_nodes = 1
        self.batch_size =100

        self.num_workers =0

class ElectricityArgs:
    def __init__(self):
        self.device ='cuda:1'
        self.data ='electricity' 
        self.root_path ='./dataset/'
        self.data_path ='electricity.txt'

        self.adj_data =''

        self.seq_len =96
        self.pred_len = 96
        
        self.num_nodes = 1
        self.batch_size =200
        self.num_workers =0

class ETTh1Args:
    def __init__(self):
        self.device ='cuda:1'
        self.data ='ETTh1' 
        self.root_path ='./dataset/'
        self.data_path ='ETTh1.csv'

        self.adj_data =''

        self.seq_len =96
        self.pred_len =96
        
        self.num_nodes = 1
        self.batch_size =200

        self.num_workers =0

class ETTh2Args:
    def __init__(self):
        self.device ='cuda:1'
        self.data ='ETTh2' 
        self.root_path ='./dataset/'
        self.data_path ='ETTh2.csv'

        self.adj_data =''

        self.seq_len =96
        self.pred_len =96
        
        self.num_nodes = 1
        self.batch_size =200

        self.num_workers =0

class ETTm1Args:
    def __init__(self):
        self.device ='cuda:1'
        self.data ='ETTm1' 
        self.root_path ='./dataset/'
        self.data_path ='ETTm1.csv'

        self.adj_data =''

        self.seq_len =96
        self.pred_len =96
        
        self.num_nodes = 1
        self.batch_size =200

        self.num_workers =0

class ETTm2Args:
    def __init__(self):
        self.device ='cuda:1'
        self.data ='ETTm2' 
        self.root_path ='./dataset/'
        self.data_path ='ETTm2.csv'

        self.adj_data =''

        self.seq_len =96
        self.pred_len =96
        
        self.num_nodes = 1
        self.batch_size =150

        self.num_workers =0

def load_arg(name):
    if name=='traffic':
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