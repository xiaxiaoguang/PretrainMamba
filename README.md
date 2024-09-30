Add data_loader before use

Add these code to offcial library , mamba-ssm , in python/site-package/mamba-ssm/modules/mamba_simple.py 

```py
def load_params(self,state_dict,frozentype):
        # print(state_dict.keys())
        with torch.no_grad():
            self.A_log.copy_(state_dict['layers.0.A_log'].to(torch.float32))
            self.D.copy_(state_dict['layers.0.D'].to(torch.float32))
            self.in_proj.weight.copy_(state_dict['layers.0.in_proj.weight'].to(torch.float32))
            self.conv1d.weight.copy_(state_dict['layers.0.conv1d.weight'].to(torch.float32))
            self.conv1d.bias.copy_(state_dict['layers.0.conv1d.bias'].to(torch.float32))
            self.x_proj.weight.copy_(state_dict['layers.0.x_proj.weight'].to(torch.float32))
            self.dt_proj.weight.copy_(state_dict['layers.0.dt_proj.weight'].to(torch.float32))
            self.dt_proj.bias.copy_(state_dict['layers.0.dt_proj.bias'].to(torch.float32))
            self.out_proj.weight.copy_(state_dict['layers.0.out_proj.weight'].to(torch.float32))
            
            if frozentype=='selectAD':
                self.A_log.requires_grad = False
                self.D.requires_grad = False
                self.in_proj.weight.requires_grad = False
                self.conv1d.weight.requires_grad = False
                self.conv1d.bias.requires_grad = False
                print('frozen selectAD')
```

* Pretrain

    python main.py --mod pretrain --datasetname NAME --device 0

* Infrecence

    First, choose the pretrained block and add the path to config.toml

    Second, change the inference params in config.toml

    Third, run:     python main.py --mod inference --datasetname NAME --device 0

* Attention

    d_model in inference config should same to d_model in pretrain config.

    pre_layer must be list.

    frozentype must in ['none' , 'selectAD']


