# REPETITIVE CONTRASTIVE LEARNING ENHANCES MAMBA’S SELECTIVITY IN TIME SERIES PREDICTION

An official code implementation for Mamba repetitive contrastive learning.

## Additional Code 

Add these code to the offcial library , mamba-ssm , in python/site-package/mamba-ssm/modules/mamba_simple.py 

```py
    def load_params(self,state_dict,frozentype):
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
            if frozentype=='select':
                self.in_proj.weight.requires_grad = False
                self.conv1d.weight.requires_grad = False
                self.conv1d.bias.requires_grad = False
                print('frozen select')
            elif frozentype=='selectAD':
                self.A_log.requires_grad = False
                self.D.requires_grad = False
                self.in_proj.weight.requires_grad = False
                self.conv1d.weight.requires_grad = False
                self.conv1d.bias.requires_grad = False
                print('frozen selectAD')
```

If you need to test your own mamba-model, you can modify the load_params function and call the one in mamba-ssm, specifically ,for a model with layers as Maamba ModuleList, it should be:

```py
    def load_params(self,layer_ids,load_path,frozentype):
        state_dict = torch.load(load_path)
        for i in layer_ids:
            self.layers[i].load_params(state_dict,frozentype)
```
## Running

### Pretraining

Refer to [`pretrain.bash`](./pretrain.bash). You can modify the basic hyperparameters of Mamba in [`config.toml`](./config.toml).

### Inference

1. Select the best pretrained blocks after pretraining and add their paths to [`config.toml`](./config.toml).
2. Alternatively, specify the path in the command-line arguments, but note that not all settings can be adjusted this way. Details can be found in [`inference.bash`](./inference.bash) and [`main.py`](./main.py).
3. Ensure that the hyperparameters of the Mamba layers remain consistent between pretraining and inference stages.
4. To run inference, refer to the command in [`inference.bash`](./inference.bash).

### Testing

- Test results for each epoch are stored in the output folders during inference.
- You can use `test_all` or `test` mode in the command, but ensure that the correct folder paths are specified in [`config.toml`](./config.toml).

## Attention

### Config Keywords

- **`frozentype`** should be one of:
  - `'none'`
  - `'select'`
  - `'selectAD'`
- **`pre_layer`** should be a list. For example, `[0, 3]` means the first and third Mamba blocks will use pretrained parameters.

### Datasets

- The only provided dataset is `ETTh1.csv`. If you need other datasets, add them to the `dataset/` folder.
- Modify [`/dataprovider/dataArgs.py`](./dataprovider/dataArgs.py) to update `root_path` and `data_path` so they correctly point to your dataset files.
- The code currently supports only the simple Mamba model.
