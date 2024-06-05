# ParamReL: Learning Parameter Space Representation  via Progressively Encoding Bayesian Flow Networks



This is the official code release for[ParamReL](https://arxiv.org/abs/2405.15268) by Zhangkai Wu, Xuhui Fan, Jin Li, Zhilin Zhao, Hui Chen, Longbing Cao.



## Setup

```bash
pip install accelerate matplotlib omegaconf rich neptune
```



## Train



###  For Discrete Data

```shell
# train BFN based model
accelerate launch --num_processes 2 --multi_gpu trainBFN.py config_file=BFNconfigs/mnist_discrete.yaml
accelerate launch trainBFN.py config_file=BFNconfigs/mnist_discrete.yaml
# train infoBFN based model
accelerate launch --num_processes 2 --multi_gpu trainInfoBFN.py config_file=infoBFNconfigs/mnist_infoBFN.yaml
accelerate launch trainInfoBFN.py config_file=infoBFNconfigs/mnist_infoBFN.yaml
```

### For Continous Data

```shell
# train BFN based model on Cifar10
accelerate launch trainBFN.py config_file=BFNconfigs/cifar10_continuous_256bins.yaml

# train infoBFN based model on Cifar10
accelerate launch trainInfoBFN.py config_file=infoBFNconfigs/cifar10_continuous_256bins.yaml

# train infoBFN based model on Cifar10
accelerate launch trainInfoBFN.py config_file=infoBFNconfigs/celeba_continuous_256bins.yaml
accelerate launch --num_processes 2 --multi_gpu trainInfoBFN.py config_file=infoBFNconfigs/celeba_continuous_256bins.yaml


accelerate launch --num_processes 2 --multi_gpu trainInfoBFN.py config_file=infoBFNconfigs/celeba_continuous_256bins.yaml
```



## Representation Learning Test
### Sample and CleanFID
```shell
# extract raw data
accelerate launch extract.py config_file=infoBFNconfigs/extract.yaml
# sampling from trained model
accelerate launch gen.py config_file=./infoBFNconfigs/celeba_continuous_Nobin.yaml
# cleanfid
python fid.py 


python sample.py seed=1 config_file=./infoBFNconfigs/celeba_continuous_256bins.yaml load_model=./nciCkpts/celeba1.pt samples_shape="[4, 64, 64, 3]" n_steps=100 a_dim=32 save_file=./celebaCon100.pt

python -c "import torch; from data import batch_to_images; batch_to_images(torch.load('./celebadis10000.pt')).savefig('celebadis10000.png')"

```

### latent quality



### Disentanglement



### Latent Traversal


