# TaskSpecificMemory
Modift memory mechanism to faciliate text match for techncal Q&amp;A

## run
use multiple gpus
'''bash
CUDA_VISIBLE_DEVICES=1,2 python -u -m torch.distributed.launch --nproc_per_node=2 main.py -n 1,2 --data_distribute -d
'''
use one gpu
'''bash
python -u main.py -n 1 
'''
