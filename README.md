# TopicAns
Utilize TopicModel to faciliate Answer recommendation on Technical QA sites

## run
use multiple gpus
```bash
CUDA_VISIBLE_DEVICES=1,2 python -u -m torch.distributed.launch --nproc_per_node=2 main.py -n 1,2 --data_distribute
```
use one gpu
```bash
python -u main.py -n 1 
```
