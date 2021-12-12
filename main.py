import argparse
import torch
from torch.backends import cudnn
import random
import numpy as np

from reformed_train_code_parallel import TrainWholeModel


def set_seed(seed):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	cudnn.deterministic = True
	cudnn.benchmark = False


def read_arguments():
	parser = argparse.ArgumentParser()

	# default arguments
	parser.add_argument("--seed", "-s", default=42, type=int)
	parser.add_argument("--load_model", "-l", action="store_true", default=False)
	parser.add_argument("--load_middle", action="store_true", default=False)
	parser.add_argument("--one_stage", action="store_true", default=False)

	parser.add_argument("--save_model_dict", default="./model/", type=str)
	parser.add_argument("--val_num_each_epoch", default=3, type=int)
	parser.add_argument("--print_num_each_epoch", default=20, type=int) # 似乎被淘汰了
	parser.add_argument("--pretrained_bert_path", default='prajjwal1/bert-small', type=str)
	parser.add_argument("--text_max_len", default=512, type=int)
	parser.add_argument("--dataset_split_num", default=20, type=int)
	parser.add_argument("--ranking_candidate_num", default=5, type=int)

	parser.add_argument("--dataset_name", "-d", type=str)  # !!!
	parser.add_argument("--val_batch_size", default=64, type=int) 	# !!!!
	parser.add_argument("--train_batch_size", default=64, type=int)  # !!!
	parser.add_argument("--val_candidate_num", default=100, type=int, help="candidates num for ranking")  # !!!
	parser.add_argument("--train_candidate_num", default=16, type=int, help="only need by cross")  # !!!
	parser.add_argument("--memory_num", "-m", default=50, type=int)  # !!!
	parser.add_argument("--latent_dim", default=100, type=int)  # !!!
	parser.add_argument("--label_num", default=4, type=int)  # !!!
	parser.add_argument("--train_vae", action="store_true", default=False)  # !!!
	parser.add_argument("--hop_num", default=1, type=int)  # !!!

	# 设置并行需要改的
	parser.add_argument('--local_rank', type=int, default=0, help = 'node rank for distributed training')
	parser.add_argument("--data_parallel", action="store_true", default=False)
	parser.add_argument("--data_distribute", action="store_true", default=False)

	# hand set arguments
	parser.add_argument("--nvidia_number", "-n", required=True, type=str)  # !!!
	parser.add_argument("--model_class", required=True, type=str)  # !!!

	parser.add_argument("--load_model_dict", type=str)
	parser.add_argument("--model_save_prefix", default="", type=str)

	# 新的玩意相关
	parser.add_argument("--distill", action="store_true", default=False)
	parser.add_argument("--teacher_path", default="./model/teacher", type=str)
	parser.add_argument("--mlm", action="store_true", default=False)
	parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
	parser.add_argument("--num_train_epochs", "-e",type=int, default=50)
	parser.add_argument("--no_initial_test", action="store_true", default=False)

	args = parser.parse_args()
	print("args:", args)
	return args


if __name__ == '__main__':
	my_args = read_arguments()

	# 设置随机种子
	set_seed(my_args.seed)

	# 创建训练类
	my_train_model = TrainWholeModel(my_args)

	# 是否训练vae
	if my_args.train_vae:
		my_train_model.train_vae()
	if my_args.model_class in ['OneSupremeMemory'] and not my_args.load_model:
		my_train_model.load_vae_model()

	# 设置训练参数
	my_train_two_stage_flag = False
	# add model
	if my_args.model_class in ['OneSupremeMemory', 'PureMemory', 'VaeAttention', 'VaeAttentionPlus', 'BasicModel',
							   'InputMemorySelfAtt', 'PureMemorySelfAtt']:
		my_train_two_stage_flag = True

	if my_args.one_stage:
		my_train_two_stage_flag = False

	if my_args.distill:
		my_train_model.train_with_teacher(model_save_path=my_args.save_model_dict + "/" + my_args.model_save_prefix +
														  my_args.model_class + "_" +
														  my_args.dataset_name,
										  train_two_stage_flag=my_train_two_stage_flag)

	if my_args.mlm:
		my_train_model.train_memory_by_mlm(memory_save_name=my_args.model_save_prefix + "_" +
															my_args.dataset_name)
	else:
		my_train_model.train(model_save_path=my_args.save_model_dict + "/" + my_args.model_save_prefix +
											 my_args.model_class + "_" +
											 my_args.dataset_name, train_two_stage_flag=my_train_two_stage_flag)

