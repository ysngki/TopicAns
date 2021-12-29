import argparse
import torch
from torch.backends import cudnn
import random
import numpy as np

from QA_trainer import TrainWholeModel


def set_seed(seed):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	cudnn.deterministic = True
	cudnn.benchmark = False


def read_arguments():
	parser = argparse.ArgumentParser()

	# must set
	# add model
	parser.add_argument("--model_class", required=True, type=str, choices=['OneSupremeMemory', 'PureMemory', 'BasicModel',
							   'InputMemorySelfAtt', 'PureMemorySelfAtt', 'QAMemory', 'QAModel', 'CrossBERT', 'ADecoder'])

	parser.add_argument("--dataset_name", "-d", required=True, type=str)
	parser.add_argument("--memory_num", "-m", default=50, type=int)
	parser.add_argument("--pretrained_bert_path", default='prajjwal1/bert-small', type=str)
	parser.add_argument("--nvidia_number", "-n", required=True, type=str)
	parser.add_argument("--one_stage", action="store_true", default=False)
	parser.add_argument("--model_save_prefix", default="", type=str)
	parser.add_argument("--memory_save_prefix", default="", type=str)
	parser.add_argument("--dataset_split_num", default=20, type=int)
	parser.add_argument("--val_batch_size", default=64, type=int)
	parser.add_argument("--train_batch_size", default=64, type=int)
	parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
	parser.add_argument("--val_num_each_epoch", default=3, type=int)
	parser.add_argument("--save_model_dict", default="./model/", type=str)
	parser.add_argument("--first_stage_lr", default=0.3, type=float, help="the lr of memory at first stage")

	# related to model
	parser.add_argument("--hop_num", default=1, type=int, help = 'hop num for pure memory')
	parser.add_argument("--composition", type=str, default='pooler', help = 'control the way to get sentence representation')

	# related to train
	parser.add_argument("--restore", action="store_true", default=False, help="use restore and only_final together to control which model to read!")

	parser.add_argument("--no_initial_test", action="store_true", default=False)

	parser.add_argument("--load_classifier", "-l", action="store_true", default=False, help="load classifier")
	parser.add_argument("--classifier_path", type=str, help="load classifier")

	parser.add_argument("--load_memory", action="store_true", default=False)
	parser.add_argument("--load_middle", action="store_true", default=False)
	parser.add_argument("--load_model_dict", type=str)

	parser.add_argument("--distill", action="store_true", default=False)
	parser.add_argument("--teacher_path", default="./model/teacher", type=str)
	parser.add_argument("--mlm", action="store_true", default=False)
	parser.add_argument("--only_final", action="store_true", default=False,
						help="using two stage setting but only train last stage")

	# 设置并行需要改的
	parser.add_argument('--local_rank', type=int, default=0, help='node rank for distributed training')
	parser.add_argument("--data_parallel", action="store_true", default=False)
	parser.add_argument("--data_distribute", action="store_true", default=False)

	# default arguments
	parser.add_argument("--seed", "-s", default=42, type=int)
	parser.add_argument("--text_max_len", default=512, type=int)
	parser.add_argument("--ranking_candidate_num", default=5, type=int)
	parser.add_argument("--label_num", default=4, type=int)  # !!!
	parser.add_argument("--num_train_epochs", "-e",type=int, default=50)

	# outdated
	parser.add_argument("--print_num_each_epoch", default=20, type=int)
	parser.add_argument("--val_candidate_num", default=100, type=int, help="candidates num for ranking")
	parser.add_argument("--train_candidate_num", default=16, type=int, help="only need by cross")
	parser.add_argument("--latent_dim", default=100, type=int)
	parser.add_argument("--train_vae", action="store_true", default=False)


	args = parser.parse_args()
	print("args:", args)
	return args


if __name__ == '__main__':
	my_args = read_arguments()

	# 设置随机种子
	set_seed(my_args.seed)

	# 创建训练类
	my_train_model = TrainWholeModel(my_args)

	# 设置训练参数
	my_train_two_stage_flag = False
	# add model
	if my_args.model_class in ['OneSupremeMemory', 'PureMemory', 'VaeAttention', 'VaeAttentionPlus', 'BasicModel',
							   'InputMemorySelfAtt', 'PureMemorySelfAtt', 'QAMemory', 'ADecoder']:
		my_train_two_stage_flag = True

	if my_args.one_stage:
		my_train_two_stage_flag = False

	if my_args.distill:
		raise Exception("Distillation is not supported yes!")

	# 通过mlm训练memory
	if my_args.mlm:
		my_train_model.train_memory_by_mlm(memory_save_name=my_args.memory_save_prefix + "_" +
															my_args.dataset_name)

	# 如果读取memory，或者不训练mlm，就要train
	if my_args.load_memory or (not my_args.mlm):
		my_train_model.train(model_save_path=my_args.save_model_dict + "/" + my_args.model_save_prefix +
											 my_args.model_class + "_" +
											 my_args.dataset_name, train_two_stage_flag=my_train_two_stage_flag,
							 memory_save_name=my_args.memory_save_prefix + "_" +
											  my_args.dataset_name, only_final=my_args.only_final)

