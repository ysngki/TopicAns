import time
from torch.backends import cudnn
import random
import numpy as np
import torch
import torch.nn.functional as F


def sum_average_tuple(in_tuple):
	sum_result = 0
	for i in in_tuple:
		sum_result += i
	return sum_result, sum_result/len(in_tuple)


def raise_dataset_error():
	raise Exception("This dataset is not supported now!")


def raise_test_error():
	raise Exception("Test stop here!!")


def print_recall_precise(label_hit_num, label_shoot_num, label_target_num, label_num):
	print(label_hit_num, label_shoot_num, label_target_num)
	# 统计命中率
	all_hit_num = 0
	for hit_num in label_hit_num:
		all_hit_num += hit_num

	all_shoot_num = 0
	for shoot_num in label_shoot_num:
		all_shoot_num += shoot_num
	accuracy = all_hit_num * 100 / all_shoot_num
	print(f"accuracy: {accuracy}%")

	# 统计召回率和准确率
	recall = []
	for i in range(0, label_num):
		recall.append(label_hit_num[i] / (label_target_num[i] + 1e-8))

	precise = []
	for i in range(0, label_num):
		precise.append(label_hit_num[i] / (label_shoot_num[i] + 1e-8))

	print("recall:\t")
	for index in range(label_num):
		print(f"Class_{index}: {recall[index]}\t", end="")
	print("\nprecise:\t")
	for index in range(label_num):
		print(f"Class_{index}: {precise[index]}\t", end="")
	print()

	return accuracy


def load_model(model, load_model_path):
	if load_model_path is None:
		print("you should offer model paths!")

	load_path = load_model_path
	model.load_state_dict(torch.load(load_path)['model'])

	# if self.data_distribute:
	# 	load_path += ".pt"
	# 	checkpoint = torch.load(load_path)
	# 	self.model.module.load_state_dict(checkpoint['model'])
	# 	amp.load_state_dict(checkpoint['amp'])
	# elif self.data_parallel:
	# 	load_path += ".pt"
	# 	checkpoint = torch.load(load_path)
	# 	self.model.module.load_state_dict(checkpoint['model'])
	# else:
	# 	self.model.load_state_dict(torch.load(load_path))

	print("model is loaded from", load_path)

	return model


def print_optimizer(optimizer):
	print("*"*100)
	print("Now optimizer lr:")
	for o in optimizer.state_dict()['param_groups']:
		print(o['lr'], end="\t")
	print("\n" + "*" * 100)


def get_rep_by_avg(embeddings, token_type_ids=None, attention_mask=None):
	"""
	do average at dim -2 without remove this dimension
	:param embeddings: (..., sequence len, dim)
	:param token_type_ids: optional
	:param attention_mask: optional
	:return: (..., 1, dim)
	"""
	# should create mask
	if token_type_ids is not None and attention_mask is not None:
		with torch.no_grad():
			# (batch_size, sequence_length)
			temp_mask = token_type_ids.clone().detach()

			# remove tokens whose type id is 1
			temp_mask[temp_mask == 0] = 2
			temp_mask -= 1

			# remove tokens whose attention mask is 0
			temp_attention_mask = attention_mask.clone().detach()
			temp_mask = temp_mask * temp_attention_mask
			sequence_len = temp_mask.sum(dim=-1).unsqueeze(-1)

			# (batch_size, sequence_length, 1)
			temp_mask = temp_mask.unsqueeze(-1)

		# (batch_size, sequence_length, hidden state size)
		embeddings = embeddings * temp_mask

		# get average embedding
		representations = embeddings.sum(dim=-2)

		representations = representations / sequence_len
	else:
		representations = torch.mean(embeddings, dim=-2)

	representations = representations.unsqueeze(-2)

	return representations


def dot_attention(q, k, v, v_mask=None):
	"""
	use q to aggregate v according to k
	:param q: (..., q_num, dim)
	:param k: (..., v_num, dim)
	:param v: (..., v_num, dim)
	:param v_mask: (..., v_num)
	:return: (..., q_num, dim)
	"""

	# (..., q_num, v_num)
	attention_weights = torch.matmul(q, k.transpose(-1, -2))

	if v_mask is not None:
		# # (..., 1, v_num)
		extended_v_mask = (1.0 - v_mask.clone().detach().unsqueeze(-2)) * -100000.0
		attention_weights += extended_v_mask

	attention_weights = F.softmax(attention_weights, -1)

	representations = torch.matmul(attention_weights, v)

	return representations


def get_elapse_time(t0):
	elapse_time = time.time() - t0
	if elapse_time > 3600:
		hour = int(elapse_time // 3600)
		minute = int((elapse_time % 3600) // 60)
		return "{}h{}m".format(hour, minute)
	else:
		minute = int((elapse_time % 3600) // 60)
		return "{}m".format(minute)


def set_seed(seed):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	cudnn.deterministic = True
	cudnn.benchmark = False

