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
	if attention_mask is not None:
		with torch.no_grad():
			# remove tokens whose attention mask is 0
			temp_attention_mask = attention_mask.clone().detach()
			sequence_len = temp_attention_mask.sum(dim=-1).unsqueeze(-1)

			# (batch_size, sequence_length, 1)
			temp_mask = temp_attention_mask.unsqueeze(-1)

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
		return "{}h{}m{}s".format(hour, minute, elapse_time)
	elif elapse_time > 60:
		minute = int((elapse_time % 3600) // 60)
		return "{}m{}s".format(minute, elapse_time)
	else:
		return "{}s".format(elapse_time)


def set_seed(seed):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	cudnn.deterministic = True
	cudnn.benchmark = False


def tokenize_and_truncate_from_head(tokenizer, block_data, text_max_len):
	possible_max_len = 999999
	encoded_block_texts = tokenizer(
		block_data, padding=True, verbose=False, add_special_tokens=True,
		truncation=True, max_length=possible_max_len, return_tensors='pt')
	this_seq_max_len = encoded_block_texts['input_ids'].shape[-1]

	this_block_input_ids_list = []
	this_block_token_type_ids_list = []
	this_block_attention_mask_list = []

	if this_seq_max_len == possible_max_len:
		raise Exception("possible_max_len is reached!!!!")
	elif this_seq_max_len <= text_max_len:
		# re-tokenize to max len
		encoded_block_texts = tokenizer(
			block_data, padding='max_length', verbose=False, add_special_tokens=True,
			truncation=True, max_length=text_max_len, return_tensors='pt')

		# read data
		this_block_input_ids_list.append(encoded_block_texts['input_ids'])
		this_block_token_type_ids_list.append(encoded_block_texts['token_type_ids'])
		this_block_attention_mask_list.append(encoded_block_texts['attention_mask'])
	else:
		# these should be truncated from head
		this_input_ids = encoded_block_texts['input_ids']
		this_token_type_ids = encoded_block_texts['token_type_ids']
		this_attention_mask = encoded_block_texts['attention_mask']

		pad_id = tokenizer.convert_tokens_to_ids('[PAD]')

		# truncate row by row
		for text_input_ids, text_token_type_ids, text_attention_mask in zip(this_input_ids, this_token_type_ids,
																			this_attention_mask):
			pad_len = (text_input_ids == pad_id).sum(-1)
			text_len = text_input_ids.shape[0] - pad_len

			if text_len < text_max_len:
				temp_input_ids = text_input_ids[:text_max_len].unsqueeze(0)
				temp_token_type_ids = text_token_type_ids[:text_max_len].unsqueeze(0)
				temp_attention_mask = text_attention_mask[:text_max_len].unsqueeze(0)
			else:
				begin_index = text_len - text_max_len
				temp_input_ids = text_input_ids[begin_index:text_len].unsqueeze(0)
				temp_token_type_ids = text_token_type_ids[begin_index:text_len].unsqueeze(0)
				temp_attention_mask = text_attention_mask[begin_index:text_len].unsqueeze(0)

			this_block_input_ids_list.append(temp_input_ids)
			this_block_token_type_ids_list.append(temp_token_type_ids)
			this_block_attention_mask_list.append(temp_attention_mask)

	return this_block_input_ids_list, this_block_token_type_ids_list, this_block_attention_mask_list, this_seq_max_len


def clean_input_ids(input_ids, attention_mask, token_type_ids):
	max_seq_len = torch.max(attention_mask.sum(-1))

	# ensure only pad be filtered
	# dropped_input_ids = input_ids[:, max_seq_len:]
	# assert torch.max(dropped_input_ids.sum(-1)) == 0

	input_ids = input_ids[:, :max_seq_len]
	token_type_ids = token_type_ids[:, :max_seq_len]
	attention_mask = attention_mask[:, :max_seq_len]

	return input_ids, attention_mask, token_type_ids


def clean_input_embeddings(attention_mask, embeddings):
	max_seq_len = torch.max(attention_mask.sum(-1))
	embeddings = embeddings[:, :max_seq_len, :]
	attention_mask = attention_mask[:, :max_seq_len]

	return attention_mask, embeddings

