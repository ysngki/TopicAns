# coding=utf-8
import pandas as pd
from datasets import load_from_disk
import numpy as np
from torch import nn, multiprocessing
import torch
import time
import sys
import os

# import multiprocessing


class CosineSimilarityTest(torch.nn.Module):
	def __init__(self):
		super(CosineSimilarityTest, self).__init__()

	def forward(self, x1, x2):
		x2 = x2.t()
		x = x1.mm(x2)

		x1_frobenius = x1.norm(dim=1).unsqueeze(0).t()
		x2_frobenins = x2.norm(dim=0).unsqueeze(0)
		x_frobenins = x1_frobenius.mm(x2_frobenins)

		final = x.mul(1 / x_frobenins)
		return final


# 多进程函数
# 需要专属的矩阵，专属的question_len，专属的idf
def multi_process_function(final_value, out_each_question_len, in_each_question_len,
						   this_idf, in_idfs, device, top_k, offset):
	# final_value = o_final_value.to(device)
	# 历史遗留问题
	len_this_indices = len(this_idf)

	out_now_start_index = 0
	in2out_matrix = None

	for first_index in range(len_this_indices):
		# 截取
		new_out_start_index = out_now_start_index + out_each_question_len[first_index]
		temp_relatedness_matrix = final_value[out_now_start_index:new_out_start_index]

		# 如果这个问题没有单词
		if out_now_start_index == new_out_start_index:
			new_data = torch.zeros(final_value.shape[1], device=device)
		else:
			out_now_start_index = new_out_start_index

			# 按列求最大值,长度等于列数
			new_data = torch.max(temp_relatedness_matrix, dim=0)[0]

		if in2out_matrix is None:
			in2out_matrix = new_data
		else:
			in2out_matrix = torch.cat((in2out_matrix, new_data), dim=0)

	in2out_matrix = in2out_matrix.view(len_this_indices, -1)

	# 乘上idf
	all_idf = []
	for temp_idf in in_idfs:
		all_idf += temp_idf

	tensor_idf = torch.tensor(all_idf, device=device)
	in2out_matrix *= tensor_idf

	# 求和处理下, 同时计算下一步的矩阵
	in_now_start_index = 0
	final_in2out_matrix = None

	out2in_matrix = None

	for first_index in range(len(in_idfs)):
		# 截取
		new_in_start_index = in_now_start_index + in_each_question_len[first_index]

		# 如果这个问题没有单词，那么它和所有问题的相似度都是0
		if in_now_start_index == new_in_start_index:
			new_data = torch.zeros(len_this_indices, device=device)
		else:
			temp_matrix = in2out_matrix[:, in_now_start_index:new_in_start_index]
			temp_idf = tensor_idf[in_now_start_index:new_in_start_index]

			# 对行求和
			new_data = temp_matrix.sum(dim=1)

			# 除idf的总和
			new_data /= temp_idf.sum()

		if final_in2out_matrix is None:
			final_in2out_matrix = new_data
		else:
			final_in2out_matrix = torch.cat((final_in2out_matrix, new_data), dim=0)

		# 我是分割线-----------
		# 截取
		temp_relatedness_matrix = final_value[:, in_now_start_index:new_in_start_index]

		if in_now_start_index == new_in_start_index:
			new_data = torch.zeros(final_value.shape[0], device=device)
		else:
			in_now_start_index = new_in_start_index
			# 按列求最大值
			new_data = torch.max(temp_relatedness_matrix, dim=1)[0]

		if out2in_matrix is None:
			out2in_matrix = new_data
		else:
			out2in_matrix = torch.cat((out2in_matrix, new_data), dim=0)

	# 进来的问题 与 这一块问题 的 相似度
	final_in2out_matrix = final_in2out_matrix.view(-1, len_this_indices).permute(1, 0)

	out2in_matrix = out2in_matrix.view(-1, final_value.shape[0])

	# 乘上idf
	all_idf = []
	for temp_idf in this_idf:
		all_idf += temp_idf

	tensor_idf = torch.tensor(all_idf, device=device)
	out2in_matrix *= tensor_idf
	out2in_matrix = out2in_matrix.permute(1, 0)

	# 求和处理下
	out_now_start_index = 0
	final_out2in_matrix = None

	for first_index in range(len_this_indices):
		# 截取
		new_out_start_index = out_now_start_index + out_each_question_len[first_index]
		if out_now_start_index == new_out_start_index:
			new_data = torch.zeros(len(in_idfs), device=device)
		else:
			temp_matrix = out2in_matrix[out_now_start_index:new_out_start_index, :]
			temp_idf = tensor_idf[out_now_start_index:new_out_start_index]

			out_now_start_index = new_out_start_index

			# 对列求和
			new_data = temp_matrix.sum(dim=0)

			# 除idf的总和
			new_data /= temp_idf.sum()

		if final_out2in_matrix is None:
			final_out2in_matrix = new_data
		else:
			final_out2in_matrix = torch.cat((final_out2in_matrix, new_data), dim=0)

	# 进来的问题 与 这一块问题 的 相似度
	final_out2in_matrix = final_out2in_matrix.view(len_this_indices, -1)

	temp_matrix = (final_in2out_matrix + final_out2in_matrix) / 2.0

	# 更新新的topk
	new_top_value, new_top_indices = temp_matrix.topk(top_k, dim=1)
	new_top_indices += offset

	final_top_value = new_top_value
	final_top_indices = new_top_indices

	torch.cuda.empty_cache()

	return final_top_value, final_top_indices


def updated_process_function(start_index, end_index, this_indices, this_idf, top_k):
	# 存储结果
	final_top_value = None
	final_top_indices = None

	# 创建模型
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	model = CosineSimilarityTest().to(device)

	# 输入的词向量矩阵
	out_emebddings = []
	# 输入的句子的长度
	out_each_question_len = []

	for question_words in this_indices:
		# 给每个问题存储结果
		temp_count = 0
		for word in question_words:
			out_emebddings.append(tensor_index2embedding(torch.tensor(word)))
			temp_count += 1

		# 每个问题的长度
		out_each_question_len.append(temp_count)

	out_emebddings = torch.stack(out_emebddings, dim=0).to(device)

	# 循环的步长
	IN_LOOP_STEP = 500

	in_now_index = start_index

	while in_now_index < end_index:

		# 得到必要的素材
		in_questions = q_title_indices[in_now_index: min(in_now_index + IN_LOOP_STEP, end_index)]
		in_idfs = q_title_idf[in_now_index: min(in_now_index + IN_LOOP_STEP, end_index)]

		in_now_index += IN_LOOP_STEP

		# 每个问题的单词数
		in_each_question_len = []
		in_accumulate_len = [0]

		# 所有问题的向量集合
		in_embeddings = []

		for question_words in in_questions:
			# 记录这个问题的单词数
			temp_count = 0
			for word in question_words:
				temp_count += 1
				in_embeddings.append(tensor_index2embedding(torch.tensor(word)))

			in_each_question_len.append(temp_count)
			in_accumulate_len.append(in_accumulate_len[-1] + temp_count)

		in_embeddings = torch.stack(in_embeddings, dim=0).to(device)

		# 计算单词间的相似度
		final_value = model(out_emebddings, in_embeddings)

		# -----------------
		offset = in_now_index - IN_LOOP_STEP
		len_this_indices = len(this_idf)

		out_now_start_index = 0
		in2out_matrix = None

		for first_index in range(len_this_indices):
			# 截取
			new_out_start_index = out_now_start_index + out_each_question_len[first_index]
			temp_relatedness_matrix = final_value[out_now_start_index:new_out_start_index]

			# 如果这个问题没有单词
			if out_now_start_index == new_out_start_index:
				new_data = torch.zeros(final_value.shape[1], device=device)
			else:
				out_now_start_index = new_out_start_index

				# 按列求最大值,长度等于列数
				new_data = torch.max(temp_relatedness_matrix, dim=0)[0]

			if in2out_matrix is None:
				in2out_matrix = new_data
			else:
				in2out_matrix = torch.cat((in2out_matrix, new_data), dim=0)

		in2out_matrix = in2out_matrix.view(len_this_indices, -1)

		# 乘上idf
		all_idf = []
		for temp_idf in in_idfs:
			all_idf += temp_idf

		tensor_idf = torch.tensor(all_idf, device=device)
		in2out_matrix *= tensor_idf

		# 求和处理下, 同时计算下一步的矩阵
		in_now_start_index = 0
		final_in2out_matrix = None

		out2in_matrix = None

		for first_index in range(len(in_idfs)):
			# 截取
			new_in_start_index = in_now_start_index + in_each_question_len[first_index]

			# 如果这个问题没有单词，那么它和所有问题的相似度都是0
			if in_now_start_index == new_in_start_index:
				new_data = torch.zeros(len_this_indices, device=device)
			else:
				temp_matrix = in2out_matrix[:, in_now_start_index:new_in_start_index]
				temp_idf = tensor_idf[in_now_start_index:new_in_start_index]

				# 对行求和
				new_data = temp_matrix.sum(dim=1)

				# 除idf的总和
				new_data /= temp_idf.sum()

			if final_in2out_matrix is None:
				final_in2out_matrix = new_data
			else:
				final_in2out_matrix = torch.cat((final_in2out_matrix, new_data), dim=0)

			# 我是分割线-----------
			# 截取
			temp_relatedness_matrix = final_value[:, in_now_start_index:new_in_start_index]

			if in_now_start_index == new_in_start_index:
				new_data = torch.zeros(final_value.shape[0], device=device)
			else:
				in_now_start_index = new_in_start_index
				# 按列求最大值
				new_data = torch.max(temp_relatedness_matrix, dim=1)[0]

			if out2in_matrix is None:
				out2in_matrix = new_data
			else:
				out2in_matrix = torch.cat((out2in_matrix, new_data), dim=0)

		# 进来的问题 与 这一块问题 的 相似度
		final_in2out_matrix = final_in2out_matrix.view(-1, len_this_indices).permute(1, 0)

		out2in_matrix = out2in_matrix.view(-1, final_value.shape[0])

		# 乘上idf
		all_idf = []
		for temp_idf in this_idf:
			all_idf += temp_idf

		tensor_idf = torch.tensor(all_idf, device=device)
		out2in_matrix *= tensor_idf
		out2in_matrix = out2in_matrix.permute(1, 0)

		# 求和处理下
		out_now_start_index = 0
		final_out2in_matrix = None

		for first_index in range(len_this_indices):
			# 截取
			new_out_start_index = out_now_start_index + out_each_question_len[first_index]
			if out_now_start_index == new_out_start_index:
				new_data = torch.zeros(len(in_idfs), device=device)
			else:
				temp_matrix = out2in_matrix[out_now_start_index:new_out_start_index, :]
				temp_idf = tensor_idf[out_now_start_index:new_out_start_index]

				out_now_start_index = new_out_start_index

				# 对列求和
				new_data = temp_matrix.sum(dim=0)

				# 除idf的总和
				new_data /= temp_idf.sum()

			if final_out2in_matrix is None:
				final_out2in_matrix = new_data
			else:
				final_out2in_matrix = torch.cat((final_out2in_matrix, new_data), dim=0)

		# 进来的问题 与 这一块问题 的 相似度
		final_out2in_matrix = final_out2in_matrix.view(len_this_indices, -1)

		temp_matrix = (final_in2out_matrix + final_out2in_matrix) / 2.0

		# 更新新的topk
		new_top_value, new_top_indices = temp_matrix.topk(top_k, dim=1)
		new_top_indices += offset

		torch.cuda.empty_cache()

		# -------------------------------------------
		if final_top_value is None:
			final_top_value = new_top_value
			final_top_indices = new_top_indices
		else:
			# 和以前的top_value拼接一下
			temp_top_value = torch.cat((final_top_value, new_top_value), dim=1)
			temp_top_indices = torch.cat((final_top_indices, new_top_indices), dim=1)

			final_top_value, temp_relative_indices = temp_top_value.topk(top_k, dim=1)
			final_top_indices = temp_top_indices.gather(1, temp_relative_indices)

		torch.cuda.empty_cache()

	del model
	return final_top_indices


if __name__ == '__main__':
	program_flag = eval(sys.argv[1])
	cuda_flag = eval(sys.argv[2])

	from_flag = eval(sys.argv[3])
	to_flag = eval(sys.argv[4])

	os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_flag)

	# 读取my_word2index
	my_word2index = np.load('./generated_file/my_word2index.npy', allow_pickle=True).item()
	my_index2word = {v: k for k, v in my_word2index.items()}

	# 读取word_idf
	word_idf = []
	with open("./generated_file/word_idf", 'r') as f:
		for index, line in enumerate(f):
			temp_result = line.split()
			if len(temp_result) > 0:
				word_idf.append(eval(temp_result[0]))
			else:
				break

	# 读取训练好的glove词向量
	my_word2embedding = {}
	with open("/data/yuanhang/important_code/so_python/glove/vectors.txt", mode='r') as f:
		for line in f:
			line_list = line.split()
			word = line_list[0]
			embed = line_list[1:]
			embed = [float(num) for num in embed]
			my_word2embedding[word] = embed

	# 以后都是 word -》 index -》 embeddding
	my_index2embedding = {}
	for word in my_word2index.keys():
		embedding = my_word2embedding.get(word)

		if embedding is not None:
			my_index2embedding[my_word2index[word]] = embedding
		else:
			my_index2embedding[my_word2index[word]] = [0.0] * 128

	# 把embedding稍稍处理一下，方便后面的计算，大概
	temp_data = [my_index2embedding[ix] for ix in range(len(my_word2index))]
	tensor_index2embedding = nn.Embedding.from_pretrained(torch.tensor(temp_data))

	qa_info = load_from_disk("./generated_file/python_qa_info.dataset")
	q_ids = qa_info['q_id']
	print(qa_info)

	# 做找相关问题的准备
	q_csv_reader = pd.read_csv('generated_file/processed_python_q_id_content.csv', index_col='id')

	related_q_ids = []

	# 为了加速计算，提前处理下数据
	q_titles = []
	for q_id in q_ids:
		q_titles.append(q_csv_reader.loc[q_id]['title'])

	q_title_indices = []
	for title in q_titles:
		target_words = title.split()
		target_indices = []
		for w in target_words:
			if my_word2index.get(w) is not None:
				target_indices.append(my_word2index.get(w))
		target_indices = tuple(target_indices)
		q_title_indices.append(target_indices)

	q_title_idf = []
	for title in q_title_indices:
		target_idf = []
		for w in title:
			target_idf.append(word_idf[w])
		# target_idf = torch.FloatTensor(target_idf)
		q_title_idf.append(target_idf)

	# 问题间的相似度矩阵
	question_num = len(q_ids)

	# 清理log文件
	with open("flag" + str( 个x), "w") as f:
		pass
	with open("related_questions" + str(program_flag), "w") as f:
		pass

	now_index = 0
	LOOP_STEP = 500

	with open("flag" + str(program_flag), "a") as f:
		f.write(str(now_index) + "\t" + time.asctime(time.localtime(time.time())) + "\n")

	multiprocessing.set_start_method('spawn', force=True)

	while now_index < question_num:

		if now_index < 10000*from_flag:
			now_index += 1
			continue

		if now_index >= 10000*to_flag:
			break

		# 得到title中每个词在字典中的index,记得去重, idf
		this_indices = q_title_indices[now_index:(now_index + LOOP_STEP)]
		this_idf = q_title_idf[now_index:(now_index + LOOP_STEP)]

		# 记录结果
		print(time.asctime(time.localtime(time.time())))

		result = updated_process_function(0, question_num, this_indices, this_idf, 21)

		print(time.asctime(time.localtime(time.time())))

		torch.cuda.empty_cache()

		# print(q_csv_reader.loc[q_ids[now_index + 1]]['title'])
		#
		# for inner_index in result[1]:
		# 	# print(q_csv_reader.loc[q_ids[inner_index.item()]]['title'])
		# 	print(inner_index.item())
		#
		# for inner_index in result[1]:
		# 	print(q_csv_reader.loc[q_ids[inner_index.item()]]['title'])
		# 	print(inner_index.item())
		# break

		for inner_index in range(len(this_indices)):
			with open("related_questions" + str(program_flag), "a") as f:
				for i in result[inner_index]:
					if i.item() != (now_index + inner_index):
						f.write(str(i.item()) + " ")
				f.write("\n")

		now_index += LOOP_STEP

		print(f"\r{now_index}/{question_num}", end="")

		with open("flag" + str(program_flag), "a") as f:
			f.write(str(now_index) + "\t" + time.asctime(time.localtime(time.time())) + "\n")

	with open("flag" + str(program_flag), "a") as f:
		f.write("finished!" + "\t" + time.asctime(time.localtime(time.time())))

	print("\nfinished!")
