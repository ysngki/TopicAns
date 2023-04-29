# coding=utf-8
import os
import sys
import time
import numpy as np
import pandas as pd
from datasets import load_from_disk

import torch
from torch import nn, multiprocessing



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


# Multi-process functions
# Need exclusive matrix, exclusive question_len, exclusive idf
def multi_process_function(final_value, out_each_question_len, in_each_question_len,
						   this_idf, in_idfs, device, top_k, offset):
	# final_value = o_final_value.to(device)
	len_this_indices = len(this_idf)

	out_now_start_index = 0
	in2out_matrix = None

	for first_index in range(len_this_indices):
		# Interception
		new_out_start_index = out_now_start_index + out_each_question_len[first_index]
		temp_relatedness_matrix = final_value[out_now_start_index:new_out_start_index]

		# If this question has no word
		if out_now_start_index == new_out_start_index:
			new_data = torch.zeros(final_value.shape[1], device=device)
		else:
			out_now_start_index = new_out_start_index

			# The maximum value by column, the length is equal to the number of columns
			new_data = torch.max(temp_relatedness_matrix, dim=0)[0]

		if in2out_matrix is None:
			in2out_matrix = new_data
		else:
			in2out_matrix = torch.cat((in2out_matrix, new_data), dim=0)

	in2out_matrix = in2out_matrix.view(len_this_indices, -1)

	# Multiply idf
	all_idf = []
	for temp_idf in in_idfs:
		all_idf += temp_idf

	tensor_idf = torch.tensor(all_idf, device=device)
	in2out_matrix *= tensor_idf

	# Under the summation process, the matrix of the next step is also calculated
	in_now_start_index = 0
	final_in2out_matrix = None

	out2in_matrix = None

	for first_index in range(len(in_idfs)):
		# Interception
		new_in_start_index = in_now_start_index + in_each_question_len[first_index]

		# If this question has no word，那么它和所有问题的相似度都是0
		if in_now_start_index == new_in_start_index:
			new_data = torch.zeros(len_this_indices, device=device)
		else:
			temp_matrix = in2out_matrix[:, in_now_start_index:new_in_start_index]
			temp_idf = tensor_idf[in_now_start_index:new_in_start_index]

			# Summation over rows
			new_data = temp_matrix.sum(dim=1)

			# Excluding the sum of idf
			new_data /= temp_idf.sum()

		if final_in2out_matrix is None:
			final_in2out_matrix = new_data
		else:
			final_in2out_matrix = torch.cat((final_in2out_matrix, new_data), dim=0)

		# Interception
		temp_relatedness_matrix = final_value[:, in_now_start_index:new_in_start_index]

		if in_now_start_index == new_in_start_index:
			new_data = torch.zeros(final_value.shape[0], device=device)
		else:
			in_now_start_index = new_in_start_index
			# Find the maximum value by column
			new_data = torch.max(temp_relatedness_matrix, dim=1)[0]

		if out2in_matrix is None:
			out2in_matrix = new_data
		else:
			out2in_matrix = torch.cat((out2in_matrix, new_data), dim=0)

	# The similarity between the incoming problem and this piece of the problem
	final_in2out_matrix = final_in2out_matrix.view(-1, len_this_indices).permute(1, 0)

	out2in_matrix = out2in_matrix.view(-1, final_value.shape[0])

	# Multiply idf
	all_idf = []
	for temp_idf in this_idf:
		all_idf += temp_idf

	tensor_idf = torch.tensor(all_idf, device=device)
	out2in_matrix *= tensor_idf
	out2in_matrix = out2in_matrix.permute(1, 0)

	# Summation
	out_now_start_index = 0
	final_out2in_matrix = None

	for first_index in range(len_this_indices):
		# Interception
		new_out_start_index = out_now_start_index + out_each_question_len[first_index]
		if out_now_start_index == new_out_start_index:
			new_data = torch.zeros(len(in_idfs), device=device)
		else:
			temp_matrix = out2in_matrix[out_now_start_index:new_out_start_index, :]
			temp_idf = tensor_idf[out_now_start_index:new_out_start_index]

			out_now_start_index = new_out_start_index

			# Summation over columns
			new_data = temp_matrix.sum(dim=0)

			# Excluding the sum of idf
			new_data /= temp_idf.sum()

		if final_out2in_matrix is None:
			final_out2in_matrix = new_data
		else:
			final_out2in_matrix = torch.cat((final_out2in_matrix, new_data), dim=0)

	# The similarity between the incoming problem and this piece of the problem
	final_out2in_matrix = final_out2in_matrix.view(len_this_indices, -1)

	temp_matrix = (final_in2out_matrix + final_out2in_matrix) / 2.0

	# Update the new topk
	new_top_value, new_top_indices = temp_matrix.topk(top_k, dim=1)
	new_top_indices += offset

	final_top_value = new_top_value
	final_top_indices = new_top_indices

	torch.cuda.empty_cache()

	return final_top_value, final_top_indices


def updated_process_function(start_index, end_index, this_indices, this_idf, top_k):
	# Store results
	final_top_value = None
	final_top_indices = None

	# Create Model
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	model = CosineSimilarityTest().to(device)

	# The input word vector matrix
	out_emebddings = []
	# Length of the input sentence
	out_each_question_len = []

	for question_words in this_indices:
		# Give each question Store results
		temp_count = 0
		for word in question_words:
			out_emebddings.append(tensor_index2embedding(torch.tensor(word)))
			temp_count += 1

		# Length of each question
		out_each_question_len.append(temp_count)

	out_emebddings = torch.stack(out_emebddings, dim=0).to(device)

	# Step size of the loop
	IN_LOOP_STEP = 500

	in_now_index = start_index

	while in_now_index < end_index:

		# Get the necessary material
		in_questions = q_title_indices[in_now_index: min(in_now_index + IN_LOOP_STEP, end_index)]
		in_idfs = q_title_idf[in_now_index: min(in_now_index + IN_LOOP_STEP, end_index)]

		in_now_index += IN_LOOP_STEP

		# Number of words per question
		in_each_question_len = []
		in_accumulate_len = [0]

		# The set of vectors for all problems
		in_embeddings = []

		for question_words in in_questions:
			# Record the number of words in this question
			temp_count = 0
			for word in question_words:
				temp_count += 1
				in_embeddings.append(tensor_index2embedding(torch.tensor(word)))

			in_each_question_len.append(temp_count)
			in_accumulate_len.append(in_accumulate_len[-1] + temp_count)

		in_embeddings = torch.stack(in_embeddings, dim=0).to(device)

		# Calculate the similarity between words
		final_value = model(out_emebddings, in_embeddings)

		# -----------------
		offset = in_now_index - IN_LOOP_STEP
		len_this_indices = len(this_idf)

		out_now_start_index = 0
		in2out_matrix = None

		for first_index in range(len_this_indices):
			# Interception
			new_out_start_index = out_now_start_index + out_each_question_len[first_index]
			temp_relatedness_matrix = final_value[out_now_start_index:new_out_start_index]

			# If this question has no word
			if out_now_start_index == new_out_start_index:
				new_data = torch.zeros(final_value.shape[1], device=device)
			else:
				out_now_start_index = new_out_start_index

				# The maximum value by column, the length is equal to the number of columns
				new_data = torch.max(temp_relatedness_matrix, dim=0)[0]

			if in2out_matrix is None:
				in2out_matrix = new_data
			else:
				in2out_matrix = torch.cat((in2out_matrix, new_data), dim=0)

		in2out_matrix = in2out_matrix.view(len_this_indices, -1)

		# Multiply idf
		all_idf = []
		for temp_idf in in_idfs:
			all_idf += temp_idf

		tensor_idf = torch.tensor(all_idf, device=device)
		in2out_matrix *= tensor_idf

		# Under the summation process, the matrix of the next step is also calculated
		in_now_start_index = 0
		final_in2out_matrix = None

		out2in_matrix = None

		for first_index in range(len(in_idfs)):
			# Interception
			new_in_start_index = in_now_start_index + in_each_question_len[first_index]

			# If this question has no word，那么它和所有问题的相似度都是0
			if in_now_start_index == new_in_start_index:
				new_data = torch.zeros(len_this_indices, device=device)
			else:
				temp_matrix = in2out_matrix[:, in_now_start_index:new_in_start_index]
				temp_idf = tensor_idf[in_now_start_index:new_in_start_index]

				# Summation over rows
				new_data = temp_matrix.sum(dim=1)

				# Excluding the sum of idf
				new_data /= temp_idf.sum()

			if final_in2out_matrix is None:
				final_in2out_matrix = new_data
			else:
				final_in2out_matrix = torch.cat((final_in2out_matrix, new_data), dim=0)

			# Interception
			temp_relatedness_matrix = final_value[:, in_now_start_index:new_in_start_index]

			if in_now_start_index == new_in_start_index:
				new_data = torch.zeros(final_value.shape[0], device=device)
			else:
				in_now_start_index = new_in_start_index
				# Find the maximum value by column
				new_data = torch.max(temp_relatedness_matrix, dim=1)[0]

			if out2in_matrix is None:
				out2in_matrix = new_data
			else:
				out2in_matrix = torch.cat((out2in_matrix, new_data), dim=0)

		# The similarity between the incoming problem and this piece of the problem
		final_in2out_matrix = final_in2out_matrix.view(-1, len_this_indices).permute(1, 0)

		out2in_matrix = out2in_matrix.view(-1, final_value.shape[0])

		# Multiply idf
		all_idf = []
		for temp_idf in this_idf:
			all_idf += temp_idf

		tensor_idf = torch.tensor(all_idf, device=device)
		out2in_matrix *= tensor_idf
		out2in_matrix = out2in_matrix.permute(1, 0)

		# Summation
		out_now_start_index = 0
		final_out2in_matrix = None

		for first_index in range(len_this_indices):
			# Interception
			new_out_start_index = out_now_start_index + out_each_question_len[first_index]
			if out_now_start_index == new_out_start_index:
				new_data = torch.zeros(len(in_idfs), device=device)
			else:
				temp_matrix = out2in_matrix[out_now_start_index:new_out_start_index, :]
				temp_idf = tensor_idf[out_now_start_index:new_out_start_index]

				out_now_start_index = new_out_start_index

				# Summation over columns
				new_data = temp_matrix.sum(dim=0)

				# Excluding the sum of idf
				new_data /= temp_idf.sum()

			if final_out2in_matrix is None:
				final_out2in_matrix = new_data
			else:
				final_out2in_matrix = torch.cat((final_out2in_matrix, new_data), dim=0)

		# The similarity between the incoming problem and this piece of the problem
		final_out2in_matrix = final_out2in_matrix.view(len_this_indices, -1)

		temp_matrix = (final_in2out_matrix + final_out2in_matrix) / 2.0

		# Update the new topk
		new_top_value, new_top_indices = temp_matrix.topk(top_k, dim=1)
		new_top_indices += offset

		torch.cuda.empty_cache()

		# -------------------------------------------
		if final_top_value is None:
			final_top_value = new_top_value
			final_top_indices = new_top_indices
		else:
			# Splice it with the previous top_value
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

	# Read my_word2index
	my_word2index = np.load('./generated_file/my_word2index.npy', allow_pickle=True).item()
	my_index2word = {v: k for k, v in my_word2index.items()}

	# Read word_idf
	word_idf = []
	with open("./generated_file/word_idf", 'r') as f:
		for index, line in enumerate(f):
			temp_result = line.split()
			if len(temp_result) > 0:
				word_idf.append(eval(temp_result[0]))
			else:
				break

	# Read the trained glove word vector
	my_word2embedding = {}
	with open("/data/yuanhang/important_code/so_python/glove/vectors.txt", mode='r') as f:
		for line in f:
			line_list = line.split()
			word = line_list[0]
			embed = line_list[1:]
			embed = [float(num) for num in embed]
			my_word2embedding[word] = embed

	#  word -> index -> embeddding
	my_index2embedding = {}
	for word in my_word2index.keys():
		embedding = my_word2embedding.get(word)

		if embedding is not None:
			my_index2embedding[my_word2index[word]] = embedding
		else:
			my_index2embedding[my_word2index[word]] = [0.0] * 128

	# The embedding is processed a little bit to facilitate the calculation later
	temp_data = [my_index2embedding[ix] for ix in range(len(my_word2index))]
	tensor_index2embedding = nn.Embedding.from_pretrained(torch.tensor(temp_data))

	qa_info = load_from_disk("./generated_file/python_qa_info.dataset")
	q_ids = qa_info['q_id']
	print(qa_info)

	# Be prepared to find relevant questions
	q_csv_reader = pd.read_csv('generated_file/processed_python_q_id_content.csv', index_col='id')

	related_q_ids = []

	# To speed up the calculation, process the data in advance
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

	# Similarity matrix between problems
	question_num = len(q_ids)

	# Clean up log files
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

		# Get the index of each word in the title in the dictionary, remember to de-duplicate, idf
		this_indices = q_title_indices[now_index:(now_index + LOOP_STEP)]
		this_idf = q_title_idf[now_index:(now_index + LOOP_STEP)]

		# Recorded results
		print(time.asctime(time.localtime(time.time())))

		result = updated_process_function(0, question_num, this_indices, this_idf, 21)

		print(time.asctime(time.localtime(time.time())))

		torch.cuda.empty_cache()

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
