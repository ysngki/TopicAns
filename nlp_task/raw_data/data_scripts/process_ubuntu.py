import json
import os

from datasets import Dataset
import random


# ---------------------------------------------------------------------------
# process train data for bi models, each record is ( context, its best response)
if not os.path.exists("/data/yuanhang/TaskSpecificMemory/nlp_task/dataset/string_bi_train_ubuntu"):
	bi_train_dict = {}
	bi_train_a = []
	bi_train_b = []
	bi_train_idx = []
	bi_train_label = []

	with open("../ubuntu/train.txt", encoding='utf-8') as f:
		index = 0
		for line in f:
			split = line.strip('\n').split('\t')
			lbl, context, response = int(split[0]), split[1:-1], split[-1]

			# new (context, reponses) pairs
			if lbl == 1:
				joined_context = '\n'.join(context)
				bi_train_a.append(joined_context)
				bi_train_b.append(response)
				bi_train_idx.append(index)
				bi_train_label.append(1)
				index += 1

	bi_train_dict = {'sentence_a': bi_train_a, 'sentence_b': bi_train_b, 'idx': bi_train_idx, 'label': bi_train_label}

	whole_datasets = Dataset.from_dict(bi_train_dict)
	whole_datasets.save_to_disk("/data/yuanhang/TaskSpecificMemory/nlp_task/dataset/string_bi_train_ubuntu")
	print("Datasets is saved at /data/yuanhang/TaskSpecificMemory/nlp_task/dataset/string_bi_train_ubuntu")


# ---------------------------------------------------------------------------
# process dev data, each record is ( context, 100 candidates (best at last))
if not os.path.exists("/data/yuanhang/TaskSpecificMemory/nlp_task/dataset/string_dev_ubuntu"):
	dev_dict = {}

	text_a = []
	whole_candidates = []
	whole_labels = []
	dev_idx = []

	with open("../ubuntu/valid.txt", encoding='utf-8') as f:
		index = 0
		this_candidates = []
		this_labels = []
		best_answer = None
		wait_next_context = False

		for line in f:
			split = line.strip('\n').split('\t')
			lbl, context, response = int(split[0]), split[1:-1], split[-1]
			joined_context = '\n'.join(context)

			if lbl == 1:
				# get all candidates for previous context
				if wait_next_context:
					this_candidates.append(best_answer)
					this_labels.append(1)

					if len(this_candidates) != 10:
						raise Exception("Candidates len should be 10!")
					whole_candidates.append(this_candidates)
					whole_labels.append(this_labels)
					# reset
					this_candidates, this_labels = [], []

				# new context
				text_a.append(joined_context)
				dev_idx.append(index)

				best_answer = response
				index += 1

				wait_next_context = True
			else:
				this_candidates.append(response)
				this_labels.append(0)

				if joined_context != text_a[-1]:
					raise Exception("Context error!")

		if len(this_candidates) > 0:
			if len(this_candidates) != 9:
				raise Exception("Candidate error!")

			this_candidates.append(best_answer)
			this_labels.append(1)

			whole_candidates.append(this_candidates)
			whole_labels.append(this_labels)

	print(len(dev_idx), len(whole_candidates), len(text_a), len(whole_labels))

	dev_dict = {'sentence_a': text_a, 'candidates': whole_candidates, 'idx': dev_idx, 'label': whole_labels}

	whole_datasets = Dataset.from_dict(dev_dict)
	whole_datasets.save_to_disk("/data/yuanhang/TaskSpecificMemory/nlp_task/dataset/string_dev_ubuntu")
	print("Datasets is saved at /data/yuanhang/TaskSpecificMemory/nlp_task/dataset/string_dev_ubuntu")

# ---------------------------------------------------------------------------
# process test data, each record is (context, 100 candidates (best at last))
if not os.path.exists("/data/yuanhang/TaskSpecificMemory/nlp_task/dataset/string_test_ubuntu"):
	dev_dict = {}

	text_a = []
	whole_candidates = []
	whole_labels = []
	dev_idx = []

	with open("../ubuntu/test.txt", encoding='utf-8') as f:
		index = 0
		this_candidates = []
		this_labels = []
		best_answer = None
		wait_next_context = False

		for line in f:
			split = line.strip('\n').split('\t')
			lbl, context, response = int(split[0]), split[1:-1], split[-1]
			joined_context = '\n'.join(context)

			if lbl == 1:
				# get all candidates for previous context
				if wait_next_context:
					this_candidates.append(best_answer)
					this_labels.append(1)

					if len(this_candidates) != 10:
						raise Exception("Candidates len should be 10!")
					whole_candidates.append(this_candidates)
					whole_labels.append(this_labels)
					# reset
					this_candidates, this_labels = [], []

				# new context
				text_a.append(joined_context)
				dev_idx.append(index)

				best_answer = response
				index += 1

				wait_next_context = True
			else:
				this_candidates.append(response)
				this_labels.append(0)

				if joined_context != text_a[-1]:
					raise Exception("Context error!")

		if len(this_candidates) > 0:
			if len(this_candidates) != 9:
				raise Exception("Candidate error!")

			this_candidates.append(best_answer)
			this_labels.append(1)

			whole_candidates.append(this_candidates)
			whole_labels.append(this_labels)

	print(len(dev_idx), len(whole_candidates), len(text_a), len(whole_labels))

	dev_dict = {'sentence_a': text_a, 'candidates': whole_candidates, 'idx': dev_idx, 'label': whole_labels}

	whole_datasets = Dataset.from_dict(dev_dict)
	whole_datasets.save_to_disk("/data/yuanhang/TaskSpecificMemory/nlp_task/dataset/string_test_ubuntu")
	print("Datasets is saved at /data/yuanhang/TaskSpecificMemory/nlp_task/dataset/string_test_ubuntu")


# ---------------------------------------------------------------------------
# process training data for cross models
if not os.path.exists("/data/yuanhang/TaskSpecificMemory/nlp_task/dataset/string_cross_train_ubuntu"):

	cross_train_dict = {}
	cross_train_a = []
	cross_train_candidates = []
	cross_train_idx = []
	cross_whole_labels = []

	with open("../ubuntu/train.txt", encoding='utf-8') as f:
		index = 0
		this_candidates = []
		this_labels = []
		best_answer = None
		wait_next_context = False

		for line in f:
			split = line.strip('\n').split('\t')
			lbl, context, response = int(split[0]), split[1:-1], split[-1]
			joined_context = '\n'.join(context)

			if lbl == 1:
				# get all candidates for previous context
				if wait_next_context:
					this_candidates.append(best_answer)
					this_labels.append(1)

					if len(this_candidates) != 2:
						raise Exception("Candidates len should be 2!")
					cross_train_candidates.append(this_candidates)
					cross_whole_labels.append(this_labels)
					# reset
					this_candidates, this_labels = [], []

				# new context
				cross_train_a.append(joined_context)
				cross_train_idx.append(index)

				best_answer = response
				index += 1

				wait_next_context = True
			else:
				this_candidates.append(response)
				this_labels.append(0)

				if joined_context != cross_train_a[-1]:
					raise Exception("Context error!")

		if len(this_candidates) > 0:
			if len(this_candidates) != 1:
				raise Exception("Candidate error!")

			this_candidates.append(best_answer)
			this_labels.append(1)

			cross_train_candidates.append(this_candidates)
			cross_whole_labels.append(this_labels)

	print(len(cross_train_idx), len(cross_train_candidates), len(cross_train_a), len(cross_whole_labels), index)

	assert len(cross_train_a) == len(cross_train_candidates)

	# 采样足够的负样本（100个）
	query_num = len(cross_train_idx)
	new_train_candidates = []
	new_labels = []

	for i, old_candidates in enumerate(cross_train_candidates):
		negative_indices = [i]
		while i in negative_indices:
			negative_indices = random.sample(range(query_num), 31)
		negative_indices.append(i)

		this_candidates = []
		for n in negative_indices:
			this_candidates.append(cross_train_candidates[n][-1])

		this_labels = [0]*99 + [1]

		new_labels.append(this_labels)
		new_train_candidates.append(this_candidates)

	cross_train_dict = {'sentence_a': cross_train_a, 'candidates': new_train_candidates, 'idx': cross_train_idx,
						'label': new_labels}

	# save
	whole_datasets = Dataset.from_dict(cross_train_dict)
	whole_datasets.save_to_disk("/data/yuanhang/TaskSpecificMemory/nlp_task/dataset/string_cross_train_ubuntu")
	print("Datasets is saved at /data/yuanhang/TaskSpecificMemory/nlp_task/dataset/string_cross_train_ubuntu")
