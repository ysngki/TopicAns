import json
import os

from datasets import Dataset


# ---------------------------------------------------------------------------
# process train data for bi models, each record is ( context, its best response)
if not os.path.exists("/data/yuanhang/memory/nlp_task/dataset/string_bi_train_dstc7"):

	with open('../dstc7/ubuntu_train_subtask_1_augmented.json') as infile:
		data = json.load(infile)

	# ---------------------------------------------------------------------------
	# process training data for bi models
	bi_train_dict = {}
	bi_train_a = []
	bi_train_b = []
	bi_train_idx = []
	bi_train_label = []

	for idx, content in enumerate(data):
		# get context
		messages = content['messages-so-far']
		context = []

		for message in messages:
			text = message['speaker'].replace('_', ' ') + ': ' + message['utterance']
			context.append(text)

		joined_context = '\n'.join(context)

		# get best_answer
		best_answer = content['options-for-correct-answers'][0]['utterance'].strip()

		bi_train_a.append(joined_context)
		bi_train_b.append(best_answer)
		bi_train_idx.append(idx)
		bi_train_label.append(1)

	bi_train_dict = {'sentence_a': bi_train_a, 'sentence_b':bi_train_b, 'idx':bi_train_idx, 'label':bi_train_label}

	whole_datasets = Dataset.from_dict(bi_train_dict)
	whole_datasets.save_to_disk("/data/yuanhang/memory/nlp_task/dataset/string_bi_train_dstc7")
	print("Datasets is saved at /data/yuanhang/memory/nlp_task/dataset/string_bi_train_dstc7")

# ---------------------------------------------------------------------------
# process dev data, each record is ( context, 100 candidates (best at last))
if not os.path.exists("/data/yuanhang/memory/nlp_task/dataset/string_dev_dstc7"):

	with open('../dstc7/ubuntu_dev_subtask_1.json') as infile:
		data = json.load(infile)

	dev_dict = {}
	text_a = []
	whole_candidates = []
	dev_idx = []
	whole_labels = []

	for idx, content in enumerate(data):
		this_labels = []

		# get context
		messages = content['messages-so-far']
		context = []
		for message in messages:
			text = message['speaker'].replace('_', ' ') + ': ' + message['utterance']
			context.append(text)

		joined_context = '\n'.join(context)

		# get best_answer
		best_answer = content['options-for-correct-answers'][0]['utterance'].strip()
		correct_id = content['options-for-correct-answers'][0]['candidate-id']

		# get negs
		this_candidates = []
		negs = content['options-for-next']
		for neg in negs:
			if neg['candidate-id'] != correct_id:
				this_candidates.append(neg['utterance'].strip())
				this_labels.append(0)
		# put best answer at last
		this_candidates.append(best_answer)
		this_labels.append(1)
		assert len(this_candidates) == 100
		assert len(this_labels) == 100

		text_a.append(joined_context)
		whole_candidates.append(this_candidates)
		dev_idx.append(idx)
		whole_labels.append(this_labels)

	dev_dict = {'sentence_a': text_a, 'candidates':whole_candidates, 'idx':dev_idx, 'label':whole_labels}

	whole_datasets = Dataset.from_dict(dev_dict)
	whole_datasets.save_to_disk("/data/yuanhang/memory/nlp_task/dataset/string_dev_dstc7")
	print("Datasets is saved at /data/yuanhang/memory/nlp_task/dataset/string_dev_dstc7")

# ---------------------------------------------------------------------------
# process test data, each record is (context, 100 candidates (best at last))
if not os.path.exists("/data/yuanhang/memory/nlp_task/dataset/string_test_dstc7"):
	with open('../dstc7/ubuntu_test_subtask_1.json') as infile:
		data = json.load(infile)

	with open('../dstc7/ubuntu_responses_subtask_1.tsv') as infile:
		for line, content in zip(infile, data):
			_, candidate_id, utterance = line.split('\t')
			content['options-for-correct-answers'] = [{'candidate-id': candidate_id, 'utterance': utterance}]

	test_dict = {}
	text_a = []
	whole_candidates = []
	test_idx = []
	whole_labels = []

	for idx, content in enumerate(data):
		this_labels = []
		# get context
		messages = content['messages-so-far']
		context = []
		for message in messages:
			text = message['speaker'].replace('_', ' ') + ': ' + message['utterance']
			context.append(text)

		joined_context = '\n'.join(context)

		# get best_answer
		best_answer = content['options-for-correct-answers'][0]['utterance'].strip()
		correct_id = content['options-for-correct-answers'][0]['candidate-id']

		# get negs
		this_candidates = []
		negs = content['options-for-next']
		for neg in negs:
			if neg['candidate-id'] != correct_id:
				this_candidates.append(neg['utterance'].strip())
				this_labels.append(0)

		# put best answer at last
		this_candidates.append(best_answer)
		this_labels.append(1)
		assert len(this_candidates) == 100
		assert len(this_labels) == 100

		text_a.append(joined_context)
		whole_candidates.append(this_candidates)
		test_idx.append(idx)
		whole_labels.append(this_labels)

	# save
	test_dict = {'sentence_a': text_a, 'candidates': whole_candidates, 'idx': test_idx, 'label':whole_labels}

	whole_datasets = Dataset.from_dict(test_dict)
	whole_datasets.save_to_disk("/data/yuanhang/memory/nlp_task/dataset/string_test_dstc7")
	print("Datasets is saved at /data/yuanhang/memory/nlp_task/dataset/string_test_dstc7")

# ---------------------------------------------------------------------------
# process train data for cross model, each record is (context, 100 candidates (best at last))
if not os.path.exists("/data/yuanhang/memory/nlp_task/dataset/string_cross_train_dstc7"):

	with open('../dstc7/ubuntu_train_subtask_1.json') as infile:
		data = json.load(infile)

	# ---------------------------------------------------------------------------
	# process training data for bi models
	cross_train_dict = {}
	cross_train_a = []
	cross_train_candidates = []
	cross_train_idx = []
	cross_whole_labels = []

	for idx, content in enumerate(data):
		this_labels = []
		# get context
		messages = content['messages-so-far']
		context = []

		for message in messages:
			text = message['speaker'].replace('_', ' ') + ': ' + message['utterance']
			context.append(text)

		joined_context = '\n'.join(context)

		# get best_answer
		best_answer = content['options-for-correct-answers'][0]['utterance'].strip()
		correct_id = content['options-for-correct-answers'][0]['candidate-id']

		# get negative
		this_candidates = []
		negs = content['options-for-next']
		for neg in negs:
			if neg['candidate-id'] != correct_id:
				this_candidates.append(neg['utterance'].strip())
				this_labels.append(0)
		this_candidates.append(best_answer)
		this_labels.append(1)
		assert len(this_candidates) == 100
		assert len(this_labels) == 100

		cross_train_a.append(joined_context)
		cross_train_candidates.append(this_candidates)
		cross_train_idx.append(idx)
		cross_whole_labels.append(this_labels)

	cross_train_dict = {'sentence_a': cross_train_a, 'candidates': cross_train_candidates, 'idx':cross_train_idx, 'label':cross_whole_labels}

	# save
	whole_datasets = Dataset.from_dict(cross_train_dict)
	whole_datasets.save_to_disk("/data/yuanhang/memory/nlp_task/dataset/string_cross_train_dstc7")
	print("Datasets is saved at /data/yuanhang/memory/nlp_task/dataset/string_cross_train_dstc7")
