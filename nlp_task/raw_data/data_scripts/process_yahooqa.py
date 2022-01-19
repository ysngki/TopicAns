import pickle

from datasets import Dataset

with open('../yahooqa/env.pkl', 'rb') as f:
	data = pickle.load(f)


def process_save_data(this_data, split_name):
	text_a = []
	idx = []
	whole_candidates = []
	whole_labels = []

	for index, this_text in enumerate(this_data):
		temp_candidates = []
		temp_labels = []

		this_candidates = this_data[this_text]

		best_answer = None

		for candidate, label in this_candidates:
			if label == 0:
				if candidate in temp_candidates:
					pass
				else:
					temp_candidates.append(candidate)
					temp_labels.append(0)
			elif label == 1:
				if best_answer is None:
					best_answer = candidate
				else:
					if best_answer == candidate:
						pass
					else:
						raise Exception("Duplicate best answer!")
			else:
				raise Exception("error!!!!!")

		assert not(best_answer is None)

		if len(temp_candidates) < 4:
			continue
		else:
			temp_candidates = temp_candidates[:4]
			temp_labels = temp_labels[:4]

		# best answer at last
		temp_candidates.append(best_answer)
		temp_labels.append(1)

		text_a.append(this_text)
		idx.append(index)
		whole_candidates.append(temp_candidates)
		whole_labels.append(temp_labels)

	data_dict = {'sentence_a': text_a, 'candidates': whole_candidates, 'idx': idx, 'label': whole_labels}
	whole_datasets = Dataset.from_dict(data_dict)
	whole_datasets.save_to_disk("/data/yuanhang/TaskSpecificMemory/nlp_task/dataset/string_" + split_name + "_yahooqa")
	print("Datasets is saved at /data/yuanhang/TaskSpecificMemory/nlp_task/dataset/string_" + split_name + "_yahooqa")
	print(whole_datasets)


process_save_data(data['train'], 'train')
process_save_data(data['test'], 'test')
process_save_data(data['dev'], 'val')
# dev_dict = {'sentence_a': text_a, 'candidates': whole_candidates, 'idx': dev_idx, 'label': whole_labels}
#
