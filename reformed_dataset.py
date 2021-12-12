import datasets
import torch.utils.data


class TBA_word_bag_classify_dataset(torch.torch.utils.data.Dataset):
	def __init__(self, file_path, tokenizer, text_max_len, voc):
		self.dataset = datasets.load_from_disk(file_path)
		self.dataset.shuffle(seed=None)

		self.tokenizer = tokenizer
		self.text_max_len = text_max_len
		self.voc_size = len(voc)
		self.voc = voc

	def __len__(self):  # 返回整个数据集的大小
		return len(self.dataset)

	def __getitem__(self, index):
		item_dic = {"title": self.dataset['title'][index],
					"body": self.dataset['body'][index],
					"answer": self.dataset['answers'][index],
					'label': self.dataset['label'][index]}

		return item_dic

	def batchify_join_str(self, batch):
		all_titles = []
		all_bodies = []
		all_answers = []
		all_labels = []

		for sample in batch:
			all_titles.append(sample['title'])
			all_bodies.append(sample['body'])
			all_answers.append(sample['answer'])
			all_labels.append(sample['label'])

		# tokenize
		encoded_title = self.tokenizer(
			all_titles, padding=True, verbose=False, add_special_tokens=True,
			truncation=True, max_length=self.text_max_len, return_tensors='pt')
		encoded_body = self.tokenizer(
			all_bodies, padding=True, verbose=False, add_special_tokens=True,
			truncation=True, max_length=self.text_max_len, return_tensors='pt')
		encoded_a = self.tokenizer(
			all_answers, padding=True, verbose=False, add_special_tokens=True,
			truncation=True, max_length=self.text_max_len, return_tensors='pt')

		# get word bag
		word_bag = torch.zeros((len(all_titles), self.voc_size))

		for t_index, t in enumerate(all_titles):
			words = t.split()
			for w in words:
				w_index = self.voc.get(w)
				if w_index is not None:
					word_bag[t_index][w_index] = 1

			words = all_bodies[t_index].split()
			for w in words:
				w_index = self.voc.get(w)
				if w_index is not None:
					word_bag[t_index][w_index] = 1

		return_dict = {'title_input_ids': encoded_title['input_ids'],
					   'title_token_type_ids': encoded_title['token_type_ids'],
					   'title_attention_mask': encoded_title['attention_mask'],
					   'body_input_ids': encoded_body['input_ids'],
					   'body_token_type_ids': encoded_body['token_type_ids'],
					   'body_attention_mask': encoded_body['attention_mask'],
					   'a_input_ids': encoded_a['input_ids'],
					   'a_token_type_ids': encoded_a['token_type_ids'],
					   'a_attention_mask': encoded_a['attention_mask'],
					   'word_bag': word_bag,
					   'label': torch.tensor(all_labels)}

		return return_dict


class Test_TBA_word_bag_classify_dataset(torch.torch.utils.data.Dataset):
	def __init__(self, data, tokenizer, text_max_len, voc):
		# 读取一块数据
		self.dataset = data

		# 保存传入的参数
		self.tokenizer = tokenizer
		self.text_max_len = text_max_len
		self.voc_size = len(voc)
		self.voc = voc

		# 读取数据到内存
		self.all_titles = data['title']
		self.all_bodies = data['body']
		all_answers = data['answers']
		all_labels = data['label']

		# tokenize
		self.encoded_title = self.tokenizer(
			self.all_titles, padding=True, verbose=False, add_special_tokens=True,
			truncation=True, max_length=self.text_max_len, return_tensors='pt')
		self.encoded_body = self.tokenizer(
			self.all_bodies, padding=True, verbose=False, add_special_tokens=True,
			truncation=True, max_length=self.text_max_len, return_tensors='pt')
		self.encoded_a = self.tokenizer(
			all_answers, padding=True, verbose=False, add_special_tokens=True,
			truncation=True, max_length=self.text_max_len, return_tensors='pt')

		self.all_labels = all_labels

	def __len__(self):  # 返回整个数据集的大小
		return len(self.encoded_title['input_ids'])

	def __getitem__(self, index):

		word_bag = torch.zeros(self.voc_size)

		words = self.all_titles[index].split()
		for w in words:
			w_index = self.voc.get(w)
			if w_index is not None:
				word_bag[w_index] = 1

		words = self.all_bodies[index].split()
		for w in words:
			w_index = self.voc.get(w)
			if w_index is not None:
				word_bag[w_index] = 1

		item_dic = {'title_input_ids': self.encoded_title['input_ids'][index],
					'title_token_type_ids': self.encoded_title['token_type_ids'][index],
					'title_attention_mask': self.encoded_title['attention_mask'][index],
					'body_input_ids': self.encoded_body['input_ids'][index],
					'body_token_type_ids': self.encoded_body['token_type_ids'][index],
					'body_attention_mask': self.encoded_body['attention_mask'][index],
					'a_input_ids': self.encoded_a['input_ids'][index],
					'a_token_type_ids': self.encoded_a['token_type_ids'][index],
					'a_attention_mask': self.encoded_a['attention_mask'][index],
					'word_bag': word_bag,
					'label': torch.tensor(self.all_labels[index])}

		return item_dic


class TBAClassifyDataset(torch.torch.utils.data.Dataset):
	def __init__(self, data, tokenizer, text_max_len):
		# 读取一块数据
		self.dataset = data

		# 保存传入的参数
		self.tokenizer = tokenizer
		self.text_max_len = text_max_len

		# 读取数据到内存
		all_titles = data['title']
		all_bodies = data['body']
		all_answers = data['answers']
		all_labels = data['label']

		# tokenize
		self.encoded_title = self.tokenizer(
			all_titles, padding=True, verbose=False, add_special_tokens=True,
			truncation=True, max_length=self.text_max_len, return_tensors='pt')
		self.encoded_body = self.tokenizer(
			all_bodies, padding=True, verbose=False, add_special_tokens=True,
			truncation=True, max_length=self.text_max_len, return_tensors='pt')
		self.encoded_a = self.tokenizer(
			all_answers, padding=True, verbose=False, add_special_tokens=True,
			truncation=True, max_length=self.text_max_len, return_tensors='pt')

		self.all_labels = all_labels

	def __len__(self):  # 返回整个数据集的大小
		return len(self.encoded_title['input_ids'])

	def __getitem__(self, index):
		item_dic = {'title_input_ids': self.encoded_title['input_ids'][index],
					'title_token_type_ids': self.encoded_title['token_type_ids'][index],
					'title_attention_mask': self.encoded_title['attention_mask'][index],
					'body_input_ids': self.encoded_body['input_ids'][index],
					'body_token_type_ids': self.encoded_body['token_type_ids'][index],
					'body_attention_mask': self.encoded_body['attention_mask'][index],
					'a_input_ids': self.encoded_a['input_ids'][index],
					'a_token_type_ids': self.encoded_a['token_type_ids'][index],
					'a_attention_mask': self.encoded_a['attention_mask'][index],
					'label': torch.tensor(self.all_labels[index])}

		return item_dic


class MLMDataset(torch.torch.utils.data.Dataset):
	def __init__(self, data, tokenizer, text_max_len, memory_num, memory_start_index, ratio=1):
		# 读取一块数据
		self.dataset = data

		# 保存传入的参数
		self.tokenizer = tokenizer
		self.text_max_len = text_max_len

		# 获得memory合并的字符串
		memory_sequence = " "
		for i in range(memory_num):
			memory_sequence += '<MEM' + str(i) + '>' + " "

		# 读取数据到内存
		all_titles = data['title']
		all_bodies = data['body']
		all_answers = data['answers']

		len_title = len(all_titles)
		len_body = len(all_bodies)
		len_answer = len(all_answers)

		all_texts = []
		for s in all_titles[:int(ratio*len_title)]:
			all_texts.append((s, memory_sequence))
		for s in all_bodies[:int(ratio*len_body)]:
			all_texts.append((s, memory_sequence))
		for s in all_answers[:int(ratio*len_answer)]:
			all_texts.append((s, memory_sequence))

		# tokenize
		encoded_text = self.tokenizer(
			all_texts, padding=True, verbose=False, add_special_tokens=True,
			return_special_tokens_mask=True,
			truncation=True, max_length=self.text_max_len, return_tensors='pt')

		self.all_input_ids = encoded_text['input_ids']

		self.all_token_type_ids = encoded_text['token_type_ids']

		self.all_attention_mask = encoded_text['attention_mask']

		self.all_special_tokens_mask = encoded_text['special_tokens_mask']

		self.all_special_tokens_mask[self.all_input_ids >= memory_start_index] = 1

		# print("\ntext:\t", self.tokenizer.decode(self.all_input_ids[10]))
		# print("input_ids:\t", self.all_input_ids[10])
		# print("attention_mask:\t", self.all_attention_mask[10])
		# print("token_type_id:\t", self.all_token_type_ids[10])
		# print("special_tokens_mask:\t", self.all_special_tokens_mask[10])

	def __len__(self):  # 返回整个数据集的大小
		return len(self.all_input_ids)

	def __getitem__(self, index):
		item_dic = {'input_ids': self.all_input_ids[index],
					'token_type_ids': self.all_token_type_ids[index],
					'attention_mask': self.all_attention_mask[index],
					'special_tokens_mask': self.all_special_tokens_mask[index]}

		return item_dic

