from torch import device
import torch.utils.data
import re


class TBAClassifyDataset(torch.torch.utils.data.Dataset):
	def __init__(self, data, tokenizer, text_max_len, deformer=False):
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

		if deformer:
			title_max_len = 64
			body_max_len = text_max_len - title_max_len
			answer_max_len = text_max_len
		else:
			title_max_len = body_max_len = answer_max_len = text_max_len

		# tokenize
		self.encoded_title = self.tokenizer(
			all_titles, padding=True, verbose=False, add_special_tokens=True,
			truncation=True, max_length=title_max_len, return_tensors='pt')
		self.encoded_body = self.tokenizer(
			all_bodies, padding=True, verbose=False, add_special_tokens=True,
			truncation=True, max_length=body_max_len, return_tensors='pt')
		self.encoded_a = self.tokenizer(
			all_answers, padding=True, verbose=False, add_special_tokens=True,
			truncation=True, max_length=answer_max_len, return_tensors='pt')

		try:
			self.body_token_type_ids = self.encoded_body['token_type_ids']
			self.a_token_type_ids = self.encoded_a['token_type_ids']
			self.title_token_type_ids = self.encoded_title['token_type_ids']
		except KeyError:
			device = self.encoded_body['input_ids'].device

			self.body_token_type_ids = torch.zeros_like(self.encoded_body['input_ids'], device=device)
			self.a_token_type_ids = torch.zeros_like(self.encoded_a['input_ids'], device=device)
			self.title_token_type_ids = torch.zeros_like(self.encoded_title['input_ids'], device=device)

		self.all_labels = all_labels

	def __len__(self):  # 返回整个数据集的大小
		return len(self.encoded_title['input_ids'])

	def __getitem__(self, index):
		item_dic = {'title_input_ids': self.encoded_title['input_ids'][index],
					'title_token_type_ids': self.title_token_type_ids[index],
					'title_attention_mask': self.encoded_title['attention_mask'][index],
					'body_input_ids': self.encoded_body['input_ids'][index],
					'body_token_type_ids': self.body_token_type_ids[index],
					'body_attention_mask': self.encoded_body['attention_mask'][index],
					'a_input_ids': self.encoded_a['input_ids'][index],
					'a_token_type_ids': self.a_token_type_ids[index],
					'a_attention_mask': self.encoded_a['attention_mask'][index],
					'label': torch.tensor(self.all_labels[index])}

		return item_dic


class TBATopicClassifyDataset(torch.torch.utils.data.Dataset):
	def __init__(self, data, tokenizer, text_max_len, dictionary):
		self.dictionary = dictionary
		self.voc_size = len(self.dictionary)

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

		self.q_bows = []
		self.a_bows = []

		for d, a in zip(all_bodies, all_answers):
			new_d = ''.join([i for i in d if not i.isdigit()])
			split_d = new_d.split()
			d_bow = self.dictionary.doc2bow(split_d)

			new_a = ''.join([i for i in a if not i.isdigit()])
			split_a = new_a.split()
			a_bow = self.dictionary.doc2bow(split_a)

			self.q_bows.append(d_bow)
			self.a_bows.append(a_bow)

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

		try:
			self.body_token_type_ids = self.encoded_body['token_type_ids']
			self.a_token_type_ids = self.encoded_a['token_type_ids']
			self.title_token_type_ids = self.encoded_title['token_type_ids']
		except KeyError:
			device = self.encoded_body['input_ids'].device

			self.body_token_type_ids = torch.zeros_like(self.encoded_body['input_ids'], device=device)
			self.a_token_type_ids = torch.zeros_like(self.encoded_a['input_ids'], device=device)
			self.title_token_type_ids = torch.zeros_like(self.encoded_title['input_ids'], device=device)

		self.all_labels = all_labels

	def __len__(self):  # 返回整个数据集的大小
		return len(self.encoded_title['input_ids'])

	def __getitem__(self, index):

		this_q_bow = torch.zeros(self.voc_size)
		# bow = [[token_id1,token_id2,...],[freq1,freq2,...]]
		item = list(zip(*self.q_bows[index])) 
		this_q_bow[list(item[0])] = torch.tensor(list(item[1])).float()

		this_a_bow = torch.zeros(self.voc_size)
		# bow = [[token_id1,token_id2,...],[freq1,freq2,...]]
		item = list(zip(*self.a_bows[index])) 
		this_a_bow[list(item[0])] = torch.tensor(list(item[1])).float()
		
		item_dic = {'title_input_ids': self.encoded_title['input_ids'][index],
					'title_token_type_ids': self.title_token_type_ids[index],
					'title_attention_mask': self.encoded_title['attention_mask'][index],
					'body_input_ids': self.encoded_body['input_ids'][index],
					'body_token_type_ids': self.body_token_type_ids[index],
					'body_attention_mask': self.encoded_body['attention_mask'][index],
					'a_input_ids': self.encoded_a['input_ids'][index],
					'a_token_type_ids': self.a_token_type_ids[index],
					'a_attention_mask': self.encoded_a['attention_mask'][index],
					'q_bow': this_q_bow,
					'a_bow': this_a_bow,
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


class QAMemClassifyDataset(torch.torch.utils.data.Dataset):
	def __init__(self, data, tokenizer, text_max_len, memory_num):
		# 保存传入的参数
		self.tokenizer = tokenizer
		self.text_max_len = text_max_len

		# 获得memory合并的字符串
		q_memory_sequence = " "
		for i in range(memory_num):
			q_memory_sequence += '<QMEM' + str(i) + '>' + " "

		a_memory_sequence = " "
		for i in range(memory_num):
			a_memory_sequence += '<AMEM' + str(i) + '>' + " "

		# 读取数据到内存
		all_titles = data['title']
		all_bodies = data['body']
		all_answers = data['answers']
		all_labels = data['label']

		# process question
		new_questions = []
		for index, title in enumerate(all_titles):
			new_questions.append((title + " " + all_bodies[index], q_memory_sequence))

		# process answers
		new_answers = []
		for index, answer in enumerate(all_answers):
			new_answers.append((answer, a_memory_sequence))

		# tokenize
		self.encoded_questions = self.tokenizer(
			new_questions, padding=True, verbose=False, add_special_tokens=True,
			truncation=True, max_length=self.text_max_len, return_tensors='pt')

		self.encoded_answers = self.tokenizer(
			new_answers, padding=True, verbose=False, add_special_tokens=True,
			truncation=True, max_length=self.text_max_len, return_tensors='pt')

		self.all_labels = all_labels

	def __len__(self):  # 返回整个数据集的大小
		return len(self.encoded_questions['input_ids'])

	def __getitem__(self, index):
		item_dic = {'q_input_ids': self.encoded_questions['input_ids'][index],
					'q_token_type_ids': self.encoded_questions['token_type_ids'][index],
					'q_attention_mask': self.encoded_questions['attention_mask'][index],
					'a_input_ids': self.encoded_answers['input_ids'][index],
					'a_token_type_ids': self.encoded_answers['token_type_ids'][index],
					'a_attention_mask': self.encoded_answers['attention_mask'][index],
					'label': torch.tensor(self.all_labels[index])}

		return item_dic


class QAClassifyDataset(torch.torch.utils.data.Dataset):
	def __init__(self, data, tokenizer, text_max_len):
		# 保存传入的参数
		self.tokenizer = tokenizer
		self.text_max_len = text_max_len

		# 读取数据到内存
		all_titles = data['title']
		all_bodies = data['body']
		all_answers = data['answers']
		all_labels = data['label']

		# process question
		new_questions = []
		for index, title in enumerate(all_titles):
			new_questions.append(title + " " + all_bodies[index])

		# process answers
		new_answers = []
		for index, answer in enumerate(all_answers):
			new_answers.append(answer)

		# tokenize
		self.encoded_questions = self.tokenizer(
			new_questions, padding=True, verbose=False, add_special_tokens=True,
			truncation=True, max_length=self.text_max_len, return_tensors='pt')

		self.encoded_answers = self.tokenizer(
			new_answers, padding=True, verbose=False, add_special_tokens=True,
			truncation=True, max_length=self.text_max_len, return_tensors='pt')
		
		try:
			self.question_token_type_ids = self.encoded_questions['token_type_ids']
			self.a_token_type_ids = self.encoded_answers['token_type_ids']
		except KeyError:
			device = self.encoded_questions['input_ids'].device

			self.question_token_type_ids = torch.zeros_like(self.encoded_questions['input_ids'], device=device)
			self.a_token_type_ids = torch.zeros_like(self.encoded_answers['input_ids'], device=device)
			
		self.all_labels = all_labels

	def __len__(self):  # 返回整个数据集的大小
		return len(self.encoded_questions['input_ids'])

	def __getitem__(self, index):
		item_dic = {'q_input_ids': self.encoded_questions['input_ids'][index],
					'q_token_type_ids': self.question_token_type_ids[index],
					'q_attention_mask': self.encoded_questions['attention_mask'][index],
					'a_input_ids': self.encoded_answers['input_ids'][index],
					'a_token_type_ids': self.a_token_type_ids[index],
					'a_attention_mask': self.encoded_answers['attention_mask'][index],
					'label': torch.tensor(self.all_labels[index])}

		return item_dic


class CrossClassifyDataset(torch.torch.utils.data.Dataset):
	def __init__(self, data, tokenizer, text_max_len):
		# 保存传入的参数
		self.tokenizer = tokenizer
		self.text_max_len = text_max_len

		# 读取数据到内存
		all_titles = data['title']
		all_bodies = data['body']
		all_answers = data['answers']
		all_labels = data['label']

		# process question
		new_questions = []
		for index, title in enumerate(all_titles):
			new_questions.append(title + " " + all_bodies[index])

		# process answers
		all_texts = []
		for index, answer in enumerate(all_answers):
			all_texts.append((new_questions[index], answer))

		# tokenize
		self.encoded_texts = self.tokenizer(
			all_texts, padding=True, verbose=False, add_special_tokens=True,
			truncation=True, max_length=self.text_max_len, return_tensors='pt')

		self.all_labels = all_labels

	def __len__(self):  # 返回整个数据集的大小
		return len(self.encoded_texts['input_ids'])

	def __getitem__(self, index):
		item_dic = {'input_ids': self.encoded_texts['input_ids'][index],
					'token_type_ids': self.encoded_texts['token_type_ids'][index],
					'attention_mask': self.encoded_texts['attention_mask'][index],
					'label': torch.tensor(self.all_labels[index])}

		return item_dic


class VaeSignleTextDataset(torch.torch.utils.data.Dataset):
	def __init__(self, docs, bows, voc_size):
		self.docs = docs
		self.bows = bows
		self.voc_size = voc_size

	def __len__(self):  # 返回整个数据集的大小
		return len(self.bows)

	def __getitem__(self, index):
		this_bow = torch.zeros(self.voc_size)

		# bow = [[token_id1,token_id2,...],[freq1,freq2,...]]
		item = list(zip(*self.bows[index])) 

		this_bow[list(item[0])] = torch.tensor(list(item[1])).float()

		# text = self.docs[index]
		max_num = 1000
		this_bow[this_bow > max_num] = max_num

		return this_bow


class QATopicClassifyDataset(torch.torch.utils.data.Dataset):
	def __init__(self, data, tokenizer, text_max_len, dictionary):
		self.dictionary = dictionary
		self.voc_size = len(self.dictionary)

		# 读取一块数据
		self.dataset = data

		self.GOOD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')

		# 保存传入的参数
		self.tokenizer = tokenizer
		self.text_max_len = text_max_len

		# 读取数据到内存
		all_titles = data['title']
		all_bodies = data['body']
		all_answers = data['answers']
		all_labels = data['label']

		self.q_bows = []
		self.a_bows = []

		for t, d, a in zip(all_titles, all_bodies, all_answers):
			new_d = t + " " + d
			new_d = self.process_text(new_d)

			split_d = new_d.split()
			d_bow = self.dictionary.doc2bow(split_d)

			new_a = self.process_text(a)
			split_a = new_a.split()
			a_bow = self.dictionary.doc2bow(split_a)

			self.q_bows.append(d_bow)
			self.a_bows.append(a_bow)

		# process question
		new_questions = []
		for index, title in enumerate(all_titles):
			new_questions.append(title + " " + all_bodies[index])

		# tokenize
		self.encoded_questions = self.tokenizer(
			new_questions, padding=True, verbose=False, add_special_tokens=True,
			truncation=True, max_length=self.text_max_len, return_tensors='pt')
		self.encoded_answers = self.tokenizer(
			all_answers, padding=True, verbose=False, add_special_tokens=True,
			truncation=True, max_length=self.text_max_len, return_tensors='pt')

		try:
			self.question_token_type_ids = self.encoded_questions['token_type_ids']
			self.a_token_type_ids = self.encoded_answers['token_type_ids']
		except KeyError:
			device = self.encoded_questions['input_ids'].device

			self.question_token_type_ids = torch.zeros_like(self.encoded_questions['input_ids'], device=device)
			self.a_token_type_ids = torch.zeros_like(self.encoded_answers['input_ids'], device=device)

		self.all_labels = all_labels

	def process_text(self, text):
		new_text = ''.join([i for i in text if not i.isdigit()])

		new_text = re.sub(r"[\.\[\]()=]", " ", new_text)
		
		new_text = self.GOOD_SYMBOLS_RE.sub('', new_text)

		return new_text
   
	def __len__(self):  # 返回整个数据集的大小
		return len(self.encoded_questions['input_ids'])

	def __getitem__(self, index):

		this_q_bow = torch.zeros(self.voc_size)
		# bow = [[token_id1,token_id2,...],[freq1,freq2,...]]
		item = list(zip(*self.q_bows[index])) 
		if len(item) != 0:
			this_q_bow[list(item[0])] = torch.tensor(list(item[1])).float()

		# set max num
		max_num = 1000
		this_q_bow[this_q_bow > max_num] = max_num

		this_a_bow = torch.zeros(self.voc_size)
		# bow = [[token_id1,token_id2,...],[freq1,freq2,...]]
		item = list(zip(*self.a_bows[index])) 
		if len(item) != 0:
			this_a_bow[list(item[0])] = torch.tensor(list(item[1])).float()
		
		this_a_bow[this_a_bow > max_num] = max_num

		item_dic = {'q_input_ids': self.encoded_questions['input_ids'][index],
					'q_token_type_ids': self.question_token_type_ids[index],
					'q_attention_mask': self.encoded_questions['attention_mask'][index],
					'a_input_ids': self.encoded_answers['input_ids'][index],
					'a_token_type_ids': self.a_token_type_ids[index],
					'a_attention_mask': self.encoded_answers['attention_mask'][index],
					'q_bow': this_q_bow,
					'a_bow': this_a_bow,
					'label': torch.tensor(self.all_labels[index])}

		return item_dic