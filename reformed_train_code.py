# %%
import datasets
from nltk.corpus import stopwords
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
import torch.nn.functional
import os
import torch.utils.data
from transformers import AutoTokenizer, AutoConfig
import numpy as np
import sys

from reformed_model_lib import OneSupremeMemory, OneSupremeMemoryConfig, PureMemory, PureMemoryConfig, \
	VaeAttentionPlus, VaeAttentionConfig, VaeAttention, BasicConfig, BasicModel, InputMemorySelfAttConfig, \
	InputMemorySelfAtt, PureMemorySelfAttConfig, PureMemorySelfAtt
from reformed_dataset import Test_TBA_word_bag_classify_dataset, TBAClassifyDataset
from class_lib import VAEQuestion, VaeDataset, calculate_vae_loss


class TrainWholeModel:
	def __init__(self, args, config=None):

		# 读取一些参数并存起来
		self.__read_args_for_train(args)
		self.args = args

		# 设置gpu
		# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
		os.environ["TOKENIZERS_PARALLELISM"] = "false"
		os.environ["CUDA_VISIBLE_DEVICES"] = args.nvidia_number

		if torch.cuda.device_count() > 1:
			self.parallel_flag = True
		else:
			self.parallel_flag = False

		nvidia_number = len(args.nvidia_number.split(","))
		self.device_ids = [i for i in range(nvidia_number)]

		self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
		if not torch.cuda.is_available():
			raise Exception("No cuda available!")
		torch.cuda.set_device('cuda:0')

		# 读取tokenizer
		tokenizer_path = args.pretrained_bert_path.replace("/", "_")
		tokenizer_path = tokenizer_path.replace("\\", "_")

		if os.path.exists("./tokenizer/" + tokenizer_path):
			self.tokenizer = AutoTokenizer.from_pretrained("./tokenizer/" + tokenizer_path)
		else:
			print("first time use this tokenizer, downloading...")
			self.tokenizer = AutoTokenizer.from_pretrained(args.pretrained_bert_path)
			tokenizer_config = AutoConfig.from_pretrained(args.pretrained_bert_path)
			# self.tokenizer.add_tokens(['\n'], special_tokens=True)

			self.tokenizer.save_pretrained("./tokenizer/" + tokenizer_path)
			tokenizer_config.save_pretrained("./tokenizer/" + tokenizer_path)

		# 获得词汇表
		# add model
		if self.model_class in ['InputMemorySelfAtt', 'PureMemorySelfAtt', 'BasicModel']:
			pass
		else:
			STOPWORDS = set(stopwords.words('english'))

			self.voc = {}
			self.voc_size = 0
			threshold = 20

			with open("./" + self.dataset_name + "/glove/vocab.txt", "r") as f:
				for line in f:
					now_word = line.split()[0]

					if eval(line.split()[1]) < threshold:
						continue

					if now_word in STOPWORDS:
						continue

					if self.voc.get(now_word) is None:
						self.voc[now_word] = self.voc_size
						self.voc_size += 1

		# 获得模型配置
		if config is None:
			self.config = self.__read_args_for_config(args)
		else:
			self.config = config

		# 创建模型
		self.__create_model()

	def train(self, model_save_path, train_two_stage_flag):
		# 读取数据
		self.__model_to_device()

		# 用来判断是哪一阶段的训练
		final_stage_flag = not train_two_stage_flag
		# 如果读取了中间模型，也就说明直接训第二阶段
		if self.args.load_middle:
			final_stage_flag = True
		# final_stage_flag = True

		# 准备训练
		epochs = 100
		previous_best_r_1 = -1

		while True:
			# 如果要进行第二阶段训练，那需要先读取上一阶段最好的model
			if final_stage_flag and train_two_stage_flag:
				self.load_models(model_save_path + "_middle")
				# self.load_models("./model//value_no_vae_OneSupremeMemory_ask_ubuntu_middle")
				# self.__model_to_device()

			print("---------------------- begin train ----------------------")
			# 优化器
			optimizer, scheduler = self.__get_model_optimizer(final_stage_flag=final_stage_flag)

			# 设置早停变量
			if final_stage_flag:
				early_stop_threshold = 10
			else:
				early_stop_threshold = 5

			early_stop_count = 0

			for epoch in range(epochs):
				# get dataset from file
				train_data = datasets.load_from_disk("./" + self.dataset_name + "/string_train.dataset")
				train_data = train_data.shuffle(seed=None)

				test_data = datasets.load_from_disk("./" + self.dataset_name + "/string_val.dataset")
				test_data = test_data.shuffle(seed=None)

				# whether this epoch get a best model
				this_epoch_best = False

				# 打印一下
				print("*" * 20 + f" {epoch} " + "*" * 20)

				# 训练之前先看初始情况
				if epoch == -1:
					print("-" * 30 + "initial_test" + "-" * 30)
					val_dataloader = self.__get_dataloader(data=test_data, batch_size=self.val_batch_size,
														   split_index=0, split_num=1)
					val_loss, val_acc = self.classify_validate_model(val_dataloader)
					print(val_loss, val_acc)
					self.ranking()
					print("-" * 30 + "initial_test_end" + "-" * 30, end="\n\n")

				# 开始训练
				train_loss = 0.0
				# 计算训练集的R@1
				shoot_num = 0
				hit_num = 0

				now_batch_num = 0

				# 逐块训练
				for split_index in range(self.dataset_split_num):
					train_dataloader = self.__get_dataloader(data=train_data, batch_size=self.train_batch_size,
															 split_index=split_index, split_num=self.dataset_split_num)
					# 训练数据
					for index, batch in enumerate(train_dataloader):
						# add model
						# 迈出一步
						if self.model_class == "CrossEncoder":
							step_loss, step_shoot_num, step_hit_num = self.__train_step_for_cross(batch=batch,
																								  optimizer=optimizer)
						elif self.model_class in ['BasicModel', 'InputMemorySelfAtt', 'PureMemorySelfAtt']:
							step_loss, step_shoot_num, step_hit_num = \
								self.__train_step_for_bi_no_vae(batch=batch, optimizer=optimizer)
						else:
							step_loss, step_shoot_num, step_hit_num = \
								self.__train_step_for_bi(batch=batch, optimizer=optimizer)

						# 更新一下信息
						train_loss += step_loss.item()

						shoot_num += step_shoot_num
						hit_num += step_hit_num
						now_batch_num += 1

						# 要不要打印一下呢
						print_interval = (len(train_dataloader)*self.dataset_split_num) // self.print_num_each_epoch
						if (now_batch_num % print_interval) == 0:
							print(
								f"epoch:{epoch + 1}, "
								f"average loss = {train_loss / shoot_num},"
								f"Acc={hit_num}/{shoot_num}={hit_num * 100 / shoot_num}%")
						sys.stdout.flush()

						# 要不要是时候评测一下呢，一个epoch评测 val_num_each_epoch 次
						epoch_val_split_num = self.val_num_each_epoch + 1
						val_interval = (len(train_dataloader)*self.dataset_split_num) // epoch_val_split_num

						if now_batch_num % val_interval == 0 and now_batch_num / val_interval < epoch_val_split_num:
							# 获得评测结果
							val_dataloader = self.__get_dataloader(data=test_data, batch_size=self.val_batch_size,
																   split_index=0, split_num=1)

							val_loss, val_acc = self.classify_validate_model(val_dataloader)
							r_1 = self.ranking()
							print(
								f"{epoch + 1} epoch middle eval: " + "*" * 30 +
								f"\nval_loss:{val_loss}\tval_acc{val_acc}\tR@1:{r_1}")

							# 模型是否比之前优秀
							if r_1 > previous_best_r_1:
								previous_best_r_1 = r_1
								this_epoch_best = True

								# 保存模型
								postfix = ""
								if not final_stage_flag:
									postfix = "_middle"

								self.save_model(model_save_path=model_save_path, postfix=postfix)

				# scheduler.step()

				# 这个epoch结束
				# 重新在测试数据上完整测试一遍
				val_dataloader = self.__get_dataloader(data=test_data, batch_size=self.val_batch_size,
													   split_index=0, split_num=1)

				val_loss, val_acc = self.classify_validate_model(val_dataloader)

				r_1 = self.ranking()
				print(f"{epoch + 1} epoch end eval: " + "*" * 30 + f"\nval_loss:{val_loss}\tval_acc{val_acc}\tR@1:{r_1}")

				# 模型是否比之前优秀
				if r_1 > previous_best_r_1:
					previous_best_r_1 = r_1
					this_epoch_best = True

					# 保存模型
					postfix = ""
					if not final_stage_flag:
						postfix = "_middle"

					self.save_model(model_save_path=model_save_path, postfix=postfix)

				# 是否早停
				if this_epoch_best:
					early_stop_count = 0
				else:
					early_stop_count += 1

					if early_stop_count == early_stop_threshold:
						print("early stop!")
						break

				sys.stdout.flush()

			# 如果只是第一阶段的训练，那么还要继续训练
			if not final_stage_flag:
				final_stage_flag = True
			else:
				break

	def train_vae(self):

		print("*" * 20 + "train vae begin" + "*" * 20)

		vae_model = VAEQuestion(input_dim=self.voc_size, latent_dim=self.latent_dim).to(self.device)

		vae_model.train()
		optimizer = torch.optim.Adam([
			{'params': vae_model.parameters()}
		], lr=1e-4)

		# 开始训练
		SPLIT_NUM = 10
		previous_loss = 999999
		for epoch in range(10):
			all_lose = 0.0

			# 因为是无监督，所以都用上
			train_datasets = datasets.load_from_disk("./" + self.dataset_name + "/string_train.dataset")
			train_datasets = train_datasets.shuffle(seed=None)
			train_data_len = len(train_datasets)

			test_datasets = datasets.load_from_disk("./" + self.dataset_name + "/string_val.dataset")
			test_datasets = test_datasets.shuffle(seed=None)
			test_data_len = len(test_datasets)

			for block_index in range(0, SPLIT_NUM):

				# 先划分好
				if block_index == SPLIT_NUM - 1:
					temp_train_dataset = train_datasets[int(block_index * train_data_len / SPLIT_NUM):]
				else:
					temp_train_dataset = train_datasets[int(block_index * train_data_len / SPLIT_NUM):
														int((block_index + 1) * train_data_len / SPLIT_NUM)]

				if block_index == SPLIT_NUM - 1:
					temp_test_dataset = test_datasets[int(block_index * test_data_len / SPLIT_NUM):]
				else:
					temp_test_dataset = test_datasets[int(block_index * test_data_len / SPLIT_NUM):
													  int((block_index + 1) * test_data_len / SPLIT_NUM)]

				# 计算词袋
				word_bag = torch.zeros((len(temp_train_dataset['title']) + len(temp_test_dataset['title']),
										self.voc_size))

				for t_index, t in enumerate(temp_train_dataset['title']):
					words = t.split()
					for w in words:
						w_index = self.voc.get(w)
						if w_index is not None:
							word_bag[t_index][w_index] = 1

					words = temp_train_dataset['body'][t_index].split()
					for w in words:
						w_index = self.voc.get(w)
						if w_index is not None:
							word_bag[t_index][w_index] = 1

				for t_index, t in enumerate(temp_test_dataset['title']):
					words = t.split()
					for w in words:
						w_index = self.voc.get(w)
						if w_index is not None:
							word_bag[t_index + len(temp_train_dataset)][w_index] = 1

					words = temp_test_dataset['body'][t_index].split()
					for w in words:
						w_index = self.voc.get(w)
						if w_index is not None:
							word_bag[t_index + len(temp_train_dataset)][w_index] = 1

				# # 开始训练
				my_dataset = VaeDataset(word_bag)

				# 生成dataloader
				BATCH_SIZE = 64

				vae_loader = torch.utils.data.DataLoader(dataset=my_dataset, batch_size=BATCH_SIZE,
														 shuffle=True, num_workers=5, drop_last=True)

				train_loss = 0.0

				# 训练数据
				for index, batch in enumerate(vae_loader):
					# 读取数据
					word_bag = (batch['word_bag']).to(self.device)

					# 优化器置零
					optimizer.zero_grad()
					# 得到模型的结果
					reconstructed_input, mean, log_var, latent_v = vae_model(word_bag, out_latent_flag=True)
					vae_loss = calculate_vae_loss(word_bag, reconstructed_input, mean, log_var)

					# 计算损失
					train_loss += vae_loss.item()

					# 误差反向传播
					vae_loss.backward()
					# 更新模型参数
					optimizer.step()

				# 这个block结束后打印一次
				all_lose += train_loss
				print(
					f"epoch:{epoch + 1}, block:{block_index + 1}/{SPLIT_NUM}, block loss = {train_loss}")
				sys.stdout.flush()

			# 这个epoch结束
			print(f"epoch:{epoch + 1} all loss:{all_lose}", end="\n\n")

			# 判断是否终止训练
			if all_lose < previous_loss - previous_loss * 0.1:
				previous_loss = all_lose
				pass
			else:
				break

		vae_model.eval()
		torch.save(vae_model.state_dict(), "./model/pretrained_all_vae_" + self.dataset_name)
		print("*" * 20 + "train vae end" + "*" * 20)

	def load_vae_model(self):
		self.model.vae_model.load_state_dict(torch.load("./model/pretrained_all_vae_" + self.dataset_name))

	# classify
	def classify_validate_model(self, dataloader):
		self.model.eval()

		label_target_num = [0, 0, 0, 0]
		label_shoot_num = [0, 0, 0, 0]
		label_hit_num = [0, 0, 0, 0]

		# 开始评测
		with torch.no_grad():
			val_loss = 0.0
			cross_entropy_function = nn.CrossEntropyLoss()

			# 生成dataloader
			classify_dataloader = dataloader

			print(f"------- begin val {self.val_batch_size * len(classify_dataloader)} data--------")

			# 计算正确率
			shoot_num = 0
			hit_num = 0
			for index, batch in enumerate(classify_dataloader):
				# 读取数据
				# add model
				if self.model_class in ["OneSupremeMemory", 'PureMemory', 'VaeAttention', 'VaeAttentionPlus']:
					logits = self.__val_step_for_bi(batch)
				elif self.model_class in ['BasicModel', 'InputMemorySelfAtt', 'PureMemorySelfAtt']:
					logits = self.__val_step_for_bi_no_vae(batch)
				else:
					raise Exception("Val step is not supported for this model class!")

				qa_labels = (batch['label']).to(self.device)

				loss = cross_entropy_function(logits, qa_labels)
				val_loss += loss.item()

				# 统计命中率
				shoot_num += len(qa_labels)
				for i in range(0, 4):
					label_target_num[i] += (qa_labels == i).sum().item()

				batch_hit_num = 0
				_, row_max_indices = logits.topk(k=1, dim=-1)

				for i, max_index in enumerate(row_max_indices):
					inner_index = max_index[0]
					label_shoot_num[inner_index] += 1
					if inner_index == qa_labels[i]:
						label_hit_num[inner_index] += 1
						batch_hit_num += 1
				hit_num += batch_hit_num

			accuracy = self.__print_ranking_result(label_hit_num=label_hit_num, label_shoot_num=label_shoot_num,
												   label_target_num=label_target_num)

			self.model.train()

		return val_loss, accuracy

	def save_model(self, model_save_path, postfix=""):
		self.model.eval()

		if self.parallel_flag:
			torch.save(self.model.module.state_dict(), model_save_path + postfix)
		else:
			torch.save(self.model.state_dict(), model_save_path + postfix)

		print("!"*60)
		print(f"model is saved at {model_save_path + postfix}")
		print("!"*60)

		self.model.train()

	def load_models(self, load_model_path):
		if load_model_path is None:
			print("you should offer model paths!")
		print("load saved model from", load_model_path)
		# 这里可能会因为并行之类的出错
		self.model.load_state_dict(torch.load(load_model_path))
		print("saved model loaded!")

	def ranking(self):
		print("--------------------- begin ranking -----------------------")
		self.model.eval()

		# 稍加处理一下数据，把数据都存在元祖里
		data_from_path = "./" + self.dataset_name + "/eva.dataset"
		evaluation_data = datasets.load_from_disk(data_from_path)

		# 随机一下
		evaluation_data = evaluation_data.shuffle()

		# 汇总下数据
		evaluation_qa_pairs = []

		evaluation_title = evaluation_data['title']
		evaluation_body = evaluation_data['body']
		evaluation_answers = evaluation_data['answers']

		for row_index in range(len(evaluation_title)):
			evaluation_qa_pairs.append((evaluation_title[row_index], evaluation_body[row_index],
										evaluation_answers[row_index]))

		print(f"all {len(evaluation_qa_pairs)} qa pairs!")

		# 开始逐条排序
		model_ranking = []

		# 一点点处理数据
		now_pair_index = 0
		PAIR_STEP = 2000

		while now_pair_index < len(evaluation_qa_pairs):
			# 取一定数量的数据
			questions = []
			bodies = []
			answers = []

			for data in evaluation_qa_pairs[now_pair_index: now_pair_index + PAIR_STEP]:
				temp_question = data[0]
				body = data[1]
				candidate_answers = data[2]

				for t_a in candidate_answers[1:self.ranking_candidate_num]:
					bodies.append(body)
					questions.append(temp_question)
					answers.append(t_a)

				# 把最佳答案塞到最后
				bodies.append(body)
				questions.append(temp_question)
				answers.append(candidate_answers[0])

			# tokenize
			encoded_a = self.tokenizer(
				answers, padding=True, verbose=False, add_special_tokens=True,
				truncation=True, max_length=self.text_max_len - self.memory_num, return_tensors='pt')

			encoded_q = self.tokenizer(
				questions, padding=True, verbose=False, add_special_tokens=True,
				truncation=True, max_length=self.text_max_len - self.memory_num, return_tensors='pt')

			encoded_b = self.tokenizer(
				bodies, padding=True, verbose=False, add_special_tokens=True,
				truncation=True, max_length=self.text_max_len - self.memory_num, return_tensors='pt')

			# 获得title的词袋
			# add model
			if self.model_class in ['BasicModel', 'InputMemorySelfAtt', 'PureMemorySelfAtt']:
				pass
			else:
				word_bag = torch.zeros((len(questions), self.voc_size))

				for t_index, t in enumerate(questions):
					words = t.split()
					for w in words:
						w_index = self.voc.get(w)
						if w_index is not None:
							word_bag[t_index][w_index] = 1

					words = bodies[t_index].split()
					for w in words:
						w_index = self.voc.get(w)
						if w_index is not None:
							word_bag[t_index][w_index] = 1

			# 检查数据数量是否正确，length是问题数
			length = len(evaluation_qa_pairs[now_pair_index: now_pair_index + PAIR_STEP])

			if len(encoded_b['input_ids']) != length * self.ranking_candidate_num:
				raise Exception("encode while ranking no possible!")

			# 开始按照更小的批次进行训练，也就是每次计算step*candidate_answer_num条数据
			now_index = 0
			step = 40

			while now_index < length:
				q_input_ids = encoded_q['input_ids'][
							  now_index * self.ranking_candidate_num:
							  (now_index + step) * self.ranking_candidate_num].to(self.device)
				q_token_type_ids = encoded_q['token_type_ids'][
								   now_index * self.ranking_candidate_num:
								   (now_index + step) * self.ranking_candidate_num].to(self.device)
				q_attention_mask = encoded_q['attention_mask'][
								   now_index * self.ranking_candidate_num:
								   (now_index + step) * self.ranking_candidate_num].to(self.device)

				a_input_ids = encoded_a['input_ids'][
							  now_index * self.ranking_candidate_num:
							  (now_index + step) * self.ranking_candidate_num].to(self.device)
				a_token_type_ids = encoded_a['token_type_ids'][
								   now_index * self.ranking_candidate_num:
								   (now_index + step) * self.ranking_candidate_num].to(self.device)
				a_attention_mask = encoded_a['attention_mask'][
								   now_index * self.ranking_candidate_num:
								   (now_index + step) * self.ranking_candidate_num].to(self.device)

				b_input_ids = encoded_b['input_ids'][
							  now_index * self.ranking_candidate_num:
							  (now_index + step) * self.ranking_candidate_num].to(self.device)
				b_token_type_ids = encoded_b['token_type_ids'][
								   now_index * self.ranking_candidate_num:
								   (now_index + step) * self.ranking_candidate_num].to(self.device)
				b_attention_mask = encoded_b['attention_mask'][
								   now_index * self.ranking_candidate_num:
								   (now_index + step) * self.ranking_candidate_num].to(self.device)

				# add model
				if self.model_class in ['BasicModel', 'InputMemorySelfAtt', 'PureMemorySelfAtt']:
					pass
				else:
					t_word_bag = word_bag[now_index * self.ranking_candidate_num:
										  (now_index + step) * self.ranking_candidate_num].to(self.device)

				now_index += step

				with torch.no_grad():
					model = self.model
					if self.parallel_flag:
						model = self.model.module

					# add model
					if self.model_class in ['BasicModel', 'InputMemorySelfAtt', 'PureMemorySelfAtt']:
						# shape = (q_num*candidate_answer_num, 4)
						logits = model(
							q_input_ids=q_input_ids, q_token_type_ids=q_token_type_ids,
							q_attention_mask=q_attention_mask,
							a_input_ids=a_input_ids, a_token_type_ids=a_token_type_ids,
							a_attention_mask=a_attention_mask,
							b_input_ids=b_input_ids, b_token_type_ids=b_token_type_ids,
							b_attention_mask=b_attention_mask)
					else:
						# shape = (q_num*candidate_answer_num, 4)
						logits, _ = model(
							q_input_ids=q_input_ids, q_token_type_ids=q_token_type_ids,
							q_attention_mask=q_attention_mask,
							a_input_ids=a_input_ids, a_token_type_ids=a_token_type_ids,
							a_attention_mask=a_attention_mask,
							b_input_ids=b_input_ids, b_token_type_ids=b_token_type_ids,
							b_attention_mask=b_attention_mask, word_bag=t_word_bag)

					logits = logits.view(-1, self.ranking_candidate_num, 4)
					logits = nn.functional.softmax(logits, dim=-1)

					sorted_index_score = []
					for q_index, q_item in enumerate(logits):
						temp_index_score = []
						for inner_index, inner_item in enumerate(q_item):
							score = inner_item[0] * (-2) + inner_item[1] * (-1) + inner_item[2] * 1 + inner_item[3] * 2
							temp_index_score.append((inner_index, score.item()))

						stl = sorted(temp_index_score, key=lambda x: x[1], reverse=True)
						sorted_index_score.append(stl)

					model_ranking += [([y[0] for y in x].index(self.ranking_candidate_num - 1) + 1) for x in
									  sorted_index_score]

			now_pair_index += PAIR_STEP

			# 每排好10000个问题，打印下当前的结果
			if (now_pair_index % 10000) == 0:
				print(f"now processed: {now_pair_index}/{len(evaluation_qa_pairs)}")

				metric_list = [i for i in range(1, self.ranking_candidate_num + 1)]
				for k in metric_list:
					print(
						"DCG@%4d: %.3f | Hits@%4d: %.3f" % (k, self.dcg_score(model_ranking, k),
															k, self.hits_count(model_ranking, k)))
				sys.stdout.flush()

		print()

		# 输出最后的评估结果
		metric_list = [i for i in range(1, self.ranking_candidate_num + 1)]
		for k in metric_list:
			print(
				"DCG@%4d: %.3f | Hits@%4d: %.3f" % (k, self.dcg_score(model_ranking, k),
													k, self.hits_count(model_ranking, k)))
		print("---------------------- end ranking ------------------------")
		self.model.train()

		return self.dcg_score(model_ranking, 1)

	def __create_model(self):
		print("---------------------- create model ----------------------")
		# 创建自己的model
		# add model
		if self.model_class == 'OneSupremeMemory':
			self.model = OneSupremeMemory(config=self.config)
		elif self.model_class == 'PureMemory':
			self.model = PureMemory(config=self.config)
		elif self.model_class == 'VaeAttention':
			self.model = VaeAttention(config=self.config)
		elif self.model_class == 'VaeAttentionPlus':
			self.model = VaeAttentionPlus(config=self.config)
		elif self.model_class == 'BasicModel':
			self.model = BasicModel(config=self.config)
		elif self.model_class == 'InputMemorySelfAtt':
			self.model = InputMemorySelfAtt(config=self.config)
		elif self.model_class == 'PureMemorySelfAtt':
			self.model = PureMemorySelfAtt(config=self.config)
		else:
			raise Exception("This model class is not supported for creating!!")

		# 要不要加载现成的模型
		if self.load_model_flag:
			load_model_path = self.args.save_model_dict + "/" + self.args.model_save_prefix + \
							  self.args.model_class + "_" + self.args.dataset_name
			if self.args.load_middle:
				load_model_path += "_middle"
			self.load_models(load_model_path)

		# 训练准备
		# self.__model_to_device()
		print("--------------------- model  created ---------------------")

	# 读取命令行传入的参数
	def __read_args_for_train(self, args):
		self.val_num_each_epoch = args.val_num_each_epoch
		self.val_candidate_num = args.val_candidate_num
		self.val_batch_size = args.val_batch_size
		self.text_max_len = args.text_max_len
		self.dataset_name = args.dataset_name
		self.model_class = args.model_class
		self.memory_num = args.memory_num
		self.train_candidate_num = args.train_candidate_num
		self.print_num_each_epoch = args.print_num_each_epoch
		self.load_model_flag = args.load_model
		self.load_model_dict = args.load_model_dict
		self.dataset_split_num = args.dataset_split_num
		self.train_batch_size = args.train_batch_size
		self.ranking_candidate_num = args.ranking_candidate_num
		self.latent_dim = args.latent_dim

	# 读取命令行传入的有关config的参数
	def __read_args_for_config(self, args):
		if args.pretrained_bert_path in ['prajjwal1/bert-small', 'google/bert_uncased_L-6_H-512_A-8']:
			word_embedding_len = 512
			sentence_embedding_len = 512
		elif args.pretrained_bert_path == 'bert-base-uncased':
			word_embedding_len = 768
			sentence_embedding_len = 768
		else:
			raise Exception("word_embedding_len, sentence_embedding_len is needed!")

		# add model
		if self.model_class == "OneSupremeMemory":
			config = OneSupremeMemoryConfig(len(self.tokenizer),
											pretrained_bert_path=args.pretrained_bert_path,
											latent_dim=args.latent_dim,
											num_labels=args.label_num,
											voc_size=self.voc_size,
											word_embedding_len=word_embedding_len,
											sentence_embedding_len=sentence_embedding_len,
											memory_num=args.memory_num)
		elif self.model_class == 'PureMemory':
			config = PureMemoryConfig(len(self.tokenizer),
									  pretrained_bert_path=args.pretrained_bert_path,
									  latent_dim=args.latent_dim,
									  num_labels=args.label_num,
									  voc_size=self.voc_size,
									  word_embedding_len=word_embedding_len,
									  sentence_embedding_len=sentence_embedding_len,
									  memory_num=args.memory_num)
		elif self.model_class == 'VaeAttentionPlus':
			config = VaeAttentionConfig(len(self.tokenizer),
										pretrained_bert_path=args.pretrained_bert_path,
										latent_dim=args.latent_dim,
										num_labels=args.label_num,
										voc_size=self.voc_size,
										word_embedding_len=word_embedding_len,
										sentence_embedding_len=sentence_embedding_len)
		elif self.model_class == 'VaeAttention':
			config = VaeAttentionConfig(len(self.tokenizer),
										pretrained_bert_path=args.pretrained_bert_path,
										latent_dim=args.latent_dim,
										num_labels=args.label_num,
										voc_size=self.voc_size,
										word_embedding_len=word_embedding_len,
										sentence_embedding_len=sentence_embedding_len)
		elif self.model_class == 'BasicModel':
			config = BasicConfig(len(self.tokenizer),
								 pretrained_bert_path=args.pretrained_bert_path,
								 num_labels=args.label_num,
								 word_embedding_len=word_embedding_len,
								 sentence_embedding_len=sentence_embedding_len)
		elif self.model_class == 'InputMemorySelfAtt':
			config = InputMemorySelfAttConfig(len(self.tokenizer),
											  pretrained_bert_path=args.pretrained_bert_path,
											  num_labels=args.label_num,
											  word_embedding_len=word_embedding_len,
											  sentence_embedding_len=sentence_embedding_len,
											  memory_num=args.memory_num)
		elif self.model_class == 'PureMemorySelfAtt':
			config = PureMemorySelfAttConfig(len(self.tokenizer),
											 pretrained_bert_path=args.pretrained_bert_path,
											 num_labels=args.label_num,
											 word_embedding_len=word_embedding_len,
											 sentence_embedding_len=sentence_embedding_len,
											 memory_num=args.memory_num,
											 hop_num=args.hop_num)
		else:
			raise Exception("No config for this class!")

		return config

	# 根据model_class获取optimizer
	def __get_model_optimizer(self, final_stage_flag):
		if self.parallel_flag:
			model = self.model.module
		else:
			model = self.model

		# add model
		# 获得模型的训练参数和对应的学习率
		if self.model_class == 'OneSupremeMemory':
			parameters_dict_list = [
				{'params': model.bert_model.parameters(), 'lr': 5e-5},
				# 这几个一样
				{'params': model.key_layer.parameters(), 'lr': 1e-4},
				{'params': model.value_layer.parameters(), 'lr': 1e-4},
				{'params': model.memory_for_question, 'lr': 0.3},
				{'params': model.memory_for_answer, 'lr': 0.3},
				# {'params': model.memory_for_question, 'lr': 5e-5},
				# {'params': model.memory_for_answer, 'lr': 5e-5},
				{'params': model.vae_model.parameters(), 'lr': 1e-4},
				{'params': model.classifier.parameters(), 'lr': 1e-4},
			]
		elif self.model_class == 'PureMemory':
			parameters_dict_list = [
				# 这几个一样
				{'params': model.bert_model.parameters(), 'lr': 5e-5},
				{'params': model.vae_model.parameters(), 'lr': 5e-5},
				# 这几个一样 
				{'params': model.key_layer.parameters(), 'lr': 1e-4},
				{'params': model.value_layer.parameters(), 'lr': 1e-4},
				{'params': model.query_for_question, 'lr': 1e-4},
				{'params': model.memory_for_question, 'lr': 1e-4},
				{'params': model.query_for_answer, 'lr': 1e-4},
				{'params': model.memory_for_answer, 'lr': 1e-4},
				# 这个不设定
				{'params': model.classifier.parameters(), 'lr': 1e-4}
			]
		elif self.model_class == 'VaeAttention':
			parameters_dict_list = [
				# 这几个一样
				{'params': model.bert_model.parameters(), 'lr': 5e-5},
				{'params': model.vae_model.parameters(), 'lr': 1e-4},
				# 这几个一样
				{'params': model.key_layer.parameters(), 'lr': 1e-4},
				{'params': model.value_layer.parameters(), 'lr': 1e-4},
				# 这个不设定
				{'params': model.classifier.parameters(), 'lr': 1e-4}
			]
		elif self.model_class == 'VaeAttentionPlus':
			parameters_dict_list = [
				# 这几个一样
				{'params': model.bert_model.parameters(), 'lr': 5e-5},
				{'params': model.vae_model.parameters(), 'lr': 1e-4},
				# 这几个一样
				{'params': model.key_layer.parameters(), 'lr': 1e-4},
				{'params': model.value_layer.parameters(), 'lr': 1e-4},
				{'params': model.fusion_layer.parameters(), 'lr': 1e-4},
				{'params': model.hint_layer.parameters(), 'lr': 1e-4},
				# 这个不设定
				{'params': model.classifier.parameters(), 'lr': 1e-4}
			]
		elif self.model_class == 'BasicModel':
			parameters_dict_list = [
				# 这几个一样
				{'params': model.bert_model.parameters(), 'lr': 5e-5},
				# 这几个一样
				{'params': model.key_layer.parameters(), 'lr': 1e-4},
				{'params': model.value_layer.parameters(), 'lr': 1e-4},
				{'params': model.query_for_question, 'lr': 1e-4},
				{'params': model.query_for_answer, 'lr': 1e-4},
				# 这个不设定
				{'params': model.classifier.parameters(), 'lr': 1e-4}
			]
		elif self.model_class == 'InputMemorySelfAtt':
			parameters_dict_list = [
				# 这几个一样
				{'params': model.bert_model.parameters(), 'lr': 5e-5},
				# 这几个一样
				{'params': model.memory_for_question, 'lr': 5e-5},
				{'params': model.memory_for_answer, 'lr': 5e-5},
				{'params': model.self_attention_weight_layer.parameters(), 'lr': 1e-4},
				{'params': model.value_layer.parameters(), 'lr': 1e-4},
				# 这个不设定
				{'params': model.classifier.parameters(), 'lr': 1e-4}
			]
		elif self.model_class == 'PureMemorySelfAtt':
			parameters_dict_list = [
				# 这几个一样
				{'params': model.bert_model.parameters(), 'lr': 5e-5},
				# 这几个一样
				{'params': model.queries_for_answer.parameters(), 'lr': 1e-4},
				{'params': model.memories_for_answer.parameters(), 'lr': 1e-4},
				{'params': model.queries_for_question.parameters(), 'lr': 1e-4},
				{'params': model.memories_for_question.parameters(), 'lr': 1e-4},
				{'params': model.self_attention_weight_layer.parameters(), 'lr': 1e-4},
				{'params': model.value_layer.parameters(), 'lr': 1e-4},
				# 这个不设定
				{'params': model.classifier.parameters(), 'lr': 1e-4}
			]
		else:
			raise Exception("No optimizer supported for this model class!")

		# 对于那些有两段的,第一段训练参数不太一样
		if not final_stage_flag:
			if self.model_class == 'OneSupremeMemory':
				parameters_dict_list = [
					# 这几个一样
					{'params': model.key_layer.parameters(), 'lr': 1e-4},
					{'params': model.value_layer.parameters(), 'lr': 1e-4},
					{'params': model.memory_for_question, 'lr': 0.3},
					{'params': model.memory_for_answer, 'lr': 0.3},
					{'params': model.vae_model.parameters(), 'lr': 1e-4},
					{'params': model.classifier.parameters(), 'lr': 1e-4},
				]
			elif self.model_class == 'PureMemory':
				parameters_dict_list = [
					# 这几个一样
					{'params': model.vae_model.parameters(), 'lr': 1e-4},
					# 这几个一样
					{'params': model.key_layer.parameters(), 'lr': 1e-4},
					{'params': model.value_layer.parameters(), 'lr': 1e-4},
					{'params': model.query_for_question, 'lr': 1e-4},
					{'params': model.memory_for_question, 'lr': 1e-4},
					{'params': model.query_for_answer, 'lr': 1e-4},
					{'params': model.memory_for_answer, 'lr': 1e-4},
					# 这个不设定
					{'params': model.classifier.parameters(), 'lr': 1e-4}
				]
			elif self.model_class == 'VaeAttention':
				parameters_dict_list = [
					# 这几个一样
					{'params': model.vae_model.parameters(), 'lr': 1e-4},
					# 这几个一样
					{'params': model.key_layer.parameters(), 'lr': 1e-4},
					{'params': model.value_layer.parameters(), 'lr': 1e-4},
					# 这个不设定
					{'params': model.classifier.parameters(), 'lr': 1e-4}
				]
			elif self.model_class == 'VaeAttentionPlus':
				parameters_dict_list = [
					# 这几个一样
					{'params': model.vae_model.parameters(), 'lr': 1e-4},
					# 这几个一样
					{'params': model.key_layer.parameters(), 'lr': 1e-4},
					{'params': model.value_layer.parameters(), 'lr': 1e-4},
					{'params': model.fusion_layer.parameters(), 'lr': 1e-4},
					{'params': model.hint_layer.parameters(), 'lr': 1e-4},
					# 这个不设定
					{'params': model.classifier.parameters(), 'lr': 1e-4},
				]
			elif self.model_class == 'BasicModel':
				parameters_dict_list = [
					# 这几个一样
					{'params': model.key_layer.parameters(), 'lr': 1e-4},
					{'params': model.value_layer.parameters(), 'lr': 1e-4},
					{'params': model.query_for_question, 'lr': 1e-4},
					{'params': model.query_for_answer, 'lr': 1e-4},
					# 这个不设定
					{'params': model.classifier.parameters(), 'lr': 1e-4}
				]
			elif self.model_class == 'InputMemorySelfAtt':
				parameters_dict_list = [
					# 这几个一样
					{'params': model.memory_for_question, 'lr': 0.3},
					{'params': model.memory_for_answer, 'lr': 0.3},
					{'params': model.self_attention_weight_layer.parameters(), 'lr': 1e-4},
					{'params': model.value_layer.parameters(), 'lr': 1e-4},
					# 这个不设定
					{'params': model.classifier.parameters(), 'lr': 1e-4}
				]
			elif self.model_class == 'PureMemorySelfAtt':
				parameters_dict_list = [
					# 这几个一样
					{'params': model.queries_for_answer.parameters(), 'lr': 1e-4},
					{'params': model.memories_for_answer.parameters(), 'lr': 1e-4},
					{'params': model.queries_for_question.parameters(), 'lr': 1e-4},
					{'params': model.memories_for_question.parameters(), 'lr': 1e-4},
					{'params': model.self_attention_weight_layer.parameters(), 'lr': 1e-4},
					{'params': model.value_layer.parameters(), 'lr': 1e-4},
					# 这个不设定
					{'params': model.classifier.parameters(), 'lr': 1e-4}
				]
			else:
				raise Exception("Have Two Stage But No optimizer supported for this model class!")

		optimizer = torch.optim.Adam(parameters_dict_list, lr=5e-5)
		scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9, verbose=True)

		return optimizer, scheduler

	def __model_to_device(self):
		if torch.cuda.device_count() > 1:
			if isinstance(self.model, torch.nn.DataParallel):
				pass
			else:
				print("Let's use", torch.cuda.device_count(), "GPUs!")
				self.model = nn.DataParallel(self.model.to(self.device), device_ids=self.device_ids)
				self.parallel_flag = True
		else:
			self.parallel_flag = False
			self.model.to(self.device)

		self.model.train()

	# 双塔模型的训练步
	def __train_step_for_bi(self, batch, optimizer):
		cross_entropy_function = nn.CrossEntropyLoss()

		# 读取数据
		q_input_ids = (batch['title_input_ids']).to(self.device)
		q_token_type_ids = (batch['title_token_type_ids']).to(self.device)
		q_attention_mask = (batch['title_attention_mask']).to(self.device)

		a_input_ids = (batch['a_input_ids']).to(self.device)
		a_token_type_ids = (batch['a_token_type_ids']).to(self.device)
		a_attention_mask = (batch['a_attention_mask']).to(self.device)

		b_input_ids = (batch['body_input_ids']).to(self.device)
		b_token_type_ids = (batch['body_token_type_ids']).to(self.device)
		b_attention_mask = (batch['body_attention_mask']).to(self.device)

		word_bag = (batch['word_bag']).to(self.device)

		qa_labels = (batch['label']).to(self.device)

		# 优化器置零
		optimizer.zero_grad()
		# 得到模型的结果
		model = self.model
		if self.parallel_flag:
			model = self.model.module

		logits, vae_loss = model(
			q_input_ids=q_input_ids, q_token_type_ids=q_token_type_ids,
			q_attention_mask=q_attention_mask,
			a_input_ids=a_input_ids, a_token_type_ids=a_token_type_ids,
			a_attention_mask=a_attention_mask,
			b_input_ids=b_input_ids, b_token_type_ids=b_token_type_ids,
			b_attention_mask=b_attention_mask, word_bag=word_bag)

		# 计算损失
		vae_loss = vae_loss.mean()
		step_loss = cross_entropy_function(logits, qa_labels) + vae_loss

		# 误差反向传播
		step_loss.backward()
		# 更新模型参数
		optimizer.step()
		optimizer.zero_grad()

		# 统计命中率
		step_shoot_num = logits.shape[0]
		with torch.no_grad():
			_, row_max_indices = logits.topk(k=1, dim=-1)
			step_hit_num = 0
			for i, max_index in enumerate(row_max_indices):
				inner_index = max_index[0]
				if inner_index == qa_labels[i]:
					step_hit_num += 1

		return step_loss, step_shoot_num, step_hit_num

	# 双塔模型的训练步
	def __train_step_for_bi_no_vae(self, batch, optimizer):
		cross_entropy_function = nn.CrossEntropyLoss()

		# 读取数据
		q_input_ids = (batch['title_input_ids']).cuda()
		q_token_type_ids = (batch['title_token_type_ids']).cuda()
		q_attention_mask = (batch['title_attention_mask']).cuda()

		a_input_ids = (batch['a_input_ids']).cuda()
		a_token_type_ids = (batch['a_token_type_ids']).cuda()
		a_attention_mask = (batch['a_attention_mask']).cuda()

		b_input_ids = (batch['body_input_ids']).cuda()
		b_token_type_ids = (batch['body_token_type_ids']).cuda()
		b_attention_mask = (batch['body_attention_mask']).cuda()

		qa_labels = (batch['label']).cuda()

		# 优化器置零
		optimizer.zero_grad()
		# 得到模型的结果
		model = self.model
		if self.parallel_flag:
			model = self.model.module
		logits = self.model(
			q_input_ids=q_input_ids, q_token_type_ids=q_token_type_ids,
			q_attention_mask=q_attention_mask,
			a_input_ids=a_input_ids, a_token_type_ids=a_token_type_ids,
			a_attention_mask=a_attention_mask,
			b_input_ids=b_input_ids, b_token_type_ids=b_token_type_ids,
			b_attention_mask=b_attention_mask)

		# 计算损失
		step_loss = cross_entropy_function(logits, qa_labels)

		# 误差反向传播
		step_loss.backward()

		# 更新模型参数
		optimizer.step()
		optimizer.zero_grad()
		torch.cuda.empty_cache()

		# 统计命中率
		step_shoot_num = logits.shape[0]
		with torch.no_grad():
			_, row_max_indices = logits.topk(k=1, dim=-1)
			step_hit_num = 0
			for i, max_index in enumerate(row_max_indices):
				inner_index = max_index[0]
				if inner_index == qa_labels[i]:
					step_hit_num += 1

		return step_loss, step_shoot_num, step_hit_num

	# cross模型的训练步
	def __train_step_for_cross(self, batch, optimizer):
		# 损失函数
		cross_entropy_function = nn.CrossEntropyLoss()

		# 读取数据
		input_ids = (batch['input_ids']).to(self.device)
		token_type_ids = (batch['token_type_ids']).to(self.device)
		attention_mask = (batch['attention_mask']).to(self.device)

		# 优化器置零
		optimizer.zero_grad()

		model = self.model
		if self.parallel_flag:
			model = self.model.module
		logits = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)

		target = torch.tensor([self.train_candidate_num-1] * logits.shape[0]).to(logits.device)

		step_loss = cross_entropy_function(logits, target)

		# 误差反向传播
		step_loss.backward()

		# 更新模型参数
		optimizer.step()
		optimizer.zero_grad()

		# 统计命中率
		step_shoot_num = logits.shape[0]
		with torch.no_grad():
			_, row_max_indices = logits.topk(k=1, dim=-1)
			step_hit_num = 0
			for i, max_index in enumerate(row_max_indices):
				inner_index = max_index[0]
				if inner_index == self.train_candidate_num-1:
					step_hit_num += 1

		return step_loss, step_shoot_num, step_hit_num

	def __val_step_for_bi(self, batch):
		# 读取数据
		q_input_ids = (batch['title_input_ids']).to(self.device)
		q_token_type_ids = (batch['title_token_type_ids']).to(self.device)
		q_attention_mask = (batch['title_attention_mask']).to(self.device)

		a_input_ids = (batch['a_input_ids']).to(self.device)
		a_token_type_ids = (batch['a_token_type_ids']).to(self.device)
		a_attention_mask = (batch['a_attention_mask']).to(self.device)

		b_input_ids = (batch['body_input_ids']).to(self.device)
		b_token_type_ids = (batch['body_token_type_ids']).to(self.device)
		b_attention_mask = (batch['body_attention_mask']).to(self.device)

		word_bag = (batch['word_bag']).to(self.device)

		with torch.no_grad():
			# 得到模型的结果
			model = self.model
			if self.parallel_flag:
				model = self.model.module
			logits, _ = model(
				q_input_ids=q_input_ids, q_token_type_ids=q_token_type_ids,
				q_attention_mask=q_attention_mask,
				a_input_ids=a_input_ids, a_token_type_ids=a_token_type_ids,
				a_attention_mask=a_attention_mask,
				b_input_ids=b_input_ids, b_token_type_ids=b_token_type_ids,
				b_attention_mask=b_attention_mask, word_bag=word_bag)

		return logits

	def __val_step_for_bi_no_vae(self, batch):
		# 读取数据
		q_input_ids = (batch['title_input_ids']).to(self.device)
		q_token_type_ids = (batch['title_token_type_ids']).to(self.device)
		q_attention_mask = (batch['title_attention_mask']).to(self.device)

		a_input_ids = (batch['a_input_ids']).to(self.device)
		a_token_type_ids = (batch['a_token_type_ids']).to(self.device)
		a_attention_mask = (batch['a_attention_mask']).to(self.device)

		b_input_ids = (batch['body_input_ids']).to(self.device)
		b_token_type_ids = (batch['body_token_type_ids']).to(self.device)
		b_attention_mask = (batch['body_attention_mask']).to(self.device)

		with torch.no_grad():
			# 得到模型的结果
			model = self.model
			if self.parallel_flag:
				model = self.model.module
			logits = model(
				q_input_ids=q_input_ids, q_token_type_ids=q_token_type_ids,
				q_attention_mask=q_attention_mask,
				a_input_ids=a_input_ids, a_token_type_ids=a_token_type_ids,
				a_attention_mask=a_attention_mask,
				b_input_ids=b_input_ids, b_token_type_ids=b_token_type_ids,
				b_attention_mask=b_attention_mask)

		return logits

	def __val_step_for_cross(self, batch):
		input_ids = (batch['input_ids']).to(self.device)
		token_type_ids = (batch['token_type_ids']).to(self.device)
		attention_mask = (batch['attention_mask']).to(self.device)

		logits = self.model.evaluate(input_ids=input_ids, token_type_ids=token_type_ids,
									 attention_mask=attention_mask)

		return logits

	def __get_dataloader(self, data, batch_size, split_index, split_num):
		if split_index == split_num - 1:
			data_block = data[int(split_index * len(data) / split_num):]
		else:
			data_block = data[int(split_index * len(data) / split_num):
								   int((split_index + 1) * len(data) / split_num)]

		# add model
		if self.model_class in ["OneSupremeMemory", 'PureMemory', 'VaeAttention', 'VaeAttentionPlus']:
			dataset = Test_TBA_word_bag_classify_dataset(data=data_block,
														 tokenizer=self.tokenizer,
														 text_max_len=self.text_max_len - self.memory_num,
														 voc=self.voc)
		elif self.model_class in ['BasicModel', 'InputMemorySelfAtt', 'PureMemorySelfAtt']:
			dataset = TBAClassifyDataset(data=data_block,
										 tokenizer=self.tokenizer,
										 text_max_len=self.text_max_len - self.memory_num)
		else:
			raise Exception("No train dataset supported for this model class!")

		dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
								num_workers=0, drop_last=True)

		return dataloader

	@staticmethod
	def __print_ranking_result(label_hit_num, label_shoot_num, label_target_num):
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
		for i in range(0, 4):
			recall.append(label_hit_num[i] / (label_target_num[i] + 1e-8))

		precise = []
		for i in range(0, 4):
			precise.append(label_hit_num[i] / (label_shoot_num[i] + 1e-8))

		print("recall:")
		print(f"\tbest:{recall[3]}\tpositive:{recall[2]}\tneutral:{recall[1]}\tworst:{recall[0]}\t")
		print("precise:")
		print(f"\tbest:{precise[3]}\tpositive:{precise[2]}\tneutral:{precise[1]}\tworst:{precise[0]}\t")
		print()

		return accuracy

	@staticmethod
	def hits_count(candidate_ranks, k):
		count = 0
		for rank in candidate_ranks:
			if rank <= k:
				count += 1
		return count / (len(candidate_ranks) + 1e-8)

	@staticmethod
	def dcg_score(candidate_ranks, k):
		score = 0
		for rank in candidate_ranks:
			if rank <= k:
				score += 1 / np.log2(1 + rank)
		return score / (len(candidate_ranks) + 1e-8)


# def set_seed(seed):
# 	random.seed(seed)
# 	np.random.seed(seed)
# 	torch.manual_seed(seed)
# 	cudnn.deterministic = True
# 	cudnn.benchmark = False
#
#
# def read_arguments():
# 	parser = argparse.ArgumentParser()
#
# 	# default arguments
# 	parser.add_argument("--seed", "-s", default=42, type=int)
# 	parser.add_argument("--load_model", "-l", action="store_true", default=False)
# 	parser.add_argument("--load_middle", action="store_true", default=False)
# 	parser.add_argument("--one_stage", action="store_true", default=False)
#
# 	parser.add_argument("--save_model_dict", default="./model/", type=str)
# 	parser.add_argument("--val_num_each_epoch", default=2, type=int)
# 	parser.add_argument("--pretrained_bert_path", default='prajjwal1/bert-small', type=str)
# 	parser.add_argument("--text_max_len", default=512, type=int)
# 	parser.add_argument("--print_num_each_epoch", default=20, type=int)
# 	parser.add_argument("--dataset_split_num", default=20, type=int)
# 	parser.add_argument("--ranking_candidate_num", default=5, type=int)
#
# 	parser.add_argument("--dataset_name", "-d", type=str)  # !!!
# 	parser.add_argument("--val_batch_size", default=32, type=int) 	# !!!!
# 	parser.add_argument("--train_batch_size", default=64, type=int)  # !!!
# 	parser.add_argument("--val_candidate_num", default=100, type=int, help="candidates num for ranking")  # !!!
# 	parser.add_argument("--train_candidate_num", default=16, type=int, help="only need by cross")  # !!!
# 	parser.add_argument("--memory_num", "-m", default=50, type=int)  # !!!
# 	parser.add_argument("--latent_dim", default=100, type=int)  # !!!
# 	parser.add_argument("--label_num", default=4, type=int)  # !!!
# 	parser.add_argument("--train_vae", action="store_true", default=False)  # !!!
# 	parser.add_argument("--hop_num", default=1, type=int)  # !!!
#
# 	# hand set arguments
# 	parser.add_argument("--nvidia_number", "-n", required=True, type=str)  # !!!
# 	parser.add_argument("--model_class", required=True, type=str)  # !!!
#
# 	parser.add_argument("--load_model_dict", type=str)
# 	parser.add_argument("--model_save_prefix", default="", type=str)
#
# 	args = parser.parse_args()
# 	print("args:", args)
# 	return args
#
#
# if __name__ == '__main__':
# 	my_args = read_arguments()
#
# 	# 设置随机种子
# 	set_seed(my_args.seed)
#
# 	# 创建训练类
# 	my_train_model = TrainWholeModel(my_args)
#
# 	# 是否训练vae
# 	if my_args.train_vae:
# 		my_train_model.train_vae()
# 	if my_args.model_class in ['OneSupremeMemory'] and not my_args.load_model:
# 		my_train_model.load_vae_model()
#
# 	# 设置训练参数
# 	my_train_two_stage_flag = False
# 	# add model
# 	if my_args.model_class in ['OneSupremeMemory', 'PureMemory', 'VaeAttention', 'VaeAttentionPlus', 'BasicModel',
# 							   'InputMemorySelfAtt', 'PureMemorySelfAtt']:
# 		my_train_two_stage_flag = True
#
# 	if my_args.one_stage:
# 		my_train_two_stage_flag = False
#
# 	my_train_model.train(model_save_path=my_args.save_model_dict + "/" + my_args.model_save_prefix +
# 										 my_args.model_class + "_" +
# 										 my_args.dataset_name, train_two_stage_flag=my_train_two_stage_flag)
