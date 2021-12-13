# %%
import datasets
from nltk.corpus import stopwords
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
import torch.nn.functional
import os
import torch.utils.data
import gc
from transformers import AutoTokenizer, AutoConfig, BertForMaskedLM, \
	DataCollatorForLanguageModeling, get_linear_schedule_with_warmup
import numpy as np
import sys
from tqdm import tqdm

from reformed_model_lib import BasicConfig, BasicModel, InputMemorySelfAttConfig, \
	InputMemorySelfAtt, PureMemorySelfAttConfig, PureMemorySelfAtt
from reformed_dataset import TBAClassifyDataset, MLMDataset


class TrainWholeModel:
	def __init__(self, args, config=None):

		# 读取一些参数并存起来-------------------------------------------------------------------
		self.__read_args_for_train(args)
		self.args = args

		# 设置gpu-------------------------------------------------------------------
		# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
		os.environ["TOKENIZERS_PARALLELISM"] = "false"
		os.environ["CUDA_VISIBLE_DEVICES"] = args.nvidia_number

		# for data_parallel
		nvidia_number = len(args.nvidia_number.split(","))
		self.device_ids = [i for i in range(nvidia_number)]

		# self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

		if not torch.cuda.is_available():
			raise Exception("No cuda available!")

		local_rank = 0
		if self.data_distribute:
			local_rank = self.local_rank
			torch.cuda.set_device(local_rank)
			torch.distributed.init_process_group(
				'nccl',
				init_method='env://'
			)
		else:
			torch.cuda.set_device(local_rank)

		print(f"local rank: {local_rank}")
		self.device = torch.device(f'cuda:{local_rank}')

		# 读取tokenizer-------------------------------------------------------------------
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

		# 获得词汇表-------------------------------------------------------------------
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

		# 获得模型配置-------------------------------------------------------------------
		if config is None:
			self.config = self.__read_args_for_config(args)
		else:
			self.config = config

		# instance attribute
		self.model = None
		self.teacher_model = None

	def train(self, model_save_path, train_two_stage_flag, memory_save_name):
		# 用来判断是哪一阶段的训练
		final_stage_flag = not train_two_stage_flag

		while True:
			# 创建模型
			self.model = self.__create_model()

			# 读取预训练好的memory
			if self.load_memory_flag:
				self.load_pretrained_memory(memory_save_name)

			# 如果要进行第二阶段训练，那需要先读取上一阶段最好的model
			if final_stage_flag and train_two_stage_flag:
				self.model = self.load_models(self.model, model_save_path + "_middle")

			# 读取数据
			self.__model_to_device()

			# 这不不会出问题吧
			self.__model_parallel()

			# 优化器
			optimizer = self.__get_model_optimizer(final_stage_flag=final_stage_flag)

			# 准备训练
			previous_best_r_1 = -1

			if train_two_stage_flag:
				if final_stage_flag:
					print("~"*60 + " begin final stage train " + "~"*60)
				else:
					print("~"*60 + " begin first stage train " + "~"*60)
			else:
				print("~"*60 + " begin one stage train " + "~"*60)

			# 设置早停变量
			if final_stage_flag:
				early_stop_threshold = 10
			else:
				early_stop_threshold = 5

			early_stop_count = 0

			for epoch in range(self.num_train_epochs):
				torch.cuda.empty_cache()

				# whether this epoch get the best model
				this_epoch_best = False

				# 打印一下
				print("*" * 20 + f" {epoch} " + "*" * 20)

				# 训练之前先看初始情况,保存初始best performance
				if epoch == 0 and not self.no_initial_test:
					print("-" * 30 + "initial_test" + "-" * 30)
					test_data = datasets.load_from_disk("./" + self.dataset_name + "/string_val.dataset")
					test_data = test_data.shuffle(seed=None)
					val_dataloader = self.__get_dataloader(data=test_data, batch_size=self.val_batch_size,
														   split_index=0, split_num=1)
					val_loss, val_acc = self.classify_validate_model(val_dataloader)
					print(val_loss, val_acc)
					r_1 = self.ranking()
					if r_1 > previous_best_r_1:
						previous_best_r_1 = r_1
					print("-" * 30 + "initial_test_end" + "-" * 30, end="\n\n")

				# 开始训练
				train_loss = 0.0
				# 计算训练集的R@1
				shoot_num = 0
				hit_num = 0

				now_batch_num = 0
				next_val_num = 1

				# 读取训练数据
				train_data = datasets.load_from_disk("./" + self.dataset_name + "/string_train.dataset")
				train_data = train_data.shuffle(seed=None)

				# 逐块训练
				for split_index in range(self.dataset_split_num):
					# 获取这块数据的dataloader
					print("-" * 10 + f"data block {split_index + 1}/{self.dataset_split_num}" + "-" * 10)

					train_block_dataloader = self.__get_dataloader(data=train_data, batch_size=self.train_batch_size,
																   split_index=split_index,
																   split_num=self.dataset_split_num)

					if self.data_distribute:
						train_block_dataloader.sampler.set_epoch(epoch)

					# 获取scheduler
					if epoch == 0 and split_index == 0:
						t_total = (
										  len(train_block_dataloader) // self.gradient_accumulation_steps) * self.num_train_epochs * self.dataset_split_num
						scheduler = get_linear_schedule_with_warmup(optimizer,
																	num_warmup_steps=int(t_total * 0.02),
																	num_training_steps=t_total)
						print(f"Train {self.num_train_epochs} epochs, Block num {self.dataset_split_num}, "
							  f"Block Batch num {len(train_block_dataloader)}, "
							  f"Acc num {self.gradient_accumulation_steps}, Total update {t_total}\n")

					# 开始训练这一块
					train_loss, shoot_num, hit_num, \
					now_batch_num, previous_best_r_1, \
					this_epoch_best, next_val_num \
						= self.__train_one_data_block(
						train_block_dataloader, epoch, optimizer, train_loss, shoot_num,
						hit_num,
						now_batch_num, final_stage_flag, model_save_path,
						previous_best_r_1,
						this_epoch_best, next_val_num, scheduler)

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

	def __train_one_data_block(self, train_block_dataloader, epoch, optimizer, train_loss, shoot_num, hit_num,
							   now_batch_num, final_stage_flag, model_save_path, previous_best_r_1,
							   this_epoch_best, next_val_num, scheduler):
		# 进度条
		bar = tqdm(train_block_dataloader, total=len(train_block_dataloader))

		# 开始训练
		for batch in bar:
			# add model
			if self.model_class == "CrossEncoder":
				step_loss, step_shoot_num, step_hit_num = self.__train_step_for_cross(batch=batch,
																					  optimizer=optimizer)
			elif self.model_class in ['BasicModel', 'InputMemorySelfAtt', 'PureMemorySelfAtt']:
				step_loss, step_shoot_num, step_hit_num = \
					self.__train_step_for_bi(batch=batch, optimizer=optimizer, now_batch_num=now_batch_num,
													scheduler=scheduler)
			else:
				raise Exception("Train step have not supported this model class")


			# 更新一下信息
			train_loss += step_loss.item()

			shoot_num += step_shoot_num
			hit_num += step_hit_num
			now_batch_num += 1

			bar.set_description(
				"epoch {:>3d} loss {:.4f} Acc {:>10d}/{:>10d} = {:.4f}".format(epoch + 1, train_loss / now_batch_num,
																			  hit_num, shoot_num,
																			  hit_num / shoot_num * 100))

			# 要不要是时候评测一下呢，一个epoch评测 val_num_each_epoch 次
			val_interval = (
								   (
											   len(train_block_dataloader) - 1) * self.dataset_split_num - self.gradient_accumulation_steps) // self.val_num_each_epoch

			if now_batch_num // val_interval >= next_val_num and next_val_num <= self.val_num_each_epoch:
				# 还有梯度没更新的话，会占着显存，影响val batch size，所以就下步val了
				if now_batch_num % self.gradient_accumulation_steps != 0:
					continue

				# 更新这一epoch中已经评测的次数
				next_val_num += 1

				# 获得评测结果
				test_data = datasets.load_from_disk("./" + self.dataset_name + "/string_val.dataset")
				test_data = test_data.shuffle(seed=None)

				val_dataloader = self.__get_dataloader(data=test_data, batch_size=self.val_batch_size,
													   split_index=0, split_num=1)

				val_loss, val_acc = self.classify_validate_model(val_dataloader)
				r_1 = self.ranking()
				print(
					f"{epoch + 1} epoch middle eval: " + "*" * 30 +
					f"\nval_loss:{val_loss}\tval_acc{val_acc}\tR@1:{r_1}\tprevious best R@1:{previous_best_r_1}\tfrom rank:{self.local_rank}")

				# 模型是否比之前优秀
				if r_1 > previous_best_r_1:
					previous_best_r_1 = r_1
					this_epoch_best = True

					# 保存模型
					postfix = ""
					if not final_stage_flag:
						postfix = "_middle"

					self.save_model(model_save_path=model_save_path, postfix=postfix)

				gc.collect()
				del val_dataloader, test_data
				gc.collect()

		gc.collect()

		return train_loss, shoot_num, hit_num, \
			   now_batch_num, previous_best_r_1, \
			   this_epoch_best, next_val_num

	def train_memory_by_mlm(self, memory_save_name):
		scheduler = None

		# 获得之前的voc size，虽然voc size一直不会变
		voc_size = len(self.tokenizer)

		# 给tokenizer添加memory
		memory_token_list = []
		for i in range(self.memory_num):
			memory_token_list.append('<MEM' + str(i) + '>')

		special_tokens_dict = {'additional_special_tokens': memory_token_list}
		print("-" * 30)
		print(f"previous token num:{len(self.tokenizer)}")
		self.tokenizer.add_special_tokens(special_tokens_dict)
		print(f"now token num:{len(self.tokenizer)}")
		print("-" * 30 + "\n")

		# 遮盖数据的api，很关键
		data_collator = DataCollatorForLanguageModeling(
			tokenizer=self.tokenizer, mlm=True, mlm_probability=0.15
		)

		# 读取训练集
		print("prepare train data ...", end=" ")
		train_data = datasets.load_from_disk("./" + self.dataset_name + "/string_train.dataset")
		train_data = train_data.shuffle(seed=None)

		print("end")
		print("-" * 30)

		# 读取测试集
		print("prepare test data ...", end=" ")
		test_data = datasets.load_from_disk("./" + self.dataset_name + "/string_val.dataset")
		test_data = test_data.shuffle(seed=None)

		test_dataset = MLMDataset(data=test_data, tokenizer=self.tokenizer, text_max_len=self.text_max_len,
								  memory_num=self.memory_num, memory_start_index=voc_size)

		print("end")
		print("-" * 30)

		# 创建模型
		print("create model ...", end=" ")
		model = BertForMaskedLM.from_pretrained(self.config.pretrained_bert_path)
		model.resize_token_embeddings(len(self.tokenizer))
		model.to(self.device)
		model.train()
		print("end")
		print("-" * 30)

		# 获得optimizer，只训练那几个embedding
		model_embeddings = model.get_input_embeddings()
		print(f"embedding shape: {model_embeddings.weight.shape}")

		parameters_dict_list = [
			{'params': model_embeddings.parameters(), 'lr': 0.3},
		]
		optimizer = torch.optim.Adam(parameters_dict_list)

		# 开始训练
		print("~" * 60 + " begin MLM train " + "~" * 60)

		previous_min_loss = np.inf
		early_stop_threshold = 10
		worse_epoch_count = 0

		for epoch in range(self.num_train_epochs):
			# 一个flag
			this_epoch_best = False

			# 一个计数器
			now_batch_num = 0
			next_val_num = 1

			# 总loss
			epoch_all_loss = 0.0

			# 一块一块得tokenize
			for split_index in range(self.dataset_split_num):
				print("-" * 10 + f"data block {split_index + 1}/{self.dataset_split_num}" + "-" * 10)

				if split_index == self.dataset_split_num - 1:
					data_block = train_data[int(split_index * len(train_data) / self.dataset_split_num):]
				else:
					data_block = train_data[int(split_index * len(train_data) / self.dataset_split_num):
											int((split_index + 1) * len(train_data) / self.dataset_split_num)]

				train_dataset = MLMDataset(data=data_block, tokenizer=self.tokenizer, text_max_len=self.text_max_len,
										   memory_num=self.memory_num, memory_start_index=voc_size)

				train_dataloader = DataLoader(train_dataset, batch_size=self.train_batch_size, num_workers=2,
											  shuffle=True, drop_last=True)

				# 进度条
				bar = tqdm(train_dataloader, total=len(train_dataloader))

				# 根据参数更新的总次数获得scheduler
				if epoch == 0 and split_index == 0:
					t_total = (len(train_dataloader) // self.gradient_accumulation_steps) * self.num_train_epochs * self.dataset_split_num
					scheduler = get_linear_schedule_with_warmup(optimizer,
																num_warmup_steps=int(t_total * 0.02),
																num_training_steps=t_total)
					print(f"Train {self.num_train_epochs} epochs, Block num {self.dataset_split_num}, "
						  f"Block Batch num {len(train_dataloader)}, "
						  f"Acc num {self.gradient_accumulation_steps}, Total update {t_total}\n")

				for batch in bar:
					# 进行一个mask
					batch['input_ids'], label = data_collator.torch_mask_tokens(batch['input_ids'],
																				batch['special_tokens_mask'])

					# 读取数据
					input_ids = (batch['input_ids']).to(self.device)
					token_type_ids = (batch['token_type_ids']).to(self.device)
					attention_mask = (batch['attention_mask']).to(self.device)
					label = label.to(self.device)

					result = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask,
								   labels=label)

					# 反向传播
					loss = result['loss']
					loss.backward()

					# 更新计数器
					now_batch_num += 1
					epoch_all_loss += loss.item()
					bar.set_description("epoch {} loss {:.4f}".format(epoch + 1, epoch_all_loss/now_batch_num))

					# 是否更新参数
					if now_batch_num % self.gradient_accumulation_steps == 0:
						# Update parameters
						model_embeddings.weight.grad[:voc_size] *= 0.0
						optimizer.step()
						optimizer.zero_grad()
						scheduler.step()

					# 要不要是时候评测一下呢，一个epoch评测 val_num_each_epoch 次
					val_interval = ((
											(
														len(train_dataloader) - 1) * self.dataset_split_num) - self.gradient_accumulation_steps) // self.val_num_each_epoch

					if now_batch_num // val_interval >= next_val_num and next_val_num <= self.val_num_each_epoch:
						# 如果还有梯度存着，那就先不val，免得遇到大val_batch_size爆显存
						if now_batch_num % self.gradient_accumulation_steps != 0:
							continue

						next_val_num += 1
						test_dataloader = DataLoader(test_dataset, batch_size=self.val_batch_size, num_workers=2,
													 shuffle=True,
													 drop_last=True)

						# 获得评测结果
						model.eval()
						val_loss = 0.0

						for val_batch in test_dataloader:
							# 进行一个mask
							val_batch['input_ids'], val_label = data_collator.torch_mask_tokens(val_batch['input_ids'],
																								val_batch[
																									'special_tokens_mask'])

							# 读取数据
							input_ids = (val_batch['input_ids']).to(self.device)
							token_type_ids = (val_batch['token_type_ids']).to(self.device)
							attention_mask = (val_batch['attention_mask']).to(self.device)
							val_label = val_label.to(self.device)

							with torch.no_grad():
								result = model(input_ids=input_ids, token_type_ids=token_type_ids,
											   attention_mask=attention_mask,
											   labels=val_label)

							# 反向传播
							val_loss += result['loss'].item()

						print(
							f"{epoch + 1} epoch middle eval: " + "*" * 30 +
							f"\nval_loss:{val_loss}\tprevious min loss:{previous_min_loss}\tfrom rank:{self.local_rank}")

						# 模型是否比之前优秀
						if val_loss < previous_min_loss:
							previous_min_loss = val_loss

							# 保存模型
							save_state = {'pretrained_bert_path': self.config.pretrained_bert_path,
										  'memory_num': self.memory_num,
										  'memory_start_index': voc_size,
										  'embedding_shape': model_embeddings.weight.shape,
										  'embedding': model_embeddings.state_dict()}

							torch.save(save_state, "./model/pretrained_memory/" + memory_save_name)
							print(f"model saved at ./model/pretrained_memory/{memory_save_name}!!!")

							this_epoch_best = True

						model.train()

				# 保险起见删除数据
				gc.collect()
				del data_block, train_dataset, train_dataloader
				gc.collect()

			if not this_epoch_best:
				worse_epoch_count += 1
			else:
				worse_epoch_count = 0

			if worse_epoch_count == early_stop_threshold:
				print("training finished!!!")
				break

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
				if self.model_class in ['BasicModel', 'InputMemorySelfAtt', 'PureMemorySelfAtt']:
					logits = self.__val_step_for_bi(batch)
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

	def load_pretrained_memory(self, memory_save_name):
		saved_dict = torch.load("./model/pretrained_memory/" + memory_save_name)
		memory_start_index = saved_dict['memory_start_index']

		memory_weights = {'memory_for_answer':saved_dict['embedding']['weight'][memory_start_index:],
						  'memory_for_question':saved_dict['embedding']['weight'][memory_start_index:]}

		model_dict = self.model.state_dict()
		model_dict.update(memory_weights)
		self.model.load_state_dict(model_dict)
		print(f"Memory is loaded from ./model/pretrained_memory/{memory_save_name} !!!")

	def save_model(self, model_save_path, postfix=""):
		self.model.eval()

		save_path = model_save_path + postfix

		if self.data_distribute or self.data_parallel:
			torch.save(self.model.module.state_dict(), save_path)
		else:
			torch.save(self.model.state_dict(), save_path)

		# if self.data_distribute:
		# 	checkpoint = {
		# 		'model': self.model.module.state_dict(),
		# 		'amp': amp.state_dict()
		# 	}
		# 	save_path += ".pt"
		# 	torch.save(checkpoint, save_path)
		# elif self.data_parallel:
		# 	checkpoint = {
		# 		'model': self.model.module.state_dict(),
		# 		'amp': {}
		# 	}
		# 	save_path += ".pt"
		# 	torch.save(checkpoint, save_path)
		# else:
		# 	torch.save(self.model.state_dict(), save_path)

		print("!" * 60)
		print(f"model is saved at {save_path}")
		print("!" * 60)

		self.model.train()

	@staticmethod
	def load_models(model, load_model_path):
		if load_model_path is None:
			print("you should offer model paths!")

		load_path = load_model_path

		model.load_state_dict(torch.load(load_path))

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

				now_index += step

				with torch.no_grad():
					# add model
					if self.model_class in ['BasicModel', 'InputMemorySelfAtt', 'PureMemorySelfAtt']:
						# shape = (q_num*candidate_answer_num, 4)
						logits = self.model(
							q_input_ids=q_input_ids, q_token_type_ids=q_token_type_ids,
							q_attention_mask=q_attention_mask,
							a_input_ids=a_input_ids, a_token_type_ids=a_token_type_ids,
							a_attention_mask=a_attention_mask,
							b_input_ids=b_input_ids, b_token_type_ids=b_token_type_ids,
							b_attention_mask=b_attention_mask)

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
		if self.model_class == 'BasicModel':
			model = BasicModel(config=self.config)
		elif self.model_class == 'InputMemorySelfAtt':
			model = InputMemorySelfAtt(config=self.config)
		elif self.model_class == 'PureMemorySelfAtt':
			model = PureMemorySelfAtt(config=self.config)
		else:
			raise Exception("This model class is not supported for creating!!")

		# 要不要加载现成的模型
		if self.load_model_flag:
			load_model_path = self.args.save_model_dict + "/" + self.args.model_save_prefix + \
							  self.args.model_class + "_" + self.args.dataset_name
			if self.args.load_middle:
				load_model_path += "_middle"

			model = self.load_models(model, load_model_path)

		print("--------------------- model  created ---------------------")

		return model

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

		self.local_rank = args.local_rank
		self.data_parallel = args.data_parallel
		self.data_distribute = args.data_distribute
		self.distill_flag = args.distill
		self.teacher_path = args.teacher_path

		self.num_train_epochs = args.num_train_epochs
		self.gradient_accumulation_steps = args.gradient_accumulation_steps
		self.no_initial_test = args.no_initial_test
		self.load_memory_flag = args.load_memory

	# 读取命令行传入的有关config的参数
	def __read_args_for_config(self, args):
		if args.pretrained_bert_path in ['prajjwal1/bert-small', 'google/bert_uncased_L-6_H-512_A-8',
										 'google/bert_uncased_L-8_H-512_A-8']:
			word_embedding_len = 512
			sentence_embedding_len = 512
		elif args.pretrained_bert_path == 'bert-base-uncased':
			word_embedding_len = 768
			sentence_embedding_len = 768
		elif args.pretrained_bert_path == 'google/bert_uncased_L-2_H-128_A-2':
			word_embedding_len = 128
			sentence_embedding_len = 128
		else:
			raise Exception("word_embedding_len, sentence_embedding_len is needed!")

		# add model
		if self.model_class == 'BasicModel':
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

		if self.data_distribute or self.data_parallel:
			model = self.model.module
		else:
			model = self.model

		# add model
		# 获得模型的训练参数和对应的学习率
		if self.model_class == 'BasicModel':
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
				# {'params': model.memory_for_question, 'lr': 1e-4},
				# {'params': model.memory_for_answer, 'lr': 1e-4},
				# {'params': model.memory_for_question, 'lr': 0.3},
				# {'params': model.memory_for_answer, 'lr': 0.3},
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
			if self.model_class == 'BasicModel':
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

		return optimizer

	def __model_to_device(self):
		self.model.to(self.device)
		self.model.train()

	def __model_parallel(self):
		if self.data_distribute:
			self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model).to(self.device)
			self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[self.local_rank],
																   find_unused_parameters=True,
																   output_device=self.local_rank)
			# self.model = convert_syncbn_model(self.model).to(self.device)
			# self.model, new_optimizer = amp.initialize(self.model, optimizer, opt_level='O1')
			# self.model = DistributedDataParallel(self.model, delay_allreduce=True)

		elif self.data_parallel:
			print("Let's use", torch.cuda.device_count(), "GPUs!")
			self.model = nn.DataParallel(self.model, device_ids=self.device_ids)

	# 双塔模型的训练步
	def __train_step_for_bi(self, batch, optimizer, now_batch_num, scheduler):
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

		qa_labels = (batch['label']).to(self.device)

		# 得到模型的结果
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
		if (now_batch_num + 1) % self.gradient_accumulation_steps == 0:
			optimizer.step()
			optimizer.zero_grad()
			scheduler.step()

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

		logits = self.model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)

		target = torch.tensor([self.train_candidate_num - 1] * logits.shape[0]).to(logits.device)

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
				if inner_index == self.train_candidate_num - 1:
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

		with torch.no_grad():
			# 得到模型的结果
			logits = self.model(
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
			now_data_block = data[int(split_index * len(data) / split_num):]
		else:
			now_data_block = data[int(split_index * len(data) / split_num):
								  int((split_index + 1) * len(data) / split_num)]

		# add model
		if self.model_class in ['BasicModel', 'InputMemorySelfAtt', 'PureMemorySelfAtt']:
			now_dataset = TBAClassifyDataset(data=now_data_block,
											 tokenizer=self.tokenizer,
											 text_max_len=self.text_max_len - self.memory_num)
		else:
			raise Exception("No train dataset supported for this model class!")

		if self.data_distribute:
			sampler = torch.utils.data.distributed.DistributedSampler(now_dataset, shuffle=True, drop_last=True)
			dataloader = DataLoader(now_dataset, batch_size=batch_size, num_workers=3, sampler=sampler)
		else:
			dataloader = DataLoader(now_dataset, batch_size=batch_size, num_workers=3, shuffle=True, drop_last=True)

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
