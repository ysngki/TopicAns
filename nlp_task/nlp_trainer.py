# %%
import gc

import datasets
from datasets import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
import torch.nn.functional
import os
import torch.utils.data
from transformers import AutoTokenizer, AutoConfig, get_linear_schedule_with_warmup
import sys
from tqdm import tqdm
import csv

from nlp_model import QAClassifierModel, QAClassifierModelConfig, CrossBERT, CrossBERTConfig, ParallelEncoder, ParallelEncoderConfig


class TrainWholeModel:
	def __init__(self, args, config=None):

		# 读取一些参数并存起来-------------------------------------------------------------------
		self.__read_args_for_train(args)
		self.args = args

		# 设置gpu-------------------------------------------------------------------
		# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
		os.environ["TOKENIZERS_PARALLELISM"] = "false"
		os.environ["CUDA_VISIBLE_DEVICES"] = args.nvidia_number

		# for data_parallel-------------------------------------------------------------------
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

		if self.args.use_cpu:
			self.device = torch.device(f'cpu')
			print("Using cpu!!!!!")
		else:
			print(f"local rank: {local_rank}")
			self.device = torch.device(f'cuda:{local_rank}')

		# 读取tokenizer-------------------------------------------------------------------
		tokenizer_path = args.pretrained_bert_path.replace("/", "_")
		tokenizer_path = tokenizer_path.replace("\\", "_")

		# add model
		if self.model_class in ['QAMemory']:
			tokenizer_path += "_" + str(self.memory_num) + "_" + self.model_class

		# read from disk or save to disk
		if os.path.exists("../tokenizer/" + tokenizer_path):
			self.tokenizer = AutoTokenizer.from_pretrained("../tokenizer/" + tokenizer_path)
		else:
			print("first time use this tokenizer, downloading...")
			self.tokenizer = AutoTokenizer.from_pretrained(args.pretrained_bert_path)
			tokenizer_config = AutoConfig.from_pretrained(args.pretrained_bert_path)

			self.tokenizer.save_pretrained("../tokenizer/" + tokenizer_path)
			tokenizer_config.save_pretrained("../tokenizer/" + tokenizer_path)

		self.origin_voc_size = len(self.tokenizer) - self.memory_num*2

		# 获得模型配置-------------------------------------------------------------------
		if config is None:
			self.config = self.__read_args_for_config(args)
		else:
			self.config = config

		# avoid warning
		self.model = None
		self.teacher_model = None
		self.restore_flag = False

	def train(self, model_save_path, train_two_stage_flag, only_final=False):
		# 用来判断是哪一阶段的训练
		final_stage_flag = not train_two_stage_flag

		# two stage training but begin with final stage
		if only_final:
			final_stage_flag = True

		while True:
			# 创建模型
			self.model = self.__create_model()

			# 如果要进行第二阶段训练，那需要先读取上一阶段最好的model, for normal two stage
			if final_stage_flag and train_two_stage_flag:
				self.model = self.load_models(self.model, model_save_path + "_middle")

			# restore training model
			restore_path = None
			if self.restore_flag:
				restore_path = "./last_model/" + self.model_save_prefix + self.model_class + "_" + self.dataset_name
				if not final_stage_flag:
					restore_path += "_middle"

				self.model = self.load_models(self.model, restore_path)

			# 读取数据
			self.__model_to_device()

			# 这不不会出问题吧
			self.__model_parallel()

			# 优化器
			optimizer = self.__get_model_optimizer(final_stage_flag=final_stage_flag)

			# 准备训练
			previous_best_acc = -1

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

			# restore training settings
			early_stop_count = 0
			restore_epoch = 0
			scheduler_last_epoch = 0

			if self.restore_flag:
				restore_data = torch.load(restore_path)

				# get scheduler
				scheduler_last_epoch = restore_data['scheduler']['last_epoch']
				early_stop_count = restore_data['early_stop_count']

				# get optimizer
				optimizer.load_state_dict(restore_data['optimizer'])

				# print lr
				for o in optimizer.state_dict()['param_groups']:
					print(o['lr'], end="\t")
				print()

				# to device
				for state in optimizer.state.values():
					for k, v in state.items():
						if torch.is_tensor(v):
							state[k] = v.to(self.device)

				# get best performance
				previous_best_r_1 = restore_data['best performance']

				# get epoch
				restore_epoch = restore_data['epoch']

				print("model is restored from", restore_path)
				print(f'Restore epoch: {restore_epoch}, Previous best R@1: {previous_best_r_1}')
				print("*"*100)

			# prepare dataset
			# all tuples
			train_dataset, val_datasets, test_datasets = self.__get_datasets()

			# avoid warning
			scheduler = None

			for epoch in range(restore_epoch, self.num_train_epochs):
				torch.cuda.empty_cache()

				# whether this epoch get the best model
				this_epoch_best = False

				# 打印一下
				print("*" * 20 + f" {epoch} " + "*" * 20)

				# 训练之前先看初始情况,保存初始best performance
				if epoch == 0 and not self.no_initial_test:
					print("-" * 30 + "initial validation" + "-" * 30)
					val_dataloader = self.__get_dataloader(data=val_datasets, batch_size=self.val_batch_size, shuffle=False, drop_last=False)

					val_loss_tuple, val_acc_tuple = (), ()
					for dataloader in val_dataloader:
						this_val_loss, this_val_acc = self.classify_validate_model(dataloader)
						val_loss_tuple = val_loss_tuple + (this_val_loss, )
						val_acc_tuple = val_acc_tuple + (this_val_acc,)

					_, val_acc = sum_average_tuple(val_acc_tuple)
					_, val_loss = sum_average_tuple(val_loss_tuple)
					print(
						f"Initial eval on Validation Dataset: " + "*" * 30 +
						f"\nval_loss:{val_loss}\tval_acc:{val_acc}%\tprevious best acc:{previous_best_acc}\tfrom rank:{self.local_rank}")

					if val_acc > previous_best_acc:
						previous_best_acc = val_acc
					print("-" * 30 + "initial_test_end" + "-" * 30, end="\n\n")

				# 开始训练
				train_loss = 0.0
				# 计算训练集的acc
				shoot_num = 0
				hit_num = 0

				now_batch_num = 0

				self.model.train()

				train_dataloader = self.__get_dataloader(data=train_dataset, batch_size=self.train_batch_size)[0]

				if self.data_distribute:
					train_dataloader.sampler.set_epoch(epoch)

				# 获取scheduler
				if epoch == restore_epoch:
					t_total = (
									  len(train_dataloader) // self.gradient_accumulation_steps) * self.num_train_epochs

					# if restore training, should pass last epoch, otherwise should not pass this argument
					if self.restore_flag:
						scheduler = get_linear_schedule_with_warmup(optimizer,
																	num_warmup_steps=int(t_total * 0.1),
																	num_training_steps=t_total,
																	last_epoch=scheduler_last_epoch)
						# avoid trying to restore again
						self.restore_flag = False
					else:
						scheduler = get_linear_schedule_with_warmup(optimizer,
																	num_warmup_steps=int(t_total * 0.1),
																	num_training_steps=t_total)

					print(f"Train {self.num_train_epochs} epochs, Block num {self.dataset_split_num}, "
						  f"Accumulate num {self.gradient_accumulation_steps}, Total update {t_total}\n")

				# 开始训练----------------------------------------------------------------------------------------
				# 进度条
				bar = tqdm(train_dataloader, total=len(train_dataloader))

				# 开始训练
				for batch in bar:
					# add model
					if self.model_class == "CrossBERT":
						step_loss, step_shoot_num, step_hit_num = self.__train_step_for_cross(batch=batch,
																							  optimizer=optimizer,
																							  now_batch_num=now_batch_num,
																							  scheduler=scheduler)
					elif self.model_class in ['QAClassifierModel', 'ParallelEncoder']:
						step_loss, step_shoot_num, step_hit_num = \
							self.__train_step_for_qa_input(batch=batch, optimizer=optimizer,
														   now_batch_num=now_batch_num,
														   scheduler=scheduler, final_stage_flag=final_stage_flag)
					else:
						raise Exception("Train step have not supported this model class")

					# 更新一下信息
					train_loss += step_loss.item()

					shoot_num += step_shoot_num
					hit_num += step_hit_num
					now_batch_num += 1

					bar.set_description(
						"epoch {:>3d} loss {:.4f} Acc {:>10d}/{:>10d} = {:.4f}".format(epoch + 1,
																					   train_loss / now_batch_num,
																					   hit_num, shoot_num,
																					   hit_num / shoot_num * 100))

					# 是否评测一次
					# 忽略，变更为一次epoch结束评测一次
					# ......

				val_dataloader = self.__get_dataloader(data=val_datasets, batch_size=self.val_batch_size, shuffle=False, drop_last=False)

				val_loss_tuple, val_acc_tuple = (), ()
				for dataloader in val_dataloader:
					this_val_loss, this_val_acc = self.classify_validate_model(dataloader)
					val_loss_tuple = val_loss_tuple + (this_val_loss,)
					val_acc_tuple = val_acc_tuple + (this_val_acc,)

				_, val_acc = sum_average_tuple(val_acc_tuple)
				_, val_loss = sum_average_tuple(val_loss_tuple)
				print(
					f"{epoch + 1} epoch end eval on Validation Dataset: " + "*" * 30 +
					f"\nval_loss:{val_loss}\tval_acc:{val_acc}%\tprevious best acc:{previous_best_acc}\tfrom rank:{self.local_rank}")

				# 准备存储模型
				postfix = ""
				if not final_stage_flag:
					postfix = "_middle"

				# 存储最优模型
				if val_acc > previous_best_acc:
					previous_best_acc = val_acc

					self.save_model(model_save_path=model_save_path, epoch=epoch, optimizer=optimizer, scheduler=scheduler,
									previous_best_performance=previous_best_acc, early_stop_count=early_stop_count, postfix=postfix)
					this_epoch_best = True


				if not os.path.exists("./last_model/"):
					os.makedirs("./last_model/")

				self.save_model(model_save_path="./last_model/" + self.model_save_prefix +
											 self.model_class + "_" +
											 self.dataset_name, epoch=epoch, optimizer=optimizer, scheduler=scheduler,
								previous_best_performance=previous_best_acc, early_stop_count=early_stop_count, postfix=postfix)

				torch.cuda.empty_cache()
				gc.collect()

				# 是否早停
				if this_epoch_best:
					early_stop_count = 0
				else:
					early_stop_count += 1

					if early_stop_count == early_stop_threshold:
						print("early stop!")
						break

				sys.stdout.flush()

			# 在测试集上做最后的检验
			# 用来修改预测的文件名
			postfix = ""
			if not final_stage_flag:
				postfix = "_middle"

			# 用最好的模型
			self.model = self.load_models(self.model, model_save_path + postfix)

			self.glue_test(test_datasets=test_datasets, postfix=postfix)

			# 如果只是第一阶段的训练，那么还要继续训练
			if not final_stage_flag:
				final_stage_flag = True
			else:
				break

	# classify
	def classify_validate_model(self, dataloader):
		self.model.eval()

		label_target_num = [0] * self.label_num
		label_shoot_num = [0] * self.label_num
		label_hit_num = [0] * self.label_num

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
				if self.model_class in ['QAClassifierModel', 'ParallelEncoder']:
					logits = self.__val_step_for_qa_input(batch)
				elif self.model_class in ['CrossBERT']:
					logits = self.__val_step_for_cross(batch)
				else:
					raise Exception("Val step is not supported for this model class!")

				qa_labels = (batch['label']).to(self.device)

				if index == 0:
					print(f"First logits during validate: {logits[0]}\t Its label is {qa_labels[0]}")
					print(f"Second logits during validate: {logits[1]}\t Its label is {qa_labels[1]}")
					print(f"Third logits during validate: {logits[2]}\t Its label is {qa_labels[2]}")

				loss = cross_entropy_function(logits, qa_labels)
				val_loss += loss.item()

				# 统计命中率
				shoot_num += len(qa_labels)
				for i in range(0, self.label_num):
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

			accuracy = print_classification_result(label_hit_num=label_hit_num, label_shoot_num=label_shoot_num,
												   label_target_num=label_target_num, label_num=self.label_num)

			self.model.train()

		return val_loss, accuracy

	def glue_test(self, test_datasets=None, model_save_path=None, postfix=""):
		# add dataset
		dataset_label_dict = {'mnli': ['entailment', 'neutral', 'contradiction']}
		output_text_name = {'mnli': ['MNLI-m.tsv', 'MNLI-mm.tsv']}

		# create model if necessary
		if self.model is None:
			self.model = self.__create_model()
			self.model = self.load_models(self.model, model_save_path)
			self.model.to(self.device)

		self.model.eval()

		# read datasets if necessary
		if test_datasets is None:
			_, _, test_datasets = self.__get_datasets()

		test_dataloaders = self.__get_dataloader(data=test_datasets, batch_size=self.val_batch_size, shuffle=False, drop_last=False)

		for dataloader_index, this_dataloader in enumerate(test_dataloaders):
			all_prediction = []

			# 开始评测
			with torch.no_grad():
				print(f"------- begin test {self.val_batch_size * len(this_dataloader)} data--------")

				for index, batch in enumerate(this_dataloader):
					this_batch_index = batch['idx']

					# 读取数据
					# add model
					if self.model_class in ['QAClassifierModel', 'ParallelEncoder']:
						logits = self.__val_step_for_qa_input(batch)
					elif self.model_class in ['CrossBERT']:
						logits = self.__val_step_for_cross(batch)
					else:
						raise Exception("Val step is not supported for this model class!")

					_, row_max_indices = logits.topk(k=1, dim=-1)
					this_prediction = [ (this_batch_index[i], item[0]) for i, item in enumerate(row_max_indices)]
					all_prediction += this_prediction

				# write to disk
				if not os.path.exists("./output/"):
					os.makedirs("./output/")

				with open("./output/" + output_text_name[self.dataset_name][dataloader_index] + postfix, "w") as writer:
					tsv_writer = csv.writer(writer, delimiter='\t', lineterminator='\n')

					tsv_writer.writerow(['index', 'prediction'])
					for (index, pre) in all_prediction:
							text_pre = dataset_label_dict[self.dataset_name][pre]
							tsv_writer.writerow([index.item(), text_pre])

				print(f"Result is saved to ./output/{output_text_name[self.dataset_name][dataloader_index] + postfix}")

		self.model.train()

	def save_model(self, model_save_path, optimizer, scheduler, epoch, previous_best_performance, early_stop_count, postfix=""):
		self.model.eval()

		save_path = model_save_path + postfix

		# Only save the model it-self, maybe parallel
		model_to_save = self.model.module if hasattr(self.model, 'module') else self.model

		# 保存模型
		save_state = {'pretrained_bert_path': self.config.pretrained_bert_path,
					  'memory_num': self.memory_num,
					  'model': model_to_save.state_dict(),
					  'optimizer': optimizer.state_dict(),
					  'scheduler': scheduler.state_dict(),
					  'best performance': previous_best_performance,
					  'early_stop_count': early_stop_count,
					  'epoch': epoch + 1}

		torch.save(save_state, save_path)

		print("!" * 60)
		print(f"model is saved at {save_path}")
		print("!" * 60)

		self.model.train()

	@staticmethod
	def load_models(model, load_model_path):
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

	# return must be tuple
	def __get_dataloader(self, data, batch_size, shuffle=True, drop_last=True):
		all_dataloader = ()

		for now_data in data:
			now_data.set_format(type='torch')

			if self.data_distribute:
				sampler = torch.utils.data.distributed.DistributedSampler(now_data, shuffle=shuffle, drop_last=drop_last)
				dataloader = DataLoader(now_data, batch_size=batch_size, sampler=sampler)
			else:
				dataloader = DataLoader(now_data, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)

			all_dataloader = all_dataloader + (dataloader, )

		return all_dataloader

	def __get_datasets(self):
		# prepare dataset
		train_dataset = ()
		# must be tuple
		val_datasets = ()
		test_datasets = ()

		# add model
		if self.model_class in ['QAClassifierModel', 'ParallelEncoder']:
			temp_dataset_process_function = self.__tokenize_qa_classify_data_then_save
			load_prefix = ""
		elif self.model_class in ['CrossBERT']:
			temp_dataset_process_function = self.__tokenize_cross_classify_data_then_save
			load_prefix = "cross_"
		else:
			raise Exception("The dataset for this class is not supported!")

		# add dataset
		if self.dataset_name == 'mnli':
			# have been processed and saved to disk
			if os.path.exists("./dataset/" + load_prefix + "glue_mnli_train"):
				train_dataset = (datasets.load_from_disk("./dataset/" + load_prefix + "glue_mnli_train"), )
				val_datasets = (datasets.load_from_disk("./dataset/" + load_prefix + "glue_mnli_val_matched"),
								datasets.load_from_disk("./dataset/" + load_prefix + "glue_mnli_val_mismatched"), )
				test_datasets = (datasets.load_from_disk("./dataset/" + load_prefix + "glue_mnli_test_matched"),
								 datasets.load_from_disk("./dataset/" + load_prefix + "glue_mnli_test_mismatched"),)
			else:
				complete_dataset = datasets.load_dataset("glue", 'mnli')

				# get train dataset
				train_dataset = (temp_dataset_process_function(data=complete_dataset['train'],
															   save_name="glue_mnli_train",
															   a_column_name="premise",
															   b_column_name="hypothesis",
															   label_column_name='label'),)

				# get val dataset
				validation_matched_dataset = temp_dataset_process_function(data=complete_dataset['validation_matched'],
																		   save_name="glue_mnli_val_matched",
																		   a_column_name="premise",
																		   b_column_name="hypothesis",
																		   label_column_name='label')
				validation_mismatched_dataset = temp_dataset_process_function(
					data=complete_dataset['validation_mismatched'],
					save_name="glue_mnli_val_mismatched",
					a_column_name="premise",
					b_column_name="hypothesis",
					label_column_name='label')

				val_datasets = (validation_matched_dataset, validation_mismatched_dataset,)

				# get test dataset
				test_matched_dataset = temp_dataset_process_function(data=complete_dataset['test_matched'],
																		   save_name="glue_mnli_test_matched",
																		   a_column_name="premise",
																		   b_column_name="hypothesis",
																		   label_column_name='label')
				test_mismatched_dataset = temp_dataset_process_function(
					data=complete_dataset['test_mismatched'],
					save_name="glue_mnli_test_mismatched",
					a_column_name="premise",
					b_column_name="hypothesis",
					label_column_name='label')

				test_datasets = (test_matched_dataset, test_mismatched_dataset,)
		else:
			raise_dataset_error()

		return train_dataset, val_datasets, test_datasets

	def __create_model(self):
		print("---------------------- create model ----------------------")
		# 创建自己的model
		# add model
		if self.model_class in ['QAClassifierModel']:
			model = QAClassifierModel(config=self.config)
		elif self.model_class in ['CrossBERT']:
			model = CrossBERT(config=self.config)
		elif self.model_class in ['ParallelEncoder']:
			model = ParallelEncoder(config=self.config)
		else:
			raise Exception("This model class is not supported for creating!!")

		# 要不要加载现成的模型
		if self.load_model_flag:
			model = self.load_models(model, self.load_model_path)

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
		self.model_save_prefix = args.model_save_prefix

		self.local_rank = args.local_rank
		self.data_parallel = args.data_parallel
		self.data_distribute = args.data_distribute
		self.distill_flag = args.distill
		self.teacher_path = args.teacher_path

		self.num_train_epochs = args.num_train_epochs
		self.gradient_accumulation_steps = args.gradient_accumulation_steps
		self.no_initial_test = args.no_initial_test
		self.load_memory_flag = args.load_memory
		self.first_stage_lr = args.first_stage_lr

		self.composition = args.composition
		self.restore_flag = args.restore
		self.label_num = args.label_num
		self.load_model_path = args.load_model_path

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
		if self.model_class in ['QAClassifierModel']:
			config = QAClassifierModelConfig(len(self.tokenizer),
										 pretrained_bert_path=args.pretrained_bert_path,
										 num_labels=args.label_num,
										 word_embedding_len=word_embedding_len,
										 sentence_embedding_len=sentence_embedding_len,
										 composition=self.composition)
		elif self.model_class in ['CrossBERT']:
			config = CrossBERTConfig(len(self.tokenizer),
									 pretrained_bert_path=args.pretrained_bert_path,
									 num_labels=args.label_num,
									 word_embedding_len=word_embedding_len,
									 sentence_embedding_len=sentence_embedding_len,
									 composition=self.composition)
		elif self.model_class in ['ParallelEncoder']:
			config = ParallelEncoderConfig(len(self.tokenizer),
									pretrained_bert_path=args.pretrained_bert_path,
									num_labels=args.label_num,
									word_embedding_len=word_embedding_len,
									sentence_embedding_len=sentence_embedding_len,
									composition=self.composition)
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
		if self.model_class in ['QAClassifierModel']:
			parameters_dict_list = [
				# 这几个一样
				{'params': model.bert_model.parameters(), 'lr': 5e-5},
				# 这几个一样
				{'params': model.self_attention_weight_layer.parameters(), 'lr': 5e-5},
				{'params': model.value_layer.parameters(), 'lr': 5e-5},
				# 这个不设定
				{'params': model.classifier.parameters(), 'lr': 5e-5}
			]
		elif self.model_class in ['CrossBERT']:
			parameters_dict_list = [
				# 这几个一样
				{'params': model.bert_model.parameters(), 'lr': 5e-5},
			]
		elif self.model_class in ['ParallelEncoder']:
			parameters_dict_list = [
				# 这几个一样
				{'params': model.bert_model.parameters(), 'lr': 5e-5},
				{'params': model.composition_layer.parameters(), 'lr': 5e-5},
				{'params': model.decoder.parameters(), 'lr': 5e-5},
				{'params': model.classifier.parameters(), 'lr': 5e-5},
			]
		else:
			raise Exception("No optimizer supported for this model class!")

		# 对于那些有两段的,第一段训练参数不太一样
		if not final_stage_flag:
			if self.model_class in ['ParallelEncoder']:
				parameters_dict_list = [
					# 这几个一样
					{'params': model.decoder.parameters(), 'lr': 5e-5},
					{'params': model.composition_layer.parameters(), 'lr': 1e-4},
					{'params': model.classifier.parameters(), 'lr': 1e-4},
				]
			else:
				raise Exception("Have Two Stage But No optimizer supported for this model class!")

		# if to restore, it will be printed in other places
		if not self.restore_flag:
			print(parameters_dict_list)
		optimizer = torch.optim.Adam(parameters_dict_list, lr=5e-5)
		print("*"*30)

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

	# 输入为QA，而非title.body.answer的模型的训练步
	def __train_step_for_qa_input(self, batch, optimizer, now_batch_num, scheduler, final_stage_flag):
		cross_entropy_function = nn.CrossEntropyLoss()

		# 读取数据
		a_input_ids = (batch['a_input_ids']).to(self.device)
		a_token_type_ids = (batch['a_token_type_ids']).to(self.device)
		a_attention_mask = (batch['a_attention_mask']).to(self.device)

		b_input_ids = (batch['b_input_ids']).to(self.device)
		b_token_type_ids = (batch['b_token_type_ids']).to(self.device)
		b_attention_mask = (batch['b_attention_mask']).to(self.device)

		qa_labels = (batch['label']).to(self.device)

		# 得到模型的结果
		logits = self.model(
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
			# add model
			# only update memory in embeddings
			if self.model_class in ['QAMemory'] and not final_stage_flag:
				self.model.embeddings.weight.grad[:self.origin_voc_size] *= 0.0

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
	def __train_step_for_cross(self, batch, optimizer, now_batch_num, scheduler):
		cross_entropy_function = nn.CrossEntropyLoss()

		# 读取数据
		input_ids = (batch['input_ids']).to(self.device)
		token_type_ids = (batch['token_type_ids']).to(self.device)
		attention_mask = (batch['attention_mask']).to(self.device)

		qa_labels = (batch['label']).to(self.device)

		# 得到模型的结果
		logits = self.model(
			input_ids=input_ids, token_type_ids=token_type_ids,
			attention_mask=attention_mask)

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

	def __val_step_for_qa_input(self, batch):
		# 读取数据
		a_input_ids = (batch['a_input_ids']).to(self.device)
		a_token_type_ids = (batch['a_token_type_ids']).to(self.device)
		a_attention_mask = (batch['a_attention_mask']).to(self.device)

		b_input_ids = (batch['b_input_ids']).to(self.device)
		b_token_type_ids = (batch['b_token_type_ids']).to(self.device)
		b_attention_mask = (batch['b_attention_mask']).to(self.device)

		with torch.no_grad():
			# 得到模型的结果
			logits = self.model(
				a_input_ids=a_input_ids, a_token_type_ids=a_token_type_ids,
				a_attention_mask=a_attention_mask,
				b_input_ids=b_input_ids, b_token_type_ids=b_token_type_ids,
				b_attention_mask=b_attention_mask)

		return logits

	def __val_step_for_cross(self, batch):
		# 读取数据
		input_ids = (batch['input_ids']).to(self.device)
		token_type_ids = (batch['token_type_ids']).to(self.device)
		attention_mask = (batch['attention_mask']).to(self.device)

		with torch.no_grad():
			# 得到模型的结果
			logits = self.model(input_ids=input_ids, token_type_ids=token_type_ids,
								attention_mask=attention_mask)

		return logits

	def __tokenize_qa_classify_data_then_save(self, data, save_name, a_column_name="premise", b_column_name="hypothesis", label_column_name='label'):
		# 读取数据到内存
		all_a_text = data[a_column_name]
		all_b_text = data[b_column_name]
		all_labels = data[label_column_name]
		all_index = data['idx']

		# tokenize
		encoded_a_text = self.tokenizer(
			all_a_text, padding=True, verbose=False, add_special_tokens=True,
			truncation=True, max_length=self.text_max_len, return_tensors='pt')

		encoded_b_text = self.tokenizer(
			all_b_text, padding=True, verbose=False, add_special_tokens=True,
			truncation=True, max_length=self.text_max_len, return_tensors='pt')

		all_labels = all_labels

		items_dic = {'a_input_ids': encoded_a_text['input_ids'],
					 'a_token_type_ids': encoded_a_text['token_type_ids'],
					 'a_attention_mask': encoded_a_text['attention_mask'],
					 'b_input_ids': encoded_b_text['input_ids'],
					 'b_token_type_ids': encoded_b_text['token_type_ids'],
					 'b_attention_mask': encoded_b_text['attention_mask'],
					 'label': torch.tensor(all_labels),
					 'idx': torch.tensor(all_index)}

		dataset = Dataset.from_dict(items_dic)

		if not os.path.exists("./dataset"):
			os.makedirs("./dataset")

		dataset.save_to_disk("./dataset/" + save_name)

		return dataset

	def __tokenize_cross_classify_data_then_save(self, data, save_name, a_column_name="premise", b_column_name="hypothesis", label_column_name='label'):
		# 读取数据到内存
		all_a_text = data[a_column_name]
		all_b_text = data[b_column_name]
		all_labels = data[label_column_name]
		all_index = data['idx']

		# tokenize
		all_texts = []
		for index, a_text in enumerate(all_a_text):
			all_texts.append((a_text, all_b_text[index]))

		# tokenize
		encoded_texts = self.tokenizer(
			all_texts, padding=True, verbose=False, add_special_tokens=True,
			truncation=True, max_length=self.text_max_len, return_tensors='pt')

		all_labels = all_labels

		items_dic = {'input_ids': encoded_texts['input_ids'],
					 'token_type_ids': encoded_texts['token_type_ids'],
					 'attention_mask': encoded_texts['attention_mask'],
					 'label': torch.tensor(all_labels),
					 'idx': torch.tensor(all_index)}

		dataset = Dataset.from_dict(items_dic)

		if not os.path.exists("./dataset/"):
			os.makedirs("./dataset/")

		dataset.save_to_disk("./dataset/" + "cross_" + save_name)

		return dataset


def sum_average_tuple(in_tuple):
	sum_result = 0
	for i in in_tuple:
		sum_result += i
	return sum_result, sum_result/len(in_tuple)


def raise_dataset_error():
	raise Exception("This dataset is not supported now!")


def print_classification_result(label_hit_num, label_shoot_num, label_target_num, label_num):
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

