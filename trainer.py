# %%
import datasets
from regex import D
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
import re

from TBA_model_lib import BasicConfig, BasicModel, InputMemorySelfAttConfig, \
	InputMemorySelfAtt, PureMemorySelfAttConfig, PureMemorySelfAtt, OneSupremeMemory, OneSupremeMemoryConfig, BasicTopicModel, BasicTopicConfig
from QA_model_lib import QAModel, QAModelConfig, CrossBERT, CrossBERTConfig, ADecoder, ADecoderConfig, QATopicModel, QATopicConfig, QATopicMemoryModel
from my_dataset import TBAClassifyDataset, MLMDataset, QAMemClassifyDataset, QAClassifyDataset, CrossClassifyDataset, VaeSignleTextDataset, TBATopicClassifyDataset, QATopicClassifyDataset
from vae import VAE
from gensim.corpora import Dictionary
from nltk.corpus import stopwords


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

		print(f"local rank: {local_rank}")
		self.device = torch.device(f'cuda:{local_rank}')

		# 读取tokenizer-------------------------------------------------------------------
		tokenizer_path = args.pretrained_bert_path.replace("/", "_")
		tokenizer_path = tokenizer_path.replace("\\", "_")
		if tokenizer_path[0] != "_":
			tokenizer_path = "_" + tokenizer_path

		# add model
		# For QAMemory, memory tokens are contained in tokenizer as special tokens
		if self.model_class in ['QAMemory']:
			tokenizer_path += "_" + str(self.memory_num) + "_" + self.model_class

		if self.dataset_name in ['so_python', 'so_java']:
			tokenizer_path += "_" + str("SC_EC")

		# read from disk or save to disk
		if os.path.exists("./tokenizer/" + tokenizer_path):
			self.tokenizer = AutoTokenizer.from_pretrained("./tokenizer/" + tokenizer_path)
		else:
			print("first time use this tokenizer, downloading...")
			self.tokenizer = AutoTokenizer.from_pretrained(args.pretrained_bert_path)
			tokenizer_config = AutoConfig.from_pretrained(args.pretrained_bert_path)

			if self.dataset_name in ['so_python', 'so_java']:
				special_tokens_dict = {'additional_special_tokens': ["[SC]", "[EC]"]}

				print("-" * 30)
				print(f"previous token num:{len(self.tokenizer)}")
				self.tokenizer.add_special_tokens(special_tokens_dict)
				print(f"now token num:{len(self.tokenizer)}")
				print("-" * 30 + "\n")

			# add model
			# add memory tokens to tokenizer
			if self.model_class in ['QAMemory']:
				memory_token_list = []

				# maybe order that answer tokens are behind question tokens is important? I forgot.
				for i in range(self.memory_num):
					memory_token_list.append('<QMEM' + str(i) + '>')
				for i in range(self.memory_num):
					memory_token_list.append('<AMEM' + str(i) + '>')

				special_tokens_dict = {'additional_special_tokens': memory_token_list}

				print("-" * 30)
				print(f"previous token num:{len(self.tokenizer)}")
				self.tokenizer.add_special_tokens(special_tokens_dict)
				print(f"now token num:{len(self.tokenizer)}")
				print("-" * 30 + "\n")

			self.tokenizer.save_pretrained("./tokenizer/" + tokenizer_path)
			tokenizer_config.save_pretrained("./tokenizer/" + tokenizer_path)

		if self.dataset_name in ['so_python', 'so_java']:
			self.origin_voc_size = len(self.tokenizer) - 2

		if self.model_class in ['QAMemory']:
			self.origin_voc_size = len(self.tokenizer) - self.memory_num*2

		# 获得模型配置-------------------------------------------------------------------
		self.dictionary = None
		if self.model_class in ['BasicTopicModel', 'QATopicModel', 'QATopicMemoryModel']:
			self.train_vae(args.latent_dim)

		if config is None:
			self.config = self.__read_args_for_config(args)
		else:
			self.config = config

		# instance attribute
		self.model = None
		self.teacher_model = None

	def train(self, train_two_stage_flag, only_final=False):
		# 用来判断是哪一阶段的训练
		final_stage_flag = not train_two_stage_flag

		model_save_name = self.model_save_prefix + self.model_class + "_" + self.dataset_name
		# best model save path
		model_save_path = self.save_model_dict + "/" + model_save_name
		# last model save path
		last_model_save_path = self.last_model_dict + "/" + model_save_name
		# pretrained memory save path
		memory_save_name = self.memory_save_prefix + "_" + self.dataset_name

		# two stage training but begin with final stage
		if only_final:
			final_stage_flag = True

		while True:
			# 创建模型，根据 model_class 的选择
			self.model = self.__create_model()
			
			# 读取事先训练好的
			if self.model_class in ['BasicTopicModel', 'QATopicModel', 'QATopicMemoryModel']:
				self.model.vae.load_state_dict(torch.load("./model/vae/" + self.dataset_name + "_" + str(self.latent_dim))['vae'])
	
			# 读取预训练好的memory，适用于需要memory的模型，如 QAmemory
			if self.load_memory_flag:
				self.load_pretrained_memory(memory_save_name)

			# 如果要进行第二阶段训练，那需要先读取上一阶段最好的model, for normal two stage（只对两阶段模型有用）
			if final_stage_flag and train_two_stage_flag:
				self.model = self.load_models(self.model, model_save_path + "_middle")

			# load model to restore training
			restore_path = self.get_restore_path(model_save_name=model_save_name, final_stage_flag=final_stage_flag)
			if self.restore_flag:
				self.model = self.load_models(self.model, restore_path)

			self.__model_to_device()

			# topic_words = self.model.vae.show_topic_words(dictionary=self.dictionary, device=self.device)
			# for t in topic_words:
			# 	print(t)
			# print("*"*50)

			# 这不不会出问题吧，设置多 GPU，应该不需要用到
			self.__model_parallel()

			# 优化器，调参用的
			optimizer = self.__get_model_optimizer(final_stage_flag=final_stage_flag)

			# 准备训练，最优模型先设置为无
			previous_best_r_1 = -1

			if train_two_stage_flag:
				if final_stage_flag:
					print("~"*60 + " begin final stage train " + "~"*60)
				else:
					print("~"*60 + " begin first stage train " + "~"*60)
			else:
				print("~"*60 + " begin one stage train " + "~"*60)

			# 设置早停变量，如果超过 10 轮没有更新最优值，则结束
			if final_stage_flag:
				early_stop_threshold = 5
			else:
				early_stop_threshold = 5

			# restore training settings，保存模型用于恢复
			early_stop_count, restore_epoch, scheduler_last_epoch, previous_best_r_1 = \
				self.restore_settings(optimizer, restore_path, previous_best_r_1)

			# avoid warning
			scheduler = None

			# 选择 cross，bi，或者其他 BERT
			train_step_function = self.select_train_step_function()

			for epoch in range(restore_epoch, self.num_train_epochs):
				torch.cuda.empty_cache()

				# whether this epoch get the best model
				this_epoch_best = False

				# 打印一下
				print("*" * 20 + f" {epoch} " + "*" * 20)

				# 训练之前先看初始情况,保存初始best performance
				if epoch == restore_epoch and not self.no_initial_test:
					print("-" * 30 + "initial validation" + "-" * 30)

					"""
					In previous experiments, I do testing (ranking) and dev (classification) after each epoch,
					And stopping training based on testing result. This practice is totally wrong!!!
					From now, I decide to abort previous dev datasets and using new ranking data as dev datasets!!!
					"""

					# test_data = datasets.load_from_disk("./" + self.dataset_name + "/string_val.dataset")
					# test_data = test_data.shuffle(seed=None)
					# val_dataloader = self.__get_dataloader(data=test_data, batch_size=self.val_batch_size,
					# 									   split_index=0, split_num=1)
					# val_loss, val_acc = self.classify_validate_model(xxxval_dataloader)
					# print(val_loss, val_acc)

					# add model
					# if self.model_class in ['QAMemory', 'QAModel', 'ADecoder']:
					# 	r_1 = self.ranking_qa_input()
					# elif self.model_class in ['CrossBERT']:
					# 	r_1 = self.ranking_cross()
					# # TBA
					# else:
					# 	r_1 = self.ranking()

					if self.model_class in ['ADecoder', 'CrossBERT']:
						raise Exception("Validation is not supported for this model class yet!")
					else:
						this_best_performance = self.do_val()

					if this_best_performance > previous_best_r_1:
						previous_best_r_1 = this_best_performance

					print("-" * 30 + "initial_test_end" + "-" * 30, end="\n\n")

				# 开始训练
				train_loss = 0.0
				# 计算训练集的R@1
				shoot_num, hit_num, now_batch_num = 0, 0, 0

				# next_val_num = 1

				# 读取训练数据
				train_data = datasets.load_from_disk("./" + self.dataset_name + "/string_train.dataset")
				train_data = train_data.shuffle(seed=None)

				self.model.train()

				# 逐块训练
				# 动机是我考虑到数据可能过大，怕占内存，我的tokenizing没有提前做，或许应该尝试下
				for split_index in range(self.dataset_split_num):
					# 获取这块数据的dataloader
					print("-" * 10 + f"data block {split_index + 1}/{self.dataset_split_num}" + "-" * 10)

					train_block_dataloader = self.__get_dataloader(data=train_data, batch_size=self.train_batch_size,
																   split_index=split_index,
																   split_num=self.dataset_split_num)

					if self.data_distribute:
						train_block_dataloader.sampler.set_epoch(epoch)

					# 获取scheduler
					if epoch == restore_epoch and split_index == 0:
						scheduler = self.get_scheduler(optimizer=optimizer,
													   scheduler_last_epoch=scheduler_last_epoch,
													   train_dataloader=train_block_dataloader)

					# 开始训练----------------------------------------------------------------------------------------
					# 进度条
					bar = tqdm(train_block_dataloader, total=len(train_block_dataloader))

					# 开始训练
					for batch in bar:
						step_loss, step_shoot_num, step_hit_num = train_step_function(batch=batch, optimizer=optimizer,
																					  now_batch_num=now_batch_num,
																					  scheduler=scheduler,
																					  final_stage_flag=final_stage_flag)

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

						"""
						曾经的我，因为不自信，在每个epoch中都要测试好多次，但现在，我已经变了。
						"""
						# # 要不要是时候评测一下呢，一个epoch评测 val_num_each_epoch 次
						# val_interval = (
						# 					   (
						# 							   len(train_block_dataloader) - 1) * self.dataset_split_num - self.gradient_accumulation_steps) // self.val_num_each_epoch
						#
						# if now_batch_num // val_interval >= next_val_num and next_val_num <= self.val_num_each_epoch:
						# 	# 还有梯度没更新的话，会占着显存，影响val batch size，所以就下步val了
						# 	if now_batch_num % self.gradient_accumulation_steps != 0:
						# 		continue
						#
						# 	# 更新这一epoch中已经评测的次数
						# 	next_val_num += 1
						#
						# 	# 获得评测结果
						# 	test_data = datasets.load_from_disk("./" + self.dataset_name + "/string_val.dataset")
						# 	test_data = test_data.shuffle(seed=None)
						#
						# 	val_dataloader = self.__get_dataloader(data=test_data, batch_size=self.val_batch_size,
						# 										   split_index=0, split_num=1)
						#
						# 	val_loss, val_acc = self.classify_validate_model(val_dataloader)
						# 	# add model
						# 	if self.model_class in ['QAMemory', 'QAModel', 'ADecoder']:
						# 		r_1 = self.ranking_qa_input()
						# 	elif self.model_class in ['CrossBERT']:
						# 		r_1 = self.ranking_cross()
						# 	else:
						# 		r_1 = self.ranking()
						# 	print(
						# 		f"{epoch + 1} epoch middle eval: " + "*" * 30 +
						# 		f"\n val_loss:{val_loss}\t val_acc{val_acc}\tR@1:{r_1}\t previous best R@1:{previous_best_r_1}\t from rank:{self.local_rank}")
						#
						# 	# 模型是否比之前优秀
						# 	if r_1 > previous_best_r_1:
						# 		previous_best_r_1 = r_1
						# 		this_epoch_best = True
						#
						# 		# 保存模型
						# 		postfix = ""
						# 		if not final_stage_flag:
						# 			postfix = "_middle"
						#
						# 		self.save_model(model_save_path=model_save_path, epoch=epoch, optimizer=optimizer,
						# 						scheduler=scheduler, previous_best_performance=previous_best_r_1,
						# 						postfix=postfix)
						#
						# 	gc.collect()
						# 	del val_dataloader, test_data
						# 	gc.collect()

					gc.collect()

				this_best_r_1 = self.do_val()

				postfix = ""
				if not final_stage_flag:
					postfix = "_middle"

				# 存储最优模型
				if this_best_r_1 > previous_best_r_1:
					previous_best_r_1 = this_best_r_1
					this_epoch_best = True

					self.save_model(model_save_path=model_save_path + postfix, epoch=epoch, optimizer=optimizer,
									scheduler=scheduler,
									previous_best_performance=this_best_r_1,
									early_stop_count=early_stop_count)

				# 存储最新的模型
				self.save_model(model_save_path=last_model_save_path + postfix, epoch=epoch, optimizer=optimizer,
								scheduler=scheduler, previous_best_performance=previous_best_r_1,
								early_stop_count=early_stop_count, postfix=postfix)

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

			postfix = ""
			if not final_stage_flag:
				postfix = "_middle"

			# 用之前保存的最好的模型
			self.model = self.load_models(self.model, model_save_path + postfix)

			this_stage_result = self.do_test()
			print("#" * 15 + f" This stage result is {this_stage_result}. " + "#" * 15)

			# 如果只是第一阶段的训练，那么还要继续训练
			if not final_stage_flag:
				final_stage_flag = True
			else:
				break

	def train_vae(self, latent_dim, in_vae=None):
		
		# self.dictionary = Dictionary().load("./" + self.dataset_name + "/vae_dictionary")
		# print(len(self.dictionary))

		# vae = VAE(bow_dim=len(self.dictionary), n_topic=latent_dim).to(self.device)
		# vae.load_state_dict(torch.load("./model/vae/" + self.dataset_name + "_" + str(latent_dim))['vae'])

		
		# topic_words = vae.show_topic_words(dictionary=self.dictionary, device=self.device)
		# print(len(topic_words), len(topic_words[0]))
		# for t in topic_words:
		# 	print(t)
		# exit()

		if os.path.exists("./" + self.dataset_name + "/vae_dictionary") and os.path.exists("./model/vae/" + self.dataset_name + "_" + str(latent_dim)):
			self.dictionary = Dictionary().load("./" + self.dataset_name + "/vae_dictionary")
			print("Pretrained Dict is loaded & Pretrained VAE exists!")
			return

		train_data = datasets.load_from_disk("./" + self.dataset_name + "/string_train.dataset")
		train_data = train_data.shuffle(seed=None)
		
		# 读取数据到内存
		old_all_titles = train_data['title']
		old_all_bodies = train_data['body']
		old_all_answers = train_data['answers']
		
		GOOD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')

		# 进行文本处理
		all_bodies = []
		for t, d in zip(old_all_titles, old_all_bodies):
			new_t = ''.join([i for i in t if not i.isdigit()])
			new_d = ''.join([i for i in d if not i.isdigit()])
			this_txt = new_t + " " + new_d
   
			this_txt = re.sub(r"[\.\[\]()=]", " ", this_txt)
			all_bodies.append(GOOD_SYMBOLS_RE.sub('', this_txt))
   
		# for i in range(10):
		# 	print(all_bodies[i])
		# 	print("*"*50)
		# 	print()
		# exit()
  
		all_answers = []
		for d in old_all_answers:
			this_txt = ''.join([i for i in d if not i.isdigit()])

			this_txt = re.sub(r"[\.\[\]()=]", " ", this_txt)
			
			all_answers.append(GOOD_SYMBOLS_RE.sub('', this_txt))

		if os.path.exists("./" + self.dataset_name + "/vae_dictionary"):
			self.dictionary = Dictionary().load("./" + self.dataset_name + "/vae_dictionary")
		else:
			# 创建词汇表
			self.dictionary = Dictionary()

			for documents in [all_bodies, all_answers]:
				split_documents = [d.split() for d in documents]

				self.dictionary.add_documents(split_documents)

			# 过滤一些词
			if self.dataset_name in ["so_python", "so_java"]:
				below_num = 40
			else:
				below_num = 20

			self.dictionary.filter_extremes(no_below=below_num, no_above=0.5, keep_n=None)

			# remove stopwords
			eng_stopwords = list(stopwords.words('english'))
			target_token = self.dictionary.doc2idx(eng_stopwords)
			self.dictionary.filter_tokens(bad_ids=target_token)

			self.dictionary.compactify()
   
		print(f"[Voc size is {len(self.dictionary)}].")
		
		# because id2token is empty by default, it is a bug.
		self.dictionary.id2token = {v:k for k,v in self.dictionary.token2id.items()}

		# convert the bodies and the answers into BO representation
		q_bows = []
		valid_q_docs = []
		
		a_bows = []
		valid_a_docs = []

		for d, a in zip(all_bodies, all_answers):
			split_d = d.split()
			d_bow = self.dictionary.doc2bow(split_d)

			split_a = a.split()
			a_bow = self.dictionary.doc2bow(split_a)

			if d_bow != [] and a_bow != []:
				valid_q_docs.append(d)
				q_bows.append(d_bow)

				valid_a_docs.append(a)
				a_bows.append(a_bow)

		print(f"q: {len(q_bows)}, {len(valid_q_docs)},\ta: {len(a_bows)}, {len(valid_a_docs)} \torigin: {len(all_bodies)}")

		# save both dictionary and docs
		self.dictionary.save("./" + self.dataset_name + "/vae_dictionary")
		self.dictionary.save_as_text("./" + self.dataset_name + "/vae_text_dictionary")
		
		# 将question和answer混合训练
		q_num = len(valid_q_docs)
		a_num = len(valid_a_docs)

		train_docs = valid_q_docs[:int(0.9*q_num)] + valid_a_docs[:int(0.9*a_num)]
		train_bows = q_bows[:int(0.9*q_num)] + a_bows[:int(0.9*a_num)]

		eval_docs = valid_q_docs[int(0.9*q_num):] + valid_a_docs[int(0.9*a_num):]
		eval_bows = q_bows[int(0.9*q_num):] + a_bows[int(0.9*a_num):]

		train_data = VaeSignleTextDataset(docs=train_docs, bows=train_bows, voc_size=len(self.dictionary))
		eval_data = VaeSignleTextDataset(docs=eval_docs, bows=eval_bows, voc_size=len(self.dictionary))

		train_dataloader = DataLoader(train_data, batch_size=128, shuffle=True, num_workers=4, drop_last=True)

		# 创建模型
		if in_vae is None:
			print("VAE NUM is ", latent_dim)
			vae = VAE(bow_dim=len(self.dictionary), n_topic=latent_dim).to(self.device)
		else:
			vae = in_vae

		optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)
		optimizer.zero_grad()
		
		t_total = len(train_dataloader) * 50
		scheduler = get_linear_schedule_with_warmup(optimizer,
													num_warmup_steps=int(t_total * 0.1),
													num_training_steps=t_total)

		print(f"[Train data len is {len(train_data)}. Eval data len is {len(eval_data)}]")

		previous_min_loss = np.inf
		early_stop_threshold = 5
		early_stop_count = 0

		for epoch in range(50):
			print("*"*45 + f" EPOCH: {epoch} " + "*"*45)

			train_dataloader = DataLoader(train_data, batch_size=64, shuffle=True, num_workers=4, drop_last=True)
			vae.train()

			bar = tqdm(train_dataloader, total=len(train_dataloader))
   
			for iter, data in enumerate(bar):
				bows = data.to(self.device)

				# print(bows.shape)
				# print(torch.max(torch.max(bows, 1)[0]))
				# exit()

				bows_recon, mus, log_vars = vae(bows, lambda x: torch.softmax(x, dim=1))

				logsoftmax = torch.log_softmax(bows_recon, dim=1)
				rec_loss = -1.0 * torch.sum(bows * logsoftmax)
		
				# rec_loss = torch.nn.functional.mse_loss(logsoftmax, torch.softmax(bows, dim=1))

				kl_div = -0.5 * torch.sum(1 + log_vars - mus.pow(2) - log_vars.exp())
				# kl_div = torch.mean(-0.5 * torch.sum(1 + log_vars - mus.pow(2) - log_vars.exp()))

				loss = rec_loss + kl_div

				loss.backward()

				optimizer.step()
				optimizer.zero_grad()	
				scheduler.step()

			# do eval
			eval_dataloader = DataLoader(eval_data, batch_size=128, shuffle=False, num_workers=4, drop_last=True)
			vae.eval()

			all_loss = 0.0
			for data in eval_dataloader:
				bows = data.to(self.device)

				bows_recon, mus, log_vars = vae(bows, lambda x: torch.softmax(x, dim=1))

				logsoftmax = torch.log_softmax(bows_recon, dim=1)
				rec_loss = -1.0 * torch.sum(bows * logsoftmax)
		
				# rec_loss = torch.nn.functional.mse_loss(logsoftmax, torch.softmax(bows, dim=1))

				kl_div = -0.5 * torch.sum(1 + log_vars - mus.pow(2) - log_vars.exp())
				# kl_div = torch.mean(-0.5 * torch.sum(1 + log_vars - mus.pow(2) - log_vars.exp()))

				loss = rec_loss + kl_div

				all_loss += loss.item()
			
			if previous_min_loss > all_loss:
				print("*"*50)
				print(f"NEW MIN LOSS: {all_loss}, PREVIOUS LOSS: {previous_min_loss}")

				previous_min_loss = all_loss
				early_stop_count = 0
				
				# save model
				save_state = {'vae': vae.state_dict()}
				torch.save(save_state, "./model/vae/" + self.dataset_name + "_" + str(latent_dim))
				print(f"Model is saved at ./model/vae/{self.dataset_name }_" + str(latent_dim))

				print("*"*50)
				topic_words = vae.show_topic_words(dictionary=self.dictionary, device=self.device)
				temp_count = 0
				for t in topic_words:
					print(t)
					if temp_count == 9:
						break
					temp_count += 1

				print("*"*50)
				
			else:
				early_stop_count += 1

				if early_stop_count == early_stop_threshold:
					break
		
		print()
		self.dictionary = Dictionary().load("./" + self.dataset_name + "/vae_dictionary")

		topic_words = vae.show_topic_words(dictionary=self.dictionary, device=self.device)
		for t in topic_words:
			print(t)
		print("*"*50)

		print("*"*45 + "VAE END!" + "*"*45)

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

		# 创建模型
		print("create model ...", end=" ")
		model = BertForMaskedLM.from_pretrained(self.config.pretrained_bert_path)
		model.resize_token_embeddings(len(self.tokenizer))
		model.to(self.device)
		model.train()
		print("end")
		print("-" * 30)

		# 获得optimizer，只训练那几个embedding
		# 节省显存和时间
		for n, p in model.named_parameters():
			if 'cls' in n or 'embeddings' in n:
				continue
			else:
				p.requires_grad = False

		model_embeddings = model.get_input_embeddings()
		print(f"embedding shape: {model_embeddings.weight.shape}")
		print(f"model_embeddings is model.cls.predictions.decoder: {model_embeddings.weight is model.cls.predictions.decoder.weight}")

		parameters_dict_list = [
			{'params': model_embeddings.parameters(), 'lr': 0.3},
			# {'params': model.cls.parameters(), 'lr': 1e-4},
		]
		optimizer = torch.optim.Adam(parameters_dict_list)

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
											  shuffle=True, drop_last=True, collate_fn=data_collator)

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
					# 读取数据
					input_ids = (batch['input_ids']).to(self.device)
					token_type_ids = (batch['token_type_ids']).to(self.device)
					attention_mask = (batch['attention_mask']).to(self.device)
					label = (batch['labels']).to(self.device)

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
													 drop_last=True, collate_fn=data_collator)

						# 获得评测结果
						model.eval()
						val_loss = 0.0

						for val_batch in test_dataloader:
							# 读取数据
							input_ids = (val_batch['input_ids']).to(self.device)
							token_type_ids = (val_batch['token_type_ids']).to(self.device)
							attention_mask = (val_batch['attention_mask']).to(self.device)
							val_label = (val_batch['labels']).to(self.device)

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
										  'embedding': model_embeddings.state_dict(),
										  'optimizer': optimizer.state_dict(),
										  'scheduler': scheduler.state_dict(),
										  'min loss': previous_min_loss,
										  'epoch': epoch + 1}

							torch.save(save_state, "./model/pretrained_memory/" + memory_save_name)
							print(f"model saved at ./model/pretrained_memory/{memory_save_name}!!!")

							this_epoch_best = True

						model.train()

				# 保险起见删除数据
				gc.collect()
				del data_block, train_dataset, train_dataloader
				gc.collect()

			# epoch结束，保存最新的模型
			model.eval()
			if not os.path.exists("./last_model/pretrained_memory/"):
				os.makedirs("./last_model/pretrained_memory/")

			# Only save the model it-self, maybe parallel
			# model_to_save = model.module if hasattr(model, 'module') else model

			save_state = {'pretrained_bert_path': self.config.pretrained_bert_path,
						  'memory_num': self.memory_num,
						  'memory_start_index': voc_size,
						  'embedding_shape': model_embeddings.weight.shape,
						  'embedding': model_embeddings.state_dict(),
						  'optimizer': optimizer.state_dict(),
						  'scheduler': scheduler.state_dict(),
						  'min loss': previous_min_loss,
						  'epoch': epoch + 1}

			torch.save(save_state, "./last_model/pretrained_memory/" + memory_save_name)
			print(f"model saved at ./last_model/pretrained_memory/{memory_save_name}!!!")
			model.train()

			# 进行早停操作
			if not this_epoch_best:
				worse_epoch_count += 1
			else:
				worse_epoch_count = 0

			if worse_epoch_count == early_stop_threshold:
				print("training finished!!!")
				break

	# val use first 50% of previous test data
	def do_val(self):
		print("--------------------- begin validation -----------------------")
		# 稍加处理一下数据，把数据都存在元祖里
		data_from_path = "./" + self.dataset_name + "/eva.dataset"
		evaluation_data = datasets.load_from_disk(data_from_path)

		# 汇总下数据
		evaluation_qa_pairs = []

		evaluation_title = evaluation_data['title']
		evaluation_body = evaluation_data['body']
		evaluation_answers = evaluation_data['answers']

		# 权宜之计，懒得处理新的排序数据了，只好分一半之前的测试集当验证集，有机会改
		begin_index = 0
		end_index = len(evaluation_title) // 2

		if self.model_class in ['QAModel']:
			concatenated_question = []
			for t, b in zip(evaluation_title, evaluation_body):
				concatenated_question.append(t + " " + b)

			for row_index in range(begin_index, end_index):
				evaluation_qa_pairs.append((concatenated_question[row_index], evaluation_answers[row_index]))
			
			# pass data into ranking method
			return self.ranking_qa_input(evaluation_qa_pairs)
		elif self.model_class in ['QATopicModel', 'QATopicMemoryModel']:
			concatenated_question = []
			for t, b in zip(evaluation_title, evaluation_body):
				concatenated_question.append(t + " " + b)

			for row_index in range(begin_index, end_index):
				evaluation_qa_pairs.append((concatenated_question[row_index], evaluation_answers[row_index]))
			
			# pass data into ranking method
			return self.ranking_qa_topic_input(evaluation_qa_pairs)
		else:
			for row_index in range(begin_index, end_index):
				evaluation_qa_pairs.append((evaluation_title[row_index], evaluation_body[row_index],
											evaluation_answers[row_index]))

			# pass data into ranking method
			return self.ranking(evaluation_qa_pairs)

	# val use last 50% of previous test data
	def do_test(self):
		print("--------------------- begin testing -----------------------")
		# 稍加处理一下数据，把数据都存在元祖里
		data_from_path = "./" + self.dataset_name + "/eva.dataset"
		evaluation_data = datasets.load_from_disk(data_from_path)

		# 汇总下数据
		evaluation_qa_pairs = []

		evaluation_title = evaluation_data['title']
		evaluation_body = evaluation_data['body']
		evaluation_answers = evaluation_data['answers']

		# 权宜之计，懒得处理新的排序数据了，只好分一半之前的测试集当验证集，有机会改
		begin_index = len(evaluation_title) // 2
		end_index = len(evaluation_title)

		if self.model_class in ['QAModel']:
			concatenated_question = []
			for t, b in zip(evaluation_title, evaluation_body):
				concatenated_question.append(t + " " + b)

			for row_index in range(begin_index, end_index):
				evaluation_qa_pairs.append((concatenated_question[row_index], evaluation_answers[row_index]))

			# pass data into ranking method
			return self.ranking_qa_input(evaluation_qa_pairs)
		elif self.model_class in ['QATopicModel', 'QATopicMemoryModel']:
			concatenated_question = []
			for t, b in zip(evaluation_title, evaluation_body):
				concatenated_question.append(t + " " + b)

			for row_index in range(begin_index, end_index):
				evaluation_qa_pairs.append((concatenated_question[row_index], evaluation_answers[row_index]))
			
			# pass data into ranking method
			return self.ranking_qa_topic_input(evaluation_qa_pairs)
		else:
			for row_index in range(begin_index, end_index):
				evaluation_qa_pairs.append((evaluation_title[row_index], evaluation_body[row_index],
											evaluation_answers[row_index]))

			# pass data into ranking method
			return self.ranking(evaluation_qa_pairs)

	def only_do_test(self):
		model_save_name = self.model_save_prefix + self.model_class + "_" + self.dataset_name
		# best model save path
		model_save_path = self.save_model_dict + "/" + model_save_name

		# 创建模型，根据 model_class 的选择
		self.model = self.__create_model()
		self.model = self.load_models(self.model, model_save_path)

		self.__model_to_device()

		# 这不不会出问题吧，设置多 GPU，应该不需要用到
		self.__model_parallel()

		print("--------------------- begin testing -----------------------")
		# 稍加处理一下数据，把数据都存在元祖里
		data_from_path = "./" + self.dataset_name + "/eva.dataset"
		evaluation_data = datasets.load_from_disk(data_from_path)

		# 汇总下数据
		evaluation_qa_pairs = []

		evaluation_title = evaluation_data['title']
		evaluation_body = evaluation_data['body']
		evaluation_answers = evaluation_data['answers']

		# 权宜之计，懒得处理新的排序数据了，只好分一半之前的测试集当验证集，有机会改
		begin_index = len(evaluation_title) // 2
		end_index = len(evaluation_title)

		if self.model_class in ['QAModel']:
			concatenated_question = []
			for t, b in zip(evaluation_title, evaluation_body):
				concatenated_question.append(t + " " + b)

			for row_index in range(begin_index, end_index):
				evaluation_qa_pairs.append((concatenated_question[row_index], evaluation_answers[row_index]))

			# pass data into ranking method
			return self.ranking_qa_input(evaluation_qa_pairs)
		elif self.model_class in ['QATopicModel', 'QATopicMemoryModel']:
			concatenated_question = []
			for t, b in zip(evaluation_title, evaluation_body):
				concatenated_question.append(t + " " + b)

			for row_index in range(begin_index, end_index):
				evaluation_qa_pairs.append((concatenated_question[row_index], evaluation_answers[row_index]))
			
			# pass data into ranking method
			return self.ranking_qa_topic_input(evaluation_qa_pairs)
		else:
			for row_index in range(begin_index, end_index):
				evaluation_qa_pairs.append((evaluation_title[row_index], evaluation_body[row_index],
											evaluation_answers[row_index]))

			# pass data into ranking method
			return self.ranking(evaluation_qa_pairs)

	def ranking(self, ranking_qa_pairs):
		"""
		:param ranking_qa_pairs: List< Tuple< title: List<str>, body: List<str>, answers: List<list<str>> >>
		:return: Recall@1: float
		"""
		self.model.eval()

		# 开始逐条排序
		model_ranking = []

		# 一点点处理数据
		now_pair_index = 0
		PAIR_STEP = 2000

		while now_pair_index < len(ranking_qa_pairs):
			# 取一定数量的数据
			questions = []
			bodies = []
			answers = []

			for data in ranking_qa_pairs[now_pair_index: now_pair_index + PAIR_STEP]:
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
			length = len(ranking_qa_pairs[now_pair_index: now_pair_index + PAIR_STEP])

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
				print(f"now processed: {now_pair_index}/{len(ranking_qa_pairs)}")

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

		self.model.train()

		return self.dcg_score(model_ranking, 1)

	def get_scheduler(self, optimizer, scheduler_last_epoch, train_dataloader):
		t_total = (
						  len(train_dataloader) // self.gradient_accumulation_steps) * self.num_train_epochs * self.dataset_split_num

		# if restore training, should pass last epoch, otherwise should not pass this argument
		if self.restore_flag:
			scheduler = get_linear_schedule_with_warmup(optimizer,
														num_warmup_steps=int(t_total * 0.02),
														num_training_steps=t_total,
														last_epoch=scheduler_last_epoch)
			# avoid trying to restore again
			self.restore_flag = False
		else:
			scheduler = get_linear_schedule_with_warmup(optimizer,
														num_warmup_steps=int(t_total * 0.02),
														num_training_steps=t_total)

		print(f"Train {self.num_train_epochs} epochs, Block num {self.dataset_split_num}, "
			  f"Block Batch num {len(train_dataloader)}, "
			  f"Acc num {self.gradient_accumulation_steps}, Total update {t_total}\n")

		return scheduler

	def select_train_step_function(self):
		# add model
		if self.model_class == "CrossBERT":
			return self.__train_step_for_cross
		elif self.model_class in ['BasicModel', 'InputMemorySelfAtt', 'PureMemorySelfAtt',
								  'OneSupremeMemory']:
			return self.__train_step_for_bi
		elif self.model_class in ['QAMemory', 'QAModel', 'ADecoder']:
			return self.__train_step_for_qa_input 
		elif self.model_class in ['QATopicModel', 'QATopicMemoryModel']:
			return self.__train_step_for_qa_topic_input
		else:
			raise Exception("Training step for this class is not supported!")

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
				if self.model_class in ['BasicModel', 'InputMemorySelfAtt', 'PureMemorySelfAtt', 'OneSupremeMemory']:
					logits = self.__val_step_for_bi(batch)
				elif self.model_class in ['QAMemory', 'QAModel', 'ADecoder']:
					logits = self.__val_step_for_qa_input(batch)
				elif self.model_class in ['CrossBERT']:
					logits = self.__val_step_for_cross(batch)
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
		if self.model_class in ['QAMemory']:
			raise Exception(f"load_pretrained_memory is not supported for {self.model_class} yet!!!")

		saved_dict = torch.load("./model/pretrained_memory/" + memory_save_name)
		memory_start_index = saved_dict['memory_start_index']

		memory_weights = {'memory_for_answer': saved_dict['embedding']['weight'][memory_start_index:],
						  'memory_for_question': saved_dict['embedding']['weight'][memory_start_index:]}

		model_dict = self.model.state_dict()
		model_dict.update(memory_weights)
		self.model.load_state_dict(model_dict)
		print(f"Memory is loaded from ./model/pretrained_memory/{memory_save_name} !!!")

	def save_model(self, model_save_path, optimizer, scheduler, epoch, previous_best_performance, early_stop_count,
				   postfix=""):
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

	def ranking_qa_input(self, ranking_qa_pairs):
		print("--------------------- begin ranking -----------------------")
		self.model.eval()

		print(f"all {len(ranking_qa_pairs)} qa pairs!")

		# 开始逐条排序
		model_ranking = []

		# 一点点处理数据
		now_pair_index = 0
		PAIR_STEP = 2000

		# 统计长度和准确率相关
		len_dis = []
		acc_dis = []
		for _ in range(16):
			len_dis.append(0)
			acc_dis.append(0)

		while now_pair_index < len(ranking_qa_pairs):
			# 获得memory合并的字符串
			q_memory_sequence = " "
			for i in range(self.memory_num):
				q_memory_sequence += '<QMEM' + str(i) + '>' + " "

			a_memory_sequence = " "
			for i in range(self.memory_num):
				a_memory_sequence += '<AMEM' + str(i) + '>' + " "

			# 取一定数量的数据
			questions = []
			answers = []

			for data in ranking_qa_pairs[now_pair_index: now_pair_index + PAIR_STEP]:
				temp_question: str = data[0]
				candidate_answers: list = data[1]

				for t_a in candidate_answers[1:self.ranking_candidate_num]:
					# add model
					if self.model_class in ['QAMemory']:
						questions.append((temp_question, q_memory_sequence))
						answers.append((t_a, a_memory_sequence))
					else:
						questions.append(temp_question)
						answers.append(t_a)

				# 把最佳答案塞到最后
				if self.model_class in ['QAMemory']:
					questions.append((temp_question, q_memory_sequence))
					answers.append((candidate_answers[0], a_memory_sequence))
				else:
					questions.append(temp_question)
					answers.append(candidate_answers[0])

			# tokenize
			encoded_a = self.tokenizer(
				answers, padding=True, verbose=False, add_special_tokens=True,
				truncation=True, max_length=self.text_max_len, return_tensors='pt')

			encoded_q = self.tokenizer(
				questions, padding=True, verbose=False, add_special_tokens=True,
				truncation=True, max_length=self.text_max_len, return_tensors='pt')

			# 检查数据数量是否正确，length是问题数
			length = len(ranking_qa_pairs[now_pair_index: now_pair_index + PAIR_STEP])

			if len(encoded_q['input_ids']) != length * self.ranking_candidate_num:
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

				now_index += step

				real_step = int(a_attention_mask.shape[0] / self.ranking_candidate_num)
				question_length = q_attention_mask.sum(-1).view(-1, self.ranking_candidate_num)[:, 0]
				answer_length = torch.mean(a_attention_mask.sum(-1).view(-1, self.ranking_candidate_num).type(torch.FloatTensor), dim=-1)
				
				considered_length = question_length

				for l in considered_length:
					len_dis[int((l.item()-1)//32)] += 1

				with torch.no_grad():
					logits = self.model(
						q_input_ids=q_input_ids, q_token_type_ids=q_token_type_ids,
						q_attention_mask=q_attention_mask,
						a_input_ids=a_input_ids, a_token_type_ids=a_token_type_ids,
						a_attention_mask=a_attention_mask)

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
				
				for index, r in enumerate(model_ranking[-real_step:]):
					if r == 1:
						acc_dis[int((considered_length[index].item()-1)//32)] += 1

			now_pair_index += PAIR_STEP

			# 每排好10000个问题，打印下当前的结果
			if (now_pair_index % 10000) == 0:
				print(f"now processed: {now_pair_index}/{len(ranking_qa_pairs)}")

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
		
		print()
		# print("Len dis:\t", len_dis)
		# print("Acc dis:\t", acc_dis)
		for index, a in enumerate(acc_dis):
			# acc_dis[index] = a/len_dis[index]
			print(f"{index*32 + 32}, \t{len_dis[index]}, \t{acc_dis[index]}, \t{a/len_dis[index]}")

		# print("Acc dis:\t", acc_dis)
		print("---------------------- end ranking ------------------------")
		self.model.train()

		return self.dcg_score(model_ranking, 1)

	def ranking_qa_topic_input(self, ranking_qa_pairs):
		print("--------------------- begin ranking -----------------------")
		self.model.eval()

		if self.model_class in ['QATopicMemoryModel']:
			max_len = self.text_max_len - self.latent_dim
		else:
			max_len = self.text_max_len

		print(f"all {len(ranking_qa_pairs)} qa pairs!")

		# 开始逐条排序
		model_ranking = []

		# 一点点处理数据
		now_pair_index = 0
		PAIR_STEP = 2000

		# 统计长度和准确率相关
		len_dis = []
		acc_dis = []
		for _ in range(16):
			len_dis.append(0)
			acc_dis.append(0)

		while now_pair_index < len(ranking_qa_pairs):
			# 取一定数量的数据
			questions = []
			answers = []

			for data in ranking_qa_pairs[now_pair_index: now_pair_index + PAIR_STEP]:
				temp_question: str = data[0]
				candidate_answers: list = data[1]

				for t_a in candidate_answers[1:self.ranking_candidate_num]:
					questions.append(temp_question)
					answers.append(t_a)

				# 把最佳答案塞到最后
				questions.append(temp_question)
				answers.append(candidate_answers[0])

			# 获得词袋
			q_bows = []
			a_bows = []

			voc_size = len(self.dictionary)

			for d, a in zip(questions, answers):
				new_d = ''.join([i for i in d if not i.isdigit()])
				split_d = new_d.split()
				d_bow = self.dictionary.doc2bow(split_d)
				q_bows.append(d_bow)

				new_a = ''.join([i for i in a if not i.isdigit()])
				split_a = new_a.split()
				a_bow = self.dictionary.doc2bow(split_a)
				a_bows.append(a_bow)

			tensor_q_bows = torch.zeros((len(q_bows), voc_size))
			for index in range(len(q_bows)):
				item = list(zip(*q_bows[index]))
				if len(item) == 0:
					continue 
				tensor_q_bows[index][list(item[0])] = torch.tensor(list(item[1])).float()

			tensor_a_bows = torch.zeros((len(a_bows), voc_size))
			for index in range(len(a_bows)):
				item = list(zip(*a_bows[index])) 
				if len(item) == 0:
					continue 
				tensor_a_bows[index][list(item[0])] = torch.tensor(list(item[1])).float()

			q_bows = tensor_q_bows
			a_bows = tensor_a_bows

			# tokenize
			encoded_a = self.tokenizer(
				answers, padding=True, verbose=False, add_special_tokens=True,
				truncation=True, max_length=max_len, return_tensors='pt')

			encoded_q = self.tokenizer(
				questions, padding=True, verbose=False, add_special_tokens=True,
				truncation=True, max_length=max_len, return_tensors='pt')

			# 检查数据数量是否正确，length是问题数
			length = len(ranking_qa_pairs[now_pair_index: now_pair_index + PAIR_STEP])

			if len(encoded_q['input_ids']) != length * self.ranking_candidate_num:
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

				q_bow = q_bows[now_index * self.ranking_candidate_num:
							  (now_index + step) * self.ranking_candidate_num].to(self.device)
				
				a_bow = a_bows[now_index * self.ranking_candidate_num:
							  (now_index + step) * self.ranking_candidate_num].to(self.device)

				now_index += step

				real_step = int(a_attention_mask.shape[0] / self.ranking_candidate_num)
				question_length = q_attention_mask.sum(-1).view(-1, self.ranking_candidate_num)[:, 0]
				answer_length = torch.mean(a_attention_mask.sum(-1).view(-1, self.ranking_candidate_num).type(torch.FloatTensor), dim=-1)
				
				considered_length = question_length

				for l in considered_length:
					len_dis[int((l.item()-1)//32)] += 1

				with torch.no_grad():
					logits, _ = self.model(
						q_input_ids=q_input_ids, q_token_type_ids=q_token_type_ids,
						q_attention_mask=q_attention_mask,
						a_input_ids=a_input_ids, a_token_type_ids=a_token_type_ids,
						a_attention_mask=a_attention_mask, q_bow=q_bow, a_bow=a_bow)

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
				
				for index, r in enumerate(model_ranking[-real_step:]):
					if r == 1:
						acc_dis[int((considered_length[index].item()-1)//32)] += 1

			now_pair_index += PAIR_STEP

			# 每排好10000个问题，打印下当前的结果
			if (now_pair_index % 10000) == 0:
				print(f"now processed: {now_pair_index}/{len(ranking_qa_pairs)}")

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
				
		print()
		# print("Len dis:\t", len_dis)
		# print("Acc dis:\t", acc_dis)
		for index, a in enumerate(acc_dis):
			# acc_dis[index] = a/len_dis[index]
			print(f"{index*32 + 32}, \t{len_dis[index]}, \t{acc_dis[index]}, \t{a/len_dis[index]}")

		# print("Acc dis:\t", acc_dis)
		print("---------------------- end ranking ------------------------")
		self.model.train()

		return self.dcg_score(model_ranking, 1)


	def ranking_cross(self):
		print("--------------------- begin ranking -----------------------")
		self.model.eval()

		# 稍加处理一下数据，把数据都存在元祖里
		data_from_path = "./" + self.dataset_name + "/eva.dataset"
		evaluation_data = datasets.load_from_disk(data_from_path)

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
			texts = []

			for data in evaluation_qa_pairs[now_pair_index: now_pair_index + PAIR_STEP]:
				temp_question: str = data[0]
				body = data[1]
				candidate_answers = data[2]

				for t_a in candidate_answers[1:self.ranking_candidate_num]:
					texts.append((temp_question + ' ' + body, t_a))

				# 把最佳答案塞到最后
				texts.append((temp_question + ' ' + body, candidate_answers[0]))

			# tokenize
			encoded_texts = self.tokenizer(
				texts, padding=True, verbose=False, add_special_tokens=True,
				truncation=True, max_length=self.text_max_len, return_tensors='pt')

			# 检查数据数量是否正确，length是问题数
			length = len(evaluation_qa_pairs[now_pair_index: now_pair_index + PAIR_STEP])

			if len(encoded_texts['input_ids']) != length * self.ranking_candidate_num:
				raise Exception("encode while ranking no possible!")

			# 开始按照更小的批次进行训练，也就是每次计算step*candidate_answer_num条数据
			now_index = 0
			step = 40

			while now_index < length:
				input_ids = encoded_texts['input_ids'][
							  now_index * self.ranking_candidate_num:
							  (now_index + step) * self.ranking_candidate_num].to(self.device)
				token_type_ids = encoded_texts['token_type_ids'][
								   now_index * self.ranking_candidate_num:
								   (now_index + step) * self.ranking_candidate_num].to(self.device)
				attention_mask = encoded_texts['attention_mask'][
								   now_index * self.ranking_candidate_num:
								   (now_index + step) * self.ranking_candidate_num].to(self.device)

				now_index += step

				with torch.no_grad():
					logits = self.model(
						input_ids=input_ids, token_type_ids=token_type_ids,
						attention_mask=attention_mask)

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
		elif self.model_class == 'BasicTopicModel':
			model = BasicTopicModel(config=self.config)
		elif self.model_class == 'InputMemorySelfAtt':
			model = InputMemorySelfAtt(config=self.config)
		elif self.model_class == 'PureMemorySelfAtt':
			model = PureMemorySelfAtt(config=self.config)
		elif self.model_class in ['QAMemory', 'QAModel']:
			model = QAModel(config=self.config)
		elif self.model_class in ['QATopicModel']:
			model = QATopicModel(config=self.config)
		elif self.model_class in ['QATopicMemoryModel']:
			model = QATopicMemoryModel(config=self.config)
		elif self.model_class in ['CrossBERT']:
			model = CrossBERT(config=self.config)
		elif self.model_class in ['ADecoder']:
			model = ADecoder(config=self.config)
		elif self.model_class in ['OneSupremeMemory']:
			model = OneSupremeMemory(config=self.config)
		else:
			raise Exception("This model class is not supported for creating!!")

		# 要不要加载现成的模型
		if self.load_classifier_flag:
			model = self.load_models(model, self.classifier_path)

		print("--------------------- model  created ---------------------")

		return model

	# 读取命令行传入的参数
	def __read_args_for_train(self, args):
		self.memory_save_prefix = args.memory_save_prefix
		self.save_model_dict = args.save_model_dict
		self.last_model_dict = args.last_model_dict
		self.val_num_each_epoch = args.val_num_each_epoch
		self.val_candidate_num = args.val_candidate_num
		self.val_batch_size = args.val_batch_size
		self.text_max_len = args.text_max_len
		self.dataset_name = args.dataset_name
		self.model_class = args.model_class
		self.memory_num = args.memory_num
		self.train_candidate_num = args.train_candidate_num
		self.print_num_each_epoch = args.print_num_each_epoch
		self.load_classifier_flag = args.load_classifier
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

		self.classifier_path = args.classifier_path

	# get path to restore training
	@staticmethod
	def get_restore_path(model_save_name, final_stage_flag):
		restore_path = "./last_model/" + model_save_name
		if not final_stage_flag:
			restore_path += "_middle"
		return restore_path	
	
	def restore_settings(self, optimizer, restore_path, previous_best_performance):
		early_stop_count = 0
		restore_epoch = 0
		scheduler_last_epoch = 0
		new_previous_best_performance = previous_best_performance
		
		if self.restore_flag:
			restore_data = torch.load(restore_path)

			# get scheduler
			scheduler_last_epoch = restore_data['scheduler']['last_epoch']
			early_stop_count = restore_data['early_stop_count']

			# get optimizer
			optimizer.load_state_dict(restore_data['optimizer'])

			# # print lr
			# for o in optimizer.state_dict()['param_groups']:
			# 	print(o['lr'], end="\t")
			# print()

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
			print("*" * 100)
			
		return early_stop_count, restore_epoch, scheduler_last_epoch, new_previous_best_performance
	
	# 读取命令行传入的有关config的参数
	def __read_args_for_config(self, args):
		if args.pretrained_bert_path in ['prajjwal1/bert-small', 'google/bert_uncased_L-6_H-512_A-8',
										 'google/bert_uncased_L-8_H-512_A-8', '/data/yuanhang/pretrained_model/prajjwal1/bert-small',
										 '/data/yuanhang/pretrained_model/prajjwal1/bert-medium']:
			word_embedding_len = 512
			sentence_embedding_len = 512
		elif args.pretrained_bert_path in ['bert-base-uncased', '/data/yuanhang/pretrained_model/bert-base-uncased']:
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
								 sentence_embedding_len=sentence_embedding_len,
								 composition=self.composition)
		elif self.model_class == 'BasicTopicModel':
			config = BasicTopicConfig(len(self.tokenizer),
										len(self.dictionary), 
										pretrained_bert_path=args.pretrained_bert_path,
										num_labels=args.label_num,
										word_embedding_len=word_embedding_len,
										sentence_embedding_len=sentence_embedding_len,
										composition=self.composition,
										topic_num=args.latent_dim)
		elif self.model_class == 'InputMemorySelfAtt':
			config = InputMemorySelfAttConfig(len(self.tokenizer),
											  pretrained_bert_path=args.pretrained_bert_path,
											  num_labels=args.label_num,
											  word_embedding_len=word_embedding_len,
											  sentence_embedding_len=sentence_embedding_len,
											  memory_num=args.memory_num,
											  composition=self.composition)
		elif self.model_class == 'PureMemorySelfAtt':
			config = PureMemorySelfAttConfig(len(self.tokenizer),
											 pretrained_bert_path=args.pretrained_bert_path,
											 num_labels=args.label_num,
											 word_embedding_len=word_embedding_len,
											 sentence_embedding_len=sentence_embedding_len,
											 memory_num=args.memory_num,
											 hop_num=args.hop_num)
		elif self.model_class in ['QAMemory', 'QAModel']:
			config = QAModelConfig(len(self.tokenizer),
										 pretrained_bert_path=args.pretrained_bert_path,
										 num_labels=args.label_num,
										 word_embedding_len=word_embedding_len,
										 sentence_embedding_len=sentence_embedding_len,
										 composition=self.composition)
		elif self.model_class in ['QATopicModel', 'QATopicMemoryModel']:
			config = QATopicConfig(len(self.tokenizer),
									len(self.dictionary),
										 pretrained_bert_path=args.pretrained_bert_path,
										 num_labels=args.label_num,
										 word_embedding_len=word_embedding_len,
										 sentence_embedding_len=sentence_embedding_len,
										 composition=self.composition,
										 topic_num=self.latent_dim)
		elif self.model_class in ['CrossBERT']:
			config = CrossBERTConfig(len(self.tokenizer),
									 pretrained_bert_path=args.pretrained_bert_path,
									 num_labels=args.label_num,
									 word_embedding_len=word_embedding_len,
									 sentence_embedding_len=sentence_embedding_len,
									 composition=self.composition)
		elif self.model_class in ['ADecoder']:
			config = ADecoderConfig(len(self.tokenizer),
									pretrained_bert_path=args.pretrained_bert_path,
									num_labels=args.label_num,
									word_embedding_len=word_embedding_len,
									sentence_embedding_len=sentence_embedding_len,
									composition=self.composition)
		elif self.model_class in ['OneSupremeMemory']:
			config = OneSupremeMemoryConfig(len(self.tokenizer),
											 pretrained_bert_path=args.pretrained_bert_path,
											 num_labels=args.label_num,
											 word_embedding_len=word_embedding_len,
											 sentence_embedding_len=sentence_embedding_len,
											 memory_num=args.memory_num)
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
		elif self.model_class == 'QATopicModel':
			parameters_dict_list = [
				# 这几个一样
				{'params': model.bert_model.parameters(), 'lr': 5e-5},
				# 这几个一样
				{'params': model.query_layer.parameters(), 'lr': 1e-4},
				{'params': model.vae.parameters(), 'lr': 1e-4},
				{'params': model.LayerNorm.parameters(), 'lr': 1e-4},
				# 这个不设定
				{'params': model.classifier.parameters(), 'lr': 1e-4}
			]
		elif self.model_class == 'QATopicMemoryModel':
			parameters_dict_list = [
				# 这几个一样
				{'params': model.bert_model.parameters(), 'lr': 5e-5},
				# 这几个一样
				{'params': model.query_layer.parameters(), 'lr': 1e-4},
				{'params': model.memory_layer.parameters(), 'lr': 1e-4},
				{'params': model.vae.parameters(), 'lr': 1e-4},
				{'params': model.LayerNorm.parameters(), 'lr': 1e-4},
				{'params': model.memory_LayerNorm.parameters(), 'lr': 1e-4},
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
				# {'params': model.queries_for_answer.parameters(), 'lr': 1e-4},
				# {'params': model.memories_for_answer.parameters(), 'lr': 1e-4},
				# {'params': model.queries_for_question.parameters(), 'lr': 1e-4},
				# {'params': model.memories_for_question.parameters(), 'lr': 1e-4},
				{'params': model.queries_for_answer, 'lr': 1e-4},
				{'params': model.memories_for_answer, 'lr': 1e-4},
				{'params': model.queries_for_question, 'lr': 1e-4},
				{'params': model.memories_for_question, 'lr': 1e-4},
				{'params': model.self_attention_weight_layer.parameters(), 'lr': 1e-4},
				{'params': model.value_layer.parameters(), 'lr': 1e-4},
				# 这个不设定
				{'params': model.classifier.parameters(), 'lr': 1e-4}
			]
		elif self.model_class in ['QAMemory', 'QAModel']:
			parameters_dict_list = [
				# 这几个一样
				{'params': model.bert_model.parameters(), 'lr': 5e-5},
				# 这几个一样
				{'params': model.self_attention_weight_layer.parameters(), 'lr': 1e-4},
				{'params': model.value_layer.parameters(), 'lr': 1e-4},
				# 这个不设定
				{'params': model.classifier.parameters(), 'lr': 1e-4}
			]
		elif self.model_class in ['CrossBERT']:
			parameters_dict_list = [
				# 这几个一样
				{'params': model.bert_model.parameters(), 'lr': 5e-5},
			]
		elif self.model_class in ['ADecoder']:
			parameters_dict_list = [
				# 这几个一样
				{'params': model.bert_model.parameters(), 'lr': 5e-5},
				{'params': model.composition_layer.parameters(), 'lr': 5e-5},
				{'params': model.decoder.parameters(), 'lr': 5e-5},
				{'params': model.classifier.parameters(), 'lr': 5e-5},
			]
		elif self.model_class in ['OneSupremeMemory']:
			parameters_dict_list = [
				# 这几个一样
				{'params': model.bert_model.parameters(), 'lr': 5e-5},
				{'params': model.memory_for_answer, 'lr': 5e-5},
				{'params': model.memory_for_question, 'lr': 5e-5},
				{'params': model.key_layer.parameters(), 'lr': 5e-5},
				{'params': model.value_layer.parameters(), 'lr': 5e-5},
				{'params': model.classifier.parameters(), 'lr': 5e-5},
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
			elif self.model_class == 'QATopicModel':
				parameters_dict_list = [
					# 这几个一样
					# {'params': model.bert_model.parameters(), 'lr': 5e-5},
					# 这几个一样
					# {'params': model.query_layer.parameters(), 'lr': 1e-4},
					# {'params': model.vae.parameters(), 'lr': 1e-4},
					{'params': model.embeddings.parameters(), 'lr': 1e-4},
					{'params': model.LayerNorm.parameters(), 'lr': 1e-4},
					# 这个不设定
					{'params': model.classifier.parameters(), 'lr': 1e-4}
				]
			elif self.model_class == 'QATopicMemoryModel':
				parameters_dict_list = [
					# 这几个一样
					# {'params': model.query_layer.parameters(), 'lr': 1e-4},
					{'params': model.memory_layer.parameters(), 'lr': 1e-4},
					# {'params': model.vae.parameters(), 'lr': 1e-4},
					{'params': model.LayerNorm.parameters(), 'lr': 1e-4},
					{'params': model.memory_LayerNorm.parameters(), 'lr': 1e-4},
					# 这个很重要！！！！！需要评估下作用
					{'params': model.embeddings.parameters(), 'lr': 1e-4},
					# 这个不设定
					{'params': model.classifier.parameters(), 'lr': 1e-4}
				]
			elif self.model_class == 'InputMemorySelfAtt':
				parameters_dict_list = [
					# 这几个一样
					{'params': model.memory_for_question, 'lr': self.first_stage_lr},
					{'params': model.memory_for_answer, 'lr': self.first_stage_lr},
					# {'params': model.memory_for_question, 'lr': 1e-4},
					# {'params': model.memory_for_answer, 'lr': 1e-4},
					{'params': model.self_attention_weight_layer.parameters(), 'lr': 1e-4},
					{'params': model.value_layer.parameters(), 'lr': 1e-4},
					# 这个不设定
					{'params': model.classifier.parameters(), 'lr': 1e-4}
				]
			elif self.model_class == 'PureMemorySelfAtt':
				parameters_dict_list = [
					# 这几个一样
					# {'params': model.queries_for_answer.parameters(), 'lr': 1e-4},
					# {'params': model.memories_for_answer.parameters(), 'lr': 1e-4},
					# {'params': model.queries_for_question.parameters(), 'lr': 1e-4},
					# {'params': model.memories_for_question.parameters(), 'lr': 1e-4},
					{'params': model.queries_for_answer, 'lr': 1e-4},
					{'params': model.memories_for_answer, 'lr': 1e-4},
					{'params': model.queries_for_question, 'lr': 1e-4},
					{'params': model.memories_for_question, 'lr': 1e-4},
					{'params': model.self_attention_weight_layer.parameters(), 'lr': 1e-4},
					{'params': model.value_layer.parameters(), 'lr': 1e-4},
					# 这个不设定
					{'params': model.classifier.parameters(), 'lr': 1e-4}
				]
			elif self.model_class in ['QAMemory']:
				parameters_dict_list = [
					# 这几个一样
					{'params': model.embeddings.parameters(), 'lr': self.first_stage_lr},
					# 这几个一样
					{'params': model.self_attention_weight_layer.parameters(), 'lr': 1e-4},
					{'params': model.value_layer.parameters(), 'lr': 1e-4},
					# 这个不设定
					{'params': model.classifier.parameters(), 'lr': 1e-4}
				]
			elif self.model_class in ['ADecoder']:
				parameters_dict_list = [
					# 这几个一样
					{'params': model.decoder.parameters(), 'lr': 5e-5},
					{'params': model.composition_layer.parameters(), 'lr': 1e-4},
					{'params': model.classifier.parameters(), 'lr': 1e-4},
				]
			elif self.model_class in ['OneSupremeMemory']:
				parameters_dict_list = [
					# 这几个一样
					{'params': model.memory_for_answer, 'lr': 1e-4},
					{'params': model.memory_for_question, 'lr': 1e-4},
					{'params': model.key_layer.parameters(), 'lr': 1e-4},
					{'params': model.value_layer.parameters(), 'lr': 1e-4},
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

	# 双塔模型的训练步
	def __train_step_for_bi(self, batch, optimizer, now_batch_num, scheduler, **kwargs):
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

	# 输入为QA，而非title.body.answer的模型的训练步
	def __train_step_for_qa_input(self, batch, optimizer, now_batch_num, scheduler, final_stage_flag):
		cross_entropy_function = nn.CrossEntropyLoss()

		# 读取数据
		q_input_ids = (batch['q_input_ids']).to(self.device)
		q_token_type_ids = (batch['q_token_type_ids']).to(self.device)
		q_attention_mask = (batch['q_attention_mask']).to(self.device)

		a_input_ids = (batch['a_input_ids']).to(self.device)
		a_token_type_ids = (batch['a_token_type_ids']).to(self.device)
		a_attention_mask = (batch['a_attention_mask']).to(self.device)

		qa_labels = (batch['label']).to(self.device)

		# 得到模型的结果
		logits = self.model(
			q_input_ids=q_input_ids, q_token_type_ids=q_token_type_ids,
			q_attention_mask=q_attention_mask,
			a_input_ids=a_input_ids, a_token_type_ids=a_token_type_ids,
			a_attention_mask=a_attention_mask)

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

	# 输入为QA，以及词袋，而非title.body.answer的模型的训练步
	def __train_step_for_qa_topic_input(self, batch, optimizer, now_batch_num, scheduler, final_stage_flag):
		cross_entropy_function = nn.CrossEntropyLoss()

		# 读取数据
		q_input_ids = (batch['q_input_ids']).to(self.device)
		q_token_type_ids = (batch['q_token_type_ids']).to(self.device)
		q_attention_mask = (batch['q_attention_mask']).to(self.device)

		a_input_ids = (batch['a_input_ids']).to(self.device)
		a_token_type_ids = (batch['a_token_type_ids']).to(self.device)
		a_attention_mask = (batch['a_attention_mask']).to(self.device)

		q_bow = (batch['q_bow']).to(self.device)
		a_bow = (batch['a_bow']).to(self.device)

		qa_labels = (batch['label']).to(self.device)

		# 得到模型的结果
		logits, vae_loss = self.model(
			q_input_ids=q_input_ids, q_token_type_ids=q_token_type_ids,
			q_attention_mask=q_attention_mask,
			a_input_ids=a_input_ids, a_token_type_ids=a_token_type_ids,
			a_attention_mask=a_attention_mask, q_bow=q_bow, a_bow=a_bow)

		# 计算损失
		step_loss = cross_entropy_function(logits, qa_labels)

		step_loss += vae_loss

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
	def __train_step_for_cross(self, batch, optimizer, now_batch_num, scheduler, **kwargs):
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

	def __val_step_for_qa_input(self, batch):
		# 读取数据
		q_input_ids = (batch['q_input_ids']).to(self.device)
		q_token_type_ids = (batch['q_token_type_ids']).to(self.device)
		q_attention_mask = (batch['q_attention_mask']).to(self.device)

		a_input_ids = (batch['a_input_ids']).to(self.device)
		a_token_type_ids = (batch['a_token_type_ids']).to(self.device)
		a_attention_mask = (batch['a_attention_mask']).to(self.device)

		with torch.no_grad():
			# 得到模型的结果
			logits = self.model(
				q_input_ids=q_input_ids, q_token_type_ids=q_token_type_ids,
				q_attention_mask=q_attention_mask,
				a_input_ids=a_input_ids, a_token_type_ids=a_token_type_ids,
				a_attention_mask=a_attention_mask)

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

	def __get_dataloader(self, data, batch_size, split_index, split_num):
		if split_index == split_num - 1:
			now_data_block = data[int(split_index * len(data) / split_num):]
		else:
			now_data_block = data[int(split_index * len(data) / split_num):
								  int((split_index + 1) * len(data) / split_num)]

		# add model
		if self.model_class in ['BasicModel', 'InputMemorySelfAtt', 'PureMemorySelfAtt', 'OneSupremeMemory']:
			now_dataset = TBAClassifyDataset(data=now_data_block,
											 tokenizer=self.tokenizer,
											 text_max_len=self.text_max_len - self.memory_num)
		elif self.model_class in ['QAMemory']:
			now_dataset = QAMemClassifyDataset(data=now_data_block,
											tokenizer=self.tokenizer,
											text_max_len=self.text_max_len, memory_num=self.memory_num)
		elif self.model_class in ['QAModel', 'ADecoder']:
			now_dataset = QAClassifyDataset(data=now_data_block,
											tokenizer=self.tokenizer,
											text_max_len=self.text_max_len)
		elif self.model_class in ['QATopicModel']:
			now_dataset = QATopicClassifyDataset(data=now_data_block,
											tokenizer=self.tokenizer,
											text_max_len=self.text_max_len,
											dictionary=self.dictionary)
		elif self.model_class in ['QATopicMemoryModel']:
			now_dataset = QATopicClassifyDataset(data=now_data_block,
											tokenizer=self.tokenizer,
											text_max_len=self.text_max_len - self.latent_dim,
											dictionary=self.dictionary)
		elif self.model_class in ['CrossBERT']:
			now_dataset = CrossClassifyDataset(data=now_data_block,
											   tokenizer=self.tokenizer,
											   text_max_len=self.text_max_len)
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

# %%
