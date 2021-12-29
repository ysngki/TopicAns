import torch


def sum_average_tuple(in_tuple):
	sum_result = 0
	for i in in_tuple:
		sum_result += i
	return sum_result, sum_result/len(in_tuple)


def raise_dataset_error():
	raise Exception("This dataset is not supported now!")


def print_recall_precise(label_hit_num, label_shoot_num, label_target_num, label_num):
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


def load_model(model, load_model_path):
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


def print_optimizer(optimizer):
	print("*"*100)
	print("Now optimizer lr:")
	for o in optimizer.state_dict()['param_groups']:
		print(o['lr'], end="\t")
	print("\n" + "*" * 100)
