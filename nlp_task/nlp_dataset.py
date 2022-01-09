import torch.utils.data


class SingleInputDataset(torch.torch.utils.data.Dataset):
	def __init__(self, input_ids, token_type_ids, attention_mask, idx):
		self.input_ids = input_ids
		self.token_type_ids = token_type_ids
		self.attention_mask = attention_mask
		self.idx = idx

	def __len__(self):  # 返回整个数据集的大小
		return len(self.input_ids)

	def __getitem__(self, index):
		item_dic = {'input_ids': self.input_ids[index],
					'token_type_ids': self.token_type_ids[index],
					'attention_mask': self.attention_mask[index],
					'idx': self.idx[index]}

		return item_dic


class DoubleInputDataset(torch.torch.utils.data.Dataset):
	def __init__(self, a_input_ids, a_token_type_ids, a_attention_mask,
				 b_input_ids, b_token_type_ids, b_attention_mask, idx):

		self.a_input_ids = a_input_ids
		self.a_token_type_ids = a_token_type_ids
		self.a_attention_mask = a_attention_mask
		self.b_input_ids = b_input_ids
		self.b_token_type_ids = b_token_type_ids
		self.b_attention_mask = b_attention_mask
		self.idx = idx

	def __len__(self):  # 返回整个数据集的大小
		return len(self.a_input_ids)

	def __getitem__(self, index):
		item_dic = {'a_input_ids': self.a_input_ids[index],
					'a_token_type_ids': self.a_token_type_ids[index],
					'a_attention_mask': self.a_attention_mask[index],
					'b_input_ids': self.b_input_ids[index],
					'b_token_type_ids': self.b_token_type_ids[index],
					'b_attention_mask': self.b_attention_mask[index],
					'idx': self.idx[index]}

		return item_dic