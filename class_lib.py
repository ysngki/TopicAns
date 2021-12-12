# %%
from transformers import BertModel
import torch
import torch.nn as nn
import torch.utils.data
import numpy as np
import torch.nn.functional
from attention_module import attend


class VaeDataset(torch.torch.utils.data.Dataset):
	def __init__(self, word_bag):
		self.word_bag = word_bag

	def __len__(self):  # 返回整个数据集的大小
		return self.word_bag.shape[0]

	def __getitem__(self, index):
		train_dic = {"word_bag": self.word_bag[index]}

		return train_dic  # 返回该样本

# 通过4段标量attention统合所有单词为一个向量
class BertRepresent(nn.Module):
    def __init__(self, model_path='prajjwal1/bert-mini', sentence_embedding_len=768, head_num=4, section_len=256):
        super(BertRepresent, self).__init__()
        self.head_num = head_num
        self.bert_model = BertModel.from_pretrained(model_path)

        self.key_layer = torch.nn.Linear(sentence_embedding_len, head_num, bias=True)
        self.value_layer = torch.nn.Linear(sentence_embedding_len, section_len, bias=True)

        self.relu = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax(dim=-2)
        self.sigmoid = torch.nn.Sigmoid()
        self.tanh = torch.nn.Tanh()
        self.init_weights()

    def init_weights(self):
        torch.nn.init.xavier_uniform_(self.key_layer.weight)
        torch.nn.init.xavier_uniform_(self.value_layer.weight)

    def forward(self, input_ids, token_type_ids, attention_mask):
        batch_nums = len(input_ids)
        out = self.bert_model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        last_hidden_state = out['last_hidden_state']

        # 创作出mask
        with torch.no_grad():
            mask = attention_mask.type(dtype=torch.float)
            mask[mask == 0] = -np.inf
            mask[mask == 1] = 0.0
            mask = mask.repeat(4, 1, 1)
            mask.transpose_(0, 1)
            mask.transpose_(1, 2)

        # (batch, sequence, 4)
        weight = self.key_layer(last_hidden_state)
        weight += mask
        weight = self.softmax(weight)

        # (batch, sequence, 250)
        value = self.value_layer(last_hidden_state)

        # (4*sequence) X (sequence*250) = 4*250
        weight = torch.transpose(weight, 1, 2)
        embedding = torch.bmm(weight, value)
        embedding = embedding.view(batch_nums, -1)
        embedding = self.sigmoid(embedding)

        return embedding


# 通过多维自注意力统合
class SequenceAttentionRepresent(nn.Module):
    def __init__(self, model_path='prajjwal1/bert-mini', word_embedding_len=512, output_embedding_len=512):
        super(SequenceAttentionRepresent, self).__init__()

        self.output_embedding_len = output_embedding_len

        self.bert_model = BertModel.from_pretrained(model_path)

        # self.key_layer = torch.nn.Linear(word_embedding_len, output_embedding_len, bias=True)
        self.key_layer = nn.Sequential(
            nn.Linear(word_embedding_len, output_embedding_len),
            nn.ReLU(),
            nn.Linear(output_embedding_len, output_embedding_len),
            nn.Sigmoid()
        )

        # self.value_layer = torch.nn.Linear(word_embedding_len, output_embedding_len, bias=True)
        self.value_layer = nn.Sequential(
            nn.Linear(word_embedding_len, output_embedding_len),
            nn.ReLU(),
            nn.Linear(output_embedding_len, output_embedding_len),
            nn.Tanh()
        )

        self.relu = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax(dim=-2)
        self.sigmoid = torch.nn.Sigmoid()
        self.tanh = torch.nn.Tanh()
        # self.init_weights()

    def init_weights(self):
        torch.nn.init.xavier_uniform_(self.key_layer.weight)
        torch.nn.init.xavier_uniform_(self.value_layer.weight)

    def forward(self, input_ids, token_type_ids, attention_mask):
        # 输入的batch的数量
        batch_nums = len(input_ids)

        # 获得bert传出来的词向量
        out = self.bert_model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        last_hidden_state = out['last_hidden_state']

        # 创作出mask
        with torch.no_grad():
            mask = attention_mask.type(dtype=torch.float)
            mask[mask == 0] = -np.inf
            mask[mask == 1] = 0.0
            mask = mask.repeat(self.output_embedding_len, 1, 1)
            mask.transpose_(0, 1)
            mask.transpose_(1, 2)

        # (batch, sequence, word_embedding_len)
        weight = self.key_layer(last_hidden_state)
        mask_weight = mask + weight
        final_weight = self.softmax(mask_weight)

        # (batch, sequence, 250)
        value = self.value_layer(last_hidden_state)

        embedding = torch.mul(final_weight, value)
        final_embedding = embedding.sum(dim=-2)
        # embedding = self.tanh(embedding)

        return final_embedding


# 换一个说法，可以说是discriminator
class SimpleClassifier(nn.Module):
    def __init__(self, input_len, true_len=False, keep_prob=0.9, num_labels=4):
        super(SimpleClassifier, self).__init__()

        if true_len:
            self.linear1 = torch.nn.Linear(input_len, input_len * 2, bias=True)
        else:
            self.linear1 = torch.nn.Linear(input_len * 2, input_len * 2, bias=True)
        self.bn1 = nn.BatchNorm1d(input_len * 2)
        self.linear2 = torch.nn.Linear(input_len * 2, input_len, bias=True)
        self.bn2 = nn.BatchNorm1d(input_len)
        self.linear3 = torch.nn.Linear(input_len, input_len // 2, bias=True)
        self.bn3 = nn.BatchNorm1d(input_len // 2)
        self.linear4 = torch.nn.Linear(input_len // 2, num_labels, bias=True)

        self.relu = torch.nn.ReLU()
        self.tanh = torch.nn.Tanh()
        self.sigmoid = torch.nn.Sigmoid()
        self.dropout = torch.nn.Dropout(p=1 - keep_prob)
        self.init_weights()

    def init_weights(self):
        torch.nn.init.xavier_uniform_(self.linear1.weight)
        torch.nn.init.xavier_uniform_(self.linear2.weight)
        torch.nn.init.xavier_uniform_(self.linear3.weight)
        torch.nn.init.xavier_uniform_(self.linear4.weight)

    def forward(self, q_embedding, a_embedding, sigmoid_flag=False):
        x = torch.cat((q_embedding, a_embedding), dim=-1)

        x = self.linear1(x)
        x = self.relu(x)
        x = self.bn1(x)
        x = self.dropout(x)


        x = self.linear2(x)
        x = self.relu(x)
        x = self.bn2(x)
        x = self.dropout(x)


        x = self.linear3(x)
        x = self.relu(x)
        x = self.bn3(x)
        x = self.dropout(x)


        x = self.linear4(x)
        # if sigmoid_flag:
        #     x = self.sigmoid(x)
        # else:
        #     x = self.tanh(x)

        return x


class BodyClassifier(nn.Module):
    def __init__(self, input_len, keep_prob=0.9, num_labels=4):
        super(BodyClassifier, self).__init__()

        self.linear1 = torch.nn.Linear(input_len * 3, input_len * 3, bias=True)
        self.bn1 = nn.BatchNorm1d(input_len * 3)
        self.linear2 = torch.nn.Linear(input_len * 3, input_len, bias=True)
        self.bn2 = nn.BatchNorm1d(input_len)
        self.linear3 = torch.nn.Linear(input_len, input_len, bias=True)
        self.bn3 = nn.BatchNorm1d(input_len)
        self.linear4 = torch.nn.Linear(input_len, num_labels, bias=True)

        self.relu = torch.nn.ReLU()
        self.tanh = torch.nn.Tanh()
        self.sigmoid = torch.nn.Sigmoid()
        self.dropout = torch.nn.Dropout(p=1 - keep_prob)
        self.init_weights()

    def init_weights(self):
        torch.nn.init.xavier_uniform_(self.linear1.weight)
        torch.nn.init.xavier_uniform_(self.linear2.weight)
        torch.nn.init.xavier_uniform_(self.linear3.weight)
        torch.nn.init.xavier_uniform_(self.linear4.weight)

    def forward(self, q_embedding, a_embedding, b_embedding, sigmoid_flag=False):
        x = torch.cat((q_embedding, a_embedding), dim=-1)
        x = torch.cat((x, b_embedding), dim=-1)

        x = self.linear1(x)
        x = self.relu(x)
        # x = self.dropout(x)
        x = self.bn1(x)

        x = self.linear2(x)
        x = self.relu(x)
        # x = self.dropout(x)
        x = self.bn2(x)

        x = self.linear3(x)
        x = self.relu(x)
        # x = self.dropout(x)
        x = self.bn3(x)

        x = self.linear4(x)
        # if sigmoid_flag:
        #     x = self.sigmoid(x)
        # else:
        #     x = self.tanh(x)

        return x


# 换一个说法，可以说是discriminator
class EnhancedQuestion(nn.Module):
    def __init__(self, head_num=4, section_len=250):
        super(EnhancedQuestion, self).__init__()
        self.linear1 = torch.nn.Linear(head_num * section_len, head_num * section_len * 2, bias=True)
        self.linear2 = torch.nn.Linear(head_num * section_len * 2, head_num * section_len, bias=True)
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        torch.nn.init.xavier_uniform_(self.linear1.weight)
        torch.nn.init.xavier_uniform_(self.linear2.weight)

    def forward(self, q_embedding):
        x = q_embedding
        x = self.linear1(x)
        x = self.relu(x)

        x = self.linear2(x)
        x = self.relu(x)

        return x


class VAEQuestion(nn.Module):
    def __init__(self, input_dim=1000, hidden_dim=500, latent_dim=100):
        super(VAEQuestion, self).__init__()

        self.en_input = nn.Linear(input_dim, hidden_dim)
        self.en_mean = nn.Linear(hidden_dim, latent_dim)
        self.en_var = nn.Linear(hidden_dim, latent_dim)

        self.de_input = nn.Linear(latent_dim, hidden_dim)
        self.de_output = nn.Linear(hidden_dim, input_dim)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=-1)

    def encoder(self, input_value):
        hidden_value = self.relu(self.en_input(input_value))
        mean = self.en_mean(hidden_value)
        log_var = self.en_var(hidden_value)
        latent_v = self.reparameterize(mean, log_var)

        return latent_v, mean, log_var

    @staticmethod
    def reparameterize(mean, log_var):
        std = torch.exp(log_var * 0.5)
        eps = torch.randn_like(std)
        return mean + eps * std

    def decode(self, latent_v):
        hidden_value = self.relu(self.de_input(latent_v))
        # 注意，输出是sigmoid！！！！！！！！！！！
        out_value = self.sigmoid(self.de_output(hidden_value))
        return out_value

    def forward(self, input_value, out_latent_flag=False):
        latent_v, mean, log_var = self.encoder(input_value)
        reconstructed_input = self.decode(latent_v)

        if out_latent_flag:
            return reconstructed_input, mean, log_var, latent_v

        return reconstructed_input, mean, log_var


def calculate_vae_loss(x, reconstructed_x, mean, log_var):
    reconstruction_loss = nn.functional.mse_loss(reconstructed_x, x, reduction='mean')
    KLD = - 0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

    return reconstruction_loss + KLD


class BertCNNRepresent(nn.Module):
    def __init__(self, model_path='bert-base-uncased', sentence_embedding_len=768, head_num=4,
                 kernel_dim=100, kernel_sizes=(2, 3, 4)):
        super(BertCNNRepresent, self).__init__()
        self.head_num = head_num
        self.bert_model = BertModel.from_pretrained(model_path)

        self.convs = nn.ModuleList([nn.Conv2d(1,
                                              kernel_dim, (K, sentence_embedding_len), padding=2) for K in
                                    kernel_sizes])

        self.relu = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax(dim=-2)
        self.sigmoid = torch.nn.Sigmoid()

    def init_weights(self):
        torch.nn.init.xavier_uniform_(self.key_layer.weight)
        torch.nn.init.xavier_uniform_(self.value_layer.weight)

    def forward(self, input_ids, token_type_ids, attention_mask):
        out = self.bert_model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        last_hidden_state = out['last_hidden_state']

        result = 0
        for index, batch in enumerate(last_hidden_state):
            temp_embedding = batch.unsqueeze(0).unsqueeze(0)
            temp_sequence_len = int(attention_mask[index].sum() + 0.1)
            temp_embedding = temp_embedding[:, :, 0:temp_sequence_len, :]
            # 3*1*100*len
            temp_embedding = [conv(temp_embedding).squeeze(3) for conv in self.convs]
            temp_embedding = [nn.functional.max_pool1d(i, i.size(2)).squeeze(2) for i in temp_embedding]
            temp_embedding = torch.cat(temp_embedding, 1)

            if index == 0:
                result = temp_embedding
            else:
                result = torch.cat([result, temp_embedding], 0)

        return self.sigmoid(result)


class ACGAN_G(nn.Module):
    def __init__(self, output_dim, num_labels=4, label_embedding_len=200):
        super(ACGAN_G, self).__init__()
        self.label_embedding = nn.Embedding(num_labels, label_embedding_len)

        self.linear1 = torch.nn.Linear(output_dim + label_embedding_len, 2 * output_dim, bias=True)
        self.BatchNorm1 = nn.BatchNorm1d(2 * output_dim)

        self.linear2 = torch.nn.Linear(2 * output_dim, 2 * output_dim, bias=True)
        self.BatchNorm2 = nn.BatchNorm1d(2 * output_dim)

        self.linear3 = torch.nn.Linear(2 * output_dim, 2 * output_dim, bias=True)
        self.BatchNorm3 = nn.BatchNorm1d(2 * output_dim)

        self.linear4 = torch.nn.Linear(2 * output_dim, output_dim, bias=True)

        self.LeakyReLU = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, q_embedding, labels):
        # 把labels变成embeddings
        l_e = self.label_embedding(labels)
        x = torch.cat((q_embedding, l_e), dim=-1)

        x = self.linear1(x)
        x = self.BatchNorm1(x)
        x = self.LeakyReLU(x)

        x = self.linear2(x)
        x = self.BatchNorm2(x)
        x = self.LeakyReLU(x)

        x = self.linear3(x)
        x = self.BatchNorm3(x)
        x = self.LeakyReLU(x)

        x = self.linear4(x)
        x = self.sigmoid(x)

        return x


class ACGAN_D(nn.Module):
    def __init__(self, input_len, num_labels=4):
        super(ACGAN_D, self).__init__()
        self.classifier = SimpleClassifier(input_len=input_len, num_labels=num_labels + 1)

    def forward(self, q_embedding, a_embedding):
        # result = (batch_num, 5)
        result = self.classifier(q_embedding=q_embedding, a_embedding=a_embedding, sigmoid_flag=True)
        # dim = 2 (batch_num, 1)
        real_or_not = (result[:, 0:1]).squeeze(-1)
        # dim = 2 (batch_num, 4)
        class_logits = result[:, 1:]
        return real_or_not, class_logits


# bert cnn
class BertCNN(nn.Module):
    def __init__(self, num_filters=256, filter_sizes=(1, 2, 3), dropout=0.1,
                 model_path='prajjwal1/bert-small', word_embedding_len=512, num_classes=4):

        super(BertCNN, self).__init__()

        self.bert_model = BertModel.from_pretrained(model_path)
        for param in self.bert_model.parameters():
            param.requires_grad = True

        self.convs = nn.ModuleList(
            [nn.Conv2d(1, num_filters, (k, word_embedding_len)) for k in filter_sizes])
        self.dropout = nn.Dropout(dropout)

        self.fc_cnn = nn.Linear(3 * num_filters * len(filter_sizes), num_classes)

    @staticmethod
    def conv_and_pool(x, conv):
        x = nn.functional.relu(conv(x)).squeeze(3)
        x = nn.functional.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, q_input_ids, q_token_type_ids, q_attention_mask,
                a_input_ids, a_token_type_ids, a_attention_mask,
                b_input_ids, b_token_type_ids, b_attention_mask):

        # 获得问题的表示
        q_out = self.bert_model(input_ids=q_input_ids, token_type_ids=q_token_type_ids, attention_mask=q_attention_mask)

        q_embeddings = q_out['last_hidden_state']
        # 获得每个batch的表示
        q_final_rep = None

        for index, embeddings in enumerate(q_embeddings):
            this_attention_mask = (q_attention_mask[index] == 1)
            this_embeddings = embeddings[this_attention_mask]
            # 扩充一下
            padding_tensor = torch.zeros_like(this_embeddings[0], device=this_embeddings.device).unsqueeze(0)
            this_embeddings = (torch.cat((this_embeddings, padding_tensor), dim=0).unsqueeze(0)).unsqueeze(0)

            out = torch.cat([self.conv_and_pool(this_embeddings, conv) for conv in self.convs], 1)

            if q_final_rep is None:
                q_final_rep = out
            else:
                q_final_rep = torch.cat((q_final_rep, out), dim=0)

        # 获得问题的表示
        b_out = self.bert_model(input_ids=b_input_ids, token_type_ids=b_token_type_ids,
                                attention_mask=b_attention_mask)

        b_embeddings = b_out['last_hidden_state']
        # 获得每个batch的表示
        b_final_rep = None

        for index, embeddings in enumerate(b_embeddings):
            this_attention_mask = (b_attention_mask[index] == 1)
            this_embeddings = embeddings[this_attention_mask]
            # 扩充一下
            padding_tensor = torch.zeros_like(this_embeddings[0], device=this_embeddings.device).unsqueeze(0)
            this_embeddings = (torch.cat((this_embeddings, padding_tensor), dim=0).unsqueeze(0)).unsqueeze(0)

            out = torch.cat([self.conv_and_pool(this_embeddings, conv) for conv in self.convs], 1)

            if b_final_rep is None:
                b_final_rep = out
            else:
                b_final_rep = torch.cat((b_final_rep, out), dim=0)

        # 获得问题的表示
        a_out = self.bert_model(input_ids=a_input_ids, token_type_ids=a_token_type_ids,
                                attention_mask=a_attention_mask)

        a_embeddings = a_out['last_hidden_state']
        # 获得每个batch的表示
        a_final_rep = None

        for index, embeddings in enumerate(a_embeddings):
            this_attention_mask = (a_attention_mask[index] == 1)
            this_embeddings = embeddings[this_attention_mask]
            # 扩充一下
            padding_tensor = torch.zeros_like(this_embeddings[0], device=this_embeddings.device).unsqueeze(0)
            this_embeddings = (torch.cat((this_embeddings, padding_tensor), dim=0).unsqueeze(0)).unsqueeze(0)

            out = torch.cat([self.conv_and_pool(this_embeddings, conv) for conv in self.convs], 1)

            if a_final_rep is None:
                a_final_rep = out
            else:
                a_final_rep = torch.cat((a_final_rep, out), dim=0)

        a_final_rep = self.dropout(a_final_rep)
        b_final_rep = self.dropout(b_final_rep)
        q_final_rep = self.dropout(q_final_rep)

        out = self.fc_cnn(torch.cat((q_final_rep, b_final_rep, a_final_rep), dim=1))

        return out


# 用vae生成attention。计算attention时，用的外部向量是mean
class VaeAttention(nn.Module):
    def __init__(self, model_path='prajjwal1/bert-small', latent_dim=100, num_labels=4, input_dim=None,
                 word_embedding_len=512, output_embedding_len=512):

        super(VaeAttention, self).__init__()

        self.output_embedding_len = output_embedding_len

        # 这个学习率不一样
        self.bert_model = BertModel.from_pretrained(model_path)

        # 主题模型
        self.vae_model = VAEQuestion(input_dim=input_dim, latent_dim=latent_dim)

        # 注意力模型
        self.key_layer = nn.Sequential(
            nn.Linear(word_embedding_len + latent_dim, 2*(word_embedding_len + latent_dim)),
            nn.ReLU(),
            nn.Linear(2*(word_embedding_len + latent_dim), output_embedding_len),
            nn.Sigmoid()
        )

        # self.value_layer = nn.Sequential(
        #     nn.Linear(word_embedding_len, output_embedding_len),
        #     nn.ReLU(),
        #     nn.Linear(output_embedding_len, output_embedding_len),
        #     nn.Tanh()
        # )

        self.value_layer = nn.Sequential(
            nn.Linear(word_embedding_len + latent_dim, 2 * (word_embedding_len + latent_dim)),
            nn.ReLU(),
            nn.Linear(2 * (word_embedding_len + latent_dim), output_embedding_len),
            nn.Tanh()
        )

        # 这些的学习率一样
        self.classifier = BodyClassifier(input_len=output_embedding_len, num_labels=num_labels)

        self.softmax = torch.nn.Softmax(dim=-2)

    def get_rep_by_vae(self, input_ids, token_type_ids, attention_mask, latent_value):
        # 获得隐藏层输出
        out = self.bert_model(input_ids=input_ids, token_type_ids=token_type_ids,
                              attention_mask=attention_mask)
        last_hidden_state = out['last_hidden_state']

        # 创作出mask
        with torch.no_grad():
            mask = attention_mask.type(dtype=torch.float)
            mask[mask == 0] = -np.inf
            mask[mask == 1] = 0.0
            mask = mask.repeat(self.output_embedding_len, 1, 1)
            mask.transpose_(0, 1)
            mask.transpose_(1, 2)

        inner_latent_value = latent_value.unsqueeze(1)

        inner_latent_value = inner_latent_value.repeat(1, last_hidden_state.shape[1], 1)

        # (batch, sequence, output_embedding_len)
        weight = self.key_layer(torch.cat((last_hidden_state, inner_latent_value), dim=-1))
        mask_weight = mask + weight
        final_weight = self.softmax(mask_weight)

        # (batch, sequence, output_embedding_len)
        value = self.value_layer(torch.cat((last_hidden_state, inner_latent_value), dim=-1))

        # 求和
        embedding = torch.mul(final_weight, value)

        final_embedding = embedding.sum(dim=-2)

        return final_embedding

    def forward(self, q_input_ids, q_token_type_ids, q_attention_mask,
                a_input_ids, a_token_type_ids, a_attention_mask,
                b_input_ids, b_token_type_ids, b_attention_mask, word_bag):

        # 把词袋丢去训练VAE
        reconstructed_input, mean, log_var, latent_v = self.vae_model(word_bag, out_latent_flag=True)
        vae_loss = calculate_vae_loss(word_bag, reconstructed_input, mean, log_var)

        # 获得表示
        q_embeddings = self.get_rep_by_vae(input_ids=q_input_ids, token_type_ids=q_token_type_ids,
                                           attention_mask=q_attention_mask, latent_value=mean)

        a_embeddings = self.get_rep_by_vae(input_ids=a_input_ids, token_type_ids=a_token_type_ids,
                                           attention_mask=a_attention_mask, latent_value=mean)

        b_embeddings = self.get_rep_by_vae(input_ids=b_input_ids, token_type_ids=b_token_type_ids,
                                           attention_mask=b_attention_mask, latent_value=mean)

        # 计算得到分类概率
        logits = self.classifier(q_embedding=q_embeddings, a_embedding=a_embeddings,
                                 b_embedding=b_embeddings)

        return logits, vae_loss.unsqueeze(0)


# 用vae生成attention。计算attention时，用的外部向量是latent value，而非mean
class VaeAttentionZ(nn.Module):
    def __init__(self, model_path='prajjwal1/bert-small', latent_dim=100, num_labels=4, input_dim=None,
                 word_embedding_len=512, output_embedding_len=512):

        super(VaeAttentionZ, self).__init__()

        self.output_embedding_len = output_embedding_len

        # 这个学习率不一样
        self.bert_model = BertModel.from_pretrained(model_path)

        # 主题模型
        self.vae_model = VAEQuestion(input_dim=input_dim, latent_dim=latent_dim)

        # 注意力模型
        self.key_layer = nn.Sequential(
            nn.Linear(word_embedding_len + latent_dim, 2*(word_embedding_len + latent_dim)),
            nn.ReLU(),
            nn.Linear(2*(word_embedding_len + latent_dim), output_embedding_len),
            nn.Sigmoid()
        )

        self.value_layer = nn.Sequential(
            nn.Linear(word_embedding_len, output_embedding_len),
            nn.ReLU(),
            nn.Linear(output_embedding_len, output_embedding_len),
            nn.Tanh()
        )

        # 这些的学习率一样
        self.classifier = BodyClassifier(input_len=output_embedding_len, num_labels=num_labels)

        self.softmax = torch.nn.Softmax(dim=-2)

    def get_rep_by_vae(self, input_ids, token_type_ids, attention_mask, latent_value):
        # 获得隐藏层输出
        out = self.bert_model(input_ids=input_ids, token_type_ids=token_type_ids,
                              attention_mask=attention_mask)
        last_hidden_state = out['last_hidden_state']

        # 创作出mask
        with torch.no_grad():
            mask = attention_mask.type(dtype=torch.float)
            mask[mask == 0] = -np.inf
            mask[mask == 1] = 0.0
            mask = mask.repeat(self.output_embedding_len, 1, 1)
            mask.transpose_(0, 1)
            mask.transpose_(1, 2)

        inner_latent_value = latent_value.unsqueeze(1)

        inner_latent_value = inner_latent_value.repeat(1, last_hidden_state.shape[1], 1)

        # (batch, sequence, output_embedding_len)
        weight = self.key_layer(torch.cat((last_hidden_state, inner_latent_value), dim=-1))
        mask_weight = mask + weight
        final_weight = self.softmax(mask_weight)

        # (batch, sequence, output_embedding_len)
        value = self.value_layer(last_hidden_state)

        # 求和
        embedding = torch.mul(final_weight, value)

        final_embedding = embedding.sum(dim=-2)

        return final_embedding

    def forward(self, q_input_ids, q_token_type_ids, q_attention_mask,
                a_input_ids, a_token_type_ids, a_attention_mask,
                b_input_ids, b_token_type_ids, b_attention_mask, word_bag):

        # 把词袋丢去训练VAE
        reconstructed_input, mean, log_var, latent_v = self.vae_model(word_bag, out_latent_flag=True)
        vae_loss = calculate_vae_loss(word_bag, reconstructed_input, mean, log_var)

        # 获得表示
        q_embeddings = self.get_rep_by_vae(input_ids=q_input_ids, token_type_ids=q_token_type_ids,
                                           attention_mask=q_attention_mask, latent_value=log_var)

        a_embeddings = self.get_rep_by_vae(input_ids=a_input_ids, token_type_ids=a_token_type_ids,
                                           attention_mask=a_attention_mask, latent_value=log_var)

        b_embeddings = self.get_rep_by_vae(input_ids=b_input_ids, token_type_ids=b_token_type_ids,
                                           attention_mask=b_attention_mask, latent_value=log_var)

        # 计算得到分类概率
        logits = self.classifier(q_embedding=q_embeddings, a_embedding=a_embeddings,
                                 b_embedding=b_embeddings)

        return logits, vae_loss.unsqueeze(0)


# 用vae生成attention。计算attention时，用的外部向量是latent value，而非mean。输入给分类器的有表示的总和
class VaeAttentionZPlus(nn.Module):
    def __init__(self, model_path='prajjwal1/bert-small', latent_dim=100, num_labels=4, input_dim=None,
                 word_embedding_len=512, output_embedding_len=512):

        super(VaeAttentionZPlus, self).__init__()

        self.output_embedding_len = output_embedding_len

        # 这个学习率不一样
        self.bert_model = BertModel.from_pretrained(model_path)

        # 主题模型
        self.vae_model = VAEQuestion(input_dim=input_dim, latent_dim=latent_dim)

        # 注意力模型
        self.key_layer = nn.Sequential(
            nn.Linear(word_embedding_len + latent_dim, 2*(word_embedding_len + latent_dim)),
            nn.ReLU(),
            nn.Linear(2*(word_embedding_len + latent_dim), output_embedding_len),
            nn.Sigmoid()
        )

        self.value_layer = nn.Sequential(
            nn.Linear(word_embedding_len, output_embedding_len),
            nn.ReLU(),
            nn.Linear(output_embedding_len, output_embedding_len),
            nn.Tanh()
        )

        # 这些的学习率一样
        self.classifier = MyClassifier(input_len=output_embedding_len*4, num_labels=num_labels)

        self.softmax = torch.nn.Softmax(dim=-2)

    def get_rep_by_vae(self, input_ids, token_type_ids, attention_mask, latent_value):
        # 获得隐藏层输出
        out = self.bert_model(input_ids=input_ids, token_type_ids=token_type_ids,
                              attention_mask=attention_mask)
        last_hidden_state = out['last_hidden_state']

        # 创作出mask
        with torch.no_grad():
            mask = attention_mask.type(dtype=torch.float)
            mask[mask == 0] = -np.inf
            mask[mask == 1] = 0.0
            mask = mask.repeat(self.output_embedding_len, 1, 1)
            mask.transpose_(0, 1)
            mask.transpose_(1, 2)

        inner_latent_value = latent_value.unsqueeze(1)

        inner_latent_value = inner_latent_value.repeat(1, last_hidden_state.shape[1], 1)

        # (batch, sequence, output_embedding_len)
        weight = self.key_layer(torch.cat((last_hidden_state, inner_latent_value), dim=-1))
        mask_weight = mask + weight
        final_weight = self.softmax(mask_weight)

        # (batch, sequence, output_embedding_len)
        value = self.value_layer(last_hidden_state)

        # 求和
        embedding = torch.mul(final_weight, value)

        final_embedding = embedding.sum(dim=-2)

        return final_embedding

    def forward(self, q_input_ids, q_token_type_ids, q_attention_mask,
                a_input_ids, a_token_type_ids, a_attention_mask,
                b_input_ids, b_token_type_ids, b_attention_mask, word_bag):

        # 把词袋丢去训练VAE
        reconstructed_input, mean, log_var, latent_v = self.vae_model(word_bag, out_latent_flag=True)
        vae_loss = calculate_vae_loss(word_bag, reconstructed_input, mean, log_var)

        # 获得表示
        q_embeddings = self.get_rep_by_vae(input_ids=q_input_ids, token_type_ids=q_token_type_ids,
                                           attention_mask=q_attention_mask, latent_value=log_var)

        a_embeddings = self.get_rep_by_vae(input_ids=a_input_ids, token_type_ids=a_token_type_ids,
                                           attention_mask=a_attention_mask, latent_value=log_var)

        b_embeddings = self.get_rep_by_vae(input_ids=b_input_ids, token_type_ids=b_token_type_ids,
                                           attention_mask=b_attention_mask, latent_value=log_var)

        # 计算得到分类概率
        logits = self.classifier(torch.cat((q_embeddings, a_embeddings, b_embeddings,
                                            q_embeddings + a_embeddings + b_embeddings), dim=-1))

        return logits, vae_loss.unsqueeze(0)


# 在ZPlus的基础上，进行如下修改：bert输出的词向量与topic一起送入感知机，得到加工后的词向量
class VaeAttentionUltra(nn.Module):
    def __init__(self, model_path='prajjwal1/bert-small', latent_dim=100, num_labels=4, input_dim=None,
                 word_embedding_len=512, output_embedding_len=512):

        super(VaeAttentionUltra, self).__init__()

        self.output_embedding_len = output_embedding_len

        # 这个学习率不一样
        self.bert_model = BertModel.from_pretrained(model_path)

        # 主题模型
        self.vae_model = VAEQuestion(input_dim=input_dim, latent_dim=latent_dim)

        # 注意力模型
        self.key_layer = nn.Sequential(
            nn.Linear(word_embedding_len + latent_dim, 2*(word_embedding_len + latent_dim)),
            nn.ReLU(),
            nn.Linear(2*(word_embedding_len + latent_dim), output_embedding_len),
            nn.Sigmoid()
        )

        self.value_layer = nn.Sequential(
            nn.Linear(word_embedding_len + latent_dim, 2 * (word_embedding_len + latent_dim)),
            nn.ReLU(),
            nn.Linear(2 * (word_embedding_len + latent_dim), output_embedding_len),
            nn.Tanh()
        )

        # 这些的学习率一样
        self.classifier = MyClassifier(input_len=output_embedding_len*4, num_labels=num_labels)

        self.softmax = torch.nn.Softmax(dim=-2)

    def get_rep_by_vae(self, input_ids, token_type_ids, attention_mask, latent_value):
        # 获得隐藏层输出
        out = self.bert_model(input_ids=input_ids, token_type_ids=token_type_ids,
                              attention_mask=attention_mask)
        last_hidden_state = out['last_hidden_state']

        # 创作出mask
        with torch.no_grad():
            mask = attention_mask.type(dtype=torch.float)
            mask[mask == 0] = -np.inf
            mask[mask == 1] = 0.0
            mask = mask.repeat(self.output_embedding_len, 1, 1)
            mask.transpose_(0, 1)
            mask.transpose_(1, 2)

        inner_latent_value = latent_value.unsqueeze(1)

        inner_latent_value = inner_latent_value.repeat(1, last_hidden_state.shape[1], 1)

        # (batch, sequence, output_embedding_len)
        weight = self.key_layer(torch.cat((last_hidden_state, inner_latent_value), dim=-1))
        mask_weight = mask + weight
        final_weight = self.softmax(mask_weight)

        # (batch, sequence, output_embedding_len)
        value = self.value_layer(torch.cat((last_hidden_state, inner_latent_value), dim=-1))

        # 求和
        embedding = torch.mul(final_weight, value)

        final_embedding = embedding.sum(dim=-2)

        return final_embedding

    def forward(self, q_input_ids, q_token_type_ids, q_attention_mask,
                a_input_ids, a_token_type_ids, a_attention_mask,
                b_input_ids, b_token_type_ids, b_attention_mask, word_bag):

        # 把词袋丢去训练VAE
        reconstructed_input, mean, log_var, latent_v = self.vae_model(word_bag, out_latent_flag=True)
        vae_loss = calculate_vae_loss(word_bag, reconstructed_input, mean, log_var)

        # 获得表示
        q_embeddings = self.get_rep_by_vae(input_ids=q_input_ids, token_type_ids=q_token_type_ids,
                                           attention_mask=q_attention_mask, latent_value=mean)

        a_embeddings = self.get_rep_by_vae(input_ids=a_input_ids, token_type_ids=a_token_type_ids,
                                           attention_mask=a_attention_mask, latent_value=mean)

        b_embeddings = self.get_rep_by_vae(input_ids=b_input_ids, token_type_ids=b_token_type_ids,
                                           attention_mask=b_attention_mask, latent_value=mean)

        # 计算得到分类概率
        logits = self.classifier(torch.cat((q_embeddings, a_embeddings, b_embeddings,
                                            q_embeddings + a_embeddings + b_embeddings), dim=-1))

        return logits, vae_loss.unsqueeze(0)


# 用vae生成attention，用了两个bert
class VaeWordDoubleAttention(nn.Module):
    def __init__(self, model_path='prajjwal1/bert-small', latent_dim=100, num_labels=4, input_dim=None,
                 word_embedding_len=512, output_embedding_len=512):

        super(VaeWordDoubleAttention, self).__init__()

        self.output_embedding_len = output_embedding_len

        # 这个学习率不一样
        self.bert_model = BertModel.from_pretrained(model_path)
        self.bert_model1 = BertModel.from_pretrained(model_path)

        # 主题模型
        self.vae_model = VAEQuestion(input_dim=input_dim, latent_dim=latent_dim)

        # 注意力模型
        self.key_layer = nn.Sequential(
            nn.Linear(word_embedding_len + latent_dim, 2*(word_embedding_len + latent_dim)),
            nn.ReLU(),
            nn.Linear(2*(word_embedding_len + latent_dim), output_embedding_len),
            nn.Sigmoid()
        )

        self.value_layer = nn.Sequential(
            nn.Linear(word_embedding_len, output_embedding_len),
            nn.ReLU(),
            nn.Linear(output_embedding_len, output_embedding_len),
            nn.Tanh()
        )

        # 这些的学习率一样
        self.classifier = BodyClassifier(input_len=output_embedding_len, num_labels=num_labels)

        self.softmax = torch.nn.Softmax(dim=-2)

    def get_rep_by_vae(self, input_ids, token_type_ids, attention_mask, latent_value, is_question=True):
        # 获得隐藏层输出
        if is_question:
            out = self.bert_model(input_ids=input_ids, token_type_ids=token_type_ids,
                                  attention_mask=attention_mask)
        else:
            out = self.bert_model1(input_ids=input_ids, token_type_ids=token_type_ids,
                                   attention_mask=attention_mask)

        last_hidden_state = out['last_hidden_state']

        # 创作出mask
        with torch.no_grad():
            mask = attention_mask.type(dtype=torch.float)
            mask[mask == 0] = -np.inf
            mask[mask == 1] = 0.0
            mask = mask.repeat(self.output_embedding_len, 1, 1)
            mask.transpose_(0, 1)
            mask.transpose_(1, 2)

        inner_latent_value = latent_value.unsqueeze(1)

        inner_latent_value = inner_latent_value.repeat(1, last_hidden_state.shape[1], 1)

        # (batch, sequence, output_embedding_len)
        weight = self.key_layer(torch.cat((last_hidden_state, inner_latent_value), dim=-1))
        mask_weight = mask + weight
        final_weight = self.softmax(mask_weight)

        # (batch, sequence, output_embedding_len)
        value = self.value_layer(last_hidden_state)

        # 求和
        embedding = torch.mul(final_weight, value)

        final_embedding = embedding.sum(dim=-2)

        return final_embedding

    def forward(self, q_input_ids, q_token_type_ids, q_attention_mask,
                a_input_ids, a_token_type_ids, a_attention_mask,
                b_input_ids, b_token_type_ids, b_attention_mask, word_bag):

        # 把词袋丢去训练VAE
        reconstructed_input, mean, log_var, latent_v = self.vae_model(word_bag, out_latent_flag=True)
        vae_loss = calculate_vae_loss(word_bag, reconstructed_input, mean, log_var)

        # 获得表示
        q_embeddings = self.get_rep_by_vae(input_ids=q_input_ids, token_type_ids=q_token_type_ids,
                                           attention_mask=q_attention_mask, latent_value=mean, is_question=True)

        a_embeddings = self.get_rep_by_vae(input_ids=a_input_ids, token_type_ids=a_token_type_ids,
                                           attention_mask=a_attention_mask, latent_value=mean, is_question=False)

        b_embeddings = self.get_rep_by_vae(input_ids=b_input_ids, token_type_ids=b_token_type_ids,
                                           attention_mask=b_attention_mask, latent_value=mean, is_question=True)

        # 计算得到分类概率
        logits = self.classifier(q_embedding=q_embeddings, a_embedding=a_embeddings,
                                 b_embedding=b_embeddings)

        return logits, vae_loss.unsqueeze(0)


# 用了两个bert，一个处理问题，一个处理答案，然后问题和答案的补充向量不一样
class VaeJustForQuestionAttention(nn.Module):
    def __init__(self, model_path='prajjwal1/bert-small', latent_dim=100, num_labels=4, input_dim=None,
                 word_embedding_len=512, output_embedding_len=512):

        super(VaeJustForQuestionAttention, self).__init__()

        self.output_embedding_len = output_embedding_len

        # 这个学习率不一样
        self.bert_model = BertModel.from_pretrained(model_path)

        # 主题模型
        self.vae_model = VAEQuestion(input_dim=input_dim, latent_dim=latent_dim)

        # 注意力模型
        self.key_layer_by_topic = nn.Sequential(
            nn.Linear(word_embedding_len + latent_dim, 2*output_embedding_len),
            nn.ReLU(),
            nn.Linear(2*output_embedding_len, output_embedding_len),
            nn.Sigmoid()
        )

        self.key_layer_by_rep = nn.Sequential(
            nn.Linear(word_embedding_len + output_embedding_len, 2 * output_embedding_len),
            nn.ReLU(),
            nn.Linear(2 * output_embedding_len, output_embedding_len),
            nn.Sigmoid()
        )

        self.value_layer = nn.Sequential(
            nn.Linear(word_embedding_len, 2*output_embedding_len),
            nn.ReLU(),
            nn.Linear(2*output_embedding_len, output_embedding_len),
            nn.Tanh()
        )

        # 这些的学习率一样
        self.classifier = BodyClassifier(input_len=output_embedding_len, num_labels=num_labels)

        self.softmax = torch.nn.Softmax(dim=-2)

    def get_rep_by_vae(self, input_ids, token_type_ids, attention_mask, attention_evidence, by_rep=True):
        # 获得隐藏层输出
        out = self.bert_model(input_ids=input_ids, token_type_ids=token_type_ids,
                              attention_mask=attention_mask)

        last_hidden_state = out['last_hidden_state']

        # 创作出mask
        with torch.no_grad():
            mask = attention_mask.type(dtype=torch.float)
            mask[mask == 0] = -np.inf
            mask[mask == 1] = 0.0
            mask = mask.repeat(self.output_embedding_len, 1, 1)
            mask.transpose_(0, 1)
            mask.transpose_(1, 2)

        # -> (batch, 1, latent_len)
        inner_attention_evidence = attention_evidence.unsqueeze(1)

        inner_attention_evidence = inner_attention_evidence.repeat(1, last_hidden_state.shape[1], 1)

        # (batch, sequence, output_embedding_len)
        if by_rep:
            weight = self.key_layer_by_rep(torch.cat((last_hidden_state, inner_attention_evidence), dim=-1))
        else:
            weight = self.key_layer_by_topic(torch.cat((last_hidden_state, inner_attention_evidence), dim=-1))

        mask_weight = mask + weight
        final_weight = self.softmax(mask_weight)

        # (batch, sequence, output_embedding_len)
        value = self.value_layer(last_hidden_state)

        # 求和
        embedding = torch.mul(final_weight, value)

        final_embedding = embedding.sum(dim=-2)

        return final_embedding

    def forward(self, q_input_ids, q_token_type_ids, q_attention_mask,
                a_input_ids, a_token_type_ids, a_attention_mask,
                b_input_ids, b_token_type_ids, b_attention_mask, word_bag):

        # 把词袋丢去训练VAE
        reconstructed_input, mean, log_var, latent_v = self.vae_model(word_bag, out_latent_flag=True)
        vae_loss = calculate_vae_loss(word_bag, reconstructed_input, mean, log_var)

        # 获得表示
        q_embeddings = self.get_rep_by_vae(input_ids=q_input_ids, token_type_ids=q_token_type_ids,
                                           attention_mask=q_attention_mask,
                                           attention_evidence=mean, by_rep=False)

        b_embeddings = self.get_rep_by_vae(input_ids=b_input_ids, token_type_ids=b_token_type_ids,
                                           attention_mask=b_attention_mask,
                                           attention_evidence=mean, by_rep=False)

        a_embeddings = self.get_rep_by_vae(input_ids=a_input_ids, token_type_ids=a_token_type_ids,
                                           attention_mask=a_attention_mask,
                                           attention_evidence=b_embeddings, by_rep=True)

        # 计算得到分类概率
        logits = self.classifier(q_embedding=q_embeddings, a_embedding=a_embeddings,
                                 b_embedding=b_embeddings)

        return logits, vae_loss.unsqueeze(0)


# 三层简单分类器
class MyClassifier(nn.Module):
    def __init__(self, input_len, keep_prob=0.9, num_labels=4):
        super(MyClassifier, self).__init__()

        self.linear1 = torch.nn.Linear(input_len, input_len * 2, bias=True)
        self.bn1 = nn.BatchNorm1d(input_len * 2)

        self.linear2 = torch.nn.Linear(input_len * 2, input_len*2, bias=True)
        self.bn2 = nn.BatchNorm1d(input_len*2)

        self.linear3 = torch.nn.Linear(input_len*2, num_labels, bias=True)

        self.relu = torch.nn.ReLU()
        self.tanh = torch.nn.Tanh()
        self.sigmoid = torch.nn.Sigmoid()
        self.dropout = torch.nn.Dropout(p=1 - keep_prob)
        self.init_weights()

    def init_weights(self):
        torch.nn.init.xavier_uniform_(self.linear1.weight)
        torch.nn.init.xavier_uniform_(self.linear2.weight)
        torch.nn.init.xavier_uniform_(self.linear3.weight)

    # sigmoid_flag=False意味着不处理直接输出
    def forward(self, my_input, sigmoid_flag=False):
        x = self.linear1(my_input)
        x = self.relu(x)
        x = self.bn1(x)

        x = self.linear2(x)
        x = self.relu(x)
        x = self.bn2(x)

        x = self.linear3(x)
        if sigmoid_flag:
            x = self.sigmoid(x)

        return x


# 动态卷积核初尝试
class DynamicCNN(nn.Module):
    def __init__(self, model_path='prajjwal1/bert-small',
                 word_embedding_len=512, initial_filter_num=128, feature_len=128, conv_layer_num=9, norm='layer',
                 least_filter_num=16, head_num=1, query_len=512):

        super(DynamicCNN, self).__init__()

        # 记录参数
        self.word_embedding_len = word_embedding_len
        self.conv_layer_num = conv_layer_num
        self.feature_len = feature_len
        self.norm = norm
        self.initial_filter_num = initial_filter_num
        self.least_filter_num = least_filter_num
        self.query_len = query_len
        self.head_num = head_num

        # 这个学习率不一样
        self.q_bert_model = BertModel.from_pretrained(model_path)
        self.a_bert_model = BertModel.from_pretrained(model_path)

        # 存储用来生成卷积核的模型
        self.generate_filter = nn.ModuleList()
        self.norms_layers = nn.ModuleList()

        self.create_generate_filter()

        # 分类器，输入是CNN提取的文本答案以及问题的表示
        self.classifier = MyClassifier(input_len=(feature_len*head_num))

        # 下面这两个模型通过自注意力来生成问题的query
        # 这里的head_num加了1，多出来的一个用来作为问题的表示
        self.key_layer = nn.Sequential(
            nn.Linear(word_embedding_len, query_len*head_num*2),
            nn.ReLU(),
            nn.Linear(query_len*head_num*2, query_len*head_num),
            nn.Tanh()
        )

        self.value_layer = nn.Sequential(
            nn.Linear(word_embedding_len, query_len * head_num * 2),
            nn.ReLU(),
            nn.Linear(query_len * head_num * 2, query_len*head_num),
            nn.Tanh()
        )

        self.softmax = torch.nn.Softmax(dim=-3)

    def create_generate_filter(self):
        # 这一个轮次卷积核的个数
        next_filter_num = self.initial_filter_num
        # 上一个轮次卷积核的个数
        last_filter_num = self.query_len

        for i in range(1, self.conv_layer_num+1):
            # 获得batch norm
            if self.norm == 'layer':
                self.norms_layers.append(torch.nn.LayerNorm(next_filter_num))
            elif self.norm == 'batch':
                self.norms_layers.append(nn.BatchNorm2d(next_filter_num))
            else:
                print("Error: norm error!")
                print("Error: norm error!")
                exit()

            # 这些线性层用来生成线性层的参数和bias
            if i == 1:
                self.generate_filter.append(
                    nn.Linear(self.query_len,
                              (last_filter_num + 1) * next_filter_num + self.feature_len * (next_filter_num + 1)))

                last_filter_num = next_filter_num
                next_filter_num = max(self.least_filter_num, int(next_filter_num/2))
            else:
                self.generate_filter.append(
                    nn.Linear(self.query_len,
                              (2 * last_filter_num + 1) * next_filter_num + self.feature_len * (next_filter_num + 1)))

                last_filter_num = next_filter_num
                next_filter_num = max(self.least_filter_num, int(next_filter_num / 2))

    # 根据q的输出，生成他的关注向量，每个关注向量都可以去答案里通过卷积网络提取特征
    # 输入是(batch, sequence, query_len), (batch, sequence, query_len*head_num)
    # 输出是(batch, head_num, query_len)
    def get_q_query(self, q_embeddings, mask):
        # (batch, sequence, query_len*head_num)
        weight = self.key_layer(q_embeddings)
        mask_weight = mask + weight

        # (batch, sequence, head_num, query_len)
        mask_weight = mask_weight.view(mask_weight.shape[0], mask_weight.shape[1], self.head_num, self.query_len)
        final_weight = self.softmax(mask_weight)

        # (batch, sequence, head_num, query_len)
        value = self.value_layer(q_embeddings)
        value = value.view(value.shape[0], value.shape[1], self.head_num, self.query_len)

        # (batch, sequence, head_num, query_len)
        query = torch.mul(final_weight, value)
        final_query = query.sum(dim=-3)

        return final_query

    def get_rep_by_q(self, input_ids, token_type_ids, attention_mask, q_queries):
        # 获得答案的隐藏层输出，(batch, sequence, embedding)
        out = self.a_bert_model(input_ids=input_ids, token_type_ids=token_type_ids,
                                attention_mask=attention_mask)
        last_hidden_state = out['last_hidden_state']

        # 创作出mask, (batch, sequence, embedding)
        with torch.no_grad():
            mask = attention_mask.type(dtype=torch.float)
            mask = mask.repeat(self.query_len, 1, 1)
            mask.transpose_(0, 1)
            mask.transpose_(1, 2)

        # 把所有padding清零
        last_hidden_state = torch.mul(last_hidden_state, mask)

        # 假设所有句子都加了开头和结尾的特殊符号，所以顶得住size是2的卷积核
        # 假设输入的句子长度都被padding到了9
        max_sequence_len = last_hidden_state.shape[1]

        # 依此生成卷积核来卷积输入
        # (batch, sequence_len, embedding_len)
        my_input = last_hidden_state
        my_input.unsqueeze_(1)
        # (batch, head_num, sequence_len, embedding_len)
        my_input = my_input.repeat(1, self.head_num, 1, 1)

        output_features = None

        # 这一个轮次提取特征的数量
        this_embedding_num = self.initial_filter_num

        for my_index, my_layer in enumerate(self.generate_filter):
            if my_index == max_sequence_len:
                break
            # 根据输入的q_embeddings生成卷积核参数
            # (batch_size, head_num, para)
            my_parameter = my_layer(q_queries)

            # 取出用来卷积的参数
            conv_parameter = my_parameter[:, :, :-(self.feature_len * (this_embedding_num + 1))]

            # 调整参数形状，第一次的话卷积核只卷积单字，之后的话是双字
            if my_index == 0:
                # bias(batch_size, head_num, 1, output_len)
                conv_bias = (conv_parameter[:, :, -this_embedding_num:]).unsqueeze(-2)

                # 变成 (batch_size, head_num, input_len, output_len)
                conv_parameter = conv_parameter[:, :, :-this_embedding_num]
                conv_parameter = conv_parameter.view(-1, self.head_num, my_input.shape[-1], this_embedding_num)

                # 把所有padding清零的mask，放在这里是为了共用if判断
                with torch.no_grad():
                    mask = attention_mask.type(dtype=torch.float)
                    mask = mask.repeat(this_embedding_num, 1, 1)
                    mask.transpose_(0, 1)
                    mask.transpose_(1, 2)
                    mask.unsqueeze_(1)
                    mask = mask.repeat(1, self.head_num, 1, 1)
            else:
                # bias
                conv_bias = conv_parameter[:, :, -this_embedding_num:].unsqueeze(-2)

                # (batch_size, head_num, input_len, output_len)
                conv_parameter = conv_parameter[:, :, :-this_embedding_num]
                conv_parameter = conv_parameter.view(-1, self.head_num, 2*my_input.shape[-1], this_embedding_num)

                # 要对my_input加工一下，毕竟二元卷积，(batch, head_num, sequence_len, embedding_len)
                # 不拿第一行，我这操作真妙啊
                temp_input = my_input[:, :, 1:, :]
                my_input = torch.cat((my_input[:, :, :-1, :], temp_input), dim=-1)

                # 把所有padding清零
                with torch.no_grad():
                    mask = attention_mask.type(dtype=torch.float)
                    mask = mask.repeat(this_embedding_num, 1, 1)
                    mask.transpose_(0, 1)
                    mask.transpose_(1, 2)
                    mask = mask[:, :-my_index, :]
                    mask.unsqueeze_(1)
                    mask = mask.repeat(1, self.head_num, 1, 1)

            # 开始卷积
            # (batch_size, head_num, sequence_len-my_index, this_embedding_num)
            my_input = nn.functional.relu(torch.matmul(my_input, conv_parameter) + conv_bias)
            my_input = torch.mul(my_input, mask)

            # batch normalization一下，这时候调整一下，把this_embedding_num当作通道数，也就是说同一个特征在一个通道
            # batch是不太好，但layer呢
            if self.norm == 'layer':
                my_input = self.norms_layers[my_index](my_input)
            elif self.norm == 'batch':
                my_input.transpose_(1, 2).unsqueeze_(-1)
                my_input = self.norms_layers[my_index](my_input)
                my_input.squeeze_(-1).transpose_(1, 2)

            # 接下来通过最大池化，然后线性层，把这层提取出的特征转换成固定长的向量输出
            # 取出用来输出的参数
            # (batch_size, head_num, para)
            feature_parameter = my_parameter[:, :, -(self.feature_len * (this_embedding_num + 1)):]

            # (batch_size, head_num, 1, feature_len)
            feature_bias = feature_parameter[:, :, -self.feature_len:].unsqueeze(-2)

            # (batch_size, head_num, input_len, feature_len)
            feature_parameter = feature_parameter[:, :, :-self.feature_len]
            feature_parameter = feature_parameter.view(-1, self.head_num, this_embedding_num, self.feature_len)

            # 取出最大特征
            # 是否要用那个attention_mask过滤掉padding部分？
            # (batch_size, head_num, 1, input_len)
            max_feature = torch.nn.functional.max_pool2d(my_input, (my_input.shape[-2], 1))

            # 加权求和，这里可以考虑拼接
            if output_features is None:
                output_features = nn.functional.relu(torch.matmul(max_feature, feature_parameter) +
                                                     feature_bias).squeeze(-2)
            else:
                output_features += nn.functional.relu(torch.matmul(max_feature, feature_parameter) +
                                                      feature_bias).squeeze(-2)
                # output_features = torch.cat((output_features,
                #                              nn.functional.relu(torch.matmul(max_feature, feature_parameter) +
                #                                                 feature_bias).squeeze(-2)), dim=-1)

            # 调整下一次表示的长度
            this_embedding_num = max(self.least_filter_num, int(this_embedding_num / 2))

        return output_features

    def forward(self, q_input_ids, q_token_type_ids, q_attention_mask,
                a_input_ids, a_token_type_ids, a_attention_mask):

        # 获得问题的cls
        temp_output = self.q_bert_model(input_ids=q_input_ids, token_type_ids=q_token_type_ids,
                                        attention_mask=q_attention_mask)
        q_embeddings = temp_output['last_hidden_state']

        # 创作出mask
        with torch.no_grad():
            mask = q_attention_mask.type(dtype=torch.float)
            mask[mask == 0] = -np.inf
            mask[mask == 1] = 0.0
            mask = mask.repeat(self.query_len*self.head_num, 1, 1)
            mask.transpose_(0, 1)
            mask.transpose_(1, 2)

        q_queries = self.get_q_query(q_embeddings, mask)

        # 获得表示
        a_rep = self.get_rep_by_q(input_ids=a_input_ids, token_type_ids=a_token_type_ids,
                                  attention_mask=a_attention_mask, q_queries=q_queries)

        a_rep = a_rep.view(-1, self.head_num*self.feature_len)

        # 分类
        q_queries = q_queries.view(q_queries.shape[0], -1)
        # logits = self.classifier(torch.cat((a_rep, q_queries), dim=-1))
        logits = self.classifier(a_rep)

        return logits


# 简单来说，通过补充向量生成一层，然后充当attention的功能来处理输入向量
class DynamicAttention(nn.Module):
    def __init__(self, model_path='prajjwal1/bert-small', latent_dim=100, num_labels=4, input_dim=None,
                 word_embedding_len=512, middle_len=128, output_embedding_len=128):

        super(DynamicAttention, self).__init__()
        self.middle_len = middle_len
        self.output_embedding_len = output_embedding_len
        self.word_embedding_len = word_embedding_len

        # 这个学习率不一样
        self.bert_model = BertModel.from_pretrained(model_path)

        # 主题模型
        self.vae_model = VAEQuestion(input_dim=input_dim, latent_dim=latent_dim)

        # 用来计算得到多维注意力的感知机
        self.key_layer = nn.Sequential(
            nn.Linear(middle_len + latent_dim, 2*(middle_len + latent_dim)),
            nn.ReLU(),
            nn.Linear(2*(middle_len + latent_dim), output_embedding_len),
            nn.Sigmoid()
        )

        # 这个不一定需要
        self.value_layer = nn.Sequential(
            nn.Linear(middle_len, output_embedding_len),
            nn.ReLU(),
            nn.Linear(output_embedding_len, output_embedding_len),
            nn.Tanh()
        )

        # 生成一层，然后用来处理词向量
        self.generate_layer = nn.Sequential(
            nn.Linear(latent_dim, latent_dim*2),
            nn.ReLU(),
            nn.Linear(latent_dim*2, word_embedding_len*middle_len + middle_len),
            nn.Tanh()
        )

        # 这些的学习率一样
        self.classifier = BodyClassifier(input_len=output_embedding_len, num_labels=num_labels)

        self.softmax = torch.nn.Softmax(dim=-2)

    def get_rep_dynamically(self, input_ids, token_type_ids, attention_mask, supplement_vector):
        # 获得隐藏层输出
        out = self.bert_model(input_ids=input_ids, token_type_ids=token_type_ids,
                              attention_mask=attention_mask)
        last_hidden_state = out['last_hidden_state']

        # 根据supplement_vector生成一层
        generated_layer_parameter = self.generate_layer(supplement_vector)
        dynamic_net_parameter = generated_layer_parameter[:, :-self.middle_len]
        # (batch_size, middle_len, output_len)
        dynamic_net_parameter = dynamic_net_parameter.view(-1, self.word_embedding_len, self.middle_len)

        dynamic_bias = generated_layer_parameter[:, -self.middle_len:].unsqueeze(-2)

        # 对bert输出的词向量进行加工
        my_input = nn.functional.relu(torch.matmul(last_hidden_state, dynamic_net_parameter) + dynamic_bias)

        # 创作出mask
        with torch.no_grad():
            mask = attention_mask.type(dtype=torch.float)
            mask[mask == 0] = -np.inf
            mask[mask == 1] = 0.0
            mask = mask.repeat(self.output_embedding_len, 1, 1)
            mask.transpose_(0, 1)
            mask.transpose_(1, 2)

        inner_latent_value = supplement_vector.unsqueeze(1)
        inner_latent_value = inner_latent_value.repeat(1, my_input.shape[1], 1)

        # (batch, sequence, output_embedding_len)
        # 生成多维注意力
        weight = self.key_layer(torch.cat((my_input, inner_latent_value), dim=-1))
        mask_weight = mask + weight
        final_weight = self.softmax(mask_weight)

        # (batch, sequence, output_embedding_len)
        value = self.value_layer(my_input)

        # 求和
        embedding = torch.mul(final_weight, value)

        final_embedding = embedding.sum(dim=-2)

        return final_embedding

    def forward(self, q_input_ids, q_token_type_ids, q_attention_mask,
                a_input_ids, a_token_type_ids, a_attention_mask,
                b_input_ids, b_token_type_ids, b_attention_mask, word_bag):

        # 把词袋丢去训练VAE
        reconstructed_input, mean, log_var, latent_v = self.vae_model(word_bag, out_latent_flag=True)
        vae_loss = calculate_vae_loss(word_bag, reconstructed_input, mean, log_var)

        # 获得表示
        q_embeddings = self.get_rep_dynamically(input_ids=q_input_ids, token_type_ids=q_token_type_ids,
                                                attention_mask=q_attention_mask, supplement_vector=mean)

        a_embeddings = self.get_rep_dynamically(input_ids=a_input_ids, token_type_ids=a_token_type_ids,
                                                attention_mask=a_attention_mask, supplement_vector=mean)

        b_embeddings = self.get_rep_dynamically(input_ids=b_input_ids, token_type_ids=b_token_type_ids,
                                                attention_mask=b_attention_mask, supplement_vector=mean)

        # 计算得到分类概率
        logits = self.classifier(q_embedding=q_embeddings, a_embedding=a_embeddings,
                                 b_embedding=b_embeddings)

        return logits, vae_loss.unsqueeze(0)


# 用vae生成attention。是VaeAttentionZ的升级版
class InstantMemoryModel(nn.Module):
    def __init__(self, model_path='prajjwal1/bert-small', latent_dim=100, num_labels=4, input_dim=None,
                 word_embedding_len=512, sentence_embedding_len=512, feature_len=1024, hint_len=512, drop_prob=0.5):

        super(InstantMemoryModel, self).__init__()

        # 毕竟num_label也算是memory的一部分
        self.feature_len = feature_len
        self.num_labels = num_labels
        self.sentence_embedding_len = sentence_embedding_len
        self.hint_len = hint_len

        # 这个学习率不一样
        self.bert_model = BertModel.from_pretrained(model_path)

        # 主题模型
        self.vae_model = VAEQuestion(input_dim=input_dim, latent_dim=latent_dim)

        # 注意力模型
        self.key_layer = nn.Sequential(
            nn.Linear(word_embedding_len + latent_dim, 2*(word_embedding_len + latent_dim)),
            nn.ReLU(),
            nn.Linear(2*(word_embedding_len + latent_dim), sentence_embedding_len),
            nn.Sigmoid()
        )

        self.value_layer = nn.Sequential(
            nn.Linear(word_embedding_len, sentence_embedding_len),
            nn.ReLU(),
            nn.Linear(sentence_embedding_len, sentence_embedding_len),
            nn.Tanh()
        )

        # 前面是表示学习，模拟外部世界的输入，下面是行动和特征学习
        self.thinking_layer = nn.Sequential(
            nn.Linear(sentence_embedding_len * 3,
                      2 * sentence_embedding_len * 3),
            nn.ReLU(),
            nn.Linear(2 * sentence_embedding_len * 3,
                      2 * sentence_embedding_len * 3),
            nn.ReLU(),
            nn.Linear(2 * sentence_embedding_len * 3,
                      2 * sentence_embedding_len * 3),
            nn.ReLU(),
            nn.Linear(2 * sentence_embedding_len * 3,
                      feature_len),
            nn.Dropout(p=drop_prob),
            nn.ReLU()
        )

        self.remember_layer = nn.Sequential(
            # nn.Linear(feature_len, 2*feature_len),
            # nn.Tanh(),
            nn.Linear(hint_len, sentence_embedding_len*3),
            nn.ReLU()
        )

        self.classifier_layer = nn.Sequential(
            # nn.Linear(feature_len, 2*feature_len),
            # nn.Tanh(),
            nn.Linear(feature_len, num_labels),
            # nn.ReLU()
        )

        self.relu = torch.nn.ReLU(inplace=True)
        self.softmax = torch.nn.Softmax(dim=-2)
        self.dropout = nn.Dropout(p=drop_prob)

    def get_rep_by_vae(self, input_ids, token_type_ids, attention_mask, latent_value):
        # 获得隐藏层输出
        out = self.bert_model(input_ids=input_ids, token_type_ids=token_type_ids,
                              attention_mask=attention_mask)
        last_hidden_state = out['last_hidden_state']

        # 创作出mask
        with torch.no_grad():
            mask = attention_mask.type(dtype=torch.float)
            mask[mask == 0] = -np.inf
            mask[mask == 1] = 0.0
            mask = mask.repeat(self.sentence_embedding_len, 1, 1)
            mask.transpose_(0, 1)
            mask.transpose_(1, 2)

        inner_latent_value = latent_value.unsqueeze(1)

        inner_latent_value = inner_latent_value.repeat(1, last_hidden_state.shape[1], 1)

        # (batch, sequence, output_embedding_len)
        weight = self.key_layer(torch.cat((last_hidden_state, inner_latent_value), dim=-1))
        mask_weight = mask + weight
        final_weight = self.softmax(mask_weight)

        # (batch, sequence, output_embedding_len)
        value = self.value_layer(last_hidden_state)

        # 求和
        embedding = torch.mul(final_weight, value)

        final_embedding = self.relu(embedding.sum(dim=-2))

        return final_embedding

    def forward(self, q_input_ids, q_token_type_ids, q_attention_mask,
                a_input_ids, a_token_type_ids, a_attention_mask,
                b_input_ids, b_token_type_ids, b_attention_mask, word_bag):

        # 把词袋丢去训练VAE
        reconstructed_input, mean, log_var, latent_v = self.vae_model(word_bag, out_latent_flag=True)
        vae_loss = calculate_vae_loss(word_bag, reconstructed_input, mean, log_var)

        # 获得表示
        q_embeddings = self.get_rep_by_vae(input_ids=q_input_ids, token_type_ids=q_token_type_ids,
                                           attention_mask=q_attention_mask, latent_value=log_var)

        a_embeddings = self.get_rep_by_vae(input_ids=a_input_ids, token_type_ids=a_token_type_ids,
                                           attention_mask=a_attention_mask, latent_value=log_var)

        b_embeddings = self.get_rep_by_vae(input_ids=b_input_ids, token_type_ids=b_token_type_ids,
                                           attention_mask=b_attention_mask, latent_value=log_var)

        # 根据输入，进行思考, 思考的结果要选择性遗忘
        original_input = torch.cat((q_embeddings, b_embeddings, a_embeddings), dim=-1)
        thinking_result = self.thinking_layer(original_input)

        # 根据思想，进行分类
        classify_result = self.classifier_layer(thinking_result)

        # 随机遗忘部分思想，然后尝试回忆起看到的东西
        # forgotten_thinking = self.dropout(thinking_result)
        remember_result = self.remember_layer(thinking_result[:, :self.hint_len])

        # 得到回忆损失
        remember_loss = nn.functional.mse_loss(remember_result, original_input,  reduction='mean')

        return classify_result, vae_loss.unsqueeze(0), remember_loss.unsqueeze(0)


class PureMemory(nn.Module):
    def __init__(self, model_path='prajjwal1/bert-small',  latent_dim=100, num_labels=4, input_dim=None,
                 word_embedding_len=512, sentence_embedding_len=512, drop_prob=0.5, memory_num=10):

        super(PureMemory, self).__init__()

        # 毕竟num_label也算是memory的一部分
        self.num_labels = num_labels
        self.sentence_embedding_len = sentence_embedding_len

        # 这个学习率不一样
        self.bert_model = BertModel.from_pretrained(model_path)

        # 记忆力模块
        self.query_for_answer = nn.Parameter(torch.randn(memory_num, word_embedding_len, device='cuda:0'))
        self.memory_for_answer = nn.Parameter(torch.randn(memory_num, word_embedding_len, device='cuda:0'))

        self.query_for_question = nn.Parameter(torch.randn(memory_num, word_embedding_len, device='cuda:0'))
        self.memory_for_question = nn.Parameter(torch.randn(memory_num, word_embedding_len, device='cuda:0'))

        # 主题模型
        self.vae_model = VAEQuestion(input_dim=input_dim, latent_dim=latent_dim)

        # 注意力模型
        self.key_layer = nn.Sequential(
            nn.Linear(word_embedding_len + latent_dim, 2*(word_embedding_len + latent_dim)),
            nn.ReLU(),
            nn.Linear(2*(word_embedding_len + latent_dim), sentence_embedding_len),
            nn.Sigmoid()
        )

        # self.value_layer = nn.Sequential(
        #     nn.Linear(word_embedding_len, sentence_embedding_len),
        #     nn.ReLU(),
        #     nn.Linear(sentence_embedding_len, sentence_embedding_len),
        #     nn.Tanh()
        # )

        self.value_layer = nn.Sequential(
            nn.Linear(word_embedding_len + latent_dim, 2 * (sentence_embedding_len + latent_dim)),
            nn.ReLU(),
            nn.Linear(2 * (sentence_embedding_len + latent_dim), sentence_embedding_len),
            nn.Tanh()
        )

        self.classifier = BodyClassifier(input_len=sentence_embedding_len, num_labels=num_labels)

        self.relu = torch.nn.ReLU(inplace=True)
        self.softmax = torch.nn.Softmax(dim=-2)
        self.dropout = nn.Dropout(p=drop_prob)

    def get_rep_by_vae(self, input_ids, token_type_ids, attention_mask, latent_value, is_question):
        # 获得隐藏层输出
        out = self.bert_model(input_ids=input_ids, token_type_ids=token_type_ids,
                              attention_mask=attention_mask)
        last_hidden_state = out['last_hidden_state']

        # 根据记忆丰富一下信息，之后可以考虑把latent一起传进去
        if is_question:
            contexts = self.query_for_question.repeat(last_hidden_state.shape[0], 1, 1)
            values = self.memory_for_question.repeat(last_hidden_state.shape[0], 1, 1)

            enrich_info = attend(query=last_hidden_state, context=contexts, value=values)
        else:
            contexts = self.query_for_answer.repeat(last_hidden_state.shape[0], 1, 1)
            values = self.memory_for_answer.repeat(last_hidden_state.shape[0], 1, 1)

            enrich_info = attend(query=last_hidden_state, context=contexts, value=values)

        # 这一步也有点草率
        last_hidden_state = enrich_info + last_hidden_state

        # 开始根据主题分布，进行信息压缩
        # 创作出mask
        with torch.no_grad():
            mask = attention_mask.type(dtype=torch.float)
            mask[mask == 0] = -np.inf
            mask[mask == 1] = 0.0
            mask = mask.repeat(self.sentence_embedding_len, 1, 1)
            mask.transpose_(0, 1)
            mask.transpose_(1, 2)

        # 根据主题分布计算权重
        inner_latent_value = latent_value.unsqueeze(1)
        inner_latent_value = inner_latent_value.repeat(1, last_hidden_state.shape[1], 1)

        # (batch, sequence, output_embedding_len)
        weight = self.key_layer(torch.cat((last_hidden_state, inner_latent_value), dim=-1))
        mask_weight = mask + weight
        final_weight = self.softmax(mask_weight)

        # (batch, sequence, output_embedding_len)
        value = self.value_layer(torch.cat((last_hidden_state, inner_latent_value), dim=-1))

        # 求和
        embedding = torch.mul(final_weight, value)

        final_embedding = self.relu(embedding.sum(dim=-2))

        return final_embedding

    def forward(self, q_input_ids, q_token_type_ids, q_attention_mask,
                a_input_ids, a_token_type_ids, a_attention_mask,
                b_input_ids, b_token_type_ids, b_attention_mask, word_bag):

        # 把词袋丢去训练VAE
        reconstructed_input, mean, log_var, latent_v = self.vae_model(word_bag, out_latent_flag=True)
        vae_loss = calculate_vae_loss(word_bag, reconstructed_input, mean, log_var)

        # 获得表示
        q_embeddings = self.get_rep_by_vae(input_ids=q_input_ids, token_type_ids=q_token_type_ids,
                                           attention_mask=q_attention_mask, latent_value=mean, is_question=True)

        b_embeddings = self.get_rep_by_vae(input_ids=b_input_ids, token_type_ids=b_token_type_ids,
                                           attention_mask=b_attention_mask, latent_value=mean, is_question=True)

        a_embeddings = self.get_rep_by_vae(input_ids=a_input_ids, token_type_ids=a_token_type_ids,
                                           attention_mask=a_attention_mask, latent_value=mean, is_question=False)

        # 根据输入，进行思考, 思考的结果要选择性遗忘
        logits = self.classifier(q_embedding=q_embeddings, a_embedding=a_embeddings,
                                 b_embedding=b_embeddings)

        return logits, vae_loss.unsqueeze(0)


class OneSupremeMemory(nn.Module):
    def __init__(self, pretrained_bert_path='prajjwal1/bert-small',  latent_dim=100, num_labels=4, input_dim=None,
                 word_embedding_len=512, sentence_embedding_len=512, drop_prob=0.5, memory_num=50):

        super(OneSupremeMemory, self).__init__()

        # 毕竟num_label也算是memory的一部分
        self.num_labels = num_labels
        self.sentence_embedding_len = sentence_embedding_len
        self.memory_num = memory_num

        # 这个学习率不一样
        self.bert_model = BertModel.from_pretrained(pretrained_bert_path)
        # 这个embedding的grad会被计入bert model里，很好
        self.embeddings = self.bert_model.get_input_embeddings()

        # 记忆力模块
        self.memory_for_answer = nn.Parameter(torch.randn(memory_num, word_embedding_len, device='cuda:0'))
        self.memory_for_question = nn.Parameter(torch.randn(memory_num, word_embedding_len, device='cuda:0'))

        # 要确认训练时它有没有被修改
        self.memory_len_one_tensor = torch.tensor([1]*memory_num, requires_grad=False, device='cuda:0')
        self.one_tensor = torch.tensor([1], requires_grad=False, device='cuda:0')

        # 主题模型
        self.vae_model = VAEQuestion(input_dim=input_dim, latent_dim=latent_dim)

        # 注意力模型
        self.key_layer = nn.Sequential(
            nn.Linear(word_embedding_len + latent_dim, 2 * (word_embedding_len + latent_dim)),
            nn.ReLU(),
            nn.Linear(2 * (word_embedding_len + latent_dim), sentence_embedding_len),
            nn.Sigmoid()
        )

        self.value_layer = nn.Sequential(
            nn.Linear(word_embedding_len, sentence_embedding_len),
            nn.ReLU(),
            nn.Linear(sentence_embedding_len, sentence_embedding_len),
            nn.Tanh()
        )

        self.classifier = BodyClassifier(input_len=sentence_embedding_len, num_labels=num_labels)

        self.relu = torch.nn.ReLU(inplace=True)
        self.softmax = torch.nn.Softmax(dim=-2)
        self.dropout = nn.Dropout(p=drop_prob)

    def get_rep_by_vae(self, input_ids, token_type_ids, attention_mask, latent_value, is_question):
        # 获得隐藏层输出, (batch, sequence, embedding)
        temp_embeddings = self.embeddings(input_ids)

        final_embeddings = None
        final_attention_mask = None
        final_token_type_ids = None

        # input_ids (batch, sequence)
        for index,  batch_attention_mask in enumerate(attention_mask):
            input_embeddings = temp_embeddings[index][batch_attention_mask == 1]
            pad_embeddings = temp_embeddings[index][batch_attention_mask == 0]

            if is_question:
                whole_embeddings = torch.cat((input_embeddings, self.memory_for_question, pad_embeddings), dim=0)
            else:
                whole_embeddings = torch.cat((input_embeddings, self.memory_for_answer, pad_embeddings), dim=0)

            # 处理attention_mask
            whole_attention_mask = torch.cat((batch_attention_mask[batch_attention_mask==1], self.memory_len_one_tensor,
                                              batch_attention_mask[batch_attention_mask==0]), dim=-1)

            # 处理token_type_id
            remain_token_type_ids_len = batch_attention_mask.shape[0] + self.memory_num - input_embeddings.shape[0]
            whole_token_type_ids = torch.cat((token_type_ids[index][batch_attention_mask==1],
                                              self.one_tensor.repeat(remain_token_type_ids_len)), dim=-1)

            whole_embeddings = whole_embeddings.unsqueeze(0)
            whole_attention_mask = whole_attention_mask.unsqueeze(0)
            whole_token_type_ids = whole_token_type_ids.unsqueeze(0)

            if final_embeddings is None:
                final_embeddings = whole_embeddings
                final_attention_mask = whole_attention_mask
                final_token_type_ids = whole_token_type_ids
            else:
                final_embeddings = torch.cat((final_embeddings, whole_embeddings), dim=0)
                final_attention_mask = torch.cat((final_attention_mask, whole_attention_mask), dim=0)
                final_token_type_ids = torch.cat((final_token_type_ids, whole_token_type_ids), dim=0)

        out = self.bert_model(inputs_embeds=final_embeddings, attention_mask=final_attention_mask,
                              token_type_ids=final_token_type_ids)

        last_hidden_state = out['last_hidden_state']

        # (batch, sequence, output_embedding_len)
        value = self.value_layer(last_hidden_state)

        # 开始根据主题分布，进行信息压缩
        # 创作出mask
        with torch.no_grad():
            mask = final_token_type_ids.type(dtype=torch.float)
            mask[mask == 1] = -np.inf
            mask[mask == 0] = 0.0
            mask = mask.repeat(self.sentence_embedding_len, 1, 1)
            mask.transpose_(0, 1)
            mask.transpose_(1, 2)

        # 根据主题分布计算权重
        inner_latent_value = latent_value.unsqueeze(1)
        inner_latent_value = inner_latent_value.repeat(1, last_hidden_state.shape[1], 1)

        # (batch, sequence, output_embedding_len)
        weight = self.key_layer(torch.cat((last_hidden_state, inner_latent_value), dim=-1))
        mask_weight = mask + weight
        final_weight = self.softmax(mask_weight)

        # 求和
        embedding = torch.mul(final_weight, value)

        final_embedding = self.relu(embedding.sum(dim=-2))

        return final_embedding

    def forward(self, q_input_ids, q_token_type_ids, q_attention_mask,
                a_input_ids, a_token_type_ids, a_attention_mask,
                b_input_ids, b_token_type_ids, b_attention_mask, word_bag):
        # 把词袋丢去训练VAE
        reconstructed_input, mean, log_var, latent_v = self.vae_model(word_bag, out_latent_flag=True)
        vae_loss = calculate_vae_loss(word_bag, reconstructed_input, mean, log_var)

        # 获得表示
        q_embeddings = self.get_rep_by_vae(input_ids=q_input_ids, token_type_ids=q_token_type_ids,
                                           attention_mask=q_attention_mask, latent_value=mean, is_question=True)

        b_embeddings = self.get_rep_by_vae(input_ids=b_input_ids, token_type_ids=b_token_type_ids,
                                           attention_mask=b_attention_mask, latent_value=mean, is_question=True)

        a_embeddings = self.get_rep_by_vae(input_ids=a_input_ids, token_type_ids=a_token_type_ids,
                                           attention_mask=a_attention_mask, latent_value=mean, is_question=False)


        # 根据输入，进行思考, 思考的结果要选择性遗忘
        logits = self.classifier(q_embedding=q_embeddings, a_embedding=a_embeddings,
                                 b_embedding=b_embeddings)

        return logits, vae_loss.unsqueeze(0)


class TeacherBert(nn.Module):
    def __init__(self, model_path='prajjwal1/bert-small',  num_labels=4,
                 word_embedding_len=512, sentence_embedding_len=512):

        super(TeacherBert, self).__init__()

        # 毕竟num_label也算是memory的一部分
        self.num_labels = num_labels
        self.sentence_embedding_len = sentence_embedding_len

        # 这个学习率不一样
        self.bert_model = BertModel.from_pretrained(model_path)

        # 注意力模型
        self.key_layer = nn.Sequential(
            nn.Linear(word_embedding_len, 2 * word_embedding_len),
            nn.ReLU(),
            nn.Linear(2 * word_embedding_len, sentence_embedding_len),
            nn.Sigmoid()
        )

        self.value_layer = nn.Sequential(
            nn.Linear(word_embedding_len, 2 * word_embedding_len),
            nn.ReLU(),
            nn.Linear(2 * word_embedding_len, sentence_embedding_len),
            nn.Sigmoid()
        )

        self.classifier = SimpleClassifier(input_len=sentence_embedding_len, num_labels=num_labels)

        self.relu = torch.nn.ReLU(inplace=True)
        self.softmax = torch.nn.Softmax(dim=-2)

    def forward(self, input_ids, attention_mask, token_type_ids):
        out = self.bert_model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        last_hidden_state = out['last_hidden_state']

        value = self.value_layer(last_hidden_state)
        weight = self.key_layer(last_hidden_state)

        # 获得问题的表示--------------------------------------------
        with torch.no_grad():
            mask = token_type_ids.type(dtype=torch.float).clone()
            mask[mask == 1] = -np.inf
            mask[mask == 0] = 0.0
            mask = mask.repeat(self.sentence_embedding_len, 1, 1)
            mask.transpose_(0, 1)
            mask.transpose_(1, 2)

        # (batch, sequence, output_embedding_len)
        mask_weight = mask + weight
        final_weight = self.softmax(mask_weight)

        # 求和
        q_embeddings = torch.mul(final_weight, value)
        q_embeddings = self.relu(q_embeddings.sum(dim=-2))

        # 获得答案的表示--------------------------------------------
        with torch.no_grad():
            mask = token_type_ids.type(dtype=torch.float).clone()
            mask[token_type_ids == 1] = 0.0
            mask[attention_mask == 0] = -np.inf
            mask[token_type_ids == 0] = -np.inf

            mask = mask.repeat(self.sentence_embedding_len, 1, 1)
            mask.transpose_(0, 1)
            mask.transpose_(1, 2)

        # (batch, sequence, output_embedding_len)
        mask_weight = mask + weight
        final_weight = self.softmax(mask_weight)

        # 求和
        a_embeddings = torch.mul(final_weight, value)
        a_embeddings = self.relu(a_embeddings.sum(dim=-2))

        # 根据输入，进行思考, 思考的结果要选择性遗忘
        logits = self.classifier(q_embedding=q_embeddings, a_embedding=a_embeddings)

        return logits