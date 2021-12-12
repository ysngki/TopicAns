from transformers import BertModel
import torch
import torch.nn as nn
import torch.utils.data
import numpy as np
import torch.nn.functional
from class_lib import BodyClassifier, calculate_vae_loss, VAEQuestion
from attention_module import attend


# --------------------------------------
# fen ge xian
# --------------------------------------
class OneSupremeMemoryConfig:
    def __init__(self, tokenizer_len, pretrained_bert_path='prajjwal1/bert-small', latent_dim=100, num_labels=4,
                 voc_size=None, word_embedding_len=512, sentence_embedding_len=512, memory_num=50):

        self.tokenizer_len = tokenizer_len
        self.pretrained_bert_path = pretrained_bert_path
        self.latent_dim = latent_dim
        self.num_labels = num_labels
        self.voc_size = voc_size
        self.word_embedding_len = word_embedding_len
        self.sentence_embedding_len = sentence_embedding_len
        self.memory_num = memory_num

    def __str__(self):
        print("*"*20 + "config" + "*"*20)
        print("tokenizer_len:", self.tokenizer_len)
        print("pretrained_bert_path:", self.pretrained_bert_path)
        print("latent_dim:", self.latent_dim)
        print("num_labels:", self.num_labels)
        print("voc_size:", self.voc_size)
        print("word_embedding_len:", self.word_embedding_len)
        print("sentence_embedding_len:", self.sentence_embedding_len)
        print("memory_num:", self.memory_num)


class OneSupremeMemory(nn.Module):
    def __init__(self, config):

        super(OneSupremeMemory, self).__init__()

        self.config = config

        # 毕竟num_label也算是memory的一部分
        self.num_labels = config.num_labels
        self.sentence_embedding_len = config.sentence_embedding_len
        self.memory_num = config.memory_num

        # 这个学习率不一样
        self.bert_model = BertModel.from_pretrained(config.pretrained_bert_path)
        self.bert_model.resize_token_embeddings(config.tokenizer_len)
        # 这个embedding的grad会被计入bert model里，很好
        self.embeddings = self.bert_model.get_input_embeddings()

        # 记忆力模块
        self.memory_for_answer = nn.Parameter(torch.randn(config.memory_num, config.word_embedding_len))
        self.memory_for_question = nn.Parameter(torch.randn(config.memory_num, config.word_embedding_len))

        # 主题模型
        self.vae_model = VAEQuestion(input_dim=config.voc_size, latent_dim=config.latent_dim)

        # 注意力模型
        # self.key_layer = nn.Sequential(
        #     nn.Linear(config.word_embedding_len + config.latent_dim,
        #               2 * (config.word_embedding_len + config.latent_dim)),
        #     nn.ReLU(),
        #     nn.Linear(2 * (config.word_embedding_len + config.latent_dim), config.sentence_embedding_len),
        #     nn.Sigmoid()
        # )

        self.key_layer = nn.Sequential(
            nn.Linear(config.word_embedding_len, 2 * config.word_embedding_len),
            nn.ReLU(),
            nn.Linear(2 * config.word_embedding_len, config.sentence_embedding_len),
            nn.Sigmoid()
        )

        self.value_layer = nn.Sequential(
            nn.Linear(config.word_embedding_len, 2 * config.sentence_embedding_len),
            nn.ReLU(),
            nn.Linear(2 * config.sentence_embedding_len, config.sentence_embedding_len),
            nn.Tanh()
        )

        self.classifier = BodyClassifier(input_len=config.sentence_embedding_len, num_labels=config.num_labels)

        self.relu = torch.nn.ReLU(inplace=True)
        self.softmax = torch.nn.Softmax(dim=-2)

    def get_rep_by_vae(self, input_ids, token_type_ids, attention_mask, latent_value, is_question):
        # 要确认训练时它有没有被修改
        memory_len_one_tensor = torch.tensor([1] * self.config.memory_num, requires_grad=False,
                                            device=input_ids.device)
        one_tensor = torch.tensor([1], requires_grad=False, device=input_ids.device)

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
            whole_attention_mask = torch.cat((batch_attention_mask[batch_attention_mask == 1], memory_len_one_tensor,
                                              batch_attention_mask[batch_attention_mask == 0]), dim=-1)

            # 处理token_type_id
            remain_token_type_ids_len = batch_attention_mask.shape[0] + self.memory_num - input_embeddings.shape[0]
            whole_token_type_ids = torch.cat((token_type_ids[index][batch_attention_mask == 1],
                                              one_tensor.repeat(remain_token_type_ids_len)), dim=-1)

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

    def get_rep_no_vae(self, input_ids, token_type_ids, attention_mask, latent_value, is_question):
        # 要确认训练时它有没有被修改
        memory_len_one_tensor = torch.tensor([1] * self.config.memory_num, requires_grad=False,
                                            device=input_ids.device)
        one_tensor = torch.tensor([1], requires_grad=False, device=input_ids.device)

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
            whole_attention_mask = torch.cat((batch_attention_mask[batch_attention_mask == 1], memory_len_one_tensor,
                                              batch_attention_mask[batch_attention_mask == 0]), dim=-1)

            # 处理token_type_id
            remain_token_type_ids_len = batch_attention_mask.shape[0] + self.memory_num - input_embeddings.shape[0]
            whole_token_type_ids = torch.cat((token_type_ids[index][batch_attention_mask == 1],
                                              one_tensor.repeat(remain_token_type_ids_len)), dim=-1)

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

        # (batch, sequence, output_embedding_len)
        weight = self.key_layer(last_hidden_state)
        mask_weight = mask + weight
        final_weight = self.softmax(mask_weight)

        # 求和
        embedding = torch.mul(final_weight, value)

        final_embedding = self.relu(embedding.sum(dim=-2))

        return final_embedding

    def get_rep_by_pooler(self, input_ids, token_type_ids, attention_mask, latent_value, is_question):
        # 要确认训练时它有没有被修改
        memory_len_one_tensor = torch.tensor([1] * self.config.memory_num, requires_grad=False,
                                             device=input_ids.device)
        one_tensor = torch.tensor([1], requires_grad=False, device=input_ids.device)

        # 获得隐藏层输出, (batch, sequence, embedding)
        temp_embeddings = self.embeddings(input_ids)

        final_embeddings = None
        final_attention_mask = None
        final_token_type_ids = None

        # input_ids (batch, sequence)
        for index, batch_attention_mask in enumerate(attention_mask):
            input_embeddings = temp_embeddings[index][batch_attention_mask == 1]
            pad_embeddings = temp_embeddings[index][batch_attention_mask == 0]

            if is_question:
                whole_embeddings = torch.cat((input_embeddings, self.memory_for_question, pad_embeddings), dim=0)
            else:
                whole_embeddings = torch.cat((input_embeddings, self.memory_for_answer, pad_embeddings), dim=0)

            # 处理attention_mask
            whole_attention_mask = torch.cat((batch_attention_mask[batch_attention_mask == 1], memory_len_one_tensor,
                                              batch_attention_mask[batch_attention_mask == 0]), dim=-1)

            # 处理token_type_id
            remain_token_type_ids_len = batch_attention_mask.shape[0] + self.memory_num - input_embeddings.shape[0]
            whole_token_type_ids = torch.cat((token_type_ids[index][batch_attention_mask == 1],
                                              one_tensor.repeat(remain_token_type_ids_len)), dim=-1)

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

        pooler_output = out['pooler_output']

        return pooler_output

    def forward(self, q_input_ids, q_token_type_ids, q_attention_mask,
                a_input_ids, a_token_type_ids, a_attention_mask,
                b_input_ids, b_token_type_ids, b_attention_mask, word_bag):
        # 把词袋丢去训练VAE
        reconstructed_input, mean, log_var, latent_v = self.vae_model(word_bag, out_latent_flag=True)
        vae_loss = calculate_vae_loss(word_bag, reconstructed_input, mean, log_var)

        # # 获得表示
        # q_embeddings = self.get_rep_no_vae(input_ids=q_input_ids, token_type_ids=q_token_type_ids,
        #                                    attention_mask=q_attention_mask, latent_value=mean, is_question=True)
        #
        # b_embeddings = self.get_rep_no_vae(input_ids=b_input_ids, token_type_ids=b_token_type_ids,
        #                                    attention_mask=b_attention_mask, latent_value=mean, is_question=True)
        #
        # a_embeddings = self.get_rep_no_vae(input_ids=a_input_ids, token_type_ids=a_token_type_ids,
        #                                    attention_mask=a_attention_mask, latent_value=mean, is_question=False)

        # 获得表示
        q_embeddings = self.get_rep_by_pooler(input_ids=q_input_ids, token_type_ids=q_token_type_ids,
                                              attention_mask=q_attention_mask, latent_value=mean, is_question=True)

        b_embeddings = self.get_rep_by_pooler(input_ids=b_input_ids, token_type_ids=b_token_type_ids,
                                              attention_mask=b_attention_mask, latent_value=mean, is_question=True)

        a_embeddings = self.get_rep_by_pooler(input_ids=a_input_ids, token_type_ids=a_token_type_ids,
                                              attention_mask=a_attention_mask, latent_value=mean, is_question=False)

        # 根据输入，进行思考, 思考的结果要选择性遗忘
        logits = self.classifier(q_embedding=q_embeddings, a_embedding=a_embeddings,
                                 b_embedding=b_embeddings)

        return logits, vae_loss.unsqueeze(0)


# --------------------------------------
# fen ge xian
# --------------------------------------
class InputMemorySelfAttConfig:
    def __init__(self, tokenizer_len, pretrained_bert_path='prajjwal1/bert-small', num_labels=4,
                 word_embedding_len=512, sentence_embedding_len=512, memory_num=50):

        self.tokenizer_len = tokenizer_len
        self.pretrained_bert_path = pretrained_bert_path
        self.num_labels = num_labels
        self.word_embedding_len = word_embedding_len
        self.sentence_embedding_len = sentence_embedding_len
        self.memory_num = memory_num

    def __str__(self):
        print("*"*20 + "config" + "*"*20)
        print("tokenizer_len:", self.tokenizer_len)
        print("pretrained_bert_path:", self.pretrained_bert_path)
        print("num_labels:", self.num_labels)
        print("word_embedding_len:", self.word_embedding_len)
        print("sentence_embedding_len:", self.sentence_embedding_len)
        print("memory_num:", self.memory_num)


class InputMemorySelfAtt(nn.Module):
    def __init__(self, config):

        super(InputMemorySelfAtt, self).__init__()

        self.config = config

        # 毕竟num_label也算是memory的一部分
        self.num_labels = config.num_labels
        self.sentence_embedding_len = config.sentence_embedding_len
        self.memory_num = config.memory_num

        # 这个学习率不一样
        self.bert_model = BertModel.from_pretrained(config.pretrained_bert_path)
        self.bert_model.resize_token_embeddings(config.tokenizer_len)
        # 这个embedding的grad会被计入bert model里，很好
        self.embeddings = self.bert_model.get_input_embeddings()

        # 记忆力模块
        self.memory_for_answer = nn.Parameter(torch.randn(config.memory_num, config.word_embedding_len))
        self.memory_for_question = nn.Parameter(torch.randn(config.memory_num, config.word_embedding_len))

        self.value_layer = nn.Sequential(
            nn.Linear(config.word_embedding_len, 2 * config.sentence_embedding_len),
            nn.ReLU(),
            nn.Linear(2 * config.sentence_embedding_len, config.sentence_embedding_len),
            nn.Tanh()
        )

        self.self_attention_weight_layer = nn.Sequential(
            nn.Linear(config.word_embedding_len, 2 * config.word_embedding_len),
            nn.ReLU(),
            nn.Linear(2 * config.word_embedding_len, 1),
            nn.Sigmoid()
        )

        self.classifier = BodyClassifier(input_len=config.sentence_embedding_len, num_labels=config.num_labels)

        self.relu = torch.nn.ReLU(inplace=True)
        self.softmax = torch.nn.Softmax(dim=-2)

    def get_rep_by_self_att_basic(self, input_ids, token_type_ids, attention_mask):
        out = self.bert_model(input_ids=input_ids, attention_mask=attention_mask,
                              token_type_ids=token_type_ids)

        last_hidden_state = out['last_hidden_state']

        # 接下来通过attention汇总信息----------------------------------------
        # (batch, sequence, output_embedding_len)
        value = self.value_layer(last_hidden_state)

        # (batch, sequence, 1)
        weight = self.self_attention_weight_layer(last_hidden_state)
        weight = weight.squeeze(-1)

        # 创作出score mask
        with torch.no_grad():
            # (batch, sequence)
            mask = attention_mask.type(dtype=torch.float)
            mask[mask == 0] = -np.inf
            mask[mask == 1] = 0.0

        mask_weight = mask + weight
        final_weight = nn.functional.softmax(mask_weight, dim=-1)
        # (batch, 1, sequence)
        final_weight = final_weight.unsqueeze(1)

        # 求和
        # (batch, 1, output_embedding_len)
        embedding = final_weight.bmm(value)
        embedding = embedding.squeeze(1)

        final_embedding = self.relu(embedding)

        return final_embedding

    def get_rep_by_self_att(self, input_ids, token_type_ids, attention_mask, is_question):
        # ----------------------通过memory来丰富信息---------------------
        # 要确认训练时它有没有被修改
        memory_len_one_tensor = torch.tensor([1] * self.config.memory_num, requires_grad=False,
                                             device=input_ids.device)
        one_tensor = torch.tensor([1], requires_grad=False, device=input_ids.device)

        # 获得隐藏层输出, (batch, sequence, embedding)
        temp_embeddings = self.embeddings(input_ids)

        final_embeddings = None
        final_attention_mask = None
        final_token_type_ids = None

        # input_ids (batch, sequence)
        for index, batch_attention_mask in enumerate(attention_mask):
            input_embeddings = temp_embeddings[index][batch_attention_mask == 1]
            pad_embeddings = temp_embeddings[index][batch_attention_mask == 0]

            if is_question:
                whole_embeddings = torch.cat((input_embeddings, self.memory_for_question, pad_embeddings), dim=0)
            else:
                whole_embeddings = torch.cat((input_embeddings, self.memory_for_answer, pad_embeddings), dim=0)

            # 处理attention_mask
            whole_attention_mask = torch.cat((batch_attention_mask[batch_attention_mask == 1], memory_len_one_tensor,
                                              batch_attention_mask[batch_attention_mask == 0]), dim=-1)

            # 处理token_type_id
            remain_token_type_ids_len = batch_attention_mask.shape[0] + self.memory_num - input_embeddings.shape[0]
            whole_token_type_ids = torch.cat((token_type_ids[index][batch_attention_mask == 1],
                                              one_tensor.repeat(remain_token_type_ids_len)), dim=-1)

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

        # 接下来通过attention汇总信息----------------------------------------
        # (batch, sequence, output_embedding_len)
        value = self.value_layer(last_hidden_state)

        # (batch, sequence, 1)
        weight = self.self_attention_weight_layer(last_hidden_state)
        weight = weight.squeeze(-1)

        # 创作出score mask
        with torch.no_grad():
            # (batch, sequence)
            mask = final_token_type_ids.type(dtype=torch.float)
            mask[mask == 1] = -np.inf
            mask[mask == 0] = 0.0

        mask_weight = mask + weight
        final_weight = nn.functional.softmax(mask_weight, dim=-1)
        # (batch, 1, sequence)
        final_weight = final_weight.unsqueeze(1)

        # 求和
        # (batch, 1, output_embedding_len)
        embedding = final_weight.bmm(value)
        embedding = embedding.squeeze(1)

        final_embedding = self.relu(embedding)

        return final_embedding

    def get_rep_by_pooler(self, input_ids, token_type_ids, attention_mask, is_question):
        # ----------------------通过memory来丰富信息---------------------
        # 要确认训练时它有没有被修改
        memory_len_one_tensor = torch.tensor([1] * self.config.memory_num, requires_grad=False,
                                             device=input_ids.device)
        one_tensor = torch.tensor([1], requires_grad=False, device=input_ids.device)

        # 获得隐藏层输出, (batch, sequence, embedding)
        temp_embeddings = self.embeddings(input_ids)

        final_embeddings = None
        final_attention_mask = None
        final_token_type_ids = None

        # input_ids (batch, sequence)
        for index, batch_attention_mask in enumerate(attention_mask):
            input_embeddings = temp_embeddings[index][batch_attention_mask == 1]
            pad_embeddings = temp_embeddings[index][batch_attention_mask == 0]

            if is_question:
                whole_embeddings = torch.cat((input_embeddings, self.memory_for_question, pad_embeddings), dim=0)
            else:
                whole_embeddings = torch.cat((input_embeddings, self.memory_for_answer, pad_embeddings), dim=0)

            # 处理attention_mask
            whole_attention_mask = torch.cat((batch_attention_mask[batch_attention_mask == 1], memory_len_one_tensor,
                                              batch_attention_mask[batch_attention_mask == 0]), dim=-1)

            # 处理token_type_id
            remain_token_type_ids_len = batch_attention_mask.shape[0] + self.memory_num - input_embeddings.shape[0]
            whole_token_type_ids = torch.cat((token_type_ids[index][batch_attention_mask == 1],
                                              one_tensor.repeat(remain_token_type_ids_len)), dim=-1)

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

        out = out['pooler_output']

        return out

    def forward(self, q_input_ids, q_token_type_ids, q_attention_mask,
                a_input_ids, a_token_type_ids, a_attention_mask,
                b_input_ids, b_token_type_ids, b_attention_mask):
        # 获得表示，普普通通
        # q_embeddings = self.get_rep_by_self_att_basic(input_ids=q_input_ids, token_type_ids=q_token_type_ids,
        #                                               attention_mask=q_attention_mask)
        #
        # b_embeddings = self.get_rep_by_self_att_basic(input_ids=b_input_ids, token_type_ids=b_token_type_ids,
        #                                               attention_mask=b_attention_mask)
        #
        # a_embeddings = self.get_rep_by_self_att_basic(input_ids=a_input_ids, token_type_ids=a_token_type_ids,
        #                                               attention_mask=a_attention_mask)

        # 获得表示,结合memory
        # q_embeddings = self.get_rep_by_self_att(input_ids=q_input_ids, token_type_ids=q_token_type_ids,
        #                                         attention_mask=q_attention_mask, is_question=True)
        #
        # b_embeddings = self.get_rep_by_self_att(input_ids=b_input_ids, token_type_ids=b_token_type_ids,
        #                                         attention_mask=b_attention_mask, is_question=True)
        #
        # a_embeddings = self.get_rep_by_self_att(input_ids=a_input_ids, token_type_ids=a_token_type_ids,
        #                                         attention_mask=a_attention_mask, is_question=False)

        q_embeddings = self.get_rep_by_pooler(input_ids=q_input_ids, token_type_ids=q_token_type_ids,
                                              attention_mask=q_attention_mask, is_question=True)

        b_embeddings = self.get_rep_by_pooler(input_ids=b_input_ids, token_type_ids=b_token_type_ids,
                                              attention_mask=b_attention_mask, is_question=True)

        a_embeddings = self.get_rep_by_pooler(input_ids=a_input_ids, token_type_ids=a_token_type_ids,
                                              attention_mask=a_attention_mask, is_question=False)

        # return torch.zeros((q_input_ids.shape[0], 4), device=q_input_ids.device, requires_grad=True)

        # # 根据输入，进行思考, 思考的结果要选择性遗忘
        logits = self.classifier(q_embedding=q_embeddings, a_embedding=a_embeddings,
                                 b_embedding=b_embeddings)


        return logits


# --------------------------------------
# fen ge xian
# --------------------------------------
class PureMemorySelfAttConfig:
    def __init__(self, tokenizer_len, pretrained_bert_path='prajjwal1/bert-small', num_labels=4,
                 word_embedding_len=512, sentence_embedding_len=512, memory_num=50, hop_num=1):

        self.tokenizer_len = tokenizer_len
        self.pretrained_bert_path = pretrained_bert_path
        self.num_labels = num_labels
        self.word_embedding_len = word_embedding_len
        self.sentence_embedding_len = sentence_embedding_len
        self.memory_num = memory_num
        self.hop_num = hop_num

    def __str__(self):
        print("*"*20 + "config" + "*"*20)
        print("tokenizer_len:", self.tokenizer_len)
        print("pretrained_bert_path:", self.pretrained_bert_path)
        print("num_labels:", self.num_labels)
        print("word_embedding_len:", self.word_embedding_len)
        print("sentence_embedding_len:", self.sentence_embedding_len)
        print("memory_num:", self.memory_num)
        print("hop_num:", self.hop_num)


class PureMemorySelfAtt(nn.Module):
    def __init__(self, config: PureMemorySelfAttConfig):

        super(PureMemorySelfAtt, self).__init__()

        # 毕竟num_label也算是memory的一部分
        self.num_labels = config.num_labels
        self.sentence_embedding_len = config.sentence_embedding_len

        # 这个学习率不一样
        self.bert_model = BertModel.from_pretrained(config.pretrained_bert_path)

        # 记忆力模块
        self.hop_num = config.hop_num

        # self.query_for_answer = nn.Parameter(torch.randn(config.memory_num, config.word_embedding_len, device='cuda:0'))
        #
        # self.memory_for_answer = nn.Parameter(
        #     torch.randn(config.memory_num, config.word_embedding_len, device='cuda:0'))
        #
        # self.query_for_question = nn.Parameter(
        #     torch.randn(config.memory_num, config.word_embedding_len, device='cuda:0'))
        #
        # self.memory_for_question = nn.Parameter(
        #     torch.randn(config.memory_num, config.word_embedding_len, device='cuda:0'))

        self.queries_for_answer = nn.ParameterList([nn.Parameter(
            torch.randn(config.memory_num, config.word_embedding_len, device='cuda:0')) for i in range(self.hop_num)])

        self.memories_for_answer = nn.ParameterList([nn.Parameter(
            torch.randn(config.memory_num, config.word_embedding_len, device='cuda:0')) for i in range(self.hop_num)])

        self.queries_for_question = nn.ParameterList([nn.Parameter(
            torch.randn(config.memory_num, config.word_embedding_len, device='cuda:0')) for i in range(self.hop_num)])

        self.memories_for_question = nn.ParameterList([nn.Parameter(
            torch.randn(config.memory_num, config.word_embedding_len, device='cuda:0')) for i in range(self.hop_num)])

        # self.memories_for_question = nn.ParameterList([nn.Parameter(
        #     torch.randn(config.memory_num, config.word_embedding_len, device='cuda:0'))
        #     for i in range(self.hop_num + 1)])
        #
        # self.memories_for_answer = nn.ParameterList([nn.Parameter(
        #     torch.randn(config.memory_num, config.word_embedding_len, device='cuda:0'))
        #     for i in range(self.hop_num + 1)])

        self.value_layer = nn.Sequential(
            nn.Linear(config.word_embedding_len, 2 * config.sentence_embedding_len),
            nn.ReLU(),
            nn.Linear(2 * config.sentence_embedding_len, config.sentence_embedding_len),
            nn.Tanh()
        )

        self.self_attention_weight_layer = nn.Sequential(
            nn.Linear(config.word_embedding_len, 2 * config.word_embedding_len),
            nn.ReLU(),
            nn.Linear(2 * config.word_embedding_len, 1),
            nn.Sigmoid()
        )

        self.classifier = BodyClassifier(input_len=config.sentence_embedding_len, num_labels=config.num_labels)

        self.relu = torch.nn.ReLU(inplace=True)
        self.softmax = torch.nn.Softmax(dim=-2)


    def get_rep_by_self_att(self, input_ids, token_type_ids, attention_mask, is_question):
        out = self.bert_model(input_ids=input_ids, attention_mask=attention_mask,
                              token_type_ids=token_type_ids)

        last_hidden_state = out['last_hidden_state']

        for i in range(self.hop_num):
            # 根据记忆丰富一下信息，之后可以考虑把latent一起传进去
            if is_question:
                contexts = self.queries_for_answer[i].repeat(last_hidden_state.shape[0], 1, 1)
                values = self.memories_for_answer[i].repeat(last_hidden_state.shape[0], 1, 1)

                enrich_info = attend(query=last_hidden_state, context=contexts, value=values)
            else:
                contexts = self.queries_for_question[i].repeat(last_hidden_state.shape[0], 1, 1)
                values = self.memories_for_question[i].repeat(last_hidden_state.shape[0], 1, 1)

                enrich_info = attend(query=last_hidden_state, context=contexts, value=values)

            # 这一步也有点草率
            last_hidden_state = enrich_info + last_hidden_state
            last_hidden_state = self.value_layer(last_hidden_state)

        # 接下来通过attention汇总信息----------------------------------------
        # (batch, sequence, output_embedding_len)
        # value = self.value_layer(last_hidden_state)
        value = last_hidden_state

        # (batch, sequence, 1)
        weight = self.self_attention_weight_layer(value)
        weight = weight.squeeze(-1)

        # 创作出score mask
        with torch.no_grad():
            # (batch, sequence)
            mask = attention_mask.type(dtype=torch.float)
            mask[mask == 0] = -np.inf
            mask[mask == 1] = 0.0

        mask_weight = mask + weight
        final_weight = nn.functional.softmax(mask_weight, dim=-1)
        # (batch, 1, sequence)
        final_weight = final_weight.unsqueeze(1)

        # 求和
        # (batch, 1, output_embedding_len)
        embedding = final_weight.bmm(value)
        embedding = embedding.squeeze(1)

        final_embedding = self.relu(embedding)

        return final_embedding

    def get_rep_simply(self, input_ids, token_type_ids, attention_mask):
        out = self.bert_model(input_ids=input_ids, attention_mask=attention_mask,
                              token_type_ids=token_type_ids)

        last_hidden_state = out['last_hidden_state']

        value = self.value_layer(last_hidden_state)

        # (batch, sequence, 1)
        weight = self.self_attention_weight_layer(last_hidden_state)
        weight = weight.squeeze(-1)

        # 创作出score mask
        with torch.no_grad():
            # (batch, sequence)
            mask = attention_mask.type(dtype=torch.float)
            mask[mask == 0] = -np.inf
            mask[mask == 1] = 0.0

        mask_weight = mask + weight
        final_weight = nn.functional.softmax(mask_weight, dim=-1)
        # (batch, 1, sequence)
        final_weight = final_weight.unsqueeze(1)

        # 求和
        # (batch, 1, output_embedding_len)
        embedding = final_weight.bmm(value)
        embedding = embedding.squeeze(1)

        final_embedding = self.relu(embedding)

        return final_embedding

    def forward(self, q_input_ids, q_token_type_ids, q_attention_mask,
                a_input_ids, a_token_type_ids, a_attention_mask,
                b_input_ids, b_token_type_ids, b_attention_mask):

        # 获得表示
        # q_embeddings = self.get_rep_by_self_att(input_ids=q_input_ids, token_type_ids=q_token_type_ids,
        #                                         attention_mask=q_attention_mask, is_question=True)
        #
        # b_embeddings = self.get_rep_by_self_att(input_ids=b_input_ids, token_type_ids=b_token_type_ids,
        #                                         attention_mask=b_attention_mask, is_question=True)
        #
        # a_embeddings = self.get_rep_by_self_att(input_ids=a_input_ids, token_type_ids=a_token_type_ids,
        #                                         attention_mask=a_attention_mask, is_question=False)

        q_embeddings = self.get_rep_simply(input_ids=q_input_ids, token_type_ids=q_token_type_ids,
                                                attention_mask=q_attention_mask)

        b_embeddings = self.get_rep_simply(input_ids=b_input_ids, token_type_ids=b_token_type_ids,
                                                attention_mask=b_attention_mask)

        a_embeddings = self.get_rep_simply(input_ids=a_input_ids, token_type_ids=a_token_type_ids,
                                                attention_mask=a_attention_mask)

        # 根据输入，进行思考, 思考的结果要选择性遗忘
        logits = self.classifier(q_embedding=q_embeddings, a_embedding=a_embeddings,
                                 b_embedding=b_embeddings)

        return logits


# --------------------------------------
# fen ge xian
# --------------------------------------
class PureMemoryConfig:
    def __init__(self, tokenizer_len, pretrained_bert_path='prajjwal1/bert-small', latent_dim=100, num_labels=4,
                 voc_size=None, word_embedding_len=512, sentence_embedding_len=512, memory_num=50):

        self.tokenizer_len = tokenizer_len
        self.pretrained_bert_path = pretrained_bert_path
        self.latent_dim = latent_dim
        self.num_labels = num_labels
        self.voc_size = voc_size
        self.word_embedding_len = word_embedding_len
        self.sentence_embedding_len = sentence_embedding_len
        self.memory_num = memory_num

    def __str__(self):
        print("*"*20 + "config" + "*"*20)
        print("tokenizer_len:", self.tokenizer_len)
        print("pretrained_bert_path:", self.pretrained_bert_path)
        print("latent_dim:", self.latent_dim)
        print("num_labels:", self.num_labels)
        print("voc_size:", self.voc_size)
        print("word_embedding_len:", self.word_embedding_len)
        print("sentence_embedding_len:", self.sentence_embedding_len)
        print("memory_num:", self.memory_num)


class PureMemory(nn.Module):
    def __init__(self, config: PureMemoryConfig):

        super(PureMemory, self).__init__()

        # 毕竟num_label也算是memory的一部分
        self.num_labels = config.num_labels
        self.sentence_embedding_len = config.sentence_embedding_len

        # 这个学习率不一样
        self.bert_model = BertModel.from_pretrained(config.pretrained_bert_path)

        # 记忆力模块
        self.query_for_answer = nn.Parameter(torch.randn(config.memory_num, config.word_embedding_len, device='cuda:0'))
        self.memory_for_answer = nn.Parameter(torch.randn(config.memory_num, config.word_embedding_len,
                                                          device='cuda:0'))

        self.query_for_question = nn.Parameter(torch.randn(config.memory_num, config.word_embedding_len,
                                                           device='cuda:0'))
        self.memory_for_question = nn.Parameter(torch.randn(config.memory_num, config.word_embedding_len,
                                                            device='cuda:0'))

        # 主题模型
        self.vae_model = VAEQuestion(input_dim=config.voc_size, latent_dim=config.latent_dim)

        # 注意力模型
        # self.key_layer = nn.Sequential(
        #     nn.Linear(config.word_embedding_len + config.latent_dim, 2*(config.word_embedding_len + config.latent_dim)),
        #     nn.ReLU(),
        #     nn.Linear(2*(config.word_embedding_len + config.latent_dim), config.sentence_embedding_len),
        #     nn.Sigmoid()
        # )

        self.key_layer = nn.Sequential(
            nn.Linear(config.word_embedding_len, 2 * config.word_embedding_len),
            nn.ReLU(),
            nn.Linear(2 * config.word_embedding_len, config.sentence_embedding_len),
            nn.Sigmoid()
        )

        # self.value_layer = nn.Sequential(
        #     nn.Linear(config.word_embedding_len + config.latent_dim,
        #               2 * (config.sentence_embedding_len + config.latent_dim)),
        #     nn.ReLU(),
        #     nn.Linear(2 * (config.sentence_embedding_len + config.latent_dim), config.sentence_embedding_len),
        #     nn.Tanh()
        # )

        self.value_layer = nn.Sequential(
            nn.Linear(config.word_embedding_len, 2 * config.sentence_embedding_len),
            nn.ReLU(),
            nn.Linear(2 * config.sentence_embedding_len, config.sentence_embedding_len),
            nn.Tanh()
        )

        self.classifier = BodyClassifier(input_len=config.sentence_embedding_len, num_labels=config.num_labels)

        self.relu = torch.nn.ReLU(inplace=True)
        self.softmax = torch.nn.Softmax(dim=-2)

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
        # weight = self.key_layer(torch.cat((last_hidden_state, inner_latent_value), dim=-1))
        weight = self.key_layer(last_hidden_state)
        mask_weight = mask + weight
        final_weight = self.softmax(mask_weight)

        # (batch, sequence, output_embedding_len)
        value = self.value_layer(last_hidden_state)

        # 求和
        embedding = torch.mul(final_weight, value)

        final_embedding = self.relu(embedding.sum(dim=-2))

        return final_embedding

    def get_rep_no_vae(self, input_ids, token_type_ids, attention_mask, is_question):
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

        # (batch, sequence, output_embedding_len)
        weight = self.key_layer(last_hidden_state)
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
        # q_embeddings = self.get_rep_by_vae(input_ids=q_input_ids, token_type_ids=q_token_type_ids,
        #                                    attention_mask=q_attention_mask, latent_value=mean, is_question=True)
        #
        # b_embeddings = self.get_rep_by_vae(input_ids=b_input_ids, token_type_ids=b_token_type_ids,
        #                                    attention_mask=b_attention_mask, latent_value=mean, is_question=True)
        #
        # a_embeddings = self.get_rep_by_vae(input_ids=a_input_ids, token_type_ids=a_token_type_ids,
        #                                    attention_mask=a_attention_mask, latent_value=mean, is_question=False)

        q_embeddings = self.get_rep_no_vae(input_ids=q_input_ids, token_type_ids=q_token_type_ids,
                                           attention_mask=q_attention_mask, is_question=True)

        b_embeddings = self.get_rep_no_vae(input_ids=b_input_ids, token_type_ids=b_token_type_ids,
                                           attention_mask=b_attention_mask, is_question=True)

        a_embeddings = self.get_rep_no_vae(input_ids=a_input_ids, token_type_ids=a_token_type_ids,
                                           attention_mask=a_attention_mask, is_question=False)

        # 根据输入，进行思考, 思考的结果要选择性遗忘
        logits = self.classifier(q_embedding=q_embeddings, a_embedding=a_embeddings,
                                 b_embedding=b_embeddings)

        return logits, vae_loss.unsqueeze(0)


# --------------------------------------
# fen ge xian
# --------------------------------------
class VaeAttentionConfig:
    def __init__(self, tokenizer_len, pretrained_bert_path='prajjwal1/bert-small', latent_dim=100, num_labels=4,
                 voc_size=None, word_embedding_len=512, sentence_embedding_len=512):

        self.tokenizer_len = tokenizer_len
        self.pretrained_bert_path = pretrained_bert_path
        self.latent_dim = latent_dim
        self.num_labels = num_labels
        self.voc_size = voc_size
        self.word_embedding_len = word_embedding_len
        self.sentence_embedding_len = sentence_embedding_len

    def __str__(self):
        print("*"*20 + "config" + "*"*20)
        print("tokenizer_len:", self.tokenizer_len)
        print("pretrained_bert_path:", self.pretrained_bert_path)
        print("latent_dim:", self.latent_dim)
        print("num_labels:", self.num_labels)
        print("voc_size:", self.voc_size)
        print("word_embedding_len:", self.word_embedding_len)
        print("sentence_embedding_len:", self.sentence_embedding_len)


# 用vae生成attention。计算attention时，用的外部向量是mean
class VaeAttention(nn.Module):
    def __init__(self, config: VaeAttentionConfig):

        super(VaeAttention, self).__init__()

        self.output_embedding_len = config.sentence_embedding_len

        # 这个学习率不一样
        self.bert_model = BertModel.from_pretrained(config.pretrained_bert_path)
        self.bert_model.resize_token_embeddings(config.tokenizer_len)

        # 主题模型
        self.vae_model = VAEQuestion(input_dim=config.voc_size, latent_dim=config.latent_dim)

        # 注意力模型
        self.key_layer = nn.Sequential(
            nn.Linear(config.word_embedding_len + config.latent_dim, 2*(config.word_embedding_len + config.latent_dim)),
            nn.ReLU(),
            nn.Linear(2*(config.word_embedding_len + config.latent_dim), config.sentence_embedding_len),
            nn.Sigmoid()
        )

        self.value_layer = nn.Sequential(
            nn.Linear(config.word_embedding_len + config.latent_dim,
                      2 * (config.word_embedding_len + config.latent_dim)),
            nn.ReLU(),
            nn.Linear(2 * (config.word_embedding_len + config.latent_dim), config.sentence_embedding_len),
            nn.Tanh()
        )

        # 这些的学习率一样
        self.classifier = BodyClassifier(input_len=config.sentence_embedding_len, num_labels=config.num_labels)

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


# 用vae生成Q的attention，然后用Q的attention去算A的attention
class VaeAttentionPlus(nn.Module):
    def __init__(self, config: VaeAttentionConfig):
        super(VaeAttentionPlus, self).__init__()

        self.output_embedding_len = config.sentence_embedding_len

        # 这个学习率不一样
        self.bert_model = BertModel.from_pretrained(config.pretrained_bert_path)
        self.bert_model.resize_token_embeddings(config.tokenizer_len)

        self.hint_len = 128

        # 主题模型
        self.vae_model = VAEQuestion(input_dim=config.voc_size, latent_dim=config.latent_dim)

        # 多维注意力模型
        self.key_layer = nn.Sequential(
            nn.Linear(config.word_embedding_len + self.hint_len,
                      2 * (config.word_embedding_len + self.hint_len)),
            nn.ReLU(),
            nn.Linear(2 * (config.word_embedding_len + self.hint_len), config.sentence_embedding_len),
            nn.Sigmoid()
        )

        self.value_layer = nn.Sequential(
            nn.Linear(config.word_embedding_len + self.hint_len,
                      2 * (config.word_embedding_len + self.hint_len)),
            nn.ReLU(),
            nn.Linear(2 * (config.word_embedding_len + self.hint_len), config.sentence_embedding_len),
            nn.Tanh()
        )

        self.fusion_layer = nn.Sequential(
            nn.Linear(config.sentence_embedding_len * 2, config.sentence_embedding_len),
            nn.ReLU(),
            nn.Linear(config.sentence_embedding_len, config.latent_dim),
        )

        self.hint_layer = nn.Sequential(
            nn.Linear(config.latent_dim, config.latent_dim * 2),
            nn.ReLU(),
            nn.Linear(config.latent_dim * 2, config.latent_dim * 2),
            nn.ReLU(),
            nn.Linear(config.latent_dim * 2, self.hint_len),
            nn.ReLU()
        )

        # 这些的学习率一样
        self.classifier = BodyClassifier(input_len=config.sentence_embedding_len, num_labels=config.num_labels)

        self.softmax = torch.nn.Softmax(dim=-2)

    def get_rep_by_vae(self, input_ids, token_type_ids, attention_mask, latent_value, is_question):
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

        if True:
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
        # 计算latent_value和hidden_state的dot attention，得到hidden_state的压缩向量
        else:
            # B * L
            required_mask = attention_mask.type(dtype=torch.float)
            mask[mask == 0] = -np.inf
            mask[mask == 1] = 0.0
            # B * 1 * L
            required_mask = required_mask.unsqueeze(1)
            # B * 1 * D
            query = latent_value.unsqueeze(1)

            final_embedding = attend(query=query, context=last_hidden_state, context_mask=required_mask,
                                     value=last_hidden_state)

        return final_embedding

    def forward(self, q_input_ids, q_token_type_ids, q_attention_mask,
                a_input_ids, a_token_type_ids, a_attention_mask,
                b_input_ids, b_token_type_ids, b_attention_mask, word_bag):
        # 把词袋丢去训练VAE
        reconstructed_input, mean, log_var, latent_v = self.vae_model(word_bag, out_latent_flag=True)
        vae_loss = calculate_vae_loss(word_bag, reconstructed_input, mean, log_var)

        # 获得表示
        hint_for_q = self.hint_layer(latent_v)

        q_embeddings = self.get_rep_by_vae(input_ids=q_input_ids, token_type_ids=q_token_type_ids,
                                           attention_mask=q_attention_mask, latent_value=hint_for_q, is_question=True)

        b_embeddings = self.get_rep_by_vae(input_ids=b_input_ids, token_type_ids=b_token_type_ids,
                                           attention_mask=b_attention_mask, latent_value=hint_for_q, is_question=True)

        fusion_result = self.fusion_layer(torch.cat((q_embeddings, b_embeddings), dim=-1))
        hint_for_a = self.hint_layer(fusion_result)

        a_embeddings = self.get_rep_by_vae(input_ids=a_input_ids, token_type_ids=a_token_type_ids,
                                           attention_mask=a_attention_mask, latent_value=hint_for_a,
                                           is_question=False)

        # 计算得到分类概率
        logits = self.classifier(q_embedding=q_embeddings, a_embedding=a_embeddings,
                                 b_embedding=b_embeddings)

        return logits, vae_loss.unsqueeze(0)


# --------------------------------------
# fen ge xian
# --------------------------------------
class BasicConfig:
    def __init__(self, tokenizer_len, pretrained_bert_path='prajjwal1/bert-small', num_labels=4,
                 word_embedding_len=512, sentence_embedding_len=512):

        self.tokenizer_len = tokenizer_len
        self.pretrained_bert_path = pretrained_bert_path
        self.num_labels = num_labels
        self.word_embedding_len = word_embedding_len
        self.sentence_embedding_len = sentence_embedding_len

    def __str__(self):
        print("*"*20 + "config" + "*"*20)
        print("tokenizer_len:", self.tokenizer_len)
        print("pretrained_bert_path:", self.pretrained_bert_path)
        print("num_labels:", self.num_labels)
        print("word_embedding_len:", self.word_embedding_len)
        print("sentence_embedding_len:", self.sentence_embedding_len)


class BasicModel(nn.Module):
    def __init__(self, config: BasicConfig):

        super(BasicModel, self).__init__()

        self.output_embedding_len = config.sentence_embedding_len

        # 这个学习率不一样
        self.bert_model = BertModel.from_pretrained(config.pretrained_bert_path)
        self.bert_model.resize_token_embeddings(config.tokenizer_len)

        # 用来计算self-attention
        self.query_for_question = nn.Parameter(torch.randn(config.word_embedding_len))
        self.query_for_answer = nn.Parameter(torch.randn(config.word_embedding_len))

        # 注意力模型
        self.key_layer = nn.Sequential(
            nn.Linear(config.word_embedding_len, 2*config.word_embedding_len),
            nn.ReLU(),
            nn.Linear(2*config.word_embedding_len, config.word_embedding_len),
            nn.Sigmoid()
        )

        self.value_layer = nn.Sequential(
            nn.Linear(config.word_embedding_len, 2 * config.word_embedding_len),
            nn.ReLU(),
            nn.Linear(2 * config.word_embedding_len, config.sentence_embedding_len),
            nn.Tanh()
        )

        # 这些的学习率一样
        self.classifier = BodyClassifier(input_len=config.sentence_embedding_len, num_labels=config.num_labels)

        self.relu = torch.nn.ReLU(inplace=True)
        self.softmax = torch.nn.Softmax(dim=-2)

    def get_rep_by_pooler(self, input_ids, token_type_ids, attention_mask):
        # 获得隐藏层输出
        out = self.bert_model(input_ids=input_ids, token_type_ids=token_type_ids,
                              attention_mask=attention_mask)
        pooler_out = out['pooler_output']

        return pooler_out

    def get_rep_by_self_attention(self, input_ids, token_type_ids, attention_mask, is_question=True):

        out = self.bert_model(input_ids=input_ids, token_type_ids=token_type_ids,
                              attention_mask=attention_mask)

        last_hidden_state = out['last_hidden_state']

        # 接下来通过attention汇总信息----------------------------------------
        # (batch, sequence, word_embedding_len)
        key = self.key_layer(last_hidden_state)

        # (batch, sequence, sentence_embedding_len)
        value = self.value_layer(last_hidden_state)

        if is_question:
            # (1, word_embedding_len)
            query = self.query_for_question.unsqueeze(0)
        else:
            query = self.query_for_answer.unsqueeze(0)

        # (batch, 1, word_embedding_len)
        query = query.repeat(key.shape[0], 1, 1)

        # (batch, 1, sequence)
        weight = query.bmm(key.transpose(1, 2))

        # 创作出score mask
        with torch.no_grad():
            # (batch, sequence)
            mask = attention_mask.type(dtype=torch.float)
            mask[mask == 0] = -np.inf
            mask[mask == 1] = 0.0
            # (batch, 1, sequence)
            mask = mask.unsqueeze(1)

        mask_weight = mask + weight
        # (batch, 1, sequence)
        final_weight = nn.functional.softmax(mask_weight, dim=-1)

        # 求和
        # (batch, 1, sentence_embedding_len)
        embedding = final_weight.bmm(value)
        final_embedding = embedding.squeeze(1)

        return final_embedding

    def get_rep_by_multi_attention(self, input_ids, token_type_ids, attention_mask):
        out = self.bert_model(input_ids=input_ids, token_type_ids=token_type_ids,
                              attention_mask=attention_mask)

        last_hidden_state = out['last_hidden_state']

        # (batch, sequence, output_embedding_len)
        value = self.value_layer(last_hidden_state)

        # 开始根据主题分布，进行信息压缩
        # 创作出mask
        with torch.no_grad():
            mask = attention_mask.type(dtype=torch.float)
            mask[mask == 0] = -np.inf
            mask[mask == 1] = 0.0
            mask = mask.repeat(self.output_embedding_len, 1, 1)
            mask.transpose_(0, 1)
            mask.transpose_(1, 2)

        # (batch, sequence, output_embedding_len)
        weight = self.key_layer(last_hidden_state)
        mask_weight = mask + weight
        final_weight = self.softmax(mask_weight)

        # 求和
        embedding = torch.mul(final_weight, value)

        final_embedding = self.relu(embedding.sum(dim=-2))

        return final_embedding

    def forward(self, q_input_ids, q_token_type_ids, q_attention_mask,
                a_input_ids, a_token_type_ids, a_attention_mask,
                b_input_ids, b_token_type_ids, b_attention_mask):

        # 获得表示
        # q_embeddings = self.get_rep_by_pooler(input_ids=q_input_ids, token_type_ids=q_token_type_ids,
        #                                       attention_mask=q_attention_mask)
        #
        # a_embeddings = self.get_rep_by_pooler(input_ids=a_input_ids, token_type_ids=a_token_type_ids,
        #                                       attention_mask=a_attention_mask)
        #
        # b_embeddings = self.get_rep_by_pooler(input_ids=b_input_ids, token_type_ids=b_token_type_ids,
        #                                       attention_mask=b_attention_mask)

        q_embeddings = self.get_rep_by_self_attention(input_ids=q_input_ids, token_type_ids=q_token_type_ids,
                                                      attention_mask=q_attention_mask)

        a_embeddings = self.get_rep_by_self_attention(input_ids=a_input_ids, token_type_ids=a_token_type_ids,
                                                      attention_mask=a_attention_mask)

        b_embeddings = self.get_rep_by_self_attention(input_ids=b_input_ids, token_type_ids=b_token_type_ids,
                                                      attention_mask=b_attention_mask)

        # q_embeddings = self.get_rep_by_multi_attention(input_ids=q_input_ids, token_type_ids=q_token_type_ids,
        #                                                attention_mask=q_attention_mask)
        #
        # a_embeddings = self.get_rep_by_multi_attention(input_ids=a_input_ids, token_type_ids=a_token_type_ids,
        #                                                attention_mask=a_attention_mask)
        #
        # b_embeddings = self.get_rep_by_multi_attention(input_ids=b_input_ids, token_type_ids=b_token_type_ids,
        #                                                attention_mask=b_attention_mask)

        # 计算得到分类概率
        logits = self.classifier(q_embedding=q_embeddings, a_embedding=a_embeddings,
                                 b_embedding=b_embeddings)

        return logits
