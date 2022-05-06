import imp
from transformers import AutoModel, BertForSequenceClassification, BertConfig, AutoConfig
from transformers.models.bert.my_modeling_bert import MyBertModel, DecoderLayerChunk
import torch
import torch.nn as nn
import torch.utils.data
import numpy as np
import torch.nn.functional
from vae import VAE


# model list
# 1. QAModel
#       basic bi-encoder, support mem
# 2. CrossBERT
#       basic cross-encoder
# 3. ADecoder
#       my model


# --------------------------------------
# fen ge xian
# --------------------------------------
# No matter whether use memory mechanism, all models whose input consists of Q and A can use this class
class QAModelConfig:
    def __init__(self, tokenizer_len, pretrained_bert_path='prajjwal1/bert-small', num_labels=4,
                 word_embedding_len=512, sentence_embedding_len=512, composition='pooler'):

        self.tokenizer_len = tokenizer_len
        self.pretrained_bert_path = pretrained_bert_path
        self.num_labels = num_labels
        self.word_embedding_len = word_embedding_len
        self.sentence_embedding_len = sentence_embedding_len
        self.composition = composition

    def __str__(self):
        print("*"*20 + "config" + "*"*20)
        print("tokenizer_len:", self.tokenizer_len)
        print("pretrained_bert_path:", self.pretrained_bert_path)
        print("num_labels:", self.num_labels)
        print("word_embedding_len:", self.word_embedding_len)
        print("sentence_embedding_len:", self.sentence_embedding_len)
        print("composition:", self.composition)


class QAModel(nn.Module):
    def __init__(self, config):

        super(QAModel, self).__init__()

        self.config = config

        # 毕竟num_label也算是memory的一部分
        self.num_labels = config.num_labels
        self.sentence_embedding_len = config.sentence_embedding_len

        # 这个学习率不一样
        self.bert_model = AutoModel.from_pretrained(config.pretrained_bert_path)
        self.bert_model.resize_token_embeddings(config.tokenizer_len)
        self.embeddings = self.bert_model.get_input_embeddings()

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

        self.classifier = QAClassifier(input_len=config.sentence_embedding_len, num_labels=config.num_labels)

        self.relu = torch.nn.ReLU(inplace=True)
        self.softmax = torch.nn.Softmax(dim=-2)

    def get_rep_by_self_att(self, input_ids, token_type_ids, attention_mask):
        input_ids, attention_mask, token_type_ids = clean_input_ids(input_ids, attention_mask, token_type_ids)
        
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
            mask = token_type_ids.type(dtype=torch.float)
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

    def get_rep_by_pooler(self, input_ids, token_type_ids, attention_mask):
        input_ids, attention_mask, token_type_ids = clean_input_ids(input_ids, attention_mask, token_type_ids)

        out = self.bert_model(input_ids=input_ids, attention_mask=attention_mask,
                              token_type_ids=token_type_ids)

        out = out['pooler_output']

        return out

    # use the average embedding of last layer as sentence representation
    def get_rep_by_avg(self, input_ids, token_type_ids, attention_mask):
        input_ids, attention_mask, token_type_ids = clean_input_ids(input_ids, attention_mask, token_type_ids)

        out = self.bert_model(input_ids=input_ids, attention_mask=attention_mask,
                              token_type_ids=token_type_ids)

        # get the average embeddings of last layer, excluding memory and special token(?)
        # (batch_size, sequence_length, hidden_size)
        last_hidden_state = out['last_hidden_state']

        with torch.no_grad():
            temp_attention_mask = attention_mask.clone().detach()

            # (batch_size, sequence_length)
            temp_mask = token_type_ids.clone().detach()

            # remove memory if existing
            temp_mask[temp_mask == 0] = 2
            temp_mask -= 1

            temp_mask = temp_mask * temp_attention_mask

            # remove cls, if cls is removed, some sentences may be empty
            # temp_mask[:, 0] = 0

            # remove sep--the last token whose type id equals 0
            sequence_len = temp_mask.sum(dim=-1) - 1
            sequence_len = sequence_len.unsqueeze(-1)
            temp_mask.scatter_(dim=1, index=sequence_len, src=torch.zeros((temp_mask.shape[0], 1), device=input_ids.device, dtype=temp_mask.dtype))

            # (batch_size, sequence_length, 1)
            temp_mask = temp_mask.unsqueeze(-1)

        last_hidden_state = last_hidden_state * temp_mask

        # get average embedding
        representations = last_hidden_state.sum(dim=1)

        representations = representations / sequence_len

        return representations

    def forward(self, q_input_ids, q_token_type_ids, q_attention_mask,
                a_input_ids, a_token_type_ids, a_attention_mask):

        if self.config.composition == 'avg':
            q_embeddings = self.get_rep_by_avg(input_ids=q_input_ids, token_type_ids=q_token_type_ids,
                                               attention_mask=q_attention_mask)

            a_embeddings = self.get_rep_by_avg(input_ids=a_input_ids, token_type_ids=a_token_type_ids,
                                               attention_mask=a_attention_mask)
        elif self.config.composition == 'pooler':
            q_embeddings = self.get_rep_by_pooler(input_ids=q_input_ids, token_type_ids=q_token_type_ids,
                                                  attention_mask=q_attention_mask)

            a_embeddings = self.get_rep_by_pooler(input_ids=a_input_ids, token_type_ids=a_token_type_ids,
                                                  attention_mask=a_attention_mask)
        else:
            raise Exception(f"Composition {self.config.composition} is not supported!!")

        # # 根据输入，进行思考, 思考的结果要选择性遗忘
        logits = self.classifier(q_embedding=q_embeddings, a_embedding=a_embeddings)


        return logits


# --------------------------------------
# fen ge xian
# --------------------------------------
class QATopicConfig:
    def __init__(self, tokenizer_len, voc_size, pretrained_bert_path='prajjwal1/bert-small', num_labels=4,
                 word_embedding_len=512, sentence_embedding_len=512, composition='pooler', topic_num=50):

        self.tokenizer_len = tokenizer_len
        self.voc_size = voc_size
        self.pretrained_bert_path = pretrained_bert_path
        self.num_labels = num_labels
        self.word_embedding_len = word_embedding_len
        self.sentence_embedding_len = sentence_embedding_len
        self.composition = composition
        self.topic_num = topic_num

    def __str__(self):
        print("*"*20 + "config" + "*"*20)
        print("tokenizer_len:", self.tokenizer_len)
        print("voc_size:", self.tokenizvoc_sizeer_len)
        print("pretrained_bert_path:", self.pretrained_bert_path)
        print("num_labels:", self.num_labels)
        print("word_embedding_len:", self.word_embedding_len)
        print("sentence_embedding_len:", self.sentence_embedding_len)
        print("composition:", self.composition)
        print("topic_num:", self.topic_num)


class QATopicModel(nn.Module):
    def __init__(self, config: QATopicConfig):

        super(QATopicModel, self).__init__()

        self.vae = VAE(config.voc_size, n_topic=config.topic_num)
        self.output_embedding_len = config.sentence_embedding_len

        # 这个学习率不一样
        self.bert_model = AutoModel.from_pretrained(config.pretrained_bert_path)
        self.bert_model.resize_token_embeddings(config.tokenizer_len)

        # 这个embedding的grad会被计入bert model里，很好
        self.embeddings = self.bert_model.get_input_embeddings()
        
        # 用来计算self-attention
        # self.query_for_question = nn.Parameter(torch.randn(config.word_embedding_len))
        # self.query_for_answer = nn.Parameter(torch.randn(config.word_embedding_len))

        # 注意力模型
        self.query_layer = nn.Sequential(
            nn.Linear(config.topic_num, 2 * config.word_embedding_len),
            nn.ReLU(),
            nn.Linear(2 * config.word_embedding_len, config.word_embedding_len),
        )
        self.LayerNorm = nn.LayerNorm(config.word_embedding_len)

        # 这些的学习率一样
        # self.classifier = QAClassifier(input_len=config.sentence_embedding_len, num_labels=config.num_labels)
        self.classifier = QATopicClassifier(input_len=config.sentence_embedding_len, topic_num=config.topic_num, num_labels=config.num_labels)

        self.relu = torch.nn.ReLU(inplace=True)
        self.softmax = torch.nn.Softmax(dim=-2)
        self.composition = config.composition

    def get_rep_by_pooler(self, input_ids, token_type_ids, attention_mask):
        input_ids, attention_mask, token_type_ids = clean_input_ids(input_ids, attention_mask, token_type_ids)

        out = self.bert_model(input_ids=input_ids, attention_mask=attention_mask,
                              token_type_ids=token_type_ids)

        out = out['pooler_output']

        return out

    def get_rep_by_topic_attention(self, input_ids, token_type_ids, attention_mask, topic_vector):
        input_ids, attention_mask, token_type_ids = clean_input_ids(input_ids, attention_mask, token_type_ids)

        out = self.bert_model(input_ids=input_ids, token_type_ids=token_type_ids,
                              attention_mask=attention_mask)

        last_hidden_state = out['last_hidden_state']

        # 接下来通过attention汇总信息----------------------------------------
        # (batch size, topic num) -> (batch size, word_embedding_len)
        query = self.query_layer(topic_vector)
        query = self.LayerNorm(query)
        # (batch, 1, word_embedding_len)
        query = query.unsqueeze(1)

        # (batch, 1, sequence)
        weight = query.bmm(last_hidden_state.transpose(1, 2))

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
        embedding = final_weight.bmm(last_hidden_state)
        final_embedding = embedding.squeeze(1)

        return final_embedding

    def forward(self, q_input_ids, q_token_type_ids, q_attention_mask,
                a_input_ids, a_token_type_ids, a_attention_mask, q_bow, a_bow, word_idf=None):
        q_reconst, q_mu, q_log_var = self.vae(q_bow, lambda x: torch.softmax(x, dim=1))
        a_reconst, a_mu, a_log_var = self.vae(a_bow, lambda x: torch.softmax(x, dim=1))

        logsoftmax = torch.log_softmax(q_reconst, dim=1)
        if word_idf is None:
            q_rec_loss = -1.0 * torch.mean(q_bow * logsoftmax)
        else:
            q_rec_loss = -1.0 * torch.mean(q_bow * logsoftmax * word_idf)

        q_kl_div = -0.5 * torch.mean(1 + q_log_var - q_mu.pow(2) - q_log_var.exp())
        
        logsoftmax = torch.log_softmax(a_reconst, dim=1)
        if word_idf is None:
            a_rec_loss = -1.0 * torch.mean(a_bow * logsoftmax)
        else:
            a_rec_loss = -1.0 * torch.mean(a_bow * logsoftmax * word_idf)

        a_kl_div = -0.5 * torch.mean(1 + a_log_var - a_mu.pow(2) - a_log_var.exp())

        vae_loss = q_rec_loss + a_rec_loss + q_kl_div  + a_kl_div

        # q_embeddings = self.get_rep_by_topic_attention(input_ids=q_input_ids, token_type_ids=q_token_type_ids,
        #                                                 attention_mask=q_attention_mask, topic_vector=q_mu)

        # a_embeddings = self.get_rep_by_topic_attention(input_ids=a_input_ids, token_type_ids=a_token_type_ids,
        #                                                 attention_mask=a_attention_mask, topic_vector=a_mu)

        q_embeddings = self.get_rep_by_pooler(input_ids=q_input_ids, token_type_ids=q_token_type_ids, attention_mask=q_attention_mask)

        a_embeddings = self.get_rep_by_pooler(input_ids=a_input_ids, token_type_ids=a_token_type_ids, attention_mask=a_attention_mask)

        # 计算得到分类概率
        logits = self.classifier(q_embedding=q_embeddings, a_embedding=a_embeddings, q_topic=q_mu, a_topic=a_mu)

        return logits, vae_loss


# --------------------------------------
# fen ge xian
# --------------------------------------
class QATopicMemoryModel(nn.Module):
    def __init__(self, config: QATopicConfig):

        super(QATopicMemoryModel, self).__init__()

        self.vae = VAE(config.voc_size, n_topic=config.topic_num)
        self.output_embedding_len = config.sentence_embedding_len

        self.config = config

        # 这个学习率不一样
        self.bert_model = AutoModel.from_pretrained(config.pretrained_bert_path)
        self.bert_model.resize_token_embeddings(config.tokenizer_len)

        # 这个embedding的grad会被计入bert model里，很好
        self.embeddings = self.bert_model.get_input_embeddings()

        # topic memory
        self.topic_word_matrix = None

        # memory transformation
        self.memory_layer = nn.Sequential(
            nn.Linear(config.voc_size, 2 * config.word_embedding_len),
            nn.ReLU(),
            nn.Linear(2 * config.word_embedding_len, config.word_embedding_len),
        )
        self.memory_LayerNorm = nn.LayerNorm(config.word_embedding_len)

        # 注意力模型
        self.query_layer = nn.Sequential(
            nn.Linear(config.topic_num, 2 * config.word_embedding_len),
            nn.ReLU(),
            nn.Linear(2 * config.word_embedding_len, config.word_embedding_len),
        )
        self.LayerNorm = nn.LayerNorm(config.word_embedding_len)

        # 这些的学习率一样
        # self.classifier = QAClassifier(input_len=config.sentence_embedding_len, num_labels=config.num_labels)
        self.classifier = QATopicClassifier(input_len=config.sentence_embedding_len, topic_num=config.topic_num, num_labels=config.num_labels)

        self.relu = torch.nn.ReLU(inplace=True)
        self.softmax = torch.nn.Softmax(dim=-2)
        self.composition = config.composition

    def load_vae(self, vae_path):
        self.vae.load_state_dict(torch.load(vae_path)['vae'])

        with torch.no_grad():
            idxes = torch.eye(self.config.topic_num)
            word_dist = self.vae.decode(idxes)
            word_dist = torch.softmax(word_dist,dim=1)
            self.topic_word_matrix = word_dist

    # use topic memory to enrich 
    def get_rep_by_pooler(self, input_ids, token_type_ids, attention_mask, topic_vector):
        # top_num = 10
        top_num = self.config.topic_num
        
        # print(input_ids.shape)
        input_ids, attention_mask, token_type_ids = clean_input_ids(input_ids, attention_mask, token_type_ids)
        device = input_ids.device

        # 获得隐藏层输出, (batch, sequence, embedding)
        temp_embeddings = self.embeddings(input_ids)

        final_embeddings = None
        final_attention_mask = None
        final_token_type_ids = None

        # concatenate memory
        # input_ids (batch, sequence)
        memory_len_one_tensor = torch.tensor([1] * top_num, requires_grad=False, device=device)
        one_tensor = torch.tensor([1], requires_grad=False, device=device)

        for index, batch_attention_mask in enumerate(attention_mask):
             # ----------------------通过memory来丰富信息---------------------
            # get topic memory
            _, top_indices = topic_vector[index].topk(top_num)
            
            idxes = torch.eye(self.config.topic_num).to(device)
            idxes = torch.index_select(idxes, 0, top_indices)

            word_dist = self.vae.decode(idxes)
            word_dist = torch.softmax(word_dist,dim=1)
            
            # print(word_dist.shape)

            topic_memory = self.memory_LayerNorm(self.memory_layer(word_dist))
            # -----------------------------------------

            input_embeddings = temp_embeddings[index][batch_attention_mask == 1]
            pad_embeddings = temp_embeddings[index][batch_attention_mask == 0]

            whole_embeddings = torch.cat((input_embeddings, topic_memory, pad_embeddings), dim=0)

            # 处理attention_mask
            whole_attention_mask = torch.cat((batch_attention_mask[batch_attention_mask == 1], memory_len_one_tensor,
                                              batch_attention_mask[batch_attention_mask == 0]), dim=-1)

            # 处理token_type_id
            remain_token_type_ids_len = batch_attention_mask.shape[0] + top_num - input_embeddings.shape[0]
            whole_token_type_ids = torch.cat((token_type_ids[index][batch_attention_mask == 1],
                                              one_tensor.repeat(remain_token_type_ids_len)), dim=-1)

            whole_embeddings = whole_embeddings.unsqueeze(0)
            whole_attention_mask = whole_attention_mask.unsqueeze(0)
            whole_token_type_ids = whole_token_type_ids.unsqueeze(0)

            if final_embeddings is None:
                final_embeddings = whole_embeddings
                final_attention_mask = whole_attention_mask
                # final_token_type_ids = whole_token_type_ids
            else:
                final_embeddings = torch.cat((final_embeddings, whole_embeddings), dim=0)
                final_attention_mask = torch.cat((final_attention_mask, whole_attention_mask), dim=0)
                # final_token_type_ids = torch.cat((final_token_type_ids, whole_token_type_ids), dim=0)

        out = self.bert_model(inputs_embeds=final_embeddings, attention_mask=final_attention_mask,
                              token_type_ids=final_token_type_ids)

        out = out['pooler_output']

        return out

    def get_rep_by_topic_attention(self, input_ids, token_type_ids, attention_mask, topic_vector):
        input_ids, attention_mask, token_type_ids = clean_input_ids(input_ids, attention_mask, token_type_ids)

        out = self.bert_model(input_ids=input_ids, token_type_ids=token_type_ids,
                              attention_mask=attention_mask)

        last_hidden_state = out['last_hidden_state']

        # 接下来通过attention汇总信息----------------------------------------
        # (batch size, topic num) -> (batch size, word_embedding_len)
        query = self.query_layer(topic_vector)
        query = self.LayerNorm(query)
        # (batch, 1, word_embedding_len)
        query = query.unsqueeze(1)

        # (batch, 1, sequence)
        weight = query.bmm(last_hidden_state.transpose(1, 2))

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
        embedding = final_weight.bmm(last_hidden_state)
        final_embedding = embedding.squeeze(1)

        return final_embedding

    def forward(self, q_input_ids, q_token_type_ids, q_attention_mask,
                a_input_ids, a_token_type_ids, a_attention_mask, q_bow, a_bow, word_idf=None):
        q_reconst, q_mu, q_log_var = self.vae(q_bow, lambda x: torch.softmax(x, dim=1))
        a_reconst, a_mu, a_log_var = self.vae(a_bow, lambda x: torch.softmax(x, dim=1))

        logsoftmax = torch.log_softmax(q_reconst, dim=1)
        if word_idf is None:
            q_rec_loss = -1.0 * torch.mean(q_bow * logsoftmax)
        else:
            q_rec_loss = -1.0 * torch.mean(q_bow * logsoftmax *  word_idf)

        q_kl_div = -0.5 * torch.mean(1 + q_log_var - q_mu.pow(2) - q_log_var.exp())
        
        logsoftmax = torch.log_softmax(a_reconst, dim=1)
        if word_idf is None:
            a_rec_loss = -1.0 * torch.mean(a_bow * logsoftmax)
        else:
            a_rec_loss = -1.0 * torch.mean(a_bow * logsoftmax * word_idf)

        a_kl_div = -0.5 * torch.mean(1 + a_log_var - a_mu.pow(2) - a_log_var.exp())

        vae_loss = q_rec_loss + a_rec_loss + q_kl_div  + a_kl_div

        # q_embeddings = self.get_rep_by_topic_attention(input_ids=q_input_ids, token_type_ids=q_token_type_ids,
        #                                                 attention_mask=q_attention_mask, topic_vector=q_mu)

        # a_embeddings = self.get_rep_by_topic_attention(input_ids=a_input_ids, token_type_ids=a_token_type_ids,
        #                                                 attention_mask=a_attention_mask, topic_vector=a_mu)

        q_embeddings = self.get_rep_by_pooler(input_ids=q_input_ids, token_type_ids=q_token_type_ids, attention_mask=q_attention_mask, topic_vector=q_mu)

        a_embeddings = self.get_rep_by_pooler(input_ids=a_input_ids, token_type_ids=a_token_type_ids, attention_mask=a_attention_mask, topic_vector=a_mu)

        # 计算得到分类概率
        logits = self.classifier(q_embedding=q_embeddings, a_embedding=a_embeddings, q_topic=q_mu, a_topic=a_mu)

        # only memory
        # logits = self.classifier(q_embedding=q_embeddings, a_embedding=a_embeddings)

        return logits, vae_loss


# --------------------------------------
# fen ge xian
# --------------------------------------
class QAOnlyMemoryModel(nn.Module):
    def __init__(self, config: QATopicConfig):

        super(QAOnlyMemoryModel, self).__init__()

        self.vae = VAE(config.voc_size, n_topic=config.topic_num)
        self.output_embedding_len = config.sentence_embedding_len

        self.config = config

        # 这个学习率不一样
        self.bert_model = AutoModel.from_pretrained(config.pretrained_bert_path)
        self.bert_model.resize_token_embeddings(config.tokenizer_len)

        # 这个embedding的grad会被计入bert model里，很好
        self.embeddings = self.bert_model.get_input_embeddings()

        self.topic_word_matrix = None
        
        # memory transformation
        self.memory_layer = nn.Sequential(
            nn.Linear(config.voc_size, 2 * config.word_embedding_len),
            nn.ReLU(),
            nn.Linear(2 * config.word_embedding_len, config.word_embedding_len),
        )
        self.memory_LayerNorm = nn.LayerNorm(config.word_embedding_len)

        # 注意力模型
        self.query_layer = nn.Sequential(
            nn.Linear(config.topic_num, 2 * config.word_embedding_len),
            nn.ReLU(),
            nn.Linear(2 * config.word_embedding_len, config.word_embedding_len),
        )
        self.LayerNorm = nn.LayerNorm(config.word_embedding_len)

        # 这些的学习率一样
        # self.classifier = QAClassifier(input_len=config.sentence_embedding_len, num_labels=config.num_labels)
        self.classifier = QAClassifier(input_len=config.sentence_embedding_len, num_labels=config.num_labels)

        self.relu = torch.nn.ReLU(inplace=True)
        self.softmax = torch.nn.Softmax(dim=-2)
        self.composition = config.composition

    def load_vae(self, vae_path):
        self.vae.load_state_dict(torch.load(vae_path)['vae'])

        with torch.no_grad():
            idxes = torch.eye(self.config.topic_num)
            word_dist = self.vae.decode(idxes)
            word_dist = torch.softmax(word_dist,dim=1)
            self.topic_word_matrix = word_dist

    # use topic memory to enrich 
    def get_rep_by_pooler(self, input_ids, token_type_ids, attention_mask):
        # print(input_ids.shape)

        input_ids, attention_mask, token_type_ids = clean_input_ids(input_ids, attention_mask, token_type_ids)
        device = input_ids.device

        # ----------------------通过memory来丰富信息---------------------
        # get topic memory
        idxes = torch.eye(self.config.topic_num).to(device)

        word_dist = self.vae.decode(idxes)
        word_dist = torch.softmax(word_dist,dim=1)
        
        # print(word_dist.shape)

        topic_memory = self.memory_LayerNorm(self.memory_layer(word_dist))

        # 获得隐藏层输出, (batch, sequence, embedding)
        temp_embeddings = self.embeddings(input_ids)

        final_embeddings = None
        final_attention_mask = None
        final_token_type_ids = None

        # concatenate memory
        # input_ids (batch, sequence)
        memory_len_one_tensor = torch.tensor([1] * self.config.topic_num, requires_grad=False, device=device)
        one_tensor = torch.tensor([1], requires_grad=False, device=device)

        for index, batch_attention_mask in enumerate(attention_mask):
            input_embeddings = temp_embeddings[index][batch_attention_mask == 1]
            pad_embeddings = temp_embeddings[index][batch_attention_mask == 0]

            whole_embeddings = torch.cat((input_embeddings, topic_memory, pad_embeddings), dim=0)

            # 处理attention_mask
            whole_attention_mask = torch.cat((batch_attention_mask[batch_attention_mask == 1], memory_len_one_tensor,
                                              batch_attention_mask[batch_attention_mask == 0]), dim=-1)

            # 处理token_type_id
            remain_token_type_ids_len = batch_attention_mask.shape[0] + self.config.topic_num - input_embeddings.shape[0]
            whole_token_type_ids = torch.cat((token_type_ids[index][batch_attention_mask == 1],
                                              one_tensor.repeat(remain_token_type_ids_len)), dim=-1)

            whole_embeddings = whole_embeddings.unsqueeze(0)
            whole_attention_mask = whole_attention_mask.unsqueeze(0)
            whole_token_type_ids = whole_token_type_ids.unsqueeze(0)

            if final_embeddings is None:
                final_embeddings = whole_embeddings
                final_attention_mask = whole_attention_mask
                # final_token_type_ids = whole_token_type_ids
            else:
                final_embeddings = torch.cat((final_embeddings, whole_embeddings), dim=0)
                final_attention_mask = torch.cat((final_attention_mask, whole_attention_mask), dim=0)
                # final_token_type_ids = torch.cat((final_token_type_ids, whole_token_type_ids), dim=0)

        out = self.bert_model(inputs_embeds=final_embeddings, attention_mask=final_attention_mask,
                              token_type_ids=final_token_type_ids)

        out = out['pooler_output']

        return out

    def get_rep_by_topic_attention(self, input_ids, token_type_ids, attention_mask, topic_vector):
        input_ids, attention_mask, token_type_ids = clean_input_ids(input_ids, attention_mask, token_type_ids)

        out = self.bert_model(input_ids=input_ids, token_type_ids=token_type_ids,
                              attention_mask=attention_mask)

        last_hidden_state = out['last_hidden_state']

        # 接下来通过attention汇总信息----------------------------------------
        # (batch size, topic num) -> (batch size, word_embedding_len)
        query = self.query_layer(topic_vector)
        query = self.LayerNorm(query)
        # (batch, 1, word_embedding_len)
        query = query.unsqueeze(1)

        # (batch, 1, sequence)
        weight = query.bmm(last_hidden_state.transpose(1, 2))

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
        embedding = final_weight.bmm(last_hidden_state)
        final_embedding = embedding.squeeze(1)

        return final_embedding

    def forward(self, q_input_ids, q_token_type_ids, q_attention_mask,
                a_input_ids, a_token_type_ids, a_attention_mask, q_bow, a_bow, word_idf=None):
        q_reconst, q_mu, q_log_var = self.vae(q_bow, lambda x: torch.softmax(x, dim=1))
        a_reconst, a_mu, a_log_var = self.vae(a_bow, lambda x: torch.softmax(x, dim=1))

        logsoftmax = torch.log_softmax(q_reconst, dim=1)
        if word_idf is None:
            q_rec_loss = -1.0 * torch.mean(q_bow * logsoftmax)
        else:
            q_rec_loss = -1.0 * torch.mean(q_bow * logsoftmax *  word_idf)

        q_kl_div = -0.5 * torch.mean(1 + q_log_var - q_mu.pow(2) - q_log_var.exp())
        
        logsoftmax = torch.log_softmax(a_reconst, dim=1)
        if word_idf is None:
            a_rec_loss = -1.0 * torch.mean(a_bow * logsoftmax)
        else:
            a_rec_loss = -1.0 * torch.mean(a_bow * logsoftmax * word_idf)

        a_kl_div = -0.5 * torch.mean(1 + a_log_var - a_mu.pow(2) - a_log_var.exp())

        vae_loss = q_rec_loss + a_rec_loss + q_kl_div  + a_kl_div

        # q_embeddings = self.get_rep_by_topic_attention(input_ids=q_input_ids, token_type_ids=q_token_type_ids,
        #                                                 attention_mask=q_attention_mask, topic_vector=q_mu)

        # a_embeddings = self.get_rep_by_topic_attention(input_ids=a_input_ids, token_type_ids=a_token_type_ids,
        #                                                 attention_mask=a_attention_mask, topic_vector=a_mu)

        q_embeddings = self.get_rep_by_pooler(input_ids=q_input_ids, token_type_ids=q_token_type_ids, attention_mask=q_attention_mask)

        a_embeddings = self.get_rep_by_pooler(input_ids=a_input_ids, token_type_ids=a_token_type_ids, attention_mask=a_attention_mask)

        # 计算得到分类概率
        logits = self.classifier(q_embedding=q_embeddings, a_embedding=a_embeddings)

        # only memory
        # logits = self.classifier(q_embedding=q_embeddings, a_embedding=a_embeddings)

        return logits, vae_loss


class QAOnlyHopMemoryModel(nn.Module):
    def __init__(self, config: QATopicConfig):

        super(QAOnlyHopMemoryModel, self).__init__()

        self.vae = VAE(config.voc_size, n_topic=config.topic_num)
        self.output_embedding_len = config.sentence_embedding_len

        self.config = config

        # 这个学习率不一样
        self.bert_model = AutoModel.from_pretrained(config.pretrained_bert_path)
        self.bert_model.resize_token_embeddings(config.tokenizer_len)

        # 这个embedding的grad会被计入bert model里，很好
        self.embeddings = self.bert_model.get_input_embeddings()

        # memory transformation
        self.memory_layer = nn.Sequential(
            nn.Linear(config.voc_size, 2 * config.word_embedding_len),
            nn.ReLU(),
            nn.Linear(2 * config.word_embedding_len, config.word_embedding_len),
        )
        self.memory_LayerNorm = nn.LayerNorm(config.word_embedding_len)

        # 注意力模型
        self.query_layer = nn.Sequential(
            nn.Linear(config.topic_num, 2 * config.word_embedding_len),
            nn.ReLU(),
            nn.Linear(2 * config.word_embedding_len, config.word_embedding_len),
        )
        self.LayerNorm = nn.LayerNorm(config.word_embedding_len)

        # 这些的学习率一样
        # self.classifier = QAClassifier(input_len=config.sentence_embedding_len, num_labels=config.num_labels)
        self.classifier = QAClassifier(input_len=config.sentence_embedding_len, num_labels=config.num_labels)

        self.relu = torch.nn.ReLU(inplace=True)
        self.softmax = torch.nn.Softmax(dim=-2)
        self.composition = config.composition

        # 这个学习率不一样
        self.whole_encoder = self.bert_model.encoder.layer

    # encode a text using lower layers
    def get_rep_by_pooler(self, input_ids, attention_mask, token_type_id, **kwargs):
        input_ids, attention_mask, token_type_id = clean_input_ids(input_ids, attention_mask, token_type_id)
        
        # get memory
        idxes = torch.eye(self.config.topic_num).to(device)

        word_dist = self.vae.decode(idxes)
        word_dist = torch.softmax(word_dist,dim=1)
        
        # print(word_dist.shape)

        topic_memory = self.memory_LayerNorm(self.memory_layer(word_dist))

        # 获得隐藏层输出, (batch, sequence, embedding)
        embeddings = self.embeddings(input_ids=input_ids)
        hidden_states = embeddings

        device = input_ids.device

        for layer_module in self.whole_encoder:
            # append memory for this layer
            final_embeddings = None
            final_attention_mask = None

            # concatenate memory
            # input_ids (batch, sequence)
            memory_len_one_tensor = torch.tensor([1] * self.config.topic_num, requires_grad=False, device=device)

            for index, batch_attention_mask in enumerate(attention_mask):
                input_embeddings = hidden_states[index][batch_attention_mask == 1]
                pad_embeddings = hidden_states[index][batch_attention_mask == 0]

                whole_embeddings = torch.cat((input_embeddings, topic_memory, pad_embeddings), dim=0)

                # 处理attention_mask
                whole_attention_mask = torch.cat((batch_attention_mask[batch_attention_mask == 1], memory_len_one_tensor,
                                                batch_attention_mask[batch_attention_mask == 0]), dim=-1)

                whole_embeddings = whole_embeddings.unsqueeze(0)
                whole_attention_mask = whole_attention_mask.unsqueeze(0)

                if final_embeddings is None:
                    final_embeddings = whole_embeddings
                    final_attention_mask = whole_attention_mask
                else:
                    final_embeddings = torch.cat((final_embeddings, whole_embeddings), dim=0)
                    final_attention_mask = torch.cat((final_attention_mask, whole_attention_mask), dim=0)
                    
                extended_attention_mask = self.bert_model.get_extended_attention_mask(final_attention_mask, final_attention_mask.shape, device)

                layer_outputs = layer_module(
                    final_embeddings,
                    extended_attention_mask
                )
                hidden_states = layer_outputs[0]
                
        return hidden_states

    def forward(self, q_input_ids, q_token_type_ids, q_attention_mask,
                a_input_ids, a_token_type_ids, a_attention_mask,
                b_input_ids, b_token_type_ids, b_attention_mask, **kwargs):

        # 先编码question
        q_input_ids, q_attention_mask, q_token_type_ids = clean_input_ids(q_input_ids, q_attention_mask, q_token_type_ids)
        b_input_ids, b_attention_mask, b_token_type_ids = clean_input_ids(b_input_ids, b_attention_mask, b_token_type_ids)

        # encoding a
        t_lower_encoded_embeddings = self.lower_encoding(input_ids=q_input_ids,
                                                         attention_mask=q_attention_mask,
                                                         token_type_id=q_token_type_ids)

        # encoding b
        b_token_type_ids = b_token_type_ids + 1

        b_lower_encoded_embeddings = self.lower_encoding(input_ids=b_input_ids,
                                                         attention_mask=b_attention_mask,
                                                         token_type_id=b_token_type_ids)

        # encoding together
        joint_embeddings = self.joint_encoding(a_embeddings=t_lower_encoded_embeddings,
                                               b_embeddings=b_lower_encoded_embeddings,
                                               a_attention_mask=q_attention_mask,
                                               b_attention_mask=b_attention_mask)

        q_embeddings = self.pooler_layer(joint_embeddings)

        # 编码answer
        a_embeddings = self.get_rep_by_pooler(input_ids=a_input_ids, token_type_ids=a_token_type_ids, attention_mask=a_attention_mask)

        logits = self.classifier(q_embedding=q_embeddings, a_embedding=a_embeddings)

        return logits


# --------------------------------------
# fen ge xian
# --------------------------------------
class CrossBERTConfig:
    def __init__(self, tokenizer_len, pretrained_bert_path='prajjwal1/bert-small', num_labels=4,
                 word_embedding_len=512, sentence_embedding_len=512, composition='pooler'):

        self.tokenizer_len = tokenizer_len
        self.pretrained_bert_path = pretrained_bert_path
        self.num_labels = num_labels
        self.word_embedding_len = word_embedding_len
        self.sentence_embedding_len = sentence_embedding_len
        self.composition = composition

    def __str__(self):
        print("*"*20 + "config" + "*"*20)
        print("tokenizer_len:", self.tokenizer_len)
        print("pretrained_bert_path:", self.pretrained_bert_path)
        print("num_labels:", self.num_labels)
        print("word_embedding_len:", self.word_embedding_len)
        print("sentence_embedding_len:", self.sentence_embedding_len)
        print("composition:", self.composition)


class CrossBERT(nn.Module):
    def __init__(self, config):

        super(CrossBERT, self).__init__()

        self.config = config

        # 毕竟num_label也算是memory的一部分
        self.num_labels = config.num_labels
        self.sentence_embedding_len = config.sentence_embedding_len

        # 这个学习率不一样
        this_bert_config = BertConfig.from_pretrained(config.pretrained_bert_path)
        this_bert_config.num_labels = self.num_labels

        self.bert_model = BertForSequenceClassification.from_pretrained(config.pretrained_bert_path, config=this_bert_config)
        self.bert_model.resize_token_embeddings(config.tokenizer_len)
        self.embeddings = self.bert_model.get_input_embeddings()

    def forward(self, input_ids, token_type_ids, attention_mask):

        # # 根据输入，进行思考, 思考的结果要选择性遗忘
        out = self.bert_model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        logits = out['logits']

        return logits


# --------------------------------------
# fen ge xian
# --------------------------------------
class ADecoderConfig:
    def __init__(self, tokenizer_len, pretrained_bert_path='prajjwal1/bert-small', num_labels=4,
                 word_embedding_len=512, sentence_embedding_len=512, composition='pooler', answer_context_num=1):

        self.tokenizer_len = tokenizer_len
        self.pretrained_bert_path = pretrained_bert_path
        self.num_labels = num_labels
        self.word_embedding_len = word_embedding_len
        self.sentence_embedding_len = sentence_embedding_len
        self.composition = composition
        self.answer_context_num = answer_context_num

    def __str__(self):
        print("*"*20 + "config" + "*"*20)
        print("tokenizer_len:", self.tokenizer_len)
        print("pretrained_bert_path:", self.pretrained_bert_path)
        print("num_labels:", self.num_labels)
        print("word_embedding_len:", self.word_embedding_len)
        print("sentence_embedding_len:", self.sentence_embedding_len)
        print("composition:", self.composition)
        print("answer_context_num:", self.answer_context_num)


class ADecoder(nn.Module):
    def __init__(self, config: ADecoderConfig):

        super(ADecoder, self).__init__()

        self.config = config

        # 毕竟num_label也算是memory的一部分
        self.num_labels = config.num_labels
        self.sentence_embedding_len = config.sentence_embedding_len

        # 这个学习率不一样
        self.this_bert_config = BertConfig.from_pretrained(config.pretrained_bert_path)

        # contain the parameters of encoder
        self.bert_model = MyBertModel.from_pretrained(config.pretrained_bert_path)

        self.bert_model.resize_token_embeddings(config.tokenizer_len)
        self.embeddings = self.bert_model.get_input_embeddings()

        # create models for compress answer
        self.composition_layer = nn.ModuleList([LinearSelfAttentionLayer(self.this_bert_config.hidden_size) for _ in range(config.answer_context_num)])

        # create models for decoder
        self.num_attention_heads = self.this_bert_config.num_attention_heads
        self.attention_head_size = int(self.this_bert_config.hidden_size / self.this_bert_config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.decoder = nn.ModuleDict({
            'answer_query': nn.ModuleList([nn.Linear(self.this_bert_config.hidden_size, self.all_head_size) for _ in
                                           range(self.this_bert_config.num_hidden_layers)]),
            'answer_key': nn.ModuleList([nn.Linear(self.this_bert_config.hidden_size, self.all_head_size) for _ in
                                           range(self.this_bert_config.num_hidden_layers)]),
            'answer_value': nn.ModuleList([nn.Linear(self.this_bert_config.hidden_size, self.all_head_size) for _ in
                                           range(self.this_bert_config.num_hidden_layers)]),
            'layer_chunks': nn.ModuleList([DecoderLayerChunk(self.this_bert_config) for _ in range(self.this_bert_config.num_hidden_layers)])
        })

        # self.decode_query = nn.ModuleList([nn.Linear(self.this_bert_config.hidden_size, self.all_head_size) for _ in range(self.this_bert_config.num_hidden_layers)])
        # self.decode_layer_chunks = nn.ModuleList([DecoderLayerChunk(self.this_bert_config) for _ in range(self.this_bert_config.num_hidden_layers)])

        # create classifier
        self.classifier = QAClassifier(input_len=config.sentence_embedding_len, num_labels=config.num_labels)

        self.init_decoder_params()

    def init_decoder_params(self):
        #  read parameters for layer chunks from encoder
        for index in range(self.this_bert_config.num_hidden_layers):
            this_static = self.decoder['layer_chunks'][index].state_dict()
            pretrained_static = self.bert_model.encoder.layer[index].state_dict()

            updating_query_static = {}
            for k in this_static.keys():
                new_k = k.replace('attention.', 'attention.output.', 1)
                updating_query_static[k] = pretrained_static[new_k]

            this_static.update(updating_query_static)
            self.decoder['layer_chunks'][index].load_state_dict(this_static)

        #  read parameters for query from encoder
        pretrained_static = self.bert_model.state_dict()
        query_static = self.decoder['answer_query'].state_dict()

        updating_query_static = {}
        for k in query_static.keys():
            full_key = 'encoder.layer.' + k.split(".")[0] + '.attention.self.query.' + k.split(".")[1]
            updating_query_static[k] = pretrained_static[full_key]

        query_static.update(updating_query_static)
        self.decoder['answer_query'].load_state_dict(query_static)

    # use the average embedding of last layer as sentence representation
    @staticmethod
    def get_rep_by_avg(embeddings, token_type_ids, attention_mask):
        with torch.no_grad():
            # (batch_size, sequence_length)
            temp_mask = token_type_ids.clone().detach()

            # remove tokens whose type id is 1
            temp_mask[temp_mask == 0] = 2
            temp_mask -= 1

            # remove tokens whose attention mask is 0
            temp_attention_mask = attention_mask.clone().detach()
            temp_mask = temp_mask * temp_attention_mask
            # remove cls, if cls is removed, some sentences may be empty
            # temp_mask[:, 0] = 0

            # remove sep--the last token
            sequence_len = temp_mask.sum(dim=-1) - 1
            sequence_len = sequence_len.unsqueeze(-1)
            temp_mask.scatter_(dim=1, index=sequence_len,
                               src=torch.zeros((temp_mask.shape[0], 1), device=embeddings.device,
                                               dtype=temp_mask.dtype))

            # (batch_size, sequence_length, 1)
            temp_mask = temp_mask.unsqueeze(-1)

        embeddings = embeddings * temp_mask

        # get average embedding
        representations = embeddings.sum(dim=1)

        representations = representations / sequence_len

        return representations

    # 如果不把a传进q，就会退化成最普通的QAModel，已经被验证过了
    def forward(self, q_input_ids, q_token_type_ids, q_attention_mask,
                a_input_ids, a_token_type_ids, a_attention_mask):

        # get a
        a_out = self.bert_model(input_ids=a_input_ids, token_type_ids=a_token_type_ids, attention_mask=a_attention_mask,
                                enrich_answer_by_question=False)
        # (batch size, sequence len, dim)
        a_last_hidden_state = a_out['last_hidden_state']

        a_embeddings = None
        for index in range(self.config.answer_context_num):
            # (batch size, 1, dim)
            this_answer_context = self.composition_layer[index](a_last_hidden_state).unsqueeze(1)

            if a_embeddings is None:
                a_embeddings = this_answer_context
            else:
                a_embeddings = torch.cat((a_embeddings, this_answer_context), dim=1)

        # a_embeddings = self.get_rep_by_avg(a_last_hidden_state, token_type_ids=a_token_type_ids, attention_mask=a_attention_mask)
        # # (batch, 1, sequence dim)
        # a_embeddings = a_embeddings.unsqueeze(-2)

        # get q and new a
        q_out = self.bert_model(input_ids=q_input_ids, token_type_ids=q_token_type_ids, attention_mask=q_attention_mask,
                                enrich_answer_by_question=True, answer_embeddings=a_embeddings,
                                decoder=self.decoder)

        q_last_hidden_state, a_embeddings = q_out['last_hidden_state'], q_out['a_embedding']

        q_embeddings = self.get_rep_by_avg(q_last_hidden_state, token_type_ids=q_token_type_ids,
                                           attention_mask=q_attention_mask)

        a_embeddings = a_embeddings.squeeze(-2)
        logits = self.classifier(q_embedding=q_embeddings, a_embedding=a_embeddings)

        return logits


# --------------------------------------
# fen ge xian
# --------------------------------------
# according to Multihop Attention Networks for Question Answer Matching
class LinearSelfAttentionLayer(nn.Module):
    def __init__(self, input_dim, drop_prob=0.2):
        super(LinearSelfAttentionLayer, self).__init__()
        self.conversion_layer = nn.Linear(input_dim, input_dim*2)
        self.weight_layer = nn.Linear(input_dim*2, input_dim)

        self.dropout = torch.nn.Dropout(p=drop_prob)
        self.tanh = torch.nn.Tanh()
        self.softmax = nn.Softmax(dim=-1)

    # context shape is supposed to be (batch size, sequence dim, input_dim)
    # output is (batch size, input_dim)
    def forward(self, context):
        # (batch size, sequence dim, input_dim)
        converted_context = self.tanh(self.conversion_layer(context))
        attention_weight = self.softmax(self.weight_layer(converted_context))

        processed_context = attention_weight.mul(context)

        # sum by dim -1
        # (batch size, input_dim)
        context_representation = processed_context.sum(dim=-2)
        return context_representation


# --------------------------------------
# fen ge xian
# --------------------------------------
# 一个分类器，2个input len的向量作为输入
class QAClassifier(nn.Module):
    def __init__(self, input_len, keep_prob=0.9, num_labels=4):
        super(QAClassifier, self).__init__()

        self.linear1 = torch.nn.Linear(input_len * 2, input_len * 2, bias=True)
        self.bn1 = nn.BatchNorm1d(input_len * 2)
        self.linear2 = torch.nn.Linear(input_len * 2, input_len, bias=True)
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

    def forward(self, q_embedding, a_embedding):
        x = torch.cat((q_embedding, a_embedding), dim=-1)

        res_x = x
        x = self.linear1(x)
        x = self.relu(x)
        # print(x.shape, res_x.shape)
        x = self.dropout(x) + res_x
        # x = self.bn1(x)

        x = self.linear2(x)
        x = self.relu(x)
        x = self.dropout(x)
        # x = self.bn2(x)

        res_x = x
        x = self.linear3(x)
        x = self.relu(x)
        x = self.dropout(x) + res_x
        # x = self.bn3(x)

        x = self.linear4(x)

        return x


class QATopicClassifier(nn.Module):
    def __init__(self, input_len, topic_num, keep_prob=0.9, num_labels=4):
        super(QATopicClassifier, self).__init__()

        self.linear1 = torch.nn.Linear((input_len + topic_num)* 2, (input_len + topic_num)* 2, bias=True)
        self.bn1 = nn.BatchNorm1d((input_len + topic_num)* 2)
        self.linear2 = torch.nn.Linear((input_len + topic_num)* 2, (input_len + topic_num), bias=True)
        self.bn2 = nn.BatchNorm1d((input_len + topic_num))
        self.linear3 = torch.nn.Linear((input_len + topic_num), (input_len + topic_num), bias=True)
        self.bn3 = nn.BatchNorm1d((input_len + topic_num))
        self.linear4 = torch.nn.Linear((input_len + topic_num), num_labels, bias=True)

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

    def forward(self, q_embedding, a_embedding, q_topic, a_topic):
        x = torch.cat((q_embedding, a_embedding, q_topic, a_topic), dim=-1)

        res_x = x
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        # x = self.bn1(x)

        x = self.linear2(x)
        x = self.relu(x)
        x = self.dropout(x)
        # x = self.bn2(x)

        res_x = x
        x = self.linear3(x)
        x = self.relu(x)
        x = self.dropout(x)
        # x = self.bn3(x)

        x = self.linear4(x)

        return x


def raise_test_exception():
    raise Exception("test end!")


def clean_input_ids(input_ids, attention_mask, token_type_ids):
    max_seq_len = torch.max(attention_mask.sum(-1))

    # ensure only pad be filtered
    dropped_input_ids = attention_mask[:, max_seq_len:]
    # try:
    assert torch.max(dropped_input_ids.sum(-1)) == 0
    # except AssertionError:
    #     print(input_ids[:, max_seq_len:][0])
    #     exit()

    input_ids = input_ids[:, :max_seq_len]
    token_type_ids = token_type_ids[:, :max_seq_len]
    attention_mask = attention_mask[:, :max_seq_len]

    return input_ids, attention_mask, token_type_ids
