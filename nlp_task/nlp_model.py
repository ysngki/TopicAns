from transformers import BertModel, BertForSequenceClassification, BertConfig
from transformers.models.bert.out_vector_modeling_bert import MyBertModel, DecoderLayerChunk
import torch
import torch.nn as nn
import torch.utils.data
import numpy as np
import torch.nn.functional
import torch.nn.functional as F

from my_function import get_rep_by_avg, dot_attention

# model list
# 1. QAModel
#       basic bi-encoder, support mem
# 2. CrossBERT
#       basic cross-encoder
# 3. ParallelEncoder
#       my model


# --------------------------------------
# fen ge xian
# --------------------------------------
# No matter whether use memory mechanism, all models whose input consists of Q and A can use this class
class QAClassifierModelConfig:
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


class QAClassifierModel(nn.Module):
    def __init__(self, config):

        super(QAClassifierModel, self).__init__()

        self.config = config

        # 毕竟num_label也算是memory的一部分
        self.num_labels = config.num_labels
        self.sentence_embedding_len = config.sentence_embedding_len

        # 这个学习率不一样
        self.bert_model = BertModel.from_pretrained(config.pretrained_bert_path)
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
        out = self.bert_model(input_ids=input_ids, attention_mask=attention_mask,
                              token_type_ids=token_type_ids)

        out = out['pooler_output']

        return out

    # use the average embedding of last layer as sentence representation
    def get_rep_by_avg(self, input_ids, token_type_ids, attention_mask):
        out = self.bert_model(input_ids=input_ids, attention_mask=attention_mask,
                              token_type_ids=token_type_ids)

        # get the average embeddings of last layer, excluding memory and special token(?)
        # (batch_size, sequence_length, hidden_size)
        last_hidden_state = out['last_hidden_state']

        representations = get_rep_by_avg(embeddings=last_hidden_state, attention_mask=attention_mask, token_type_ids=token_type_ids)
        representations = representations.squeeze(-2)

        return representations

    def forward(self, a_input_ids, a_token_type_ids, a_attention_mask,
                b_input_ids, b_token_type_ids, b_attention_mask):

        if self.config.composition == 'avg':
            composition_function = self.get_rep_by_avg
        elif self.config.composition == 'pooler':
            composition_function = self.get_rep_by_pooler
        else:
            raise Exception(f"Composition {self.config.composition} is not supported!!")

        a_embeddings = composition_function(input_ids=a_input_ids, token_type_ids=a_token_type_ids,
                                            attention_mask=a_attention_mask)

        b_embeddings = composition_function(input_ids=b_input_ids, token_type_ids=b_token_type_ids,
                                            attention_mask=b_attention_mask)

        # # 根据输入，进行思考, 思考的结果要选择性遗忘
        logits = self.classifier(a_embedding=a_embeddings, b_embedding=b_embeddings)

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
class ParallelEncoderConfig:
    def __init__(self, tokenizer_len, pretrained_bert_path='prajjwal1/bert-small', num_labels=4,
                 word_embedding_len=512, sentence_embedding_len=512, composition='pooler', context_num=1):

        self.tokenizer_len = tokenizer_len
        self.pretrained_bert_path = pretrained_bert_path
        self.num_labels = num_labels
        self.word_embedding_len = word_embedding_len
        self.sentence_embedding_len = sentence_embedding_len
        self.composition = composition
        self.context_num = context_num

    def __str__(self):
        print("*"*20 + "config" + "*"*20)
        print("tokenizer_len:", self.tokenizer_len)
        print("pretrained_bert_path:", self.pretrained_bert_path)
        print("num_labels:", self.num_labels)
        print("word_embedding_len:", self.word_embedding_len)
        print("sentence_embedding_len:", self.sentence_embedding_len)
        print("composition:", self.composition)
        print("context_num:", self.context_num)


class ParallelEncoder(nn.Module):
    def __init__(self, config: ParallelEncoderConfig):

        super(ParallelEncoder, self).__init__()

        self.config = config

        # 毕竟num_label也算是memory的一部分
        self.num_labels = config.num_labels
        self.sentence_embedding_len = config.sentence_embedding_len

        # 这个学习率不一样
        self.this_bert_config = BertConfig.from_pretrained(config.pretrained_bert_path)

        # contain the parameters of encoder
        self.bert_model = MyBertModel.from_pretrained(config.pretrained_bert_path)
        self.bert_model.resize_token_embeddings(config.tokenizer_len)

        # self.embeddings = self.bert_model.get_input_embeddings()

        # used to compressing candidate to generate context vectors
        self.composition_layer = OutVectorSelfAttentionLayer(self.this_bert_config.hidden_size, config.context_num)

        # create models for decoder
        self.num_attention_heads = self.this_bert_config.num_attention_heads
        self.attention_head_size = int(self.this_bert_config.hidden_size / self.this_bert_config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.decoder = nn.ModuleDict({
            # used by candidate context embeddings to enrich themselves
            'candidate_query': nn.ModuleList([nn.Linear(self.this_bert_config.hidden_size, self.all_head_size) for _ in
                                              range(self.this_bert_config.num_hidden_layers)]),
            'candidate_key': nn.ModuleList([nn.Linear(self.this_bert_config.hidden_size, self.all_head_size) for _ in
                                            range(self.this_bert_config.num_hidden_layers)]),
            'candidate_value': nn.ModuleList([nn.Linear(self.this_bert_config.hidden_size, self.all_head_size) for _ in
                                              range(self.this_bert_config.num_hidden_layers)]),
            # used to project representations into same space
            'layer_chunks': nn.ModuleList(
                [DecoderLayerChunk(self.this_bert_config) for _ in range(self.this_bert_config.num_hidden_layers)]),
            # used to compress candidate context embeddings into a vector
            'candidate_composition_layer': MeanLayer(),
            # used to generate query to compress Text_A thus get its representation
            'compress_query': nn.ModuleList([nn.Linear(self.this_bert_config.hidden_size, self.all_head_size) for _ in
                                             range(self.this_bert_config.num_hidden_layers)]),
            # used to update hidden states with self-att output as input
            'LSTM': MyLSTMBlock(self.this_bert_config.hidden_size),
        })

        # create classifier
        self.classifier = QAClassifier(input_len=config.sentence_embedding_len, num_labels=config.num_labels)

        self.init_decoder_params()

    # use parameters of pre-trained encoder to initialize decoder
    def init_decoder_params(self):
        #  read parameters for layer chunks from encoder
        for index in range(self.this_bert_config.num_hidden_layers):
            this_static = self.decoder['layer_chunks'][index].state_dict()
            pretrained_static = self.bert_model.encoder.layer[index].state_dict()

            updating_layer_static = {}
            for k in this_static.keys():
                new_k = k.replace('attention.', 'attention.output.', 1)
                updating_layer_static[k] = pretrained_static[new_k]

            this_static.update(updating_layer_static)
            self.decoder['layer_chunks'][index].load_state_dict(this_static)

        #  read parameters for query from encoder
        pretrained_static = self.bert_model.state_dict()
        query_static = self.decoder['candidate_query'].state_dict()

        updating_query_static = {}
        for k in query_static.keys():
            full_key = 'encoder.layer.' + k.split(".")[0] + '.attention.self.query.' + k.split(".")[1]
            updating_query_static[k] = pretrained_static[full_key]

        query_static.update(updating_query_static)
        self.decoder['candidate_query'].load_state_dict(query_static)

        #  read parameters for compress_query from encoder
        compress_query_static = self.decoder['compress_query'].state_dict()

        updating_query_static = {}
        for k in compress_query_static.keys():
            full_key = 'encoder.layer.' + k.split(".")[0] + '.attention.self.query.' + k.split(".")[1]
            updating_query_static[k] = pretrained_static[full_key]

        compress_query_static.update(updating_query_static)
        self.decoder['compress_query'].load_state_dict(compress_query_static)

        #  read parameters for key from encoder
        key_static = self.decoder['candidate_key'].state_dict()

        updating_key_static = {}
        for k in key_static.keys():
            full_key = 'encoder.layer.' + k.split(".")[0] + '.attention.self.key.' + k.split(".")[1]
            updating_key_static[k] = pretrained_static[full_key]

        key_static.update(updating_key_static)
        self.decoder['candidate_key'].load_state_dict(key_static)

        #  read parameters for value from encoder
        value_static = self.decoder['candidate_value'].state_dict()

        updating_value_static = {}
        for k in value_static.keys():
            full_key = 'encoder.layer.' + k.split(".")[0] + '.attention.self.value.' + k.split(".")[1]
            updating_value_static[k] = pretrained_static[full_key]

        value_static.update(updating_value_static)
        self.decoder['candidate_value'].load_state_dict(value_static)

    # 如果不把a传进q，就会退化成最普通的QAModel，已经被验证过了
    # b_text can be pre-computed
    def forward(self, a_input_ids, a_token_type_ids, a_attention_mask,
                b_input_ids, b_token_type_ids, b_attention_mask, candidate_num=1):
        # encoding candidate texts
        # (text_b_num, sequence len, dim)
        b_out = self.bert_model(input_ids=b_input_ids, token_type_ids=b_token_type_ids, attention_mask=b_attention_mask,
                                enrich_candidate_by_question=False)
        b_last_hidden_state = b_out['last_hidden_state']

        # get (text_b_num, context_num, dim)
        b_embeddings = self.composition_layer(b_last_hidden_state, attention_mask=b_attention_mask)
        # convert to (query_num, candidate_context_num, dim)
        if candidate_num != b_embeddings.shape[0]:
            b_embeddings = b_embeddings.reshape(a_input_ids.shape[0], candidate_num*self.config.context_num, b_embeddings.shape[-1])
        elif candidate_num == b_embeddings.shape[0]:
            # need broadcast
            b_embeddings = b_embeddings.reshape(-1, b_embeddings.shape[-1]).unsqueeze(0).repeat(a_input_ids.shape[0], 1, 1)
        else:
            raise Exception("Candidate num should either equal to text_b_num or equal to (text_b_num / text_a_num)!")

        lstm_initial_state = torch.zeros(a_input_ids.shape[0], candidate_num, b_embeddings.shape[-1], device=b_embeddings.device)
        # (query_num, candidate_context_num + candidate_num, dim)
        b_embeddings = torch.cat((b_embeddings, lstm_initial_state), dim=1)

        # get q and new a
        a_out = self.bert_model(input_ids=a_input_ids, token_type_ids=a_token_type_ids, attention_mask=a_attention_mask,
                                enrich_candidate_by_question=True, candidate_embeddings=b_embeddings,
                                decoder=self.decoder, candidate_num=candidate_num)

        a_last_hidden_state, decoder_output = a_out['last_hidden_state'], a_out['decoder_output']

        # get final representations
        # (query_num, candidate_num, context_num, dim)
        b_context_embeddings = decoder_output[:,:-candidate_num,:].reshape(decoder_output.shape[0], candidate_num, self.config.context_num, decoder_output.shape[-1])
        # (query_num, candidate_num, dim)
        b_embeddings = self.decoder['candidate_composition_layer'](b_context_embeddings).squeeze(-2)

        a_embeddings = decoder_output[:,-candidate_num:,:]

        logits = self.classifier(a_embedding=a_embeddings, b_embedding=b_embeddings)
        if candidate_num == 1:
            logits = logits.squeeze(1)

        return logits


# --------------------------------------
# fen ge xian
# --------------------------------------
# A gate, a substitution of resnet
class MyLSTMBlock(nn.Module):
    def __init__(self, input_dim):
        super(MyLSTMBlock, self).__init__()
        self.weight_layer = nn.Linear(input_dim*3, input_dim)

        self.sigmoid = nn.Sigmoid()

    # shapes are supposed to be (batch size, input_dim)
    def forward(self, this_compressed_vector, last_compressed_vector, weight_hint):
        weight = self.weight_layer(torch.cat((weight_hint, this_compressed_vector, last_compressed_vector), dim=-1))
        weight = self.sigmoid(weight)

        new_compressed_vector = weight * this_compressed_vector + (1-weight)*last_compressed_vector

        return new_compressed_vector


# --------------------------------------
# Composition Layers
# 1. LinearSelfAttentionLayer: Multi-layer self-attention
# 2. OutVectorSelfAttentionLayer: maintain several queries to get several representations by attention mechanism
# 3. MeanLayer: do average to get representations
# --------------------------------------
# according to Multihop Attention Networks for Question Answer Matching
class LinearSelfAttentionLayer(nn.Module):
    def __init__(self, input_dim, drop_prob=0.2):
        super(LinearSelfAttentionLayer, self).__init__()
        self.conversion_layer = nn.Linear(input_dim, input_dim*2)
        self.weight_layer = nn.Linear(input_dim*2, input_dim)

        self.dropout = torch.nn.Dropout(p=drop_prob)
        self.tanh = torch.nn.Tanh()
        # here is different from that paper
        self.softmax = nn.Softmax(dim=-2)

    # context shape is supposed to be (batch size, sequence len, input_dim)
    # attention_mask shape is (batch size, sequence len)
    # output is (batch size, input_dim)
    def forward(self, context, attention_mask):
        # (batch size, sequence dim, input_dim)
        converted_context = self.tanh(self.conversion_layer(context))
        raw_attention_weight = self.weight_layer(converted_context)

        # 创作出mask
        with torch.no_grad():
            mask = attention_mask.type(dtype=torch.float).clone().detach()
            mask[mask == 0] = -np.inf
            mask[mask == 1] = 0.0
            mask = mask.repeat(context.shape[-1], 1, 1)
            mask.transpose_(0, 1)
            mask.transpose_(1, 2)

        raw_attention_weight = raw_attention_weight + mask
        attention_weight = self.softmax(raw_attention_weight)

        processed_context = attention_weight.mul(context)

        # sum by dim -1
        # (batch size, input_dim)
        context_representation = processed_context.sum(dim=-2)
        return context_representation


class OutVectorSelfAttentionLayer(nn.Module):
    def __init__(self, input_dim, context_num):
        super(OutVectorSelfAttentionLayer, self).__init__()
        self.query = nn.Parameter(torch.randn(context_num, input_dim))

    def forward(self, context, attention_mask=None):
        """
        :param context: (..., sequence len, input_dim)
        :param attention_mask: (..., sequence len)
        :return: (..., context_num, input_dim)
        """

        context_representation = dot_attention(q=self.query, k=context, v=context, v_mask=attention_mask)

        return context_representation


class MeanLayer(nn.Module):
    def __init__(self):
        super(MeanLayer, self).__init__()

    @staticmethod
    def forward(embeddings, token_type_ids=None, attention_mask=None):
        """
        do average at dim -2 without remove this dimension
        :param embeddings: (..., sequence len, dim)
        :param token_type_ids: optional
        :param attention_mask: optional
        :return: (..., 1, dim)
        """

        representation = get_rep_by_avg(embeddings, token_type_ids, attention_mask)

        return representation


# --------------------------------------
# fen ge xian
# --------------------------------------
class PolyEncoderConfig:
    def __init__(self, tokenizer_len, pretrained_bert_path='prajjwal1/bert-small', num_labels=4,
                 word_embedding_len=512, sentence_embedding_len=512, context_num=1):

        self.tokenizer_len = tokenizer_len
        self.pretrained_bert_path = pretrained_bert_path
        self.num_labels = num_labels
        self.word_embedding_len = word_embedding_len
        self.sentence_embedding_len = sentence_embedding_len
        self.context_num = context_num

    def __str__(self):
        print("*"*20 + "config" + "*"*20)
        print("tokenizer_len:", self.tokenizer_len)
        print("pretrained_bert_path:", self.pretrained_bert_path)
        print("num_labels:", self.num_labels)
        print("word_embedding_len:", self.word_embedding_len)
        print("sentence_embedding_len:", self.sentence_embedding_len)
        print("context_num:", self.context_num)


class PolyEncoder(nn.Module):
    def __init__(self, config: PolyEncoderConfig):

        super(PolyEncoder, self).__init__()

        self.config = config

        # 毕竟num_label也算是memory的一部分
        self.num_labels = config.num_labels
        self.sentence_embedding_len = config.sentence_embedding_len

        # 这个学习率不一样
        self.this_bert_config = BertConfig.from_pretrained(config.pretrained_bert_path)

        # contain the parameters of encoder
        self.bert_model = BertModel.from_pretrained(config.pretrained_bert_path)
        self.bert_model.resize_token_embeddings(config.tokenizer_len)

        self.query_composition_layer = OutVectorSelfAttentionLayer(self.this_bert_config.hidden_size, self.config.context_num)

        self.classifier = QAClassifier(input_len=config.sentence_embedding_len, num_labels=config.num_labels)

        torch.nn.init.normal_(self.query_composition_layer.query, self.this_bert_config.hidden_size ** -0.5)

    # 如果不把a传进q，就会退化成最普通的QAModel，已经被验证过了
    # b_text can be pre-computed
    def forward(self, a_input_ids, a_token_type_ids, a_attention_mask,
                b_input_ids, b_token_type_ids, b_attention_mask):

        batch_size = a_token_type_ids.shape[0]

        # encoding candidate texts
        # (batch_size, sequence len, dim)
        b_out = self.bert_model(input_ids=b_input_ids, token_type_ids=b_token_type_ids, attention_mask=b_attention_mask)
        b_last_hidden_state = b_out['last_hidden_state']
        # (batch_size, 1, dim) ---- pooler
        candidate_context_vectors = b_last_hidden_state[:,0,:].unsqueeze(-2)

        # encoding query texts
        a_out = self.bert_model(input_ids=a_input_ids, token_type_ids=a_token_type_ids, attention_mask=a_attention_mask)
        a_last_hidden_state = a_out['last_hidden_state']
        # (batch_size, context_num, dim)
        query_context_vectors = self.query_composition_layer(a_last_hidden_state, a_attention_mask)

        # one-one classification task
        if self.config.num_labels > 0:
            pass
        else:
            # (batch_size, batch_size, dim)
            candidate_context_vectors = candidate_context_vectors.squeeze(-2).unsqueeze(0).repeat(batch_size, batch_size, query_context_vectors.shape[-1])

        # (batch_size, 1, dim) or (batch_size, batch_size, dim)
        # i,j,k ---> i means different query, j means different candidate
        final_query_context_vec = dot_attention(q=candidate_context_vectors, k=query_context_vectors,
                                                v=query_context_vectors)

        # one-one classification task
        if self.config.num_labels > 0:
            final_query_context_vec = final_query_context_vec.squeeze(-2)
            candidate_context_vectors = candidate_context_vectors.squeeze(-2)

            logits = self.classifier(a_embedding=final_query_context_vec, b_embedding=candidate_context_vectors)

            return logits
        # cos full-match task
        else:
            # (batch size, batch size)
            # first is query, second is candidate
            dot_product = torch.sum(final_query_context_vec * candidate_context_vectors, -1)
            mask = torch.eye(batch_size).to(dot_product.device)
            loss = F.log_softmax(dot_product, dim=-1) * mask
            loss = (-loss.sum(dim=1)).mean()

            return loss


# --------------------------------------
# fen ge xian
# --------------------------------------
# 一个分类器，2个input len的向量作为输入
class QAClassifier(nn.Module):
    def __init__(self, input_len, keep_prob=0.9, num_labels=4):
        super(QAClassifier, self).__init__()

        self.linear1 = torch.nn.Linear(input_len * 3, input_len * 3, bias=True)
        self.linear2 = torch.nn.Linear(input_len * 3, num_labels, bias=True)

        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(p=1 - keep_prob)
        self.init_weights()

    def init_weights(self):
        torch.nn.init.xavier_uniform_(self.linear1.weight)
        torch.nn.init.xavier_uniform_(self.linear2.weight)

    def forward(self, a_embedding, b_embedding):
        x = torch.cat((a_embedding, b_embedding, torch.abs(a_embedding-b_embedding)), dim=-1)
        # x = torch.cat((q_embedding, a_embedding), dim=-1)

        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.linear2(x)

        return x


def raise_test_exception():
    raise Exception("test end!")
