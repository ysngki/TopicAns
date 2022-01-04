from .modeling_bert import *

import math
import os
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.utils.checkpoint
from packaging import version
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from ...activations import ACT2FN
from ...file_utils import (
	ModelOutput,
	add_code_sample_docstrings,
	add_start_docstrings,
	add_start_docstrings_to_model_forward,
	replace_return_docstrings,
)
from ...modeling_outputs import (
	BaseModelOutputWithPastAndCrossAttentions,
	BaseModelOutputWithPoolingAndCrossAttentions,
	CausalLMOutputWithCrossAttentions,
	MaskedLMOutput,
	MultipleChoiceModelOutput,
	NextSentencePredictorOutput,
	QuestionAnsweringModelOutput,
	SequenceClassifierOutput,
	TokenClassifierOutput,
)
from ...modeling_utils import (
	PreTrainedModel,
	apply_chunking_to_forward,
	find_pruneable_heads_and_indices,
	prune_linear_layer,
)
from ...utils import logging
from .configuration_bert import BertConfig

logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "bert-base-uncased"
_CONFIG_FOR_DOC = "BertConfig"
_TOKENIZER_FOR_DOC = "BertTokenizer"

BERT_PRETRAINED_MODEL_ARCHIVE_LIST = [
	"bert-base-uncased",
	"bert-large-uncased",
	"bert-base-cased",
	"bert-large-cased",
	"bert-base-multilingual-uncased",
	"bert-base-multilingual-cased",
	"bert-base-chinese",
	"bert-base-german-cased",
	"bert-large-uncased-whole-word-masking",
	"bert-large-cased-whole-word-masking",
	"bert-large-uncased-whole-word-masking-finetuned-squad",
	"bert-large-cased-whole-word-masking-finetuned-squad",
	"bert-base-cased-finetuned-mrpc",
	"bert-base-german-dbmdz-cased",
	"bert-base-german-dbmdz-uncased",
	"cl-tohoku/bert-base-japanese",
	"cl-tohoku/bert-base-japanese-whole-word-masking",
	"cl-tohoku/bert-base-japanese-char",
	"cl-tohoku/bert-base-japanese-char-whole-word-masking",
	"TurkuNLP/bert-base-finnish-cased-v1",
	"TurkuNLP/bert-base-finnish-uncased-v1",
	"wietsedv/bert-base-dutch-cased",
	# See all BERT models at https://huggingface.co/models?filter=bert
]


def raise_test_exception():
	raise Exception("test end!")


# return the results of self attention, if candidate exists, it will concanated with question returened
class MyBertSelfAttention(nn.Module):
	def __init__(self, config, position_embedding_type=None):
		super().__init__()
		if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
			raise ValueError(
				f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
				f"heads ({config.num_attention_heads})"
			)

		self.num_attention_heads = config.num_attention_heads
		self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
		self.all_head_size = self.num_attention_heads * self.attention_head_size

		self.query = nn.Linear(config.hidden_size, self.all_head_size)
		self.key = nn.Linear(config.hidden_size, self.all_head_size)
		self.value = nn.Linear(config.hidden_size, self.all_head_size)

		self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
		self.position_embedding_type = position_embedding_type or getattr(
			config, "position_embedding_type", "absolute"
		)
		if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
			self.max_position_embeddings = config.max_position_embeddings
			self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)

		self.is_decoder = config.is_decoder

	def transpose_for_scores(self, x):
		new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
		x = x.view(*new_x_shape)
		# (batch size, head num, sequence len, head dim)
		return x.permute(0, 2, 1, 3)

	def forward(
			self,
			hidden_states,
			attention_mask=None,
			head_mask=None,
			encoder_hidden_states=None,
			encoder_attention_mask=None,
			past_key_value=None,
			output_attentions=False,
			enrich_candidate_by_question=False,
			# (batch_size, candidate_context_num, dim)
			candidate_context_embeddings=None,
			lstm_hint_vector=None,
			decoder=None,
			layer_index=None,
			candidate_num=1,
	):
		mixed_query_layer = self.query(hidden_states)
		# get query for candidate
		if enrich_candidate_by_question:
			if candidate_context_embeddings is None or decoder is None or layer_index is None:
				raise Exception(
					"candidate_embeddings or decode_query_layer or layer_index missed for enrich_candidate_by_question!")

			# (query_num, candidate_context_num, vec_dim)
			# used to enrich candidate contexts themselves
			candidate_query = decoder['candidate_query'][layer_index](candidate_context_embeddings)

			# used to compress the text a into a vector, like what cls does
  			# (query_num, candidate_num, vec_dim)
			compress_query = decoder['compress_query'][layer_index](lstm_hint_vector)
			mixed_query_layer = torch.cat((mixed_query_layer, compress_query), dim=1)

		# If this is instantiated as a cross-attention module, the keys
		# and values come from an encoder; the attention mask needs to be
		# such that the encoder's padding tokens are not attended to.
		is_cross_attention = encoder_hidden_states is not None

		if is_cross_attention and past_key_value is not None:
			# reuse k,v, cross_attentions
			key_layer = past_key_value[0]
			value_layer = past_key_value[1]
			attention_mask = encoder_attention_mask
		elif is_cross_attention:
			key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
			value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
			attention_mask = encoder_attention_mask
		elif past_key_value is not None:
			key_layer = self.transpose_for_scores(self.key(hidden_states))
			value_layer = self.transpose_for_scores(self.value(hidden_states))
			key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
			value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
		# only consider here!!!!!!!!!!!!!!!!
		else:
			# (batch size, head num, sequence len, head dim)
			key_layer = self.transpose_for_scores(self.key(hidden_states))
			value_layer = self.transpose_for_scores(self.value(hidden_states))

			# get key and value for candidate
			if enrich_candidate_by_question:
				candidate_key_layer = self.transpose_for_scores(
					decoder['candidate_key'][layer_index](candidate_context_embeddings))
				candidate_value_layer = self.transpose_for_scores(
					decoder['candidate_value'][layer_index](candidate_context_embeddings))

				# candidate can see almost all tokens
				candidate_key_layer = torch.cat([key_layer, candidate_key_layer], dim=-2)
				candidate_value_layer = torch.cat([value_layer, candidate_value_layer], dim=-2)

		query_layer = self.transpose_for_scores(mixed_query_layer)
		if enrich_candidate_by_question:
			candidate_query = self.transpose_for_scores(candidate_query)

		if self.is_decoder:
			# if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
			# Further calls to cross_attention layer can then reuse all cross-attention
			# key/value_states (first "if" case)
			# if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
			# all previous decoder key/value_states. Further calls to uni-directional self-attention
			# can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
			# if encoder bi-directional self-attention `past_key_value` is always `None`
			past_key_value = (key_layer, value_layer)

		# Take the dot product between "query" and "key" to get the raw attention scores.
		# (batch size, head num, sequence len, sequence len)
		attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
		if enrich_candidate_by_question:
			# (batch size, head num, candidate_num*context_num, sequence len + candidate_num*context_num)
			candidate_attention_scores = torch.matmul(candidate_query, candidate_key_layer.transpose(-1, -2))

		if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
			seq_length = hidden_states.size()[1]
			position_ids_l = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device).view(-1, 1)
			position_ids_r = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device).view(1, -1)
			distance = position_ids_l - position_ids_r
			positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
			positional_embedding = positional_embedding.to(dtype=query_layer.dtype)  # fp16 compatibility

			if self.position_embedding_type == "relative_key":
				relative_position_scores = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
				attention_scores = attention_scores + relative_position_scores
			elif self.position_embedding_type == "relative_key_query":
				relative_position_scores_query = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
				relative_position_scores_key = torch.einsum("bhrd,lrd->bhlr", key_layer, positional_embedding)
				attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key

		attention_scores = attention_scores / math.sqrt(self.attention_head_size)
		if enrich_candidate_by_question:
			# (batch size, head num, candidate_num*context_num, sequence len + candidate_num*context_num)
			candidate_attention_scores = candidate_attention_scores / math.sqrt(self.attention_head_size)

		# attention_mask.shape = (batch size, 1, 1, question_sequence_len)
		if attention_mask is not None:
			# Apply the attention mask is (precomputed for all layers in BertModel forward() function)
			attention_scores = attention_scores + attention_mask

			if enrich_candidate_by_question:
				# prepare mask for candidate
				candidate_context_num = candidate_context_embeddings.shape[1]

				# (1, 1, 1, candidate_context_num)
				candidate_mask_tensor = torch.tensor([-10000.0] * candidate_context_num,
													 device=attention_scores.device).unsqueeze(
					0).unsqueeze(0).unsqueeze(0)
				# (batch size, 1, 1, candidate_context_num)
				candidate_mask_tensor = candidate_mask_tensor.repeat(attention_mask.shape[0], 1, 1, 1)
				# (batch size, 1, 1, question_sequence_len + candidate_context_num)
				new_attention_mask = torch.cat([attention_mask, candidate_mask_tensor], dim=-1)
				# (batch size, head num, candidate_context_num, question_sequence_len + candidate_context_num)
				new_attention_mask = new_attention_mask.repeat(1, candidate_attention_scores.shape[1],
															   candidate_attention_scores.shape[2], 1)
				# (batch size, head num, candidate_num, context_num, question_sequence_len + candidate_context_num)
				previous_shape = new_attention_mask.size()

				new_attention_mask = new_attention_mask.reshape(previous_shape[0], previous_shape[1], candidate_num, -1,
																previous_shape[-1])
				context_num = new_attention_mask.shape[-2]
				all_sequence_len = new_attention_mask.shape[-1]

				# let candidate diagno to be 0.0
				for inner_index in range(1, candidate_num + 1):
					new_attention_mask[:, :, -inner_index, :,
					-inner_index * context_num: all_sequence_len - ((inner_index - 1) * context_num)] = 0.0
				new_attention_mask = new_attention_mask.reshape(*previous_shape)
				if candidate_num > 1:
					raise Exception("Please check here whether new_attention_mask is correct!")
				# now candidate can see itself and question, while question can only see itself
				candidate_attention_scores = candidate_attention_scores + new_attention_mask

		# Normalize the attention scores to probabilities.
		attention_probs = nn.functional.softmax(attention_scores, dim=-1)
		if enrich_candidate_by_question:
			candidate_attention_probs = nn.functional.softmax(candidate_attention_scores, dim=-1)

		# This is actually dropping out entire tokens to attend to, which might
		# seem a bit unusual, but is taken from the original Transformer paper.
		# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! This is removed by yyh.
		# attention_probs = self.dropout(attention_probs)

		# Mask heads if we want to
		if head_mask is not None:
			attention_probs = attention_probs * head_mask

		# (batch size, head num, sequence len, dim)
		context_layer = torch.matmul(attention_probs, value_layer)
		if enrich_candidate_by_question:
			candidate_context = torch.matmul(candidate_attention_probs, candidate_value_layer)

		# (batch size, sequence len, head num, dim)
		context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
		if enrich_candidate_by_question:
			candidate_context = candidate_context.permute(0, 2, 1, 3).contiguous()

		new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
		if enrich_candidate_by_question:
			new_candidate_context_shape = candidate_context.size()[:-2] + (self.all_head_size,)

		# (batch size, sequence len, dim)
		context_layer = context_layer.view(*new_context_layer_shape)
		if enrich_candidate_by_question:
			candidate_context = candidate_context.view(*new_candidate_context_shape)
			lstm_hidden_states = context_layer[:, -candidate_num:, :]

			context_layer = context_layer[:, :-candidate_num, :]
		else:
			candidate_context = None
			lstm_hidden_states = None

		outputs = (context_layer, candidate_context, lstm_hidden_states, attention_probs) if output_attentions else (
			context_layer, candidate_context, lstm_hidden_states,)

		if self.is_decoder:
			outputs = outputs + (past_key_value,)
		return outputs


class MyBertAttention(nn.Module):
	def __init__(self, config, position_embedding_type=None):
		super().__init__()
		self.self = MyBertSelfAttention(config, position_embedding_type=position_embedding_type)
		self.output = BertSelfOutput(config)
		self.pruned_heads = set()

	def prune_heads(self, heads):
		if len(heads) == 0:
			return
		heads, index = find_pruneable_heads_and_indices(
			heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
		)

		# Prune linear layers
		self.self.query = prune_linear_layer(self.self.query, index)
		self.self.key = prune_linear_layer(self.self.key, index)
		self.self.value = prune_linear_layer(self.self.value, index)
		self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

		# Update hyper params and store pruned heads
		self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
		self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
		self.pruned_heads = self.pruned_heads.union(heads)

	def forward(
			self,
			hidden_states,
			attention_mask=None,
			head_mask=None,
			encoder_hidden_states=None,
			encoder_attention_mask=None,
			past_key_value=None,
			output_attentions=False,
			enrich_candidate_by_question=False,
			# (query_num, candidate_context_num + candidate_num, dim)
			candidate_embeddings=None,
			decoder=None,
			layer_index=None,
			candidate_num=1,
	):
		if enrich_candidate_by_question:
      		# (query_num, candidate_context_num, dim)
			candidate_context_embeddings = candidate_embeddings[:, :-candidate_num, :]
			# (query_num, candidate_num, context_num, dim)
			lstm_hint_vector = candidate_context_embeddings.reshape(candidate_embeddings.shape[0], candidate_num, -1,
																	candidate_embeddings.shape[-1])
			# (query_num, candidate_num, vec_dim)
			lstm_hint_vector = decoder['candidate_composition_layer'](lstm_hint_vector).squeeze(-2)
			# lstm_hint_vector = torch.mean(lstm_hint_vector, dim=-2)
			# hidden states at last step
			lstm_hidden_states = candidate_embeddings[:, -candidate_num:, :]
		else:
			candidate_context_embeddings = None
			lstm_hidden_states = None
			lstm_hint_vector = None

		self_outputs = self.self(
			hidden_states,
			attention_mask,
			head_mask,
			encoder_hidden_states,
			encoder_attention_mask,
			past_key_value,
			output_attentions,
			enrich_candidate_by_question,
			# each candiate has only one compressing hint
			candidate_context_embeddings,
			lstm_hint_vector,
			decoder,
			layer_index,
			candidate_num,
		)

		# dense + dropout + layer_norm(res)
		attention_output = self.output(self_outputs[0], hidden_states)

		if enrich_candidate_by_question:
			# dense + dropout + layer_norm(res) + intermediate ...
			new_candidate_context_embeddings = decoder['layer_chunks'][layer_index](res_flag=True,
																					new_embeddings=self_outputs[1],
																					old_embeddings=candidate_context_embeddings)

			new_lstm_hidden_states = decoder['layer_chunks'][layer_index](res_flag=False,
																		  new_embeddings=self_outputs[2],
																		  old_embeddings=lstm_hidden_states,
																		  lstm_cell=decoder['LSTM'],
																		  lstm_hint_vector=lstm_hint_vector)

			candidate_embeddings = torch.cat((new_candidate_context_embeddings, new_lstm_hidden_states), dim=1)
		else:
			candidate_embeddings = None

		outputs = (attention_output, candidate_embeddings) + self_outputs[1:]  # add attentions if we output them

		return outputs


class MyBertLayer(nn.Module):
	def __init__(self, config):
		super().__init__()
		self.chunk_size_feed_forward = config.chunk_size_feed_forward
		self.seq_len_dim = 1
		self.attention = MyBertAttention(config)

		self.is_decoder = config.is_decoder
		self.add_cross_attention = config.add_cross_attention
		if self.add_cross_attention:
			if not self.is_decoder:
				raise ValueError(f"{self} should be used as a decoder model if cross attention is added")
			self.crossattention = BertAttention(config, position_embedding_type="absolute")

		self.intermediate = BertIntermediate(config)
		self.output = BertOutput(config)

	def forward(
			self,
			hidden_states,
			attention_mask=None,
			head_mask=None,
			encoder_hidden_states=None,
			encoder_attention_mask=None,
			past_key_value=None,
			output_attentions=False,
			enrich_candidate_by_question=False,
			candidate_embeddings=None,
			decoder=None,
			layer_index=None,
			candidate_num=1,
	):
		# decoder uni-directional self-attention cached key/values tuple is at positions 1,2
		self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
		self_attention_outputs = self.attention(
			hidden_states,
			attention_mask,
			head_mask,
			output_attentions=output_attentions,
			past_key_value=self_attn_past_key_value,
			enrich_candidate_by_question=enrich_candidate_by_question,
			candidate_embeddings=candidate_embeddings,
			decoder=decoder,
			layer_index=layer_index,
			candidate_num=candidate_num,
		)

		attention_output = self_attention_outputs[0]

		# feed attention_output forward---------------------------------------------------------------------
		# if decoder, the last output is tuple of self-attn cache
		if self.is_decoder:
			outputs = self_attention_outputs[1:-1]
			present_key_value = self_attention_outputs[-1]
		else:
			outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

		cross_attn_present_key_value = None
		if self.is_decoder and encoder_hidden_states is not None:
			if not hasattr(self, "crossattention"):
				raise ValueError(
					f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers by setting `config.add_cross_attention=True`"
				)

			# cross_attn cached key/values tuple is at positions 3,4 of past_key_value tuple
			cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
			cross_attention_outputs = self.crossattention(
				attention_output,
				attention_mask,
				head_mask,
				encoder_hidden_states,
				encoder_attention_mask,
				cross_attn_past_key_value,
				output_attentions,
			)
			attention_output = cross_attention_outputs[0]
			outputs = outputs + cross_attention_outputs[1:-1]  # add cross attentions if we output attention weights

			# add cross-attn cache to positions 3,4 of present_key_value tuple
			cross_attn_present_key_value = cross_attention_outputs[-1]
			present_key_value = present_key_value + cross_attn_present_key_value

		# self.chunk_size_feed_forward seems useless.
		if self.chunk_size_feed_forward > 0:
			raise Exception("chunk_size_feed_forward > 0, Here should be comprehended!")

		layer_output = apply_chunking_to_forward(
			self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
		)
		outputs = (layer_output,) + outputs

		# if decoder, return the attn key/values as the last output
		if self.is_decoder:
			outputs = outputs + (present_key_value,)

		# hidden states + attention
		return outputs

	def feed_forward_chunk(self, attention_output):
		# dense + activation
		intermediate_output = self.intermediate(attention_output)
		# dense + dropout + layer_norm(res)
		layer_output = self.output(intermediate_output, attention_output)
		return layer_output


class MyBertEncoder(nn.Module):
	def __init__(self, config):
		super().__init__()
		self.config = config
		self.layer = nn.ModuleList([MyBertLayer(config) for _ in range(config.num_hidden_layers)])
		self.gradient_checkpointing = False

	def forward(
			self,
			hidden_states,
			attention_mask=None,
			head_mask=None,
			encoder_hidden_states=None,
			encoder_attention_mask=None,
			past_key_values=None,
			use_cache=None,
			output_attentions=False,
			output_hidden_states=False,
			return_dict=True,
			enrich_candidate_by_question=False,
			candidate_embeddings=None,
			decoder=None,
			candidate_num=1,
	):

		all_hidden_states = () if output_hidden_states else None
		all_self_attentions = () if output_attentions else None
		all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

		next_decoder_cache = () if use_cache else None

		# decode related part
		# contain context vectors of text_B as well as lstm initial states
		this_layer_decoder_state = candidate_embeddings

		for i, layer_module in enumerate(self.layer):
			if output_hidden_states:
				all_hidden_states = all_hidden_states + (hidden_states,)

			layer_head_mask = head_mask[i] if head_mask is not None else None
			past_key_value = past_key_values[i] if past_key_values is not None else None

			if self.gradient_checkpointing and self.training:

				if use_cache:
					logger.warning(
						"`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
					)
					use_cache = False

				def create_custom_forward(module):
					def custom_forward(*inputs):
						return module(*inputs, past_key_value, output_attentions)

					return custom_forward

				layer_outputs = torch.utils.checkpoint.checkpoint(
					create_custom_forward(layer_module),
					hidden_states,
					attention_mask,
					layer_head_mask,
					encoder_hidden_states,
					encoder_attention_mask,
					enrich_candidate_by_question=enrich_candidate_by_question,
					candidate_embeddings=this_layer_decoder_state,
					decoder=decoder,
					layer_index=i,
					candidate_num=candidate_num,
				)
			else:
				layer_outputs = layer_module(
					hidden_states,
					attention_mask,
					layer_head_mask,
					encoder_hidden_states,
					encoder_attention_mask,
					past_key_value,
					output_attentions,
					enrich_candidate_by_question=enrich_candidate_by_question,
					candidate_embeddings=this_layer_decoder_state,
					decoder=decoder,
					layer_index=i,
					candidate_num=candidate_num,
				)
			# (Batch, Context_num * Candidate_num + Candidate_num, Dim)
			this_layer_decoder_state = layer_outputs[1]

			# (Batch, A_Sentence, Dim)
			hidden_states = layer_outputs[0]

			if use_cache:
				next_decoder_cache += (layer_outputs[-1],)
			if output_attentions:
				all_self_attentions = all_self_attentions + (layer_outputs[1],)
				if self.config.add_cross_attention:
					all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

		if output_hidden_states:
			all_hidden_states = all_hidden_states + (hidden_states,)

		return tuple(
			v
			for v in [
				hidden_states,
				next_decoder_cache,
				all_hidden_states,
				all_self_attentions,
				all_cross_attentions,
				this_layer_decoder_state,
			]
			# if v is not None
		)


class MyBertModel(BertPreTrainedModel):
	"""

	The model can behave as an encoder (with only self-attention) as well as a decoder, in which case a layer of
	cross-attention is added between the self-attention layers, following the architecture described in `Attention is
	all you need <https://arxiv.org/abs/1706.03762>`__ by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit,
	Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.

	To behave as an decoder the model needs to be initialized with the :obj:`is_decoder` argument of the configuration
	set to :obj:`True`. To be used in a Seq2Seq model, the model needs to initialized with both :obj:`is_decoder`
	argument and :obj:`add_cross_attention` set to :obj:`True`; an :obj:`encoder_hidden_states` is then expected as an
	input to the forward pass.
	"""

	def __init__(self, config, add_pooling_layer=True):
		super().__init__(config)

		self.config = config

		self.embeddings = BertEmbeddings(config)

		self.encoder = MyBertEncoder(config)

		self.pooler = BertPooler(config) if add_pooling_layer else None

		# Initialize weights and apply final processing
		self.post_init()

	def get_input_embeddings(self):
		return self.embeddings.word_embeddings

	def set_input_embeddings(self, value):
		self.embeddings.word_embeddings = value

	def _prune_heads(self, heads_to_prune):
		"""
		Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
		class PreTrainedModel
		"""
		for layer, heads in heads_to_prune.items():
			self.encoder.layer[layer].attention.prune_heads(heads)

	@add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
	@add_code_sample_docstrings(
		processor_class=_TOKENIZER_FOR_DOC,
		checkpoint=_CHECKPOINT_FOR_DOC,
		output_type=BaseModelOutputWithPoolingAndCrossAttentions,
		config_class=_CONFIG_FOR_DOC,
	)
	def forward(
			self,
			input_ids=None,
			attention_mask=None,
			token_type_ids=None,
			position_ids=None,
			head_mask=None,
			inputs_embeds=None,
			encoder_hidden_states=None,
			encoder_attention_mask=None,
			past_key_values=None,
			use_cache=None,
			output_attentions=None,
			output_hidden_states=None,
			return_dict=None,
			# added by yyh
			enrich_candidate_by_question=False,
			candidate_embeddings=None,
			decoder=None,
			candidate_num=1,
	):
		r"""
		encoder_hidden_states  (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
			Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
			the model is configured as a decoder.
		encoder_attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
			Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
			the cross-attention if the model is configured as a decoder. Mask values selected in ``[0, 1]``:

			- 1 for tokens that are **not masked**,
			- 0 for tokens that are **masked**.
		past_key_values (:obj:`tuple(tuple(torch.FloatTensor))` of length :obj:`config.n_layers` with each tuple having 4 tensors of shape :obj:`(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
			Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.

			If :obj:`past_key_values` are used, the user can optionally input only the last :obj:`decoder_input_ids`
			(those that don't have their past key value states given to this model) of shape :obj:`(batch_size, 1)`
			instead of all :obj:`decoder_input_ids` of shape :obj:`(batch_size, sequence_length)`.
		use_cache (:obj:`bool`, `optional`):
			If set to :obj:`True`, :obj:`past_key_values` key value states are returned and can be used to speed up
			decoding (see :obj:`past_key_values`).
		"""
		output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
		output_hidden_states = (
			output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
		)
		return_dict = return_dict if return_dict is not None else self.config.use_return_dict

		if self.config.is_decoder:
			use_cache = use_cache if use_cache is not None else self.config.use_cache
		else:
			use_cache = False

		if input_ids is not None and inputs_embeds is not None:
			raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
		elif input_ids is not None:
			input_shape = input_ids.size()
		elif inputs_embeds is not None:
			input_shape = inputs_embeds.size()[:-1]
		else:
			raise ValueError("You have to specify either input_ids or inputs_embeds")

		batch_size, seq_length = input_shape
		device = input_ids.device if input_ids is not None else inputs_embeds.device

		# past_key_values_length
		past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

		if attention_mask is None:
			attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)

		if token_type_ids is None:
			if hasattr(self.embeddings, "token_type_ids"):
				buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
				buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
				token_type_ids = buffered_token_type_ids_expanded
			else:
				token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

		# We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
		# ourselves in which case we just need to make it broadcastable to all heads.
		# input_shape = (batch size, seq length)
		# extended_attention_mask: 0 denotes useful information, --10000.0 denotes tokens should be ingnored
		extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)

		# If a 2D or 3D attention mask is provided for the cross-attention
		# we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
		if self.config.is_decoder and encoder_hidden_states is not None:
			encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
			encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
			if encoder_attention_mask is None:
				encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
			encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
		else:
			encoder_extended_attention_mask = None

		# Prepare head mask if needed
		# 1.0 in head_mask indicate we keep the head
		# attention_probs has shape bsz x n_heads x N x N
		# input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
		# and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
		head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

		# input input_ids or input_embeds, plus them with token type embeddings and position embeddings
		embedding_output = self.embeddings(
			input_ids=input_ids,
			position_ids=position_ids,
			token_type_ids=token_type_ids,
			inputs_embeds=inputs_embeds,
			past_key_values_length=past_key_values_length,
		)

		# all compute in encoder
		encoder_outputs = self.encoder(
			embedding_output,
			attention_mask=extended_attention_mask,
			head_mask=head_mask,
			encoder_hidden_states=encoder_hidden_states,
			encoder_attention_mask=encoder_extended_attention_mask,
			past_key_values=past_key_values,
			use_cache=use_cache,
			output_attentions=output_attentions,
			output_hidden_states=output_hidden_states,
			return_dict=False,
			enrich_candidate_by_question=enrich_candidate_by_question,
			candidate_embeddings=candidate_embeddings,
			decoder=decoder,
			candidate_num=candidate_num,
		)
		sequence_output = encoder_outputs[0]
		pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

		return {'last_hidden_state': encoder_outputs[0], 'pooler_output': pooled_output,
				'past_key_values': encoder_outputs[1], 'hidden_states': encoder_outputs[2],
				'attentions': encoder_outputs[3], 'cross_attentions': encoder_outputs[4],
				'decoder_output': encoder_outputs[5]}


# used by decoder. support copy parameters from encoder
class DecoderLayerChunk(nn.Module):
	def __init__(self, config):
		super().__init__()
		self.chunk_size_feed_forward = config.chunk_size_feed_forward
		self.seq_len_dim = 1

		self.attention = MyBertSelfOutput(config)
		self.intermediate = BertIntermediate(config)
		self.output = BertOutput(config)

	def forward(
			self,
			# lstm_flag = not res_flag
			res_flag,
			new_embeddings=None,
			old_embeddings=None,
			lstm_cell=None,
			lstm_hint_vector=None,
	):
		# dense + dropout + layer_norm(res)
		attention_output = self.attention(new_embeddings, old_embeddings, res_flag, lstm_cell, lstm_hint_vector)
		# dense + activation
		intermediate_output = self.intermediate(attention_output)
		# dense + dropout + layer_norm(res)
		layer_output = self.output(intermediate_output, attention_output)

		return layer_output


class MyBertSelfOutput(nn.Module):
	def __init__(self, config):
		super().__init__()
		self.dense = nn.Linear(config.hidden_size, config.hidden_size)
		self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
		self.dropout = nn.Dropout(config.hidden_dropout_prob)

	def forward(self, hidden_states, input_tensor, res_flag, lstm_cell, lstm_hint_vector):
		hidden_states = self.dense(hidden_states)
		hidden_states = self.dropout(hidden_states)
		if res_flag:
			hidden_states = self.LayerNorm(hidden_states + input_tensor)
		else:
			hidden_states = self.LayerNorm(lstm_cell(hidden_states, input_tensor, lstm_hint_vector))

		return hidden_states
