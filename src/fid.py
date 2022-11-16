# coding=utf-8
# Copyright 2018 Mesh TensorFlow authors, T5 Authors and HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import types

import torch
from torch import nn
from transformers.utils import logging

from src.modeling_t5 import T5ForConditionalGeneration, T5Stack

logger = logging.get_logger(__name__)


class FiDStack(T5Stack):
    def __init__(self, config, embed_tokens=None):
        super().__init__(config, embed_tokens=embed_tokens)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        inputs_embeds=None,
        head_mask=None,
        cross_attn_head_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        if not self.is_decoder:
            input_ids = input_ids.view(input_ids.size(0) * self.config.n_context, -1)
            attention_mask = attention_mask.view(attention_mask.size(0) * self.config.n_context, -1)

        output = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            inputs_embeds=inputs_embeds,
            head_mask=head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if not self.is_decoder:
            if not return_dict:
                last_hidden_states = output[0]
                last_hidden_state = last_hidden_states.view(self.config.bsz, -1, last_hidden_states.size(-1))
                output = tuple(
                    last_hidden_state,
                    *output[1:],
                )
            else:
                last_hidden_state = output.last_hidden_state
                output.last_hidden_state = last_hidden_state.view(self.config.bsz, -1, last_hidden_state.size(-1))

        return output


class FiD(T5ForConditionalGeneration):
    _keys_to_ignore_on_load_missing = [
        r"encoder\.embed_tokens\.weight",
        r"decoder\.embed_tokens\.weight",
        r"lm_head\.weight",
    ]
    _keys_to_ignore_on_load_unexpected = [
        r"decoder\.block\.0\.layer\.1\.EncDecAttention\.relative_attention_bias\.weight",
    ]

    def __init__(self, config):
        super().__init__(config)
        self.model_dim = config.d_model

        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = FiDStack(encoder_config, self.shared)

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = FiDStack(decoder_config, self.shared)

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

        # Model parallel
        self.model_parallel = False
        self.device_map = None

    def set_checkpoint(self, use_checkpoint):
        """
        Enable or disable checkpointing in the encoder.
        See https://pytorch.org/docs/stable/checkpoint.html
        """
        for mod in self.encoder.encoder.block:
            mod.use_checkpoint = use_checkpoint

    def reset_score_storage(self):
        """
        Reset score storage, only used when cross-attention scores are saved
        to train a retriever.
        """
        for mod in self.decoder.block:
            mod.layer[1].EncDecAttention.score_storage = None
            mod.layer[1].EncDecAttention.normalized_score_storage = None
            mod.layer[1].EncDecAttention.prob_storage = None

    @torch.no_grad()
    def get_crossattention_scores(self, n_passages, mask, labels, ids, mode="all", mask_query=None):
        """
        Cross-attention scores are aggregated to obtain a single scalar per
        passage. This scalar can be seen as a similarity score between the
        question and the input passage. It is obtained by averaging the
        cross-attention scores obtained on the first decoded token over heads,
        layers, and tokens of the input passage.

        More details in Distilling Knowledge from Reader to Retriever:
        https://arxiv.org/abs/2012.04584.
        """
        scores, norms, probs = [], [], []
        for mod in self.decoder.block:
            scores.append(mod.layer[1].EncDecAttention.score_storage)
            norms.append(mod.layer[1].EncDecAttention.normalized_score_storage)
            probs.append(mod.layer[1].EncDecAttention.prob_storage)
        scores = torch.stack(scores)
        norms = torch.stack(norms)
        probs = torch.stack(probs)

        output = {}
        if "scores" in mode or "all" in mode:
            self.aggregate_value(scores, mask, labels, n_passages, ids, mask_query, output, prefix="scores")
        if "probs" in mode or "all" in mode:
            self.aggregate_value(probs, mask, labels, n_passages, ids, mask_query, output, prefix="probs")
        if "norms" in mode or "all" in mode:
            self.aggregate_value(norms, mask, labels, n_passages, ids, mask_query, output, prefix="norms")
        return output

    def aggregate_value(self, scores, mask, labels, n_passages, ids, mask_query=None, output={}, prefix=""):
        n_layers, bsz, n_tokens, total_tokens = scores.size()

        ids = ids.view(bsz, n_passages, -1)
        scores = scores.view(n_layers, bsz, n_tokens, n_passages, -1)
        mask = mask.view(bsz, n_passages, -1)
        scores = scores.masked_fill(~mask[None, :, None], 0.0)

        ntokens_sum = 256 * n_layers * (~(labels == -100)).sum(dim=[1])[:, None]
        ntokens_wquery = mask.sum(dim=[2]) * n_layers * (~(labels == -100)).sum(dim=[1])[:, None]
        ntokens_first = mask.sum(dim=[2]) * n_layers

        # Compute scores based on topk scores
        scores = scores.sum(dim=[0])
        for k in [5, 10, 20]:
            topkscores = self.get_topk_score(k, scores, mask, labels, n_layers)
            output[f"{prefix}top{k}"] = topkscores

        scores = scores.masked_fill((labels == -100)[:, :, None, None], 0.0)
        scores_wquery = scores.sum(dim=[1, 3])

        scores_wquery_sepmask = scores.masked_fill(~(ids == 1)[:, None], 0).sum(dim=[1, 3])
        output[f"{prefix}nosep"] = scores_wquery_sepmask / ntokens_sum

        output[f"{prefix}first"] = scores[:, 0].sum(dim=[2]) / ntokens_first
        output[f"{prefix}sum"] = scores_wquery / ntokens_sum
        output[f"{prefix}avg"] = scores_wquery / ntokens_wquery

        scores_woquery = None
        # Compute scores based on scores without query
        if not mask_query is None:
            output[f"{prefix}woquery"] = self.get_woquery_score(scores, mask_query, mask, labels, n_layers)

        return output

    def get_topk_score(self, topk, scores, mask, labels, n_layers):
        topkscores = torch.topk(scores, k=topk, dim=-1)[0].sum(dim=[3])
        topkscores = topkscores.masked_fill((labels == -100)[:, :, None], 0.0)
        ntokens_top = n_layers * (~(labels == -100)).sum(dim=[1])[:, None]
        topkscores = topkscores.sum(dim=1) / (topk * ntokens_top)
        return topkscores

    def get_woquery_score(self, scores, mask_query, mask, labels, n_layers):
        if scores.size(-1) > mask_query.size(-1):
            zero_padding = torch.zeros(
                [mask_query.size(0), scores.size(-1) - mask_query.size(-1)], device=mask_query.device, dtype=torch.bool
            )
            mask_query = torch.cat([mask_query, zero_padding], dim=-1)
        mask_query = mask * (~mask_query[:, None])
        scores_woquery = scores.masked_fill(~mask_query[:, None], 0.0)
        # ntokens_woquery = mask_query.sum(dim=[2]) * n_layers * (~(labels==-100)).sum(dim=[1])[:, None]
        ntokens_woquery = 256 * n_layers * (~(labels == -100)).sum(dim=[1])[:, None]
        scores_woquery = scores_woquery.sum(dim=[1, 3])
        return scores_woquery / ntokens_woquery

    def overwrite_forward_crossattention(self):
        """
        Replace cross-attention forward function, only used to save
        cross-attention scores.
        """
        for mod in self.decoder.block:
            xattn = mod.layer[1].EncDecAttention
            xattn.forward = types.MethodType(cross_attention_forward, xattn)

    def create_crossattention_storage(self):
        for mod in self.decoder.block:
            xattn = mod.layer[1].EncDecAttention
            xattn.score_storage = None
            xattn.normalized_score_storage = None
            xattn.prob_storage = None


def cross_attention_forward(
    self,
    hidden_states,
    mask=None,
    key_value_states=None,
    position_bias=None,
    past_key_value=None,
    layer_head_mask=None,
    query_length=None,
    use_cache=False,
    output_attentions=False,
):
    """
    Self-attention (if key_value_states is None) or attention over source sentence (provided by key_value_states).
    """
    # Input is (batch_size, seq_length, dim)
    # Mask is (batch_size, key_length) (non-causal) or (batch_size, key_length, key_length)
    # past_key_value[0] is (batch_size, n_heads, q_len - 1, dim_per_head)
    batch_size, seq_length = hidden_states.shape[:2]

    real_seq_length = seq_length

    if past_key_value is not None:
        assert (
            len(past_key_value) == 2
        ), f"past_key_value should have 2 past states: keys and values. Got { len(past_key_value)} past states"
        real_seq_length += past_key_value[0].shape[2] if query_length is None else query_length

    key_length = real_seq_length if key_value_states is None else key_value_states.shape[1]

    def shape(states):
        """projection"""
        return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)

    def unshape(states):
        """reshape"""
        return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)

    def project(hidden_states, proj_layer, key_value_states, past_key_value):
        """projects hidden states correctly to key/query states"""
        if key_value_states is None:
            # self-attn
            # (batch_size, n_heads, seq_length, dim_per_head)
            hidden_states = shape(proj_layer(hidden_states))
        elif past_key_value is None:
            # cross-attn
            # (batch_size, n_heads, seq_length, dim_per_head)
            hidden_states = shape(proj_layer(key_value_states))

        if past_key_value is not None:
            if key_value_states is None:
                # self-attn
                # (batch_size, n_heads, key_length, dim_per_head)
                hidden_states = torch.cat([past_key_value, hidden_states], dim=2)
            else:
                # cross-attn
                hidden_states = past_key_value
        return hidden_states

    # get query states
    query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)

    # get key/value states
    key_states = project(
        hidden_states, self.k, key_value_states, past_key_value[0] if past_key_value is not None else None
    )
    value_states = project(
        hidden_states, self.v, key_value_states, past_key_value[1] if past_key_value is not None else None
    )

    # compute scores
    scores = torch.matmul(
        query_states, key_states.transpose(3, 2)
    )  # equivalent of torch.einsum("bnqd,bnkd->bnqk", query_states, key_states), compatible with onnx op>9

    if position_bias is None:
        if not self.has_relative_attention_bias:
            position_bias = torch.zeros(
                (1, self.n_heads, real_seq_length, key_length), device=scores.device, dtype=scores.dtype
            )
            if self.gradient_checkpointing and self.training:
                position_bias.requires_grad = True
        else:
            position_bias = self.compute_bias(real_seq_length, key_length)

        # if key and values are already calculated
        # we want only the last query position bias
        if past_key_value is not None:
            position_bias = position_bias[:, :, -hidden_states.size(1) :, :]

        if mask is not None:
            position_bias = position_bias + mask  # (batch_size, n_heads, seq_length, key_length)

    scores += position_bias

    attn_weights = nn.functional.softmax(scores.float(), dim=-1)  # .type_as(scores)

    if hasattr(self, "score_storage"):
        with torch.no_grad():
            self.score_storage = scores.detach().mean(dim=1)
            self.prob_storage = attn_weights.detach().mean(dim=1)
            self.normalized_score_storage = (
                (torch.norm(value_states.float(), dim=-1)[:, :, None] * attn_weights).detach().mean(dim=1)
            )

    attn_weights = nn.functional.dropout(attn_weights.type_as(scores), p=self.dropout, training=self.training)

    # Mask heads if we want to
    if layer_head_mask is not None:
        attn_weights = attn_weights * layer_head_mask

    attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    attn_output = self.o(attn_output)

    present_key_value_state = (key_states, value_states) if (self.is_decoder and use_cache) else None
    outputs = (attn_output,) + (present_key_value_state,) + (position_bias,)

    if output_attentions:
        outputs = outputs + (attn_weights,)
    return outputs
