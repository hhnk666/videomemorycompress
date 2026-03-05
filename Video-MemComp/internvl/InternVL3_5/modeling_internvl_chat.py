# --------------------------------------------------------
# InternVL
# Copyright (c) 2024 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import torch.nn.functional as F
from transformers.cache_utils import Cache, DynamicCache
import warnings
from typing import List, Optional, Tuple, Union

import torch.utils.checkpoint
import transformers
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import GenerationConfig
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging
from transformers import LlamaForCausalLM, Qwen2ForCausalLM, Qwen3ForCausalLM, Qwen3MoeForCausalLM

from .configuration_internvl_chat import InternVLChatConfig
from .conversation import get_conv_template
from .modeling_intern_vit import InternVisionModel, has_flash_attn

logger = logging.get_logger(__name__)


def version_cmp(v1, v2, op='eq'):
    import operator

    from packaging import version
    op_func = getattr(operator, op)
    return op_func(version.parse(v1), version.parse(v2))


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class InternVLChatModel(PreTrainedModel):
    config_class = InternVLChatConfig
    main_input_name = 'pixel_values'
    base_model_prefix = 'language_model'
    _supports_flash_attn_2 = True
    supports_gradient_checkpointing = True
    _no_split_modules = [
        "InternVisionModel",
        "Qwen3DecoderLayer",
    ]

    # support transformers 4.51.+
    _tp_plan = ''

    def __init__(self, config: InternVLChatConfig, vision_model=None, language_model=None, use_flash_attn=True):
        super().__init__(config)

        assert version_cmp(transformers.__version__, '4.37.0', 'ge')
        image_size = config.force_image_size or config.vision_config.image_size
        patch_size = config.vision_config.patch_size
        self.patch_size = patch_size
        self.select_layer = config.select_layer
        self.template = config.template
        self.num_image_token = int(
            (image_size // patch_size) ** 2 * (config.downsample_ratio ** 2))
        self.downsample_ratio = config.downsample_ratio
        self.ps_version = config.ps_version
        use_flash_attn = use_flash_attn if has_flash_attn else False
        config.vision_config.use_flash_attn = True if use_flash_attn else False
        config.llm_config._attn_implementation = 'flash_attention_2' if use_flash_attn else 'eager'

        logger.info(f'num_image_token: {self.num_image_token}')
        logger.info(f'ps_version: {self.ps_version}')
        if vision_model is not None:
            self.vision_model = vision_model
        else:
            self.vision_model = InternVisionModel(config.vision_config)
        if language_model is not None:
            self.language_model = language_model
        else:
            architecture: str = config.llm_config.architectures[0]
            if architecture == 'LlamaForCausalLM':
                self.language_model = LlamaForCausalLM(config.llm_config)
            elif architecture == 'Qwen2ForCausalLM':
                self.language_model = Qwen2ForCausalLM(config.llm_config)
            elif architecture == 'Qwen3MoeForCausalLM':
                self.language_model = Qwen3MoeForCausalLM(config.llm_config)
            elif architecture == 'Qwen3ForCausalLM':
                self.language_model = Qwen3ForCausalLM(config.llm_config)
            else:
                raise NotImplementedError(
                    f'{architecture} is not implemented.')

        vit_hidden_size = config.vision_config.hidden_size
        llm_hidden_size = config.llm_config.hidden_size

        self.mlp1 = nn.Sequential(
            nn.LayerNorm(vit_hidden_size *
                         int(1 / self.downsample_ratio) ** 2),
            nn.Linear(vit_hidden_size * int(1 / self.downsample_ratio)
                      ** 2, llm_hidden_size),
            nn.GELU(),
            nn.Linear(llm_hidden_size, llm_hidden_size)
        )

        self.img_context_token_id = None
        self.conv_template = get_conv_template(self.template)
        self.system_message = self.conv_template.system_message
        # New
        self.streaming_state = {
            "past_key_values": None,
            "fusion_counts": None,
            "last_frame_hidden_states": None,
            "logical_seq_len": 0,
            "logical_positions": 0,
        }

        self.inter_frame_threshold = 0.75  # inter frame threshold
        self.intra_frame_threshold = 0.75  # intra frame threshold
        self.kv_cache_token_limit = 6000  # KV Size per layer
        self.kv_cache_token_limit_per_layer = self.kv_cache_token_limit
        # end of fix 
        self.gsim_threshold = 0.5  # κ (kappa) for GSim gating
        self.gsim_temperature = 150  # τ (tau) for GSim softmax

    def _process_visual_chunk(self, frame_embeds: torch.Tensor, original_indices: torch.Tensor) -> dict:
        """
        Processes a single visual chunk: performs inter-frame dropping, then intra-frame GSim fusion.
        Returns a dictionary with processed embeddings, pos_ids, and fusion_counts.
        """
        # 1. Inter-frame dropping
        last_hs = self.streaming_state.get("last_frame_hidden_states")
        inter_keep_mask = torch.ones(
            frame_embeds.shape[1], dtype=torch.bool, device=frame_embeds.device)
        if last_hs is not None and last_hs.shape == frame_embeds.shape:
            similarity = F.cosine_similarity(last_hs.squeeze(
                0).detach(), frame_embeds.squeeze(0).detach(), dim=-1)
            inter_keep_mask = similarity <= self.inter_frame_threshold

        # Update last_frame_hidden_states for the next iteration
        self.streaming_state["last_frame_hidden_states"] = frame_embeds.clone()

        # Apply inter-frame dropping
        surviving_embeds_inter = frame_embeds[:, inter_keep_mask, :]
        original_indices_after_inter = original_indices[inter_keep_mask]

        if surviving_embeds_inter.shape[1] < 2:
            # Not enough tokens for fusion, return as is
            return {
                "embeds": surviving_embeds_inter,
                "indices": original_indices_after_inter,
                "fusion_counts": torch.ones(surviving_embeds_inter.shape[1], device=surviving_embeds_inter.device)
            }

        # 2. Intra-frame merging (GSim)
        # First, use bipartite merge to decide *what* to drop/merge
        merged_hs_pre_gsim, dropped_indices_relative = self._bipartite_merge_and_replace(
            surviving_embeds_inter.squeeze(0), self.intra_frame_threshold)

        if dropped_indices_relative.numel() == 0:
            # No fusion happened, just return the result after inter-frame dropping
            return {
                "embeds": surviving_embeds_inter,
                "indices": original_indices_after_inter,
                "fusion_counts": torch.ones(surviving_embeds_inter.shape[1], device=surviving_embeds_inter.device)
            }

        # Identify survivors
        survivor_mask = torch.ones(
            surviving_embeds_inter.shape[1], dtype=torch.bool, device=frame_embeds.device)
        survivor_mask[dropped_indices_relative] = False
        survivor_absolute_indices = original_indices_after_inter[survivor_mask]

        # Get embeddings for GSim calculation
        survivor_tokens_pre_gsim = merged_hs_pre_gsim[survivor_mask]
        original_tokens_in_chunk = surviving_embeds_inter.squeeze(
            0)  # Tokens before this fusion step

        # GSim calculation
        survivor_norm = F.normalize(survivor_tokens_pre_gsim, p=2, dim=-1)
        original_norm = F.normalize(original_tokens_in_chunk, p=2, dim=-1)
        cosine_sim = torch.matmul(survivor_norm, original_norm.t())
        gsim_matrix = torch.where(
            cosine_sim >= self.gsim_threshold, cosine_sim, 0.0)

        m_matrix = F.softmax(gsim_matrix * self.gsim_temperature, dim=0)
        s_factors = torch.sum(m_matrix, dim=1)

        w_matrix = m_matrix / (s_factors.unsqueeze(1) + 1e-8)
        transformed_tokens = torch.matmul(
            w_matrix, original_tokens_in_chunk).unsqueeze(0)

        return {
            "embeds": transformed_tokens,
            "indices": survivor_absolute_indices,
            "fusion_counts": s_factors.float()
        }

    def _create_streaming_attention_mask(self, q_len: int, past_kv_len: int, combined_fusion_counts: torch.Tensor, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Creates a 4D attention mask/bias for streaming, incorporating causal masking and fusion counts."""
        total_kv_len = past_kv_len + q_len
        attn_bias = torch.zeros(1, 1, q_len, total_kv_len,
                                device=device, dtype=dtype)

        # Apply causal mask for the query block itself
        if q_len > 1:
            causal_mask = torch.triu(torch.ones(
                q_len, q_len, dtype=torch.bool, device=device), diagonal=1)
            attn_bias[:, :, :, past_kv_len:].masked_fill_(
                causal_mask, -torch.inf)

        # Apply fusion count bias
        if combined_fusion_counts is not None and combined_fusion_counts.numel() > 0:
            bias_len = min(total_kv_len, combined_fusion_counts.shape[0])
            fusion_bias = torch.log1p(
                combined_fusion_counts[:bias_len] - 1).to(dtype)
            attn_bias[:, :, :, :bias_len] += fusion_bias[None, None, None, :]

        return attn_bias

    def reset_streaming_state(self):
        """
        reset the state of streaming
        """
        print("--- Resetting model streaming state for new sample ---")
        self.streaming_state = {
            "past_key_values": None,
            "fusion_counts": None,
            "last_frame_hidden_states": None,
            "logical_seq_len": 0,
        }

    def _bipartite_merge_and_replace(
        self,
        frame_hs: torch.Tensor,
        in_frame_threshold: float,
        max_iterations: int = 10
    ) -> Tuple[torch.Tensor, torch.LongTensor]:
        """
        Recursively performs token merging as described in https://arxiv.org/pdf/2210.09461.
        """
        if frame_hs.shape[0] < 2:
            return frame_hs, torch.tensor([], dtype=torch.long, device=frame_hs.device)

        current_frame_hs = frame_hs
        all_dropped_indices = []
        original_to_current_mapping = torch.arange(
            frame_hs.shape[0], device=frame_hs.device)

        for iteration in range(max_iterations):
            if current_frame_hs.shape[0] < 2:
                break

            updated_frame_hs, dropped_indices_current = self._bipartite_merge_and_replace_single(
                current_frame_hs, in_frame_threshold
            )

            if dropped_indices_current.numel() == 0:
                break

            dropped_indices_original = original_to_current_mapping[dropped_indices_current]
            all_dropped_indices.append(dropped_indices_original)

            keep_mask = torch.ones(
                current_frame_hs.shape[0], dtype=torch.bool, device=frame_hs.device)
            keep_mask[dropped_indices_current] = False
            original_to_current_mapping = original_to_current_mapping[keep_mask]
            current_frame_hs = updated_frame_hs[keep_mask]

        if all_dropped_indices:
            final_dropped_indices = torch.cat(all_dropped_indices, dim=0)
        else:
            final_dropped_indices = torch.tensor(
                [], dtype=torch.long, device=frame_hs.device)

        final_frame_hs = frame_hs.clone()
        final_keep_mask = torch.ones(
            frame_hs.shape[0], dtype=torch.bool, device=frame_hs.device)
        if final_dropped_indices.numel() > 0:
            final_keep_mask[final_dropped_indices] = False
        final_frame_hs[original_to_current_mapping] = current_frame_hs

        return final_frame_hs, final_dropped_indices

    @torch.no_grad()
    def _proxy_prefill_and_get_queries(self, past_key_values: Cache):
        """
        [Refactored version] Generates proxy queries for importance scoring.
        This version uses safe APIs to access the KV Cache and dynamically generates 
        correct Attention Masks and positional encodings for each layer, precisely 
        accommodating complex scenarios where layers have varying lengths.
        """
        device = self.get_input_embeddings().weight.device
        proxy_ids = torch.tensor([[151645, 151644, 77091, 198]], device=device)
        proxy_embeds = self.get_input_embeddings()(proxy_ids)
        bsz, q_len, _ = proxy_embeds.shape

        # 1. Create a temporary, safe copy of pkv for simulating forward propagation
        pkv_clone = DynamicCache()
        if past_key_values is not None and past_key_values.get_seq_length(0) > 0:
            # Iterate through all layers to safely populate the copy
            for layer_idx in range(len(self.language_model.layers)):
                k, v = past_key_values[layer_idx]
                pkv_clone.update(k.clone(), v.clone(), layer_idx)

        proxy_queries_per_layer = []
        hidden_states = proxy_embeds

        # 2. Simulate forward propagation layer by layer
        for layer_idx, layer in enumerate(self.language_model.layers):
            layernorm_output = layer.input_layernorm(hidden_states)

            # 3. Dynamically compute positional information (RoPE) for the current layer
            #    Key point: each layer has its own independent past_len
            current_past_len = pkv_clone.get_seq_length(layer_idx)
            proxy_pos_ids = torch.arange(
                current_past_len, current_past_len + q_len, device=device
            ).view(1, -1)

            # RoPE computation logic (assuming rotary_emb and apply_rotary_pos_emb are correct)
            cos, sin = self.language_model.rotary_emb(
                proxy_embeds, proxy_pos_ids)
            position_embeddings = (cos, sin)

            # 4. Extract rotated query vectors (q_rot) for scoring
            q_vectors = layer.self_attn.q_proj(layernorm_output)
            q_vectors = q_vectors.view(
                bsz, q_len, layer.self_attn.config.num_attention_heads, layer.self_attn.head_dim
            ).transpose(1, 2)
            q_rot, _ = apply_rotary_pos_emb(q_vectors, q_vectors, cos, sin)
            proxy_queries_per_layer.append(q_rot)

            # 5. Dynamically generate the correct Attention Mask for the current layer
            total_kv_len = current_past_len + q_len
            attention_mask = torch.zeros(
                (bsz, 1, q_len, total_kv_len), dtype=hidden_states.dtype, device=device
            )
            if q_len > 1:
                causal_mask_bool = torch.triu(
                    torch.ones((q_len, q_len), device=device,
                               dtype=torch.bool),
                    diagonal=1
                )
                attention_mask[..., :, current_past_len:].masked_fill_(
                    causal_mask_bool, torch.finfo(hidden_states.dtype).min
                )

            # 6. Simulate complete attention computation to update hidden_states for the next layer
            #    Note: pkv_clone will be internally updated during this call
            attn_output, _ = layer.self_attn(
                hidden_states=layernorm_output,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                past_key_values=pkv_clone,
                use_cache=True,
                cache_position=proxy_pos_ids  
            )

            hidden_states = hidden_states + attn_output
            hidden_states = hidden_states + \
                layer.mlp(layer.post_attention_layernorm(hidden_states))

        return proxy_queries_per_layer

    def _bipartite_merge_and_replace_single(
        self,
        frame_hs: torch.Tensor,
        in_frame_threshold: float
    ) -> Tuple[torch.Tensor, torch.LongTensor]:
        """
        Performs a single-step bipartite graph merge: replaces target positions with 
        merged vectors and returns the indices of tokens that were merged away.
        This function precisely controls gradient flow: the decision-making process 
        is gradient-free, while the merging operation preserves gradients.
        """
        if frame_hs.shape[0] < 2:
            return frame_hs, torch.tensor([], dtype=torch.long, device=frame_hs.device)

        frame_hs_detached = frame_hs.detach()
        metric_norm = frame_hs_detached / \
            frame_hs_detached.norm(dim=-1, keepdim=True)
        metric_norm = metric_norm.unsqueeze(0)
        a, b = metric_norm[:, ::2, :], metric_norm[:, 1::2, :]

        if a.shape[1] == 0 or b.shape[1] == 0:
            return frame_hs, torch.tensor([], dtype=torch.long, device=frame_hs.device)

        scores = a @ b.transpose(-1, -2)
        node_max, node_idx = scores.max(dim=-1)
        edge_idx = (node_max > in_frame_threshold).squeeze(0)

        if not edge_idx.any():
            return frame_hs, torch.tensor([], dtype=torch.long, device=frame_hs.device)

        a_indices_relative = torch.arange(a.shape[1], device=frame_hs.device)
        a_dropped_indices_relative = a_indices_relative[edge_idx]
        dropped_indices_relative = a_dropped_indices_relative * 2


        updated_frame_hs = frame_hs.clone()
        a_orig, b_orig = frame_hs[::2, :], frame_hs[1::2, :]
        merged_sum = torch.zeros_like(b_orig, dtype=frame_hs.dtype)
        counts = torch.zeros(
            b_orig.shape[0], dtype=torch.int64, device=frame_hs.device)
        b_dest_map = node_idx.squeeze(0)[edge_idx]
        merged_sum.index_add_(0, b_dest_map, a_orig[edge_idx])
        counts.index_add_(0, b_dest_map, torch.ones_like(
            b_dest_map, dtype=torch.int64))
        b_consumed_indices_relative = torch.unique(b_dest_map)
        merged_sum.index_add_(0, b_consumed_indices_relative,
                              b_orig[b_consumed_indices_relative])
        counts.index_add_(0, b_consumed_indices_relative, torch.ones_like(
            b_consumed_indices_relative, dtype=torch.int64))
        final_merged_vectors = merged_sum[b_consumed_indices_relative] / \
            counts[b_consumed_indices_relative].unsqueeze(
                -1).to(frame_hs.dtype)


        original_b_consumed_indices = b_consumed_indices_relative * 2 + 1
        updated_frame_hs[original_b_consumed_indices] = final_merged_vectors

        return updated_frame_hs, dropped_indices_relative

    def forward(
            self,
            pixel_values: torch.FloatTensor,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            image_flags: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        image_flags = image_flags.squeeze(-1)
        input_embeds = self.language_model.get_input_embeddings()(input_ids).clone()

        vit_embeds = self.extract_feature(pixel_values)
        vit_embeds = vit_embeds[image_flags == 1]
        vit_batch_size = pixel_values.shape[0]

        B, N, C = input_embeds.shape
        input_embeds = input_embeds.reshape(B * N, C)

        # if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
        #     print(f'dynamic ViT batch size: {vit_batch_size}, images per sample: {vit_batch_size / B}, dynamic token length: {N}')

        input_ids = input_ids.reshape(B * N)
        selected = (input_ids == self.img_context_token_id)
        try:
            input_embeds[selected] = input_embeds[selected] * \
                0.0 + vit_embeds.reshape(-1, C)
        except Exception as e:
            vit_embeds = vit_embeds.reshape(-1, C)
            print(f'warning: {e}, input_embeds[selected].shape={input_embeds[selected].shape}, '
                  f'vit_embeds.shape={vit_embeds.shape}')
            n_token = min(selected.sum(), vit_embeds.size(0))
            input_embeds[selected][:n_token] = input_embeds[selected][:n_token] * \
                0.0 + vit_embeds[:n_token]

        input_embeds = input_embeds.reshape(B, N, C)

        outputs = self.language_model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        logits = outputs.logits

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1,
                                             self.language_model.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def pixel_shuffle(self, x, scale_factor=0.5):
        n, w, h, c = x.size()
        # N, W, H, C --> N, W, H * scale, C // scale
        x = x.view(n, w, int(h * scale_factor), int(c / scale_factor))
        # N, W, H * scale, C // scale --> N, H * scale, W, C // scale
        x = x.permute(0, 2, 1, 3).contiguous()
        # N, H * scale, W, C // scale --> N, H * scale, W * scale, C // (scale ** 2)
        x = x.view(n, int(h * scale_factor), int(w * scale_factor),
                   int(c / (scale_factor * scale_factor)))
        if self.ps_version == 'v1':
            warnings.warn("In ps_version 'v1', the height and width have not been swapped back, "
                          'which results in a transposed image.')
        else:
            x = x.permute(0, 2, 1, 3).contiguous()
        return x

    def extract_feature(self, pixel_values):
        """
        Extracts visual features from pixel values.
        [FIX] Ensures that pixel_values are on the same device as the vision_model.
        """

        model_device = self.vision_model.device

        pixel_values = pixel_values.to(
            device=model_device, dtype=self.vision_model.dtype)

        if self.select_layer == -1:
            vit_embeds = self.vision_model(
                pixel_values=pixel_values,
                output_hidden_states=False,
                return_dict=True).last_hidden_state
        else:
            vit_embeds = self.vision_model(
                pixel_values=pixel_values,
                output_hidden_states=True,
                return_dict=True).hidden_states[self.select_layer]

        vit_embeds = vit_embeds[:, 1:, :]

        h = w = int(vit_embeds.shape[1] ** 0.5)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)
        vit_embeds = self.pixel_shuffle(
            vit_embeds, scale_factor=self.downsample_ratio)
        vit_embeds = vit_embeds.reshape(
            vit_embeds.shape[0], -1, vit_embeds.shape[-1])
        vit_embeds = self.mlp1(vit_embeds)
        return vit_embeds

    def batch_chat(self, tokenizer, pixel_values, questions, generation_config, num_patches_list=None,
                   history=None, return_history=False, IMG_START_TOKEN='<img>', IMG_END_TOKEN='</img>',
                   IMG_CONTEXT_TOKEN='<IMG_CONTEXT>', verbose=False, image_counts=None):
        if history is not None or return_history:
            print('Now multi-turn chat is not supported in batch_chat.')
            raise NotImplementedError

        if image_counts is not None:
            num_patches_list = image_counts
            print(
                'Warning: `image_counts` is deprecated. Please use `num_patches_list` instead.')

        img_context_token_id = tokenizer.convert_tokens_to_ids(
            IMG_CONTEXT_TOKEN)
        self.img_context_token_id = img_context_token_id

        if verbose and pixel_values is not None:
            image_bs = pixel_values.shape[0]
            print(f'dynamic ViT batch size: {image_bs}')

        queries = []
        for idx, num_patches in enumerate(num_patches_list):
            question = questions[idx]
            if pixel_values is not None and '<image>' not in question:
                question = '<image>\n' + question
            template = get_conv_template(self.template)
            template.system_message = self.system_message
            template.append_message(template.roles[0], question)
            template.append_message(template.roles[1], None)
            query = template.get_prompt()

            image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * \
                self.num_image_token * num_patches + IMG_END_TOKEN
            query = query.replace('<image>', image_tokens, 1)
            queries.append(query)

        tokenizer.padding_side = 'left'
        model_inputs = tokenizer(queries, return_tensors='pt', padding=True)
        input_ids = model_inputs['input_ids'].to(self.device)
        attention_mask = model_inputs['attention_mask'].to(self.device)
        eos_token_id = tokenizer.convert_tokens_to_ids(template.sep.strip())
        generation_config['eos_token_id'] = eos_token_id
        generation_output = self.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            **generation_config
        )
        responses = tokenizer.batch_decode(
            generation_output, skip_special_tokens=True)
        responses = [response.split(template.sep.strip())[
            0].strip() for response in responses]
        return responses

    def chat(self, tokenizer, pixel_values, question, generation_config, history=None, return_history=False,
             num_patches_list=None, IMG_START_TOKEN='<img>', IMG_END_TOKEN='</img>', IMG_CONTEXT_TOKEN='<IMG_CONTEXT>',
             verbose=False):

        if history is None and pixel_values is not None and '<image>' not in question:
            question = '<image>\n' + question

        if num_patches_list is None:
            num_patches_list = [pixel_values.shape[0]
                                ] if pixel_values is not None else []
        assert pixel_values is None or len(
            pixel_values) == sum(num_patches_list)

        img_context_token_id = tokenizer.convert_tokens_to_ids(
            IMG_CONTEXT_TOKEN)
        self.img_context_token_id = img_context_token_id

        template = get_conv_template(self.template)
        template.system_message = self.system_message
        eos_token_id = tokenizer.convert_tokens_to_ids(template.sep.strip())

        history = [] if history is None else history
        for (old_question, old_answer) in history:
            template.append_message(template.roles[0], old_question)
            template.append_message(template.roles[1], old_answer)
        template.append_message(template.roles[0], question)
        template.append_message(template.roles[1], None)
        query = template.get_prompt()

        if verbose and pixel_values is not None:
            image_bs = pixel_values.shape[0]
            print(f'dynamic ViT batch size: {image_bs}')

        for num_patches in num_patches_list:
            image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * \
                self.num_image_token * num_patches + IMG_END_TOKEN
            query = query.replace('<image>', image_tokens, 1)

        model_inputs = tokenizer(query, return_tensors='pt')
        input_ids = model_inputs['input_ids'].to(self.device)
        print(input_ids)
        attention_mask = model_inputs['attention_mask'].to(self.device)
        generation_config['eos_token_id'] = eos_token_id
        generation_output = self.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            **generation_config
        )
        response = tokenizer.batch_decode(
            generation_output, skip_special_tokens=True)[0]
        response = response.split(template.sep.strip())[0].strip()
        history.append((question, response))
        if return_history:
            return response, history
        else:
            query_to_print = query.replace(IMG_CONTEXT_TOKEN, '')
            query_to_print = query_to_print.replace(
                f'{IMG_START_TOKEN}{IMG_END_TOKEN}', '<image>')
            if verbose:
                print(query_to_print, response)
            return response

    @torch.no_grad()
    def _proxy_prefill_and_get_queries(self, past_key_values: Cache):
        """
        [v3 - Corrected Cache API] Generates proxy queries for importance scoring.
        This version uses safe APIs to access the KV Cache and dynamically generates 
        correct Attention Masks and positional encodings for each layer, precisely 
        accommodating complex scenarios where layers have varying sequence lengths.
        """
        device = self.get_input_embeddings().weight.device
        proxy_ids = torch.tensor([[151645, 151644, 77091, 198]], device=device)
        proxy_embeds = self.get_input_embeddings()(proxy_ids)
        bsz, q_len, _ = proxy_embeds.shape

        
        if hasattr(self.language_model, 'model'):
            layers = self.language_model.model.layers
            rotary_emb = self.language_model.model.rotary_emb
        else:
            layers = self.language_model.layers
            rotary_emb = self.language_model.rotary_emb


        pkv_clone = DynamicCache()
        if past_key_values is not None and past_key_values.get_seq_length(layer_idx=0) > 0:

            num_cached_layers = len(past_key_values)
            for layer_idx in range(min(len(layers), num_cached_layers)):
                k, v = past_key_values[layer_idx]
                pkv_clone.update(k.clone(), v.clone(), layer_idx=layer_idx)

        proxy_queries_per_layer = []
        hidden_states = proxy_embeds


        for layer_idx, layer in enumerate(layers):
            layernorm_output = layer.input_layernorm(hidden_states)

            current_past_len = pkv_clone.get_seq_length(layer_idx)
            proxy_pos_ids = torch.arange(
                current_past_len, current_past_len + q_len, device=device
            ).view(1, -1)

            cos, sin = rotary_emb(proxy_embeds, proxy_pos_ids)

            q_vectors = layer.self_attn.q_proj(layernorm_output)

            q_vectors = q_vectors.view(
                bsz, q_len, layer.self_attn.config.num_attention_heads, layer.self_attn.head_dim
            ).transpose(1, 2)

            q_rot, _ = apply_rotary_pos_emb(q_vectors, q_vectors, cos, sin)
            proxy_queries_per_layer.append(q_rot)


            total_kv_len = current_past_len + q_len
            attention_mask = torch.ones(
                bsz, 1, q_len, total_kv_len, device=device, dtype=torch.bool)
            attention_mask = attention_mask.tril(diagonal=current_past_len)
            attention_mask = torch.where(
                attention_mask, 0.0, -torch.inf).to(hidden_states.dtype)


            attn_output, _ = layer.self_attn(
                hidden_states=layernorm_output,
                position_embeddings=(cos, sin),
                attention_mask=attention_mask,
                past_key_values=pkv_clone,
                use_cache=True,
                cache_position=proxy_pos_ids
            )

            hidden_states = hidden_states + attn_output
            hidden_states = hidden_states + \
                layer.mlp(layer.post_attention_layernorm(hidden_states))

        return proxy_queries_per_layer

    def _compress_kv_cache(self):
        """
        KV Cache compression using a 'Uniform + Fixed Ratio + Dynamic Entropy' triple-hybrid strategy.
        """
        pkv = self.streaming_state["past_key_values"]
        if pkv is None or pkv.get_seq_length(layer_idx=0) == 0:
            return

        if hasattr(self.language_model, 'model'):
            layers = self.language_model.model.layers
        else:
            layers = self.language_model.layers

        # 1. Initialization and Configuration
        try:
            kv_dtype = pkv.key_cache[0].dtype if hasattr(
                pkv, 'key_cache') and pkv.key_cache else torch.bfloat16
        except (IndexError, AttributeError):
            kv_dtype = torch.bfloat16
        try:
            device = pkv.key_cache[0].device if hasattr(
                pkv, 'key_cache') and pkv.key_cache else self.device
        except (IndexError, AttributeError):
            device = self.device

        num_layers = len(layers)
        fusion_counts = self.streaming_state["fusion_counts"]
        total_budget = self.kv_cache_token_limit_per_layer * num_layers
        current_total_len = sum(pkv.get_seq_length(layer_idx=i)
                                for i in range(num_layers))

        if current_total_len <= total_budget:
            return

        # 2. Calculate Importance Scores and Entropy
        pkv_clone = DynamicCache()
        if pkv.get_seq_length() > 0:
            for layer_idx in range(num_layers):
                k, v = pkv[layer_idx]
                pkv_clone.update(k.clone(), v.clone(), layer_idx)

        proxy_queries = self._proxy_prefill_and_get_queries(pkv_clone)
        num_kv_groups = layers[0].self_attn.num_key_value_groups

        layer_scores = []
        layer_entropies = []  

        for layer_idx in range(num_layers):
            k_layer, v_layer = pkv[layer_idx]
            q_proxy = proxy_queries[layer_idx]

            with torch.autocast(device_type=k_layer.device.type, dtype=kv_dtype):
                k_layer_repeated = repeat_kv(k_layer, num_kv_groups)
                attn_scores = torch.matmul(
                    q_proxy, k_layer_repeated.transpose(-2, -1))
                scores = attn_scores.mean(dim=2).mean(dim=1).squeeze(0)

                fc_layer = fusion_counts[layer_idx]
                if fc_layer.numel() == scores.numel():
                    # Compress fc_layer to [N] to match scores dimensions, preventing incorrect broadcasting
                    if fc_layer.dim() > 1:
                        attention_bias = torch.log1p(
                            fc_layer.squeeze(0) - 1).to(scores.device, scores.dtype)
                    else:
                        attention_bias = torch.log1p(
                            fc_layer - 1).to(scores.device, scores.dtype)
                    scores = scores + attention_bias

                v_norm = v_layer.norm(p=2, dim=-1).mean(dim=1).squeeze(0)
                log_v_norm = torch.log(
                    v_norm + 1e-9).to(scores.device, scores.dtype)
                scores = scores + log_v_norm

            layer_scores.append(scores)

            # --- Restore Entropy Calculation Logic ---
            if scores.numel() > 0:
                probs = F.softmax(scores.to(torch.float32), dim=0)
                entropy = -torch.sum(probs * torch.log(probs + 1e-9))
            else:
                entropy = torch.tensor(0.0, device=scores.device)
            # Weight entropy using v_norm to focus more on tokens with larger values
            layer_entropies.append(torch.log1p_(entropy) * log_v_norm.mean())

        # 3. Dynamic Budget Allocation (Using Triple-Hybrid Strategy)
        new_keys = [None] * num_layers
        new_values = [None] * num_layers
        new_fusion_counts = [None] * num_layers
        layer_lens = [pkv.get_seq_length(i) for i in range(num_layers)]

        if total_budget > 0:
            weight_uniform = 0.3
            weight_static = 0.6
            weight_entropy = 0.1

            uniform_proportions = torch.full(
                (num_layers,), 1.0 / num_layers, device=device)


            static_ratios_list = [16.91, 11.62, 9.92, 8.35, 8, 7.8, 7.23, 6.33, 5.47, 4.94, 4.30, 3.53, 2.87,
                                  2.32, 1.93, 1.59, 1.24, 0.98, 0.53, 0.40, 0.29, 0.20, 0.15, 0.13, 0.11, 0.09, 0.07, 0.05]
            if len(static_ratios_list) != num_layers:
                raise ValueError(
                    f"The number of provided static ratios ({len(static_ratios_list)}) does not match the number of model layers ({num_layers}).")
            static_ratios_tensor = torch.tensor(
                static_ratios_list, dtype=torch.float32, device=device)
            static_proportions = static_ratios_tensor / static_ratios_tensor.sum()

            entropies_tensor = torch.stack(layer_entropies)
            entropies_tensor = entropies_tensor - entropies_tensor.min()
            total_entropy = entropies_tensor.sum()

            if total_entropy <= 1e-9: 
                entropy_proportions = uniform_proportions.clone()
            else:
                entropy_proportions = entropies_tensor / total_entropy

            proportions_to_use = (weight_uniform * uniform_proportions +
                                  weight_static * static_proportions +
                                  weight_entropy * entropy_proportions)


            layer_lens_tensor = torch.tensor(layer_lens, device=device)
            num_to_keep = proportions_to_use * total_budget


            over_allocated_mask = num_to_keep > layer_lens_tensor.float()
            excess_budget = (num_to_keep[over_allocated_mask] -
                             layer_lens_tensor[over_allocated_mask].float()).sum()
            num_to_keep = torch.min(num_to_keep, layer_lens_tensor.float())


            if excess_budget > 0:
                under_allocated_mask = ~over_allocated_mask
                if under_allocated_mask.any():
                    proportions_under = proportions_to_use[under_allocated_mask]
                    if proportions_under.sum() > 0:
                        proportions_under /= proportions_under.sum()
                        redistribution = proportions_under * excess_budget
                        current_under_values = num_to_keep[under_allocated_mask]
                        capacity_under = layer_lens_tensor[under_allocated_mask]
                        final_additions = torch.min(
                            redistribution, (capacity_under - current_under_values).float())
                        num_to_keep[under_allocated_mask] += final_additions

            floored_keeps = num_to_keep.floor().long()
            remainder = int(total_budget - floored_keeps.sum())
            if remainder > 0:
                fractional_parts = num_to_keep - floored_keeps.float()
                fractional_parts[floored_keeps >= layer_lens_tensor] = -1
                top_k_indices = torch.topk(fractional_parts, min(
                    remainder, (fractional_parts > 0).sum().item())).indices
                floored_keeps.scatter_add_(
                    0, top_k_indices, torch.ones_like(top_k_indices, dtype=torch.long))

            budgets_for_all_layers = floored_keeps.tolist()
        else:
            budgets_for_all_layers = [0] * num_layers

        print(
            f"KV Cache compression budgets per layer: {budgets_for_all_layers}")



        new_keys = [None] * num_layers
        new_values = [None] * num_layers
        new_fusion_counts = [None] * num_layers

        for layer_idx in range(num_layers):
            budget_for_this_layer = int(budgets_for_all_layers[layer_idx])
            k_layer, v_layer = pkv[layer_idx]
            fc_layer = fusion_counts[layer_idx]
            current_len = k_layer.shape[2]

            if current_len <= budget_for_this_layer:
                new_keys[layer_idx] = k_layer
                new_values[layer_idx] = v_layer
                new_fusion_counts[layer_idx] = fc_layer
                continue

            if budget_for_this_layer == 0:
                new_keys[layer_idx] = torch.empty_like(k_layer[..., :0, :])
                new_values[layer_idx] = torch.empty_like(v_layer[..., :0, :])

                new_fusion_counts[layer_idx] = fc_layer[..., :0]

                continue

            scores_this_layer = layer_scores[layer_idx]
            _, elite_indices = torch.topk(
                scores_this_layer, k=budget_for_this_layer)
            final_indices = torch.sort(elite_indices).values

            new_keys[layer_idx] = k_layer.index_select(2, final_indices)
            new_values[layer_idx] = v_layer.index_select(2, final_indices)

            new_fusion_counts[layer_idx] = fc_layer.index_select(
                0, final_indices)


        new_cache = DynamicCache()


        for i in range(num_layers):

            if new_keys[i] is not None and new_keys[i].shape[2] > 0:

                new_cache.update(new_keys[i], new_values[i], layer_idx=i)

        self.streaming_state["past_key_values"] = new_cache

        self.streaming_state["fusion_counts"] = new_fusion_counts

    @torch.no_grad()
    def generate(
            self,
            pixel_values: Optional[torch.FloatTensor] = None,
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.LongTensor] = None,
            generation_config: Optional[GenerationConfig] = None,
            use_frame_prefill: bool = True,
            **generate_kwargs,
    ) -> torch.LongTensor:
        if not use_frame_prefill or input_ids.shape[1] <= 1 or input_ids.shape[0] > 1:
            if pixel_values is not None:
                inputs_embeds = self.language_model.get_input_embeddings()(input_ids)
                vit_embeds = self.extract_feature(pixel_values)
                mask = (input_ids == self.img_context_token_id)
                if mask.sum() == vit_embeds.view(-1, vit_embeds.shape[-1]).shape[0]:
                    inputs_embeds[mask] = vit_embeds.view(
                        -1, vit_embeds.shape[-1])
                else:
                    logger.warning(
                        "Mismatch between image tokens and vit_embeds, skipping embedding replacement.")
            else:
                inputs_embeds = None

            return self.language_model.generate(
                input_ids=input_ids if inputs_embeds is None else None,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                generation_config=generation_config,
                **generate_kwargs,
            )


        print("\n=== Streaming Mode with Token & Cache Compression (v7) ===")
        from transformers.generation.utils import LogitsProcessorList, StoppingCriteriaList

        # 1. Prepare configuration and utilities
        has_default_max_length = 'max_length' not in generate_kwargs
        gen_config, model_kwargs = self.language_model._prepare_generation_config(
            generation_config, **generate_kwargs)
        self.language_model._prepare_special_tokens(
            gen_config, device=self.device)

        input_ids_len = input_ids.shape[1]
        if gen_config.max_new_tokens is not None:
            if not has_default_max_length:
                logger.warning(
                    f"Both `max_new_tokens` and `max_length` seem to have been set. `max_new_tokens` will take precedence.")
            gen_config.max_length = gen_config.max_new_tokens + input_ids_len

        logits_processor = self.language_model._get_logits_processor(
            generation_config=gen_config, input_ids_seq_length=input_ids_len,
            encoder_input_ids=input_ids, logits_processor=LogitsProcessorList()
        )
        stopping_criteria = self.language_model._get_stopping_criteria(
            generation_config=gen_config, stopping_criteria=StoppingCriteriaList()
        )

        # 2. Reset state and extract visual features
        self.reset_streaming_state()
        vit_embeds = self.extract_feature(pixel_values).view(
            -1, self.language_model.config.hidden_size) if pixel_values is not None else None

        # 3. Chunk and execute streaming Prefill
        input_ids_cpu = input_ids.squeeze(0).cpu()
        is_image_mask = (input_ids_cpu == self.img_context_token_id)
        change_mask = (is_image_mask[1:] != is_image_mask[:-1])
        boundaries = torch.cat([torch.tensor([0]), torch.where(change_mask)[
                               0] + 1, torch.tensor([len(input_ids_cpu)])])
        chunks = [{'start': s.item(), 'end': e.item(), 'is_image': is_image_mask[s.item()].item()}
                  for s, e in zip(boundaries[:-1], boundaries[1:]) if s.item() != e.item()]

        pkv = self.streaming_state["past_key_values"]
        if pkv is None:
            pkv = DynamicCache()
            self.streaming_state["past_key_values"] = pkv
            num_layers = len(self.language_model.model.layers) if hasattr(
                self.language_model, 'model') else len(self.language_model.layers)
            self.streaming_state["fusion_counts"] = [torch.tensor(
                [], device=self.device) for _ in range(num_layers)]

        vit_embeds_offset = 0
        hidden_states = None


        for chunk in chunks:
            self._compress_kv_cache()

            start, end = chunk['start'], chunk['end']
            original_indices = torch.arange(start, end, device=self.device)

            if chunk['is_image']:
                frame_embeds = vit_embeds[vit_embeds_offset:vit_embeds_offset + (
                    end - start)].unsqueeze(0)
                vit_embeds_offset += (end - start)
                processed = self._process_visual_chunk(
                    frame_embeds, original_indices)
                processed_embeds = processed["embeds"]
                # Ensure 1D
                processed_fusion_counts = processed["fusion_counts"].view(-1)
            else:
                processed_embeds = self.language_model.get_input_embeddings()(
                    input_ids[:, start:end])
                processed_fusion_counts = torch.ones(
                    processed_embeds.shape[1], device=self.device)

            if processed_embeds.shape[1] == 0:
                continue

            # ==================== Manual Layer-by-Layer Prefill Execution ====================
            pkv = self.streaming_state["past_key_values"]
            q_len = processed_embeds.shape[1]

            # 1. Prepare shared Position Embeddings
            past_kv_len = self.streaming_state["logical_seq_len"]
            position_ids = torch.arange(
                past_kv_len, past_kv_len + q_len, device=self.device).unsqueeze(0)

            # RoPE encoding only needs to be computed once, then passed to all layers
            position_embeddings = self.language_model.model.rotary_emb(
                processed_embeds, position_ids)

            # 2. Initialize hidden_states
            current_hidden_states = processed_embeds

            # 3. Manually iterate through each layer
            model_layers = self.language_model.model.layers
            fusion_counts_history = self.streaming_state["fusion_counts"]

            for layer_idx, layer in enumerate(model_layers):
                
                combined_fc = torch.cat(
                    [fusion_counts_history[layer_idx].view(-1), processed_fusion_counts])

                #pkv.get_seq_length(layer_idx)retrieves the actual physical length of this layer after compression
                attn_bias = self._create_streaming_attention_mask(
                    q_len=q_len,
                    past_kv_len=pkv.get_seq_length(layer_idx),
                    combined_fusion_counts=combined_fc,
                    device=self.device,
                    dtype=processed_embeds.dtype
                )

                # Note: Settinguse_cache=Truehere triggers automatic pkv updates
                current_hidden_states = layer(
                    hidden_states=current_hidden_states,
                    attention_mask=attn_bias,
                    position_ids=position_ids,
                    past_key_values=pkv,
                    use_cache=True,
                    cache_position=position_ids,  
                    position_embeddings=position_embeddings,
                )

            # 4. Update global state
            # a. Update fusion_counts history
            for i in range(len(fusion_counts_history)):
                fusion_counts_history[i] = torch.cat(
                    [fusion_counts_history[i], processed_fusion_counts])

            # b. Update logical sequence length
            self.streaming_state["logical_seq_len"] += q_len

            # c. Feed the final layer's output as input for the next iteration
            hidden_states = current_hidden_states

        # 4. Autoregressive generation loop
        if hidden_states is None:
            # If no output exists after prefill... return immediately
            return input_ids

        # a. Compute logits for the first decoding step following prefill completion
        last_hidden_state_normalized = self.language_model.model.norm(
            hidden_states)
        last_token_hidden_state = last_hidden_state_normalized[:, -1, :]
        next_token_logits = self.language_model.lm_head(
            last_token_hidden_state)

        generated_ids = input_ids

        while True:
            if stopping_criteria(generated_ids, None):
                break

            # b. Apply logits processor... to select the next token
            next_token_scores = logits_processor(
                generated_ids, next_token_logits)
            next_tokens = torch.argmax(next_token_scores, dim=-1, keepdim=True)
            generated_ids = torch.cat([generated_ids, next_tokens], dim=-1)

            pkv = self.streaming_state["past_key_values"]
            model_layers = self.language_model.model.layers
            fusion_counts_history = self.streaming_state["fusion_counts"]

            # c. Prepare shared inputs across all layers for this decoding step
            next_position_ids = torch.tensor(
                [[self.streaming_state["logical_seq_len"]]], device=self.device)
            next_token_embeds = self.language_model.get_input_embeddings()(next_tokens)

            # Compute RoPE for the current token
            position_embeddings = self.language_model.model.rotary_emb(
                next_token_embeds, next_position_ids)

            # d. Initialize hidden_states for the current decoding step
            current_hidden_states = next_token_embeds

            # e. Iterate through each layer manually, mirroring the prefill process
            for layer_idx, layer in enumerate(model_layers):
                # fusion_counts must be one-dimensional
                new_fc_token = torch.tensor([1.0], device=self.device)
                combined_fc = torch.cat(
                    [fusion_counts_history[layer_idx].view(-1), new_fc_token])

                attn_bias = self._create_streaming_attention_mask(
                    q_len=1,  
                    past_kv_len=pkv.get_seq_length(layer_idx),
                    combined_fusion_counts=combined_fc,
                    device=self.device, dtype=next_token_embeds.dtype
                )

                current_hidden_states = layer(
                    hidden_states=current_hidden_states,
                    attention_mask=attn_bias,
                    position_ids=next_position_ids,
                    past_key_values=pkv,  
                    use_cache=True,
                    cache_position=next_position_ids,
                    position_embeddings=position_embeddings,
                )

            # f. After processing all layers, apply the final LayerNorm and lm_head
            last_hidden_state_normalized_decode = self.language_model.model.norm(
                current_hidden_states)
            next_token_logits = self.language_model.lm_head(
                last_hidden_state_normalized_decode).squeeze(1)

            # g. Update global state to prepare for the next decoding iteration
            new_fc_token_update = torch.tensor([1.0], device=self.device)
            for i in range(len(fusion_counts_history)):
                fusion_counts_history[i] = torch.cat(
                    [fusion_counts_history[i], new_fc_token_update])

            self.streaming_state["logical_seq_len"] += 1

        return generated_ids

    @property
    def lm_head(self):
        return self.language_model.get_output_embeddings()

    def get_output_embeddings(self):
        return self.language_model.get_output_embeddings()

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        return self.language_model.set_input_embeddings(value)

    def set_output_embeddings(self, value):
        return self.language_model.set_output_embeddings(value)
