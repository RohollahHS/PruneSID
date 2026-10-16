"""PruneSID-style image-token compression for Qwen2.5-VL (Transformers 5.3.0).

Inference-only reference implementation. Supported path: batch size 1, one image,
no video, standard 2-D attention_mask, and eager/dynamic-cache generate().
"""

from __future__ import annotations

import math
from types import MethodType
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLModelOutputWithPast


@torch.no_grad()
def _pca_group(features: torch.Tensor, min_components: int):
    # features: [num_visual_tokens, hidden_size]
    x = torch.sigmoid(features.float()).transpose(0, 1)  # [hidden_size, num_visual_tokens]
    q = max(1, min(int(min_components), min(x.shape) - 1))
    if q < 1:
        return x.new_ones((features.shape[0], 1)), torch.zeros(features.shape[0], dtype=torch.long, device=x.device)
    _, _, v = torch.pca_lowrank(x, q=q)
    v = v.abs()
    return v, v.argmax(dim=1)


def _nms(similarity: np.ndarray, scores: np.ndarray, threshold: float) -> list[int]:
    scores = scores.astype(np.float32, copy=True)
    keep: list[int] = []
    while scores.size and np.any(scores > 0):
        idx = int(scores.argmax())
        if scores[idx] <= 0:
            break
        keep.append(idx)
        scores[idx] = 0
        scores[similarity[idx] > threshold] = 0
    return keep


@torch.no_grad()
def prunesid_select_tokens(features: torch.Tensor, need_token_num: int = 64) -> torch.Tensor:
    """Return selected token indices, preserving PruneSID's PCA/group/NMS strategy."""
    n = int(features.shape[0])
    if n <= 16:
        return torch.arange(n, device=features.device)

    target = max(round(n * int(need_token_num) / 576), 16)
    target = min(target, n)
    if target >= n:
        return torch.arange(n, device=features.device)

    components, belongs = _pca_group(features, min_components=max(int(target / 4), 4))
    num_groups = components.shape[1]
    token_ids = torch.arange(n, device=features.device)
    belongs_mask = belongs[:, None] == torch.arange(num_groups, device=features.device)[None, :]
    components = components * belongs_mask
    scores = torch.zeros_like(components)

    normalized = F.normalize(features.float(), p=2, dim=-1)
    similarity = (normalized @ normalized.T).clamp(-1, 1)
    sim_mean = similarity.triu(diagonal=0).mean().item()
    similarity_np = similarity.cpu().numpy()
    ratio = max(target / (n / 18), 1.0)
    threshold = float(ratio * sim_mean)

    group_members: list[torch.Tensor] = []
    nms_members: list[list[int]] = []
    group_counts = torch.zeros(num_groups, dtype=torch.long, device=features.device)
    nms_counts = torch.zeros_like(group_counts)
    for g in range(num_groups):
        members = torch.where(belongs == g)[0]
        group_members.append(members)
        group_counts[g] = members.numel()
        if members.numel() == 0:
            nms_members.append([])
            continue
        member_np = members.cpu().numpy()
        group_sim = similarity_np[np.ix_(member_np, member_np)]
        group_scores = components[members, g].cpu().numpy()
        local_keep = _nms(group_sim, group_scores, threshold)
        global_keep = members[torch.as_tensor(local_keep, device=features.device, dtype=torch.long)]
        nms_members.append(global_keep.cpu().tolist())
        nms_counts[g] = global_keep.numel()
        # Rank diversified candidates above non-NMS tokens, as in the original method.
        if global_keep.numel():
            ranks = torch.arange(n, n - global_keep.numel(), -1, device=features.device, dtype=scores.dtype)
            scores[global_keep, g] = ranks

    if int(nms_counts.sum()) == 0:
        return torch.linspace(0, n - 1, target, device=features.device).round().long().unique()[:target]

    # Start with one token per non-empty group, then apportion the remainder by NMS counts.
    lower = (group_counts > 0).long()
    cap = 5 * math.ceil(target / 64)
    upper = torch.minimum(group_counts, torch.full_like(group_counts, cap))
    upper = torch.minimum(upper, nms_counts)
    while int(upper.sum()) < target:
        upper = torch.minimum(upper + (upper < group_counts).long(), group_counts)

    lower_sum = int(lower.sum())
    remaining = max(0, target - lower_sum)
    weights = nms_counts.float()
    weights = weights / weights.sum()
    cumulative = torch.round(torch.cumsum(weights, dim=0) * remaining).long()
    previous = torch.cat([torch.zeros(1, dtype=torch.long, device=features.device), cumulative[:-1]])
    group_budget = torch.minimum(lower + cumulative - previous, upper)

    # The per-group cap may clip the proportional allocation; redistribute clipped tokens.
    while int(group_budget.sum()) < target:
        candidates = torch.where(group_budget < upper)[0]
        if candidates.numel() == 0:
            # Relax only the initial cap; never exceed the number of tokens in a group.
            upper = torch.minimum(upper + (upper < group_counts).long(), group_counts)
            candidates = torch.where(group_budget < upper)[0]
            if candidates.numel() == 0:
                break
        best = max(candidates.tolist(), key=lambda i: (int(nms_counts[i]), int(group_counts[i]), -i))
        group_budget[best] += 1

    selected: list[torch.Tensor] = []
    for g, members in enumerate(group_members):
        count = int(group_budget[g])
        if count <= 0 or members.numel() == 0:
            continue
        # Restrict sorting to this group so zero-score ties cannot select a token from another group.
        order = torch.argsort(scores[members, g], descending=True)
        selected.append(members[order[:count]])

    if not selected:
        return torch.arange(target, device=features.device)
    important = torch.cat(selected)
    important = important[:target]
    # Match the original code's raster-order restoration for larger budgets.
    if important.numel() > 150:
        important = important.sort().values
    return important.long()


def _cache_seq_length(cache) -> int:
    if cache is None:
        return 0
    if hasattr(cache, "get_seq_length"):
        return int(cache.get_seq_length())
    # Legacy tuple cache fallback.
    return int(cache[0][0].shape[-2])


def _prunesid_qwen2_5_vl_model_forward(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values=None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    pixel_values: Optional[torch.Tensor] = None,
    pixel_values_videos: Optional[torch.FloatTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    rope_deltas: Optional[torch.LongTensor] = None,
    mm_token_type_ids: Optional[torch.IntTensor] = None,
    cache_position: Optional[torch.LongTensor] = None,
    second_per_grid_ts: Optional[torch.Tensor] = None,
    **kwargs,
):
    """Patched Qwen2_5_VLModel.forward. Keeps the official text/vision modules intact."""
    original_forward = self._prunesid_original_forward
    past_len = _cache_seq_length(past_key_values)

    # Decode after a compressed prefill: the KV cache contains the shortened prompt,
    # while GenerationMixin's external input_ids/attention_mask still describe the full prompt.
    if (
        getattr(self, "_prunesid_active", False)
        and past_len > 0
        and pixel_values is None
        and pixel_values_videos is None
    ):
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("input_ids or inputs_embeds must be provided during decoding")
            inputs_embeds = self.get_input_embeddings()(input_ids)
        batch_size, query_length, _ = inputs_embeds.shape
        cache_position = torch.arange(past_len, past_len + query_length, device=inputs_embeds.device)
        prefix_mask = self._prunesid_prefix_attention_mask.to(inputs_embeds.device)
        generated_past = max(0, past_len - prefix_mask.shape[-1])
        decode_mask = torch.ones(
            (batch_size, generated_past + query_length),
            dtype=prefix_mask.dtype,
            device=inputs_embeds.device,
        )
        attention_mask = torch.cat([prefix_mask, decode_mask], dim=-1)
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            cache_position=cache_position,
            **kwargs,
        )
        result = Qwen2_5_VLModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=getattr(self, "rope_deltas", rope_deltas),
        )
        return result if return_dict else result.to_tuple()

    # Use the stock path for text-only calls and non-PruneSID video calls.
    if pixel_values is None:
        return original_forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            rope_deltas=rope_deltas,
            mm_token_type_ids=mm_token_type_ids,
            cache_position=cache_position,
            second_per_grid_ts=second_per_grid_ts,
            **kwargs,
        )

    if pixel_values_videos is not None:
        raise NotImplementedError("This reference patch compresses image prompts only; image+video prompts are not supported.")
    if input_ids is None or inputs_embeds is not None:
        raise ValueError("PruneSID patch expects input_ids and does not support externally supplied inputs_embeds.")
    if input_ids.shape[0] != 1:
        raise NotImplementedError("This reference implementation supports batch_size=1 only.")
    if image_grid_thw is None or image_grid_thw.shape[0] != 1:
        raise NotImplementedError("This reference implementation supports exactly one image per prompt.")
    if attention_mask is not None and attention_mask.ndim != 2:
        raise ValueError("Expected the standard 2-D attention_mask returned by AutoProcessor.")

    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    use_cache = use_cache if use_cache is not None else self.config.use_cache
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict
    inputs_embeds = self.get_input_embeddings()(input_ids)

    # Transformers 5.3.0 Qwen2.5-VL returns merged features in pooler_output.
    pixel_values = pixel_values.type(self.visual.dtype)
    with torch.no_grad():
        vision_outputs = self.visual(pixel_values, grid_thw=image_grid_thw, return_dict=True)
    split_sizes = (image_grid_thw.prod(-1) // self.visual.spatial_merge_size**2).tolist()
    image_chunks = list(torch.split(vision_outputs.pooler_output, split_sizes))
    if len(image_chunks) != 1:
        raise RuntimeError(f"Expected one image feature chunk, got {len(image_chunks)}")
    image_features = image_chunks[0].to(device=inputs_embeds.device, dtype=inputs_embeds.dtype)

    image_positions = torch.where(input_ids[0] == self.config.image_token_id)[0]
    if image_positions.numel() != image_features.shape[0]:
        raise ValueError(
            f"Image placeholder count ({image_positions.numel()}) does not match vision tokens ({image_features.shape[0]})."
        )
    if image_positions.numel() and not torch.equal(
        image_positions,
        torch.arange(image_positions[0], image_positions[0] + image_positions.numel(), device=image_positions.device),
    ):
        raise NotImplementedError("The image placeholder tokens must form one contiguous span.")

    image_mask, _ = self.get_placeholder_mask(input_ids, inputs_embeds=inputs_embeds, image_features=image_features)
    inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_features)
    important_indices = prunesid_select_tokens(image_features, self._prunesid_need_token_num).to(input_ids.device)
    image_start = int(image_positions[0])
    image_end = int(image_positions[-1])
    keep_ids = torch.cat(
        [
            torch.arange(0, image_start, device=input_ids.device),
            important_indices + image_start,
            torch.arange(image_end + 1, input_ids.shape[1], device=input_ids.device),
        ]
    ).long()

    # Compute multimodal positions on the original placeholder sequence, then apply the same
    # selected sequence indices to embeddings and positions. v5.3.0 uses 4-way position_ids.
    if position_ids is None:
        position_ids = self.compute_3d_position_ids(
            input_ids=input_ids,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            second_per_grid_ts=second_per_grid_ts,
            mm_token_type_ids=mm_token_type_ids,
        )

    inputs_embeds = inputs_embeds[:, keep_ids, :]
    if position_ids is not None:
        position_ids = position_ids[..., keep_ids]
    if attention_mask is not None:
        attention_mask = attention_mask[:, keep_ids]

    # The actual KV cache must use compact sequence coordinates, not the unpruned prompt length.
    cache_position = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device)
    self._prunesid_active = True
    if attention_mask is None:
        self._prunesid_prefix_attention_mask = torch.ones(
            (1, keep_ids.numel()), dtype=torch.long, device=inputs_embeds.device
        )
    else:
        self._prunesid_prefix_attention_mask = attention_mask.detach().clone()

    outputs = self.language_model(
        input_ids=None,
        position_ids=position_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=True,
        cache_position=cache_position,
        **kwargs,
    )
    result = Qwen2_5_VLModelOutputWithPast(
        last_hidden_state=outputs.last_hidden_state,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        rope_deltas=getattr(self, "rope_deltas", rope_deltas),
    )
    return result if return_dict else result.to_tuple()


def prunesid_qwen2_5_vl(model, need_token_num: int = 64):
    """Install the PruneSID-style forward patch on a loaded Qwen2.5-VL model.

    Example:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(...)
        model = prunesid_qwen2_5_vl(model, need_token_num=64)
        generated_ids = model.generate(**inputs, max_new_tokens=128)
    """
    if not hasattr(model, "model") or not hasattr(model.model, "visual") or not hasattr(model.model, "language_model"):
        raise TypeError("Expected a Qwen2_5_VLForConditionalGeneration model.")
    if not 1 <= int(need_token_num):
        raise ValueError("need_token_num must be a positive integer.")
    base_model = model.model
    if not hasattr(base_model, "_prunesid_original_forward"):
        base_model._prunesid_original_forward = base_model.forward
    base_model._prunesid_need_token_num = int(need_token_num)
    base_model._prunesid_active = False
    base_model.forward = MethodType(_prunesid_qwen2_5_vl_model_forward, base_model)
    return model
