"""C96 training objectives and interaction masks."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .relation import PackedScenes, future_relation_labels, pack_scenes


def per_agent_wta_loss(
    predictions: torch.Tensor, logits: torch.Tensor, target: torch.Tensor
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    displacement = torch.linalg.vector_norm(predictions - target[:, None], dim=-1)
    error = displacement.sum(dim=-1)
    winner = error.detach().argmin(dim=-1)
    batch = torch.arange(target.shape[0], device=target.device)
    regression = F.smooth_l1_loss(predictions[batch, winner], target)
    classification = F.cross_entropy(logits, winner)
    return regression + classification, {
        "regression": regression.detach(),
        "classification": classification.detach(),
        "winner": winner,
    }


def scene_wta_loss(
    predictions: torch.Tensor,
    logits: torch.Tensor,
    target: torch.Tensor,
    scene_index: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    packed = pack_scenes(scene_index)
    displacement = torch.linalg.vector_norm(predictions - target[:, None], dim=-1)
    actor_error = displacement.sum(dim=-1)
    scene_error = actor_error.new_zeros((packed.scene_count, predictions.shape[1]))
    scene_error.index_add_(0, packed.inverse, actor_error)
    scene_error = scene_error / packed.counts[:, None]
    scene_winner = scene_error.detach().argmin(dim=-1)
    actor_winner = scene_winner[packed.inverse]
    batch = torch.arange(target.shape[0], device=target.device)
    regression = F.smooth_l1_loss(predictions[batch, actor_winner], target)
    scene_logits = logits.new_zeros((packed.scene_count, logits.shape[1]))
    scene_logits.index_add_(0, packed.inverse, logits)
    scene_logits = scene_logits / packed.counts[:, None]
    classification = F.cross_entropy(scene_logits, scene_winner)
    return regression + classification, {
        "regression": regression.detach(),
        "classification": classification.detach(),
        "winner": actor_winner,
        "scene_winner": scene_winner,
    }


def relation_supervision(
    target: torch.Tensor,
    scene_index: torch.Tensor,
    horizontal_threshold: float = 1.0,
    vertical_threshold: float = 0.3,
) -> tuple[torch.Tensor, torch.Tensor, PackedScenes]:
    return future_relation_labels(
        target,
        scene_index,
        horizontal_threshold=horizontal_threshold,
        vertical_threshold=vertical_threshold,
    )


def relation_classification_loss(
    relation_logits: torch.Tensor,
    labels: torch.Tensor,
    valid_pairs: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    selected_logits = relation_logits[valid_pairs]
    selected_labels = labels[valid_pairs]
    if selected_labels.numel() == 0:
        zero = relation_logits.sum() * 0.0
        return zero, {
            "relation_accuracy": zero.detach(),
            "relation_positive_fraction": zero.detach(),
        }
    counts = torch.bincount(selected_labels, minlength=3).to(relation_logits.dtype)
    weights = counts.sum() / counts.clamp_min(1.0)
    weights = weights / weights.mean()
    loss = F.cross_entropy(selected_logits, selected_labels, weight=weights)
    prediction = selected_logits.argmax(dim=-1)
    return loss, {
        "relation_accuracy": (prediction == selected_labels).float().mean().detach(),
        "relation_positive_fraction": (selected_labels != 0).float().mean().detach(),
    }


def future_proximity_actor_mask(
    target: torch.Tensor,
    scene_index: torch.Tensor,
    horizontal_threshold: float = 1.0,
    vertical_threshold: float = 0.3,
) -> torch.Tensor:
    packed = pack_scenes(scene_index)
    future = packed.pack(target)
    relative = future[:, :, None] - future[:, None, :]
    horizontal = torch.linalg.vector_norm(relative[..., :2], dim=-1)
    vertical = relative[..., 2].abs()
    close = ((horizontal <= horizontal_threshold) & (vertical <= vertical_threshold)).any(dim=-1)
    eye = torch.eye(packed.max_actors, dtype=torch.bool, device=target.device)[None]
    valid_pair = packed.valid[:, :, None] & packed.valid[:, None, :] & ~eye
    actor_close = (close & valid_pair).any(dim=-1)
    return actor_close[packed.inverse, packed.rank]
