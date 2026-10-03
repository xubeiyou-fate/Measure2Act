# ASCENT component redistributed with upstream attribution; see
# docs/ASCENT_NOTICE.md. Measure2Act-specific changes remain identified by the
# release boundary and do not change the upstream attribution.
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence

from causal_mode_generation.model import CausalModeMemory
from continuous_geometry.basis import bspline_basis, reconstruct_curve
from mode_state_query import ModeStateQueryDecoder
from proper_set_ascent.seeds import regular_simplex
from .transformer_blocks import Block
from .scene_interaction import RelationalSceneEncoder
from .spatiotemporal_fusion import SpatioTemporalHistoryEncoder
from .temporal_memory import (
    ModeHistoryCrossAttention,
    NeuralCDEHistory,
    StableDiagonalHistorySSM,
)
from .utils import flight_params_to_pos, ptsToGlobal, ptsToLocal


class Projector(nn.Module):
    """A multi-layer perceptron (MLP) module used to project feature vectors
    from an arbitrary input dimension to a specified embedding dimension.

    Attributes:
        embed_dim (int): The dimensionality of the output embedding.
        proj (nn.Sequential): Sequential container of Linear and ReLU layers.
    """
    def __init__(self, dim, embed_dim=128):
        super().__init__()

        self.embed_dim = embed_dim
        self.proj = nn.Sequential(
            nn.Linear(dim, self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, self.embed_dim),
        )

    def forward(self, data):
        """Forward pass for projecting input features.
        Args:
            data (torch.Tensor): Input tensor of shape (..., dim).

        Returns:
            torch.Tensor: Projected features of shape (..., embed_dim).
        """
        return self.proj(data)


class Ascent(nn.Module):
    """ASCENT Trajectory Prediction network module designed to model agent dynamics
    and context configurations using Transformer blocks, outputting multi-modal
    future trajectories and flight parameters.

    Attributes:
        attn_depth (int): Number of Transformer layers in the agent/scene blocks.
        k (int): Number of prediction modes (multimodal outputs).
        embed_dim (int): Internal hidden size/embedding dimension.
        global_pos_embedding (bool): If True, adds global position embedding to agent features.
        normalize_coords (bool): If True, standardizes trajectories into local reference frames.
        future_steps (int): Total number of discrete future time steps to predict.
        decoder (str/obj): Configuration or indicator for the chosen decoding strategy.
        agent_blks (nn.ModuleList): Sequence of Transformer Blocks for agent-level encoding.
        agent_xy_proj (Projector): Projector for 2D horizontal coordinates.
        agent_z_proj (Projector): Projector for vertical depth/altitude coordinates.
        agent_ts_proj (Projector): Projector for temporal step/index tracking.
        loc (nn.Sequential): MLP mapping to direct 3D future coordinate regressions.
        fp1 (nn.Sequential): MLP predicting the 1st flight parameter feature set over time (speed)
        fp2 (nn.Sequential): MLP predicting the 2nd flight parameter feature set (yaw).
        fp3 (nn.Sequential): MLP predicting the 3rd flight parameter feature set (pitch).
        pi (nn.Sequential): MLP calculating probability logit distributions across modes.
        pos_embed (nn.Sequential): MLP encoding spatial orientation attributes.
        norm (nn.LayerNorm): Layer normalization applied prior to core decoding sequences.
        type_embed (nn.Parameter): Learnable parameters identifying agent categories.
        mode1_embed (nn.Parameter): Learnable multimodal query embeddings for trajectory branching.
        obs (int): Context historical observation timeline duration length.
        obs_steps (int): Total discrete temporal ticks monitored in the history.
        cache (dict): Runtime storage dictionary for analytical metrics or state variables.
    """
    def __init__(self, config):
        super(Ascent, self).__init__()
        self.attn_depth = int(config.get("attn_depth", 2))
        self.k = config["k"]
        self.embed_dim = int(config.get("embed_dim", 128))
        if self.attn_depth < 1:
            raise ValueError("attn_depth must be positive")
        if self.embed_dim % 8:
            raise ValueError("embed_dim must be divisible by eight")

        self.global_pos_embedding = config["global_pos_embedding"]
        self.normalize_coords = config["normalize_coords"]
        self.future_steps = config["pred_len"] // config["pred_step"]
        self.pred_step = int(config["pred_step"])
        self.decoder = config["decoder"]
        self.kinematic_decoder_variant = config.get(
            "kinematic_decoder_variant", "signed_speed_pitch"
        )
        if self.kinematic_decoder_variant not in {
            "signed_speed_pitch",
            "identifiable_horizontal",
            "positive_speed_pitch",
            "cartesian_velocity",
        }:
            raise ValueError("unsupported kinematic_decoder_variant")
        self.near_far_physical_decoder = bool(
            config.get("near_far_physical_decoder", False)
        )
        self.near_far_seconds = int(config.get("near_far_seconds", 60))
        if self.near_far_physical_decoder and self.kinematic_decoder_variant != "positive_speed_pitch":
            raise ValueError(
                "near/far physical decoder requires positive_speed_pitch"
            )
        self.mode_head_variant = config.get("mode_head_variant", "shared")
        if self.mode_head_variant not in {"shared", "private"}:
            raise ValueError("unsupported mode_head_variant")
        self.scene_interaction = bool(config.get("scene_interaction", False))
        self.scene_interaction_stage = config.get("scene_interaction_stage", "actor")
        self.scene_shared_logits = bool(config.get("scene_shared_logits", False))
        self.causal_mode_variant = config.get("causal_mode_variant")
        self.causal_mode_generation = self.causal_mode_variant is not None
        self.continuous_geometry_variant = config.get("continuous_geometry_variant")
        self.continuous_geometry = self.continuous_geometry_variant is not None
        if self.mode_head_variant == "private" and self.continuous_geometry:
            raise ValueError("private mode heads cannot be combined with continuous geometry")
        self.wind_relative_motion = bool(config.get("wind_relative_motion", False))
        self.wind_variant = config.get("wind_variant", "real")
        self.mode_state_query = bool(config.get("mode_state_query", False))
        self.proper_set = bool(config.get("proper_set", False))
        self.broadcast_audio_dim = int(config.get("broadcast_audio_dim", 0))
        self.aligned_weather_dim = int(config.get("aligned_weather_dim", 0))
        self.history_encoder_variant = config.get(
            "history_encoder_variant", "max_pool"
        )
        if self.history_encoder_variant not in {
            "max_pool", "stable_ssm", "neural_cde", "mode_history_attention"
        }:
            raise ValueError("unsupported history_encoder_variant")
        self.spatiotemporal_history_variant = config.get(
            "spatiotemporal_history_variant"
        )
        if self.spatiotemporal_history_variant not in {
            None,
            "temporal_only",
            "full_agent_time",
        }:
            raise ValueError("unsupported spatiotemporal_history_variant")
        if (
            self.spatiotemporal_history_variant is not None
            and self.history_encoder_variant != "max_pool"
        ):
            raise ValueError(
                "spatiotemporal history cannot be combined with legacy history variants"
            )
        if self.broadcast_audio_dim < 0:
            raise ValueError("broadcast_audio_dim cannot be negative")
        if self.aligned_weather_dim < 0:
            raise ValueError("aligned_weather_dim cannot be negative")
        if self.scene_interaction_stage not in {"actor", "mode"}:
            raise ValueError("scene_interaction_stage must be 'actor' or 'mode'")
        if self.scene_shared_logits and (
            not self.scene_interaction or self.scene_interaction_stage != "mode"
        ):
            raise ValueError("scene_shared_logits requires mode-stage scene interaction")
        if self.causal_mode_generation and self.causal_mode_variant not in {
            "latent", "physical"
        }:
            raise ValueError("causal_mode_variant must be 'latent' or 'physical'")
        if self.causal_mode_generation and self.scene_interaction:
            raise ValueError("Causal C1 modes cannot be combined with scene interaction")
        if self.continuous_geometry_variant not in {None, "bspline"}:
            raise ValueError("continuous_geometry_variant must be None or 'bspline'")
        if self.continuous_geometry and self.causal_mode_generation:
            raise ValueError("continuous geometry cannot be combined with causal modes")
        if self.wind_variant not in {"real", "zero"}:
            raise ValueError("wind_variant must be 'real' or 'zero'")
        if self.wind_relative_motion and (
            self.scene_interaction or self.causal_mode_generation or self.continuous_geometry
        ):
            raise ValueError("C3 wind-relative motion must remain an isolated ASCENT variant")
        if self.wind_relative_motion and not self.normalize_coords:
            raise ValueError("wind-relative motion requires normalized local coordinates")
        if self.mode_state_query and (
            self.scene_interaction
            or self.causal_mode_generation
            or self.continuous_geometry
            or self.wind_relative_motion
        ):
            raise ValueError("mode-state query must remain an isolated C4 variant")
        if self.proper_set and (
            self.scene_interaction
            or self.causal_mode_generation
            or self.continuous_geometry
            or self.wind_relative_motion
            or self.mode_state_query
        ):
            raise ValueError("C15 proper-set mode must remain an isolated ASCENT variant")

        # Encode agent dynamics
        dpr = [x.item() for x in torch.linspace(0, 0.2, self.attn_depth)]
        self.agent_blks = nn.ModuleList(
            Block(
                dim=self.embed_dim,
                num_heads=8,
                mlp_ratio=4.0,
                qkv_bias=False,
                drop_path=dpr[i],
            )
            for i in range(self.attn_depth)
        )

        self.hist_embed = nn.Linear(3*12, self.embed_dim) # unused (legacy)

        # Project to feature space
        self.agent_xyz_proj = Projector(3, self.embed_dim) # unused (legacy)
        self.agent_xy_proj = Projector(2, self.embed_dim)
        self.agent_z_proj = Projector(1, self.embed_dim)
        self.agent_ts_proj = Projector(1, self.embed_dim)

        # Unused (legacy)
        self.scene_blks = nn.ModuleList(
                Block(
                    dim=self.embed_dim,
                    num_heads=8,
                    mlp_ratio=4.0,
                    qkv_bias=False,
                    drop_path=dpr[i],
                )
                for i in range(self.attn_depth)
            )

        #############################################################
        # Decoders & Predictive Heads
        #############################################################

        # Direct 3D Cartesian sequence output head
        self.loc = nn.Sequential(
            nn.Linear(self.embed_dim, 256),
            nn.ReLU(),
            nn.Linear(256, self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, self.future_steps * 3),
        )

        # Kinematic / Flight Parameter prediction heads (e.g. speed, yaw, pitch) for structured trajectory reconstruction.
        # C126 can give each fixed mode an independent physical head. This is
        # deliberately static mode ownership, not a learned router or gate.
        def make_head(output_dim: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(self.embed_dim, 256),
                nn.ReLU(),
                nn.Linear(256, self.embed_dim),
                nn.ReLU(),
                nn.Linear(self.embed_dim, output_dim),
            )

        if self.mode_head_variant == "private":
            self.fp1 = None
            self.fp2 = None
            self.fp3 = None
            self.fp1_heads = nn.ModuleList(
                [make_head(self.future_steps) for _ in range(self.k)]
            )
        else:
            self.fp1 = make_head(self.future_steps)
        fp2_width = (
            self.future_steps
            if self.kinematic_decoder_variant == "cartesian_velocity"
            else self.future_steps * 2
        )
        if self.mode_head_variant == "private":
            self.fp2_heads = nn.ModuleList(
                [make_head(fp2_width) for _ in range(self.k)]
            )
        else:
            self.fp2 = make_head(fp2_width)
        fp3_width = (
            self.future_steps
            if self.kinematic_decoder_variant
            in {"identifiable_horizontal", "cartesian_velocity"}
            else self.future_steps * 2
        )
        if self.mode_head_variant == "private":
            self.fp3_heads = nn.ModuleList(
                [make_head(fp3_width) for _ in range(self.k)]
            )
        else:
            self.fp3 = make_head(fp3_width)

        if self.near_far_physical_decoder:
            if self.pred_step <= 0 or self.future_steps <= 0:
                raise ValueError("near/far decoder requires a regular prediction grid")
            self.near_far_steps = self.near_far_seconds // self.pred_step
            if not 0 < self.near_far_steps < self.future_steps:
                raise ValueError("near/far split must be inside the prediction horizon")
            self.near_steps = self.near_far_steps
            self.far_steps = self.future_steps - self.near_steps
            self.nf_near_fp1 = make_head(self.near_steps)
            self.nf_near_fp2 = make_head(self.near_steps * 2)
            self.nf_near_fp3 = make_head(self.near_steps * 2)
            self.nf_far_fp1 = make_head(self.far_steps)
            self.nf_far_fp2 = make_head(self.far_steps * 2)
            self.nf_far_fp3 = make_head(self.far_steps * 2)

        self.history_capacity_variant = config.get("history_capacity_variant")
        if self.history_capacity_variant not in {None, "mlp"}:
            raise ValueError("unsupported history_capacity_variant")
        if self.history_capacity_variant == "mlp":
            hidden = int(config.get("history_capacity_hidden", self.embed_dim))
            if hidden < 1:
                raise ValueError("history_capacity_hidden must be positive")
            self.history_capacity_mlp = nn.Sequential(
                nn.LayerNorm(self.embed_dim),
                nn.Linear(self.embed_dim, hidden),
                nn.GELU(),
                nn.Linear(hidden, self.embed_dim),
            )

        if self.continuous_geometry:
            self.spline_control_points = int(config.get("spline_control_points", 8))
            self.spline_near_control_points = int(
                config.get("spline_near_control_points", 2)
            )
            predicted_control_points = self.spline_control_points - 1
            if self.spline_control_points <= 3:
                raise ValueError("cubic splines require at least four control points")
            if not 0 < self.spline_near_control_points < predicted_control_points:
                raise ValueError("invalid near/far spline control-point split")
            self.spline_far_control_points = (
                predicted_control_points - self.spline_near_control_points
            )
            self.spline_near_head = nn.Sequential(
                nn.Linear(self.embed_dim, 256),
                nn.ReLU(),
                nn.Linear(256, self.embed_dim),
                nn.ReLU(),
                nn.Linear(self.embed_dim, self.spline_near_control_points * 3),
            )
            self.spline_far_head = nn.Sequential(
                nn.Linear(self.embed_dim, 256),
                nn.ReLU(),
                nn.Linear(256, self.embed_dim),
                nn.ReLU(),
                nn.Linear(self.embed_dim, self.spline_far_control_points * 3),
            )
            self.register_buffer(
                "spline_basis",
                bspline_basis(self.spline_control_points, self.future_steps),
                persistent=True,
            )
            for head in (self.loc, self.fp1, self.fp2, self.fp3):
                for parameter in head.parameters():
                    parameter.requires_grad_(False)

        # Score head for each predicted mode
        if self.proper_set:
            self.pi = None
        else:
            self.pi = nn.Sequential(
                nn.Linear(self.embed_dim, 256),
                nn.ReLU(),
                nn.Linear(256, self.embed_dim),
                nn.ReLU(),
                nn.Linear(self.embed_dim, 1),
            )

        # Positional embedding
        self.pos_embed = nn.Sequential(
            nn.Linear(7, self.embed_dim),
            nn.GELU(),
            nn.Linear(self.embed_dim, self.embed_dim),
        )

        self.norm = nn.LayerNorm(self.embed_dim)

        if self.broadcast_audio_dim:
            self.broadcast_audio_projection = nn.Sequential(
                nn.LayerNorm(self.broadcast_audio_dim),
                nn.Linear(self.broadcast_audio_dim, self.embed_dim),
                nn.ReLU(),
                nn.Linear(self.embed_dim, self.embed_dim),
            )

        if self.aligned_weather_dim:
            self.aligned_weather_projection = nn.Sequential(
                nn.LayerNorm(self.aligned_weather_dim),
                nn.Linear(self.aligned_weather_dim, self.embed_dim),
                nn.GELU(),
                nn.Linear(self.embed_dim, self.embed_dim),
            )
            self.aligned_weather_fusion = nn.Sequential(
                nn.LayerNorm(2 * self.embed_dim),
                nn.Linear(2 * self.embed_dim, self.embed_dim),
                nn.GELU(),
                nn.Linear(self.embed_dim, self.embed_dim),
            )

        # Learnable categorical class tokens
        self.type_embed = nn.Parameter(torch.Tensor(5, self.embed_dim))

        # Mode tokens
        if self.proper_set:
            self.register_buffer(
                "proper_set_seed_bank", regular_simplex(self.k), persistent=True
            )
            self.proper_set_seed_projection = nn.Linear(
                self.k - 1, self.embed_dim, bias=True
            )
        else:
            self.mode1_embed = nn.Parameter(torch.Tensor(self.k, self.embed_dim))

        if self.mode_state_query:
            self.mode_state_decoder = ModeStateQueryDecoder(
                embed_dim=self.embed_dim,
                future_steps=self.future_steps,
                num_heads=int(config.get("mode_state_heads", 4)),
            )

        self.obs = config["obs_len"]
        self.obs_steps = config["obs_steps"]

        if self.scene_interaction:
            self.scene_encoder = RelationalSceneEncoder(
                embed_dim=self.embed_dim,
                num_heads=int(config.get("scene_attention_heads", 4)),
                horizon_seconds=float(config.get("pred_len", 120)),
            )

        self.cache = {}
        self.init_weights()
        # Construct C52 modules only after every baseline parameter has been
        # initialized. Resetting the same seed therefore gives identical
        # shared ASCENT parameters across all history-readout ablations.
        if self.history_encoder_variant == "stable_ssm":
            self.history_sequence_encoder = StableDiagonalHistorySSM(
                dim=self.embed_dim,
                state_size=int(config.get("history_ssm_state_size", 4)),
            )
        elif self.history_encoder_variant == "neural_cde":
            self.history_sequence_encoder = NeuralCDEHistory(
                dim=self.embed_dim,
                control_dim=int(config.get("history_cde_control_dim", 8)),
            )
        elif self.history_encoder_variant == "mode_history_attention":
            self.mode_history_attention = ModeHistoryCrossAttention(
                dim=self.embed_dim,
                heads=int(config.get("history_attention_heads", 4)),
            )
        if self.causal_mode_generation:
            self.causal_mode_decoder = CausalModeMemory(
                embed_dim=self.embed_dim,
                future_steps=self.future_steps,
                step_seconds=int(config["pred_step"]),
                variant=self.causal_mode_variant,
                num_heads=int(config.get("causal_mode_heads", 4)),
            )
        if self.spatiotemporal_history_variant is not None:
            self.spatiotemporal_history_encoder = SpatioTemporalHistoryEncoder(
                dim=self.embed_dim,
                heads=int(config.get("spatiotemporal_heads", 8)),
                depth=int(config.get("spatiotemporal_depth", 2)),
                cross_actor=self.spatiotemporal_history_variant
                == "full_agent_time",
            )

        return


    def init_weights(self):
        """Initializes model parameters using random normal distributions."""
        nn.init.normal_(self.type_embed, std=0.02)
        if self.proper_set:
            nn.init.normal_(self.proper_set_seed_projection.weight, std=0.02)
            nn.init.zeros_(self.proper_set_seed_projection.bias)
        else:
            nn.init.normal_(self.mode1_embed, std=0.02)
        return

    def _mode_embeddings(self) -> torch.Tensor:
        if self.proper_set:
            return self.proper_set_seed_projection(self.proper_set_seed_bank)
        return self.mode1_embed


    def compute_angles(self, actor_feat):
            """Computes the heading (yaw) and elevation (pitch) angles of an actor based on
            the displacement between its last two observed positions.

            This method projects 3D spatial velocity into trigonometric components (sine and cosine)
            to provide a continuous, wrap-around-safe directional encoding for the model.

            Args:
                actor_feat (torch.Tensor): Tensor containing observed actor trajectories of shape
                    (Batch, Time_Steps, 3), where the last dimension represents (X, Y, Z) coordinates.

            Returns:
                tuple: A tuple containing:
                    - actor_angles_sincos (torch.Tensor): Normalized directional features of shape (Batch, 4),
                    ordered as [cos(yaw), sin(yaw), cos(pitch), sin(pitch)].
                    - yaw (torch.Tensor): Raw yaw (azimuth) angles in radians of shape (Batch,).
                    - pitch (torch.Tensor): Raw pitch (elevation) angles in radians of shape (Batch,).
            """
            delta = actor_feat[:, -1, :] - actor_feat[:, -2, :]  # [B, 3] — [dx, dy, dz]
            # Compute yaw (azimuth) angle in XY plane
            yaw = torch.atan2(delta[:, 1], delta[:, 0])  # atan2(dy, dx)
            # Compute pitch (elevation) angle in Z relative to XY plane
            horizontal_norm = torch.norm(delta[:, :2], dim=-1) + 1e-8  # sqrt(dx^2 + dy^2)
            pitch = torch.atan2(delta[:, 2], horizontal_norm)  # atan2(dz, √(dx² + dy²))
            # Final directional encoding: sin/cos for yaw and pitch
            yaw_sincos = torch.stack([torch.cos(yaw), torch.sin(yaw)], dim=-1)         # [B, 2]
            pitch_sincos = torch.stack([torch.cos(pitch), torch.sin(pitch)], dim=-1)  # [B, 2]
            # Concatenate for final representation: [cos(yaw), sin(yaw), cos(pitch), sin(pitch)]
            actor_angles_sincos = torch.cat([yaw_sincos, pitch_sincos], dim=-1)  # [B, 4]
            return actor_angles_sincos, yaw, pitch

    def _wind_vector(self, data: dict, batch: int, dtype: torch.dtype) -> torch.Tensor:
        if "context" not in data:
            raise KeyError("wind-relative ASCENT requires data['context']")
        context = data["context"]
        if context.ndim != 3 or context.shape[1] != batch or context.shape[2] != 2:
            raise ValueError("context must have shape [observation_steps, batch, 2]")
        wind = context[-1].to(dtype=dtype)
        if self.wind_variant == "zero":
            wind = torch.zeros_like(wind)
        return wind

    def _remove_observed_wind(
        self, actor_feat: torch.Tensor, wind: torch.Tensor
    ) -> torch.Tensor:
        """Convert ground positions to an air-relative history in physical units."""
        elapsed = (
            torch.arange(actor_feat.shape[1], device=actor_feat.device, dtype=actor_feat.dtype)
            - (actor_feat.shape[1] - 1)
        ) * float(self.obs_steps)
        adjusted = actor_feat.clone()
        adjusted[..., :2] = adjusted[..., :2] - (
            wind[:, None, :] * elapsed[None, :, None] / 1000.0
        )
        return adjusted

    def _restore_future_wind(
        self, trajectories: torch.Tensor, wind: torch.Tensor
    ) -> torch.Tensor:
        """Convert air-relative global predictions back to ground trajectories."""
        elapsed = torch.arange(
            1,
            self.future_steps + 1,
            device=trajectories.device,
            dtype=trajectories.dtype,
        ) * float(self.pred_step)
        drift = trajectories.new_zeros((wind.shape[0], self.future_steps, 3))
        drift[..., :2] = wind[:, None, :] * elapsed[None, :, None] / 1000.0
        return trajectories + drift[:, None]

    def _flight_parameters_from_feature(self, feature: torch.Tensor) -> torch.Tensor:
        """Decode one complete parameterized trajectory from one mode feature."""
        batch = feature.shape[0]
        speed = self.fp1(feature)
        if self.kinematic_decoder_variant == "positive_speed_pitch":
            speed = F.softplus(speed)
        heading_raw = self.fp2(feature).view(batch, self.future_steps, 2)
        heading_norm = torch.sqrt(heading_raw.square().sum(dim=-1) + 1e-8)
        heading = torch.atan2(
            heading_raw[..., 0] / heading_norm,
            heading_raw[..., 1] / heading_norm,
        )
        pitch_raw = self.fp3(feature).view(batch, self.future_steps, 2)
        pitch_norm = torch.sqrt(pitch_raw.square().sum(dim=-1) + 1e-8)
        pitch = torch.atan2(
            pitch_raw[..., 0] / pitch_norm,
            pitch_raw[..., 1] / pitch_norm,
        )
        return torch.stack([speed, heading, pitch], dim=-1)

    def _apply_mode_head(self, features: torch.Tensor, name: str) -> torch.Tensor:
        """Apply a shared or fixed mode-owned head to ``[B,K,D]`` features."""
        if self.mode_head_variant == "shared":
            return getattr(self, name)(features)
        heads = getattr(self, f"{name}_heads")
        return torch.stack(
            [head(features[:, mode]) for mode, head in enumerate(heads)], dim=1
        )

    def _decode_causal_modes(
        self,
        actor_context: torch.Tensor,
        actor_centers: torch.Tensor,
        yaw: torch.Tensor,
        pitch: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        latent_memory: list[torch.Tensor] = []
        physical_memory: list[torch.Tensor] = []
        local_trajectories = []
        flight_parameters = []
        mode_features = []
        mode_logits = []

        for mode_index in range(self.k):
            query = actor_context + self._mode_embeddings()[mode_index]
            feature = self.causal_mode_decoder(
                query, latent_memory, physical_memory
            )
            parameters = self._flight_parameters_from_feature(feature)
            local_trajectory = flight_params_to_pos(
                parameters, torch.zeros_like(actor_centers)
            )[:, 1:]

            mode_features.append(feature)
            mode_logits.append(self.pi(feature).squeeze(-1))
            flight_parameters.append(parameters)
            local_trajectories.append(local_trajectory)
            latent_memory.append(feature)
            physical_memory.append(self.causal_mode_decoder.encode_physical(
                local_trajectory.detach(), parameters.detach()
            ))

        features = torch.stack(mode_features, dim=1)
        logits = torch.stack(mode_logits, dim=1)
        parameters = torch.stack(flight_parameters, dim=1)
        local = torch.stack(local_trajectories, dim=1)
        if self.normalize_coords:
            trajectories = local.clone()
            for mode_index in range(self.k):
                trajectories[:, mode_index] = ptsToGlobal(
                    actor_centers, yaw, pitch, local[:, mode_index]
                )
        else:
            trajectories = self.loc(features).view(
                actor_context.shape[0], self.k, self.future_steps, 3
            )
        auxiliary = {
            "variant": self.causal_mode_variant,
            "physical_anchor_indices": self.causal_mode_decoder.anchor_indices,
            "memory_modes": self.k - 1,
        }
        return trajectories, logits, parameters, features, auxiliary

    def _decode_continuous_geometry(
        self,
        features: torch.Tensor,
        actor_centers: torch.Tensor,
        yaw: torch.Tensor,
        pitch: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        """Directly decode one complete clamped cubic spline per mode query."""
        batch, modes, _ = features.shape
        near = self.spline_near_head(features).view(
            batch, modes, self.spline_near_control_points, 3
        )
        far = self.spline_far_head(features).view(
            batch, modes, self.spline_far_control_points, 3
        )
        controls = features.new_zeros(
            (batch, modes, self.spline_control_points, 3)
        )
        controls[:, :, 1 : 1 + self.spline_near_control_points] = near
        controls[:, :, 1 + self.spline_near_control_points :] = far
        local = reconstruct_curve(controls, self.spline_basis)
        trajectories = local.clone()
        for mode_index in range(self.k):
            trajectories[:, mode_index] = ptsToGlobal(
                actor_centers, yaw, pitch, local[:, mode_index]
            )
        return trajectories, {
            "representation": "clamped_cubic_bspline",
            "control_points": controls,
            "near_control_points": self.spline_near_control_points,
            "far_control_points": self.spline_far_control_points,
        }


    def _decode_near_far_physical(
        self,
        feat: torch.Tensor,
        actor_centers: torch.Tensor,
        yaw: torch.Tensor,
        pitch: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Decode two fixed physical intervals with a deterministic boundary state."""
        batch, modes, _ = feat.shape

        def decode_interval(
            features: torch.Tensor,
            fp1_head: nn.Module,
            fp2_head: nn.Module,
            fp3_head: nn.Module,
            steps: int,
            initial: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            speed = F.softplus(fp1_head(features))
            heading_raw = fp2_head(features).view(batch, modes, steps, 2)
            heading_norm = torch.sqrt(heading_raw.square().sum(dim=-1) + 1e-8)
            heading = torch.atan2(
                heading_raw[..., 0] / heading_norm,
                heading_raw[..., 1] / heading_norm,
            )
            pitch_raw = fp3_head(features).view(batch, modes, steps, 2)
            pitch_norm = torch.sqrt(pitch_raw.square().sum(dim=-1) + 1e-8)
            pitch_angle = torch.atan2(
                pitch_raw[..., 0] / pitch_norm,
                pitch_raw[..., 1] / pitch_norm,
            )
            flight_params = torch.stack([speed, heading, pitch_angle], dim=-1)
            positions = flight_params_to_pos(
                flight_params.reshape(batch * modes, steps, 3),
                initial.reshape(batch * modes, 3),
            ).reshape(batch, modes, steps + 1, 3)[:, :, 1:]
            return positions, flight_params, speed

        zero = torch.zeros_like(actor_centers[:, None].expand(batch, modes, 3))
        near_local, near_params, near_speed = decode_interval(
            feat,
            self.nf_near_fp1,
            self.nf_near_fp2,
            self.nf_near_fp3,
            self.near_steps,
            zero,
        )
        boundary = near_local[:, :, -1]
        far_local, far_params, far_speed = decode_interval(
            feat,
            self.nf_far_fp1,
            self.nf_far_fp2,
            self.nf_far_fp3,
            self.far_steps,
            boundary,
        )
        local = torch.cat([near_local, far_local], dim=2)
        trajectories = local.clone()
        for mode_index in range(self.k):
            trajectories[:, mode_index] = ptsToGlobal(
                actor_centers, yaw, pitch, local[:, mode_index]
            )
        return trajectories, {
            "near_flight_params": near_params,
            "far_flight_params": far_params,
            "near_horizontal_control": near_speed,
            "far_horizontal_control": far_speed,
            "boundary_local_state": boundary,
            "near_steps": self.near_steps,
            "far_steps": self.far_steps,
            "transition": "fixed_kinematic_boundary_at_60_seconds",
        }

    def forward(self, data):
        """Executes full multi-modal trajectory inference.

        Args:
            data (dict): Dictionary comprising tracking data elements:
                - "obs_traj": Observed tracking tensor (Steps, Batch, 3)
                - "context": Ambient frame features or states (Steps, Batch, Channels)
                - "adj": System adjacency or connectivity maps.

        Returns:
            tuple: Contains:
                - loc (torch.Tensor): Final absolute trajectories [B, k, future_steps, 3]
                - pi (torch.Tensor): Scored probability distribution across tracks [B, k]
                - dict: Diagnostic metadata (flight parameters, centers, yaw offsets).
        """
        # Adjust dimensions for internal tracking structures
        actor_feat = torch.transpose(data["obs_traj"], 1, 0)

        B = actor_feat.shape[0]
        ts = torch.arange(actor_feat.shape[1]).view(1, -1, 1).repeat(actor_feat.shape[0], 1, 1).to(actor_feat.device).float()

        ### ACTOR ENCODING ###
        actor_centers = actor_feat[:, -1]
        wind = None
        if self.wind_relative_motion:
            wind = self._wind_vector(data, B, actor_feat.dtype)
            actor_feat = self._remove_observed_wind(actor_feat, wind)
        global_history = actor_feat.clone()
        actor_velocities = (
            actor_feat[:, -1] - actor_feat[:, -2]
        ) / float(self.obs_steps)

        # Positional Embedding
        actor_angles_sincos, yaw, pitch = self.compute_angles(actor_feat)
        pos_feat = torch.cat([actor_centers, actor_angles_sincos], dim=-1)
        pos_embed = self.pos_embed(pos_feat)

        # Coordinate transformation to egocentric local frame
        if self.normalize_coords:
            actor_feat = ptsToLocal(actor_centers, yaw, pitch, actor_feat).float() # actor_feat - actor_centers.unsqueeze(1) #

        # Route variables through specialized sub-dimension MLPs or unified 3D blocks
        actor_feat = self.agent_xy_proj(actor_feat[..., :2]) + self.agent_z_proj(actor_feat[..., 2].unsqueeze(-1))

        # Inject timeline markers and process through Agent Attention blocks
        actor_feat += self.agent_ts_proj(ts)
        for blk in self.agent_blks:
            actor_feat = blk(actor_feat)
        history_tokens = actor_feat
        spatiotemporal_aux = None
        if self.spatiotemporal_history_variant is not None:
            if "adj" not in data:
                raise KeyError("spatiotemporal ASCENT requires data['adj']")
            actor_feat, spatiotemporal_aux = self.spatiotemporal_history_encoder(
                history_tokens, global_history, data["adj"]
            )
        elif self.history_encoder_variant in {"stable_ssm", "neural_cde"}:
            actor_feat = self.history_sequence_encoder(history_tokens)
        else:
            actor_feat = torch.max(history_tokens, axis=1).values
        if self.history_capacity_variant == "mlp":
            actor_feat = self.history_capacity_mlp(actor_feat)

        # Layer semantic identifier and global spatial biases
        actor_types = self.type_embed[0].unsqueeze(0)
        actor_feat = actor_feat + actor_types
        if self.global_pos_embedding and self.normalize_coords:
            actor_feat += pos_embed

        mode_history_tokens = None
        if self.history_encoder_variant == "mode_history_attention":
            mode_history_tokens = history_tokens + actor_types[:, None]
            if self.global_pos_embedding and self.normalize_coords:
                mode_history_tokens = mode_history_tokens + pos_embed[:, None]

        broadcast_audio = None
        if self.broadcast_audio_dim:
            if "broadcast_audio" not in data:
                raise KeyError("audio-conditioned ASCENT requires data['broadcast_audio']")
            broadcast_audio = data["broadcast_audio"]
            if broadcast_audio.ndim != 2 or broadcast_audio.shape != (
                B,
                self.broadcast_audio_dim,
            ):
                raise ValueError(
                    "broadcast_audio must have shape "
                    f"[{B}, {self.broadcast_audio_dim}]"
                )
            actor_feat = actor_feat + self.broadcast_audio_projection(
                broadcast_audio.to(dtype=actor_feat.dtype)
            )

        aligned_weather = None
        if self.aligned_weather_dim:
            if "aligned_weather" not in data:
                raise KeyError("aligned-weather ASCENT requires data['aligned_weather']")
            aligned_weather = data["aligned_weather"]
            if aligned_weather.ndim != 3:
                raise ValueError("aligned_weather must have shape [T,B,C] or [B,T,C]")
            if aligned_weather.shape[0] == B and aligned_weather.shape[2] == self.aligned_weather_dim:
                weather_batch = aligned_weather
            elif aligned_weather.shape[1] == B and aligned_weather.shape[2] == self.aligned_weather_dim:
                weather_batch = aligned_weather.transpose(0, 1)
            else:
                raise ValueError(
                    "aligned_weather must have batch dimension matching obs_traj and "
                    f"last dimension {self.aligned_weather_dim}"
                )
            weather_context = self.aligned_weather_projection(
                weather_batch.to(dtype=actor_feat.dtype)
            ).mean(dim=1)
            actor_feat = self.aligned_weather_fusion(
                torch.cat((actor_feat, weather_context), dim=-1)
            )

        scene_aux = None
        if self.scene_interaction and self.scene_interaction_stage == "actor":
            if "adj" not in data:
                raise KeyError("scene-conditioned ASCENT requires data['adj']")
            actor_feat, scene_aux = self.scene_encoder(
                actor_feat,
                actor_centers,
                actor_velocities,
                yaw,
                pitch,
                data["adj"],
            )

        ### SCENE ENCODING ###
        feat = torch.stack([actor_feat], dim=1)
        feat = self.norm(feat)

        ### DECODING ###
        feat = feat[:, 0]
        actor_context = feat

        if self.causal_mode_generation:
            loc, pi, flight_params, feat, causal_aux = self._decode_causal_modes(
                feat, actor_centers, yaw, pitch
            )
            return loc, pi, {
                "flight_params": flight_params,
                "actor_centers": actor_centers,
                "actor_angles": yaw,
                "actor_pitch": pitch,
                "actor_context": actor_context,
                "mode_features": feat,
                "causal_mode": causal_aux,
                **(
                    {"spatiotemporal_history": spatiotemporal_aux}
                    if spatiotemporal_aux is not None
                    else {}
                ),
            }

        # Broadcast feature embeddings to multimodal query tokens for parallel mode-specific decoding
        modes1 = self._mode_embeddings().view(1, self.k, self.embed_dim).repeat(B, 1, 1)
        feat = feat.unsqueeze(1).repeat(1, self.k, 1) + modes1

        history_attention_weights = None
        if self.history_encoder_variant == "mode_history_attention":
            feat, history_attention_weights = self.mode_history_attention(
                feat, mode_history_tokens
            )

        if self.scene_interaction and self.scene_interaction_stage == "mode":
            if "adj" not in data:
                raise KeyError("joint mode-query ASCENT requires data['adj']")
            conditioned_modes = []
            mode_auxiliary = []
            for mode_index in range(self.k):
                conditioned, auxiliary = self.scene_encoder(
                    feat[:, mode_index],
                    actor_centers,
                    actor_velocities,
                    yaw,
                    pitch,
                    data["adj"],
                )
                conditioned_modes.append(conditioned)
                mode_auxiliary.append(auxiliary)
            feat = torch.stack(conditioned_modes, dim=1)
            scene_aux = {
                "scene_sizes": mode_auxiliary[0]["scene_sizes"],
                "interaction_mask": mode_auxiliary[0]["interaction_mask"],
                "attention_entropy": torch.stack(
                    [item["attention_entropy"] for item in mode_auxiliary], dim=1
                ).mean(dim=1),
            }

        # Flight parameter prediction heads and score head
        pi = (
            feat.new_zeros((B, self.k))
            if self.proper_set
            else self.pi(feat).squeeze(dim=-1)
        )
        scene_mode_logits = None
        if self.scene_shared_logits:
            _, inverse, counts = torch.unique(
                data["adj"].to(torch.long),
                sorted=True,
                return_inverse=True,
                return_counts=True,
            )
            scene_mode_logits = pi.new_zeros((counts.shape[0], self.k))
            scene_mode_logits.index_add_(0, inverse, pi)
            scene_mode_logits = scene_mode_logits / counts[:, None]
            pi = scene_mode_logits[inverse]

        if self.near_far_physical_decoder:
            loc, near_far_aux = self._decode_near_far_physical(
                feat, actor_centers, yaw, pitch
            )
            auxiliary = {
                "actor_centers": actor_centers,
                "actor_angles": yaw,
                "actor_pitch": pitch,
                "actor_context": actor_context,
                "mode_features": feat,
                "horizontal_control": torch.cat(
                    [
                        near_far_aux["near_horizontal_control"],
                        near_far_aux["far_horizontal_control"],
                    ],
                    dim=-1,
                ),
                "near_far_physical": near_far_aux,
                "kinematic_decoder": {
                    "variant": self.kinematic_decoder_variant,
                    "near_far_physical_decoder": True,
                    "trajectory_residual": False,
                    "control_residual": False,
                    "learned_gate": False,
                    "token_codebook": False,
                    "future_autoregression": False,
                    "post_generation_selector": False,
                },
            }
            if scene_aux is not None:
                auxiliary["scene_interaction"] = scene_aux
            if scene_mode_logits is not None:
                auxiliary["scene_mode_logits"] = scene_mode_logits
            if spatiotemporal_aux is not None:
                auxiliary["spatiotemporal_history"] = spatiotemporal_aux
            return loc, pi, auxiliary

        if self.continuous_geometry:
            loc, spline_aux = self._decode_continuous_geometry(
                feat, actor_centers, yaw, pitch
            )
            auxiliary = {
                "actor_centers": actor_centers,
                "actor_angles": yaw,
                "actor_pitch": pitch,
                "actor_context": actor_context,
                "mode_features": feat,
                "continuous_geometry": spline_aux,
            }
            if scene_aux is not None:
                auxiliary["scene_interaction"] = scene_aux
            if scene_mode_logits is not None:
                auxiliary["scene_mode_logits"] = scene_mode_logits
            if spatiotemporal_aux is not None:
                auxiliary["spatiotemporal_history"] = spatiotemporal_aux
            return loc, pi, auxiliary

        if self.mode_state_query:
            flight_params, state_features = self.mode_state_decoder(feat)
            init_pos = torch.zeros_like(
                actor_centers.unsqueeze(1).repeat(1, self.k, 1).view(-1, 3)
            )
            local = flight_params_to_pos(
                flight_params.view(B * self.k, -1, 3), init_pos
            ).view(B, self.k, -1, 3)[:, :, 1:]
            loc = local.clone()
            if self.normalize_coords:
                for mode_index in range(self.k):
                    loc[:, mode_index] = ptsToGlobal(
                        actor_centers, yaw, pitch, local[:, mode_index]
                    )
            else:
                loc = self.loc(feat).view(B, self.k, self.future_steps, 3)
            auxiliary = {
                "flight_params": flight_params,
                "actor_centers": actor_centers,
                "actor_angles": yaw,
                "actor_pitch": pitch,
                "actor_context": actor_context,
                "mode_features": feat,
                "state_features": state_features,
                "mode_state_query": {
                    "future_steps": self.future_steps,
                    "query_factorization": "mode_intention_plus_horizon_state",
                },
            }
            if spatiotemporal_aux is not None:
                auxiliary["spatiotemporal_history"] = spatiotemporal_aux
            return loc, pi, auxiliary

        fp1 = self._apply_mode_head(feat, "fp1")
        fp2_raw = self._apply_mode_head(feat, "fp2")
        fp3_raw = self._apply_mode_head(feat, "fp3")

        if self.kinematic_decoder_variant == "cartesian_velocity":
            increments = torch.stack([fp1, fp2_raw, fp3_raw], dim=-1)
            flight_params = increments
            horizontal_control = torch.linalg.vector_norm(
                increments[..., :2], dim=-1
            )
            loc = torch.cumsum(increments, dim=-2)
        else:
            heading_raw = fp2_raw.view(B, self.k, -1, 2)
            heading_norm = torch.sqrt(heading_raw.square().sum(dim=-1) + 1e-8)
            heading = torch.atan2(
                heading_raw[..., 0] / heading_norm,
                heading_raw[..., 1] / heading_norm,
            )
            if self.kinematic_decoder_variant == "identifiable_horizontal":
                horizontal_speed = F.softplus(fp1)
                vertical_rate = fp3_raw
                increments = torch.stack(
                    [
                        horizontal_speed * torch.cos(heading),
                        horizontal_speed * torch.sin(heading),
                        vertical_rate,
                    ],
                    dim=-1,
                )
                flight_params = torch.stack(
                    [horizontal_speed, heading, vertical_rate], dim=-1
                )
                horizontal_control = horizontal_speed
                loc = torch.cumsum(increments, dim=-2)
            else:
                pitch_raw = fp3_raw.view(B, self.k, -1, 2)
                pitch_norm = torch.sqrt(pitch_raw.square().sum(dim=-1) + 1e-8)
                pitch_angle = torch.atan2(
                    pitch_raw[..., 0] / pitch_norm,
                    pitch_raw[..., 1] / pitch_norm,
                )
                speed = (
                    F.softplus(fp1)
                    if self.kinematic_decoder_variant == "positive_speed_pitch"
                    else fp1
                )
                flight_params = torch.stack([speed, heading, pitch_angle], dim=-1)
                horizontal_control = speed
                init_pos = torch.zeros_like(
                    actor_centers.unsqueeze(1).repeat(1, self.k, 1).view(-1, 3)
                )
                loc = flight_params_to_pos(
                    flight_params.view(B * self.k, -1, 3), init_pos
                ).view(B, self.k, -1, 3)[:, :, 1:]

        # Project coordinate fields back into global space or construct spatial projections
        if self.normalize_coords:
            for k in range(self.k):
                loc[:, k] = ptsToGlobal(actor_centers, yaw, pitch, loc[:, k])
        else:
            loc = self.loc(feat).view(B, self.k, self.future_steps, 3)
        if wind is not None:
            loc = self._restore_future_wind(loc, wind)

        auxiliary = {
            "flight_params": flight_params,
            "horizontal_control": horizontal_control,
            "actor_centers": actor_centers,
            "actor_angles": yaw,
            "actor_pitch": pitch,
            "actor_context": actor_context,
            # Exposed for post-generation ranking; this does not change checkpoint parameters.
            "mode_features": feat,
            "kinematic_decoder": {
                "variant": self.kinematic_decoder_variant,
                "mode_head_variant": self.mode_head_variant,
                "trajectory_residual": False,
                "learned_gate": False,
                "future_autoregression": False,
                "post_generation_selector": False,
            },
        }
        if self.history_encoder_variant != "max_pool":
            auxiliary["history_encoder"] = {
                "variant": self.history_encoder_variant,
                "trajectory_residual": False,
                "sample_router": False,
            }
            if history_attention_weights is not None:
                auxiliary["history_encoder"]["attention_weights"] = (
                    history_attention_weights
                )
        if broadcast_audio is not None:
            auxiliary["broadcast_audio"] = {
                "conditioning": "additive_scene_broadcast_after_actor_encoding",
                "input_dimension": self.broadcast_audio_dim,
                "speaker_assignment": False,
                "routing_gate": False,
            }
        if self.proper_set:
            auxiliary["proper_set"] = {
                "equal_weight": True,
                "seed_geometry": "regular_simplex",
                "probability_head": False,
            }
        if wind is not None:
            auxiliary["wind_relative_motion"] = {
                "variant": self.wind_variant,
                "wind_mps": wind,
                "conversion": "mps_times_seconds_div_1000_to_km",
            }
        if scene_aux is not None:
            auxiliary["scene_interaction"] = scene_aux
        if scene_mode_logits is not None:
            auxiliary["scene_mode_logits"] = scene_mode_logits
        if spatiotemporal_aux is not None:
            auxiliary["spatiotemporal_history"] = spatiotemporal_aux
        return loc, pi, auxiliary
