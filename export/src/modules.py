import torch
from torch import nn

from sam2.modeling.sam2_base import SAM2Base
from sam2.modeling.sam2_utils import get_1d_sine_pe


class ImageEncoder(nn.Module):
    def __init__(self, sam_model) -> None:
        super().__init__()
        self.model = sam_model
        self.no_mem_embed = sam_model.no_mem_embed
        self.image_encoder = sam_model.image_encoder
        self.prepare_backbone_features = sam_model._prepare_backbone_features

    @torch.no_grad()
    def forward(self, image: torch.Tensor):
        backbone_out = self.image_encoder(image)

        backbone_out["backbone_fpn"][0] = self.model.sam_mask_decoder.conv_s0(
            backbone_out["backbone_fpn"][0]
        )
        backbone_out["backbone_fpn"][1] = self.model.sam_mask_decoder.conv_s1(
            backbone_out["backbone_fpn"][1]
        )

        vision_pos_enc = backbone_out["vision_pos_enc"]
        backbone_fpn = backbone_out["backbone_fpn"]
        pix_feat = backbone_out["vision_features"]

        for i in range(len(backbone_fpn)):
            if backbone_fpn[i].dim() == 3:
                backbone_fpn[i] = backbone_fpn[i].unsqueeze(0)
        for i in range(len(vision_pos_enc)):
            if vision_pos_enc[i].dim() == 3:
                vision_pos_enc[i] = vision_pos_enc[i].unsqueeze(0)

        _, current_vision_feats, current_vision_pos_embeds, _ = self.prepare_backbone_features(
            {
                "backbone_fpn": backbone_fpn,
                "vision_pos_enc": vision_pos_enc,
            }
        )

        current_vision_feat = current_vision_feats[-1] + self.no_mem_embed
        current_vision_feat2 = current_vision_feat.reshape(64, 64, 1, 256).permute(2, 3, 0, 1)

        high_res_features_0 = current_vision_feats[0].reshape(256, 256, 1, 32).permute(2, 3, 0, 1)
        high_res_features_1 = current_vision_feats[1].reshape(128, 128, 1, 64).permute(2, 3, 0, 1)

        return (
            pix_feat,
            high_res_features_0,
            high_res_features_1,
            current_vision_feat2,
            current_vision_pos_embeds[-1],
        )


class _DecoderBase(nn.Module):
    def __init__(self, sam_model: SAM2Base) -> None:
        super().__init__()
        self.model = sam_model
        self.sigmoid_scale_for_mem_enc = getattr(sam_model, "sigmoid_scale_for_mem_enc", 1.0)
        self.sigmoid_bias_for_mem_enc = getattr(sam_model, "sigmoid_bias_for_mem_enc", 0.0)

    @staticmethod
    def _point_inputs(point_coords: torch.Tensor, point_labels: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "point_coords": point_coords,
            "point_labels": point_labels,
        }

    def _decode(
        self,
        image_embed: torch.Tensor,
        high_res_feats_0: torch.Tensor,
        high_res_feats_1: torch.Tensor,
        point_inputs: dict[str, torch.Tensor] | None,
        single_token_obj_ptr: bool = False,
        multimask_output: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        high_res_feats = [high_res_feats_0, high_res_feats_1]
        decoder = self.model.sam_mask_decoder
        orig_multimask_obj_ptr = getattr(decoder, "use_multimask_token_for_obj_ptr", False)
        if single_token_obj_ptr:
            decoder.use_multimask_token_for_obj_ptr = False
        try:
            (
                _,
                _,
                _,
                low_res_masks,
                high_res_masks,
                obj_ptr,
                _,
            ) = self.model._forward_sam_heads(
                backbone_features=image_embed,
                point_inputs=point_inputs,
                mask_inputs=None,
                high_res_features=high_res_feats,
                multimask_output=multimask_output,
            )
        finally:
            if single_token_obj_ptr:
                decoder.use_multimask_token_for_obj_ptr = orig_multimask_obj_ptr

        mask_for_mem = torch.sigmoid(high_res_masks)
        mask_for_mem = mask_for_mem * self.sigmoid_scale_for_mem_enc
        mask_for_mem = mask_for_mem + self.sigmoid_bias_for_mem_enc
        pred_mask = low_res_masks
        return obj_ptr, mask_for_mem, pred_mask


class ImageDecoder(_DecoderBase):
    @torch.no_grad()
    def forward(
        self,
        point_coords: torch.Tensor,
        point_labels: torch.Tensor,
        image_embed: torch.Tensor,
        high_res_feats_0: torch.Tensor,
        high_res_feats_1: torch.Tensor,
    ):
        point_inputs = self._point_inputs(point_coords, point_labels)
        return self._decode(image_embed, high_res_feats_0, high_res_feats_1, point_inputs)


class ImageDecoderPredMask(_DecoderBase):
    @torch.no_grad()
    def forward(
        self,
        point_coords: torch.Tensor,
        point_labels: torch.Tensor,
        image_embed: torch.Tensor,
        high_res_feats_0: torch.Tensor,
        high_res_feats_1: torch.Tensor,
    ) -> torch.Tensor:
        point_inputs = self._point_inputs(point_coords, point_labels)
        _, _, pred_mask = self._decode(image_embed, high_res_feats_0, high_res_feats_1, point_inputs)
        return pred_mask


class VideoDecoderInit(_DecoderBase):
    @torch.no_grad()
    def forward(
        self,
        point_coords: torch.Tensor,
        point_labels: torch.Tensor,
        image_embed: torch.Tensor,
        high_res_feats_0: torch.Tensor,
        high_res_feats_1: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        point_inputs = self._point_inputs(point_coords, point_labels)
        obj_ptr, mask_for_mem, pred_mask = self._decode(
            image_embed,
            high_res_feats_0,
            high_res_feats_1,
            point_inputs,
            single_token_obj_ptr=True,
            multimask_output=True,
        )
        return obj_ptr, mask_for_mem, pred_mask


class VideoDecoderPropagate(_DecoderBase):
    @torch.no_grad()
    def forward(
        self,
        image_embed: torch.Tensor,
        high_res_feats_0: torch.Tensor,
        high_res_feats_1: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        obj_ptr, mask_for_mem, pred_mask = self._decode(
            image_embed,
            high_res_feats_0,
            high_res_feats_1,
            point_inputs=None,
            single_token_obj_ptr=True,
            multimask_output=True,
        )
        return obj_ptr, mask_for_mem, pred_mask


class _MemAttentionBase(nn.Module):
    def __init__(self, sam_model: SAM2Base) -> None:
        super().__init__()
        self.model = sam_model
        self.memory_attention = sam_model.memory_attention
        self.no_mem_embed = sam_model.no_mem_embed

    def _prepare_current_features(
        self, current_vision_feat: torch.Tensor
    ) -> tuple[torch.Tensor, int, int, int, int]:
        batch, channels, height, width = current_vision_feat.shape
        feat_hwbc = current_vision_feat.permute(2, 3, 0, 1).reshape(height * width, batch, channels)
        feat_hwbc = feat_hwbc - self.no_mem_embed
        return feat_hwbc, batch, channels, height, width

    @staticmethod
    def _flatten_memory_bank(memory_1: torch.Tensor) -> torch.Tensor:
        mem_frames = memory_1.shape[0]
        memory_1 = memory_1.reshape(mem_frames, 64, 64 * 64)
        memory_1 = memory_1.permute(0, 2, 1)
        return memory_1.reshape(-1, 1, 64)

    @staticmethod
    def _flatten_object_pointers(memory_0: torch.Tensor) -> tuple[torch.Tensor, int]:
        num_obj_ptr = memory_0.shape[0]
        memory_0 = memory_0.reshape(num_obj_ptr, 4, 64)
        memory_0 = memory_0.unsqueeze(1)
        memory_0 = memory_0.permute(0, 2, 1, 3)
        memory_0 = memory_0.reshape(num_obj_ptr * 4, 1, 64)
        return memory_0, num_obj_ptr * 4


class MemAttention(_MemAttentionBase):
    @torch.no_grad()
    def forward(
        self,
        current_vision_feat: torch.Tensor,
        current_vision_pos_embed: torch.Tensor,
        memory_0: torch.Tensor,
        memory_1: torch.Tensor,
        memory_pos_embed: torch.Tensor,
    ) -> torch.Tensor:
        feat_hwbc, batch, channels, height, width = self._prepare_current_features(current_vision_feat)

        memory_0, num_obj_ptr_tokens = self._flatten_object_pointers(memory_0)
        memory_1 = self._flatten_memory_bank(memory_1)
        memory = torch.cat((memory_1, memory_0), dim=0)

        fused_hwbc = self.memory_attention(
            curr=feat_hwbc,
            curr_pos=current_vision_pos_embed,
            memory=memory,
            memory_pos=memory_pos_embed,
            num_obj_ptr_tokens=num_obj_ptr_tokens,
        )

        return fused_hwbc.permute(1, 2, 0).reshape(batch, channels, height, width)


class MemAttentionNoObjPtr(_MemAttentionBase):
    @torch.no_grad()
    def forward(
        self,
        current_vision_feat: torch.Tensor,
        current_vision_pos_embed: torch.Tensor,
        memory_1: torch.Tensor,
        memory_pos_embed: torch.Tensor,
    ) -> torch.Tensor:
        feat_hwbc, batch, channels, height, width = self._prepare_current_features(current_vision_feat)
        memory_1 = self._flatten_memory_bank(memory_1)

        fused_hwbc = self.memory_attention(
            curr=feat_hwbc,
            curr_pos=current_vision_pos_embed,
            memory=memory_1,
            memory_pos=memory_pos_embed,
            num_obj_ptr_tokens=0,
        )

        return fused_hwbc.permute(1, 2, 0).reshape(batch, channels, height, width)


class MemAttentionObjPtr(_MemAttentionBase):
    def __init__(self, sam_model: SAM2Base) -> None:
        super().__init__(sam_model)
        self.mem_dim = sam_model.mem_dim
        self.max_obj_ptrs_in_encoder = getattr(sam_model, "max_obj_ptrs_in_encoder", 16)
        self.add_tpos_enc_to_obj_ptrs = getattr(sam_model, "add_tpos_enc_to_obj_ptrs", True)
        self.proj_tpos_enc_in_obj_ptrs = getattr(sam_model, "proj_tpos_enc_in_obj_ptrs", False)
        self.obj_ptr_tpos_proj = sam_model.obj_ptr_tpos_proj

    @torch.no_grad()
    def forward(
        self,
        current_vision_feat: torch.Tensor,
        current_vision_pos_embed: torch.Tensor,
        memory_0: torch.Tensor,
        obj_ptr_offsets: torch.Tensor,
        memory_1: torch.Tensor,
        memory_pos_embed: torch.Tensor,
    ) -> torch.Tensor:
        feat_hwbc, batch, channels, height, width = self._prepare_current_features(current_vision_feat)

        num_obj_ptr = memory_0.shape[0]
        if self.add_tpos_enc_to_obj_ptrs:
            t_diff_max = max(1, self.max_obj_ptrs_in_encoder - 1)
            tpos_dim = channels if self.proj_tpos_enc_in_obj_ptrs else self.mem_dim
            obj_pos = get_1d_sine_pe(obj_ptr_offsets / t_diff_max, dim=tpos_dim)
            obj_pos = self.obj_ptr_tpos_proj(obj_pos)
            obj_pos = obj_pos.unsqueeze(1).expand(-1, batch, self.mem_dim)
        else:
            obj_pos = memory_0.new_zeros((num_obj_ptr, batch, self.mem_dim))

        memory_0, num_obj_ptr_tokens = self._flatten_object_pointers(memory_0)
        if self.mem_dim < channels:
            obj_pos = obj_pos.repeat_interleave(channels // self.mem_dim, dim=0)

        memory_1 = self._flatten_memory_bank(memory_1)
        memory = torch.cat((memory_1, memory_0), dim=0)
        memory_pos = torch.cat((memory_pos_embed, obj_pos), dim=0)

        fused_hwbc = self.memory_attention(
            curr=feat_hwbc,
            curr_pos=current_vision_pos_embed,
            memory=memory,
            memory_pos=memory_pos,
            num_obj_ptr_tokens=num_obj_ptr_tokens,
        )

        return fused_hwbc.permute(1, 2, 0).reshape(batch, channels, height, width)


class MemAttentionNoObjPtr1Frame(_MemAttentionBase):
    def __init__(self, sam_model: SAM2Base) -> None:
        super().__init__(sam_model)
        image_size = int(getattr(sam_model, "image_size", 1024))
        with torch.no_grad():
            dummy = torch.zeros(1, 3, image_size, image_size, dtype=torch.float32)
            backbone_out = sam_model.image_encoder(dummy)
            _, _, current_vision_pos_embeds, _ = sam_model._prepare_backbone_features(backbone_out)
            current_vision_pos_embed = current_vision_pos_embeds[-1].detach().clone()
        self.register_buffer("current_vision_pos_embed", current_vision_pos_embed, persistent=False)

    @staticmethod
    def _flatten_single_memory_frame(memory_1: torch.Tensor) -> torch.Tensor:
        return memory_1.reshape(1, 64, 64 * 64).permute(0, 2, 1).reshape(4096, 1, 64)

    @torch.no_grad()
    def forward(
        self,
        current_vision_feat: torch.Tensor,
        memory_1: torch.Tensor,
        memory_pos_embed: torch.Tensor,
    ) -> torch.Tensor:
        feat_hwbc, batch, channels, height, width = self._prepare_current_features(current_vision_feat)
        memory_1 = self._flatten_single_memory_frame(memory_1)

        fused_hwbc = self.memory_attention(
            curr=feat_hwbc,
            curr_pos=self.current_vision_pos_embed,
            memory=memory_1,
            memory_pos=memory_pos_embed,
            num_obj_ptr_tokens=0,
        )

        return fused_hwbc.permute(1, 2, 0).reshape(batch, channels, height, width)


class _MemEncoderBase(nn.Module):
    def __init__(self, sam_model: SAM2Base) -> None:
        super().__init__()
        self.model = sam_model
        self.maskmem_tpos_enc = sam_model.maskmem_tpos_enc
        self.feat_sizes = [(256, 256), (128, 128), (64, 64)]

    def _encode(
        self,
        mask_for_mem: torch.Tensor,
        pix_feat: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, channels, height, width = pix_feat.shape
        flattened = pix_feat.view(batch, channels, height * width).permute(2, 0, 1)
        # Default to "object present" when exporting the standalone memory encoder.
        # SAM2 base-plus uses no_obj_embed_spatial, and a neutral 0 logit would route
        # every frame through the no-object spatial embedding path.
        object_score_logits = torch.full((batch, 1), 10.0, device=pix_feat.device)

        maskmem_features, maskmem_pos_enc = self.model._encode_new_memory(
            current_vision_feats=[flattened],
            feat_sizes=self.feat_sizes,
            pred_masks_high_res=mask_for_mem,
            object_score_logits=object_score_logits,
            is_mask_from_pts=True,
        )

        maskmem_pos_enc_tensor = maskmem_pos_enc[-1]
        maskmem_pos_enc_tensor = maskmem_pos_enc_tensor.view(1, 64, height * width).permute(2, 0, 1)
        return maskmem_features, maskmem_pos_enc_tensor, self.maskmem_tpos_enc


class MemEncoder(_MemEncoderBase):
    @torch.no_grad()
    def forward(
        self,
        mask_for_mem: torch.Tensor,
        pix_feat: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self._encode(mask_for_mem, pix_feat)


class MemEncoderLite(_MemEncoderBase):
    @torch.no_grad()
    def forward(
        self,
        mask_for_mem: torch.Tensor,
        pix_feat: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        maskmem_features, maskmem_pos_enc, _ = self._encode(mask_for_mem, pix_feat)
        return maskmem_features, maskmem_pos_enc
