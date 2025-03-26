import torch
from torch import nn
from sam2.modeling.sam2_base import SAM2Base

class ImageEncoder(nn.Module):
    def __init__(self, sam_model: SAM2Base) -> None:
        super().__init__()
        self.model = sam_model
        # Feature map sizes for different levels (in ascending order).
        # Let's assume index 0 => largest spatial, 2 => smallest spatial.
        self._bb_feat_sizes = [
            (256, 256),
            (128, 128),
            (64, 64),
        ]

    @torch.no_grad()
    def forward(self, input: torch.Tensor):
        # Run the SAM2 image encoder.
        backbone_out = self.model.forward_image(input)
        # Flatten feature maps and get positional encodings.
        backbone_out, vision_feats, vision_pos_embeds, feat_sizes = \
            self.model._prepare_backbone_features(backbone_out)

        # Optionally add no_mem_embed (used for video/multi-frame mode).
        if self.model.directly_add_no_mem_embed:
            vision_feats[-1] = vision_feats[-1] + self.model.no_mem_embed

        # vision_feats is typically a list: [feat0, feat1, feat2] each shape (HW, B, C).
        # We want them in shapes [B,C,H,W], specifically:
        #   feat0 -> [1,  32, 256, 256]
        #   feat1 -> [1,  64, 128, 128]
        #   feat2 -> [1, 256,  64,  64]
        # We'll do a straightforward approach that matches self._bb_feat_sizes.

        # CAREFUL: Ensure you're matching the correct feat size with each index
        # from your actual backbone order.
        f0_h, f0_w = self._bb_feat_sizes[0]  # (256,256)
        f1_h, f1_w = self._bb_feat_sizes[1]  # (128,128)
        f2_h, f2_w = self._bb_feat_sizes[2]  # (64,64)

        feat0 = vision_feats[0]  # shape (256*256, B=1, ?)
        feat1 = vision_feats[1]
        feat2 = vision_feats[2]

        # Convert each to (B,C,H,W):
        feat0 = feat0.permute(1, 2, 0).reshape(1, -1, f0_h, f0_w)
        feat1 = feat1.permute(1, 2, 0).reshape(1, -1, f1_h, f1_w)
        feat2 = feat2.permute(1, 2, 0).reshape(1, -1, f2_h, f2_w)

        # Now feat0 => [1, 32, 256, 256], feat1 => [1, 64, 128, 128], feat2 => [1, 256, 64, 64]
        # According to your original code, you want:
        #   high_res_features1 = feats[0]  => [1,32,256,256]
        #   high_res_features2 = feats[1]  => [1,64,128,128]
        #   image_embeddings   = feats[2]  => [1,256,64,64]
        high_res_features1 = feat0
        high_res_features2 = feat1
        image_embeddings   = feat2

        # For multi-frame memory: get flattened vision features and positional encodings.
        current_vision_feat = vision_feats[-1]   # [HW, B, C], e.g. [4096, 1, 256]
        vision_pos_embed    = vision_pos_embeds[-1]  # [HW, B, 256]

        return (image_embeddings, high_res_features1, high_res_features2,
                current_vision_feat, vision_pos_embed)


class ImageDecoder(nn.Module):
    def __init__(self, sam_model: SAM2Base) -> None:
        super().__init__()
        self.model = sam_model
        self.sigmoid_scale_for_mem_enc = getattr(sam_model, "sigmoid_scale_for_mem_enc", 1.0)
        self.sigmoid_bias_for_mem_enc  = getattr(sam_model, "sigmoid_bias_for_mem_enc", 0.0)

    @torch.no_grad()
    def forward(
        self,
        point_coords: torch.Tensor,      # [num_labels, num_points, 2]
        point_labels: torch.Tensor,      # [num_labels, num_points]
        image_embed: torch.Tensor,       # [1,256,64,64]
        high_res_feats_0: torch.Tensor,  # [1,32,256,256]
        high_res_feats_1: torch.Tensor,  # [1,64,128,128]
    ):
        """
        A simplified "points-only" decoder that calls the higher-level 
        _forward_sam_heads to generate a final mask. No existing-mask prompt.
        Returns the low-res mask in shape [1, num_masks, 256, 256].
        We'll do final upsampling in Python *outside* ONNX.
        """
        # 1) Prepare dict for point inputs
        point_inputs = {
            "point_coords": point_coords,
            "point_labels": point_labels,
        }
        high_res_feats = [high_res_feats_0, high_res_feats_1]

        # 2) Forward pass using the internal SAM logic
        (
            _,
            _,
            _,
            low_res_masks,   # [1, num_masks, 256, 256]
            high_res_masks,  # [1, num_masks, 1024, 1024]
            obj_ptr,         
            _
        ) = self.model._forward_sam_heads(
            backbone_features=image_embed,
            point_inputs=point_inputs,
            mask_inputs=None,          # no prior-mask prompt
            high_res_features=high_res_feats,
            multimask_output=True
        )

        # 3) If you plan to feed the predicted mask into memory:
        mask_for_mem = torch.sigmoid(high_res_masks)
        mask_for_mem = mask_for_mem * self.sigmoid_scale_for_mem_enc
        mask_for_mem = mask_for_mem + self.sigmoid_bias_for_mem_enc

        # 4) Just return low_res_masks directly as pred_mask
        pred_mask = low_res_masks
        return obj_ptr, mask_for_mem, pred_mask


class MemAttention(nn.Module):
    """
    Wraps SAM2's memory_attention for multi-frame usage. We'll cut down on repeated
    reshape/permutations, still returning [1,256,64,64].
    """

    def __init__(self, sam_model: SAM2Base) -> None:
        super().__init__()
        self.model = sam_model
        self.memory_attention = sam_model.memory_attention
        self.no_mem_embed = sam_model.no_mem_embed  # if your model uses this

    @torch.no_grad()
    def forward(self,
                current_vision_feat: torch.Tensor,      # [1,256,64,64]
                current_vision_pos_embed: torch.Tensor, # [4096,1,256]
                memory_0: torch.Tensor,                 # [num_obj_ptr,256]
                memory_1: torch.Tensor,                 # [n,64,64,64]
                memory_pos_embed: torch.Tensor          # [N,1,64]
               ) -> torch.Tensor:
        """
        Returns fused feature shaped like [1,256,64,64].
        """

        # current_vision_feat is (B=1, C=256, H=64, W=64).
        B, C, H, W = current_vision_feat.shape

        # Flatten from (B,C,H,W) => (HW,B,C).
        # This is one permute+reshape step:
        feat_hwbc = current_vision_feat.permute(2, 3, 0, 1).reshape(H * W, B, C)

        # Subtract no_mem_embed if required:
        feat_hwbc = feat_hwbc - self.no_mem_embed

        # memory_0 => shape [num_obj_ptr, 256]. Suppose each pointer is 4 tokens of dim 64 => total 256.
        # If you do something like:
        #   memory_0.reshape(-1,1,4,64) -> (num_obj_ptr, 1, 4, 64)
        #   permute/flatten => (num_obj_ptr*4, 1, 64)
        # We'll do fewer steps in a single chain:
        num_obj_ptr = memory_0.shape[0]
        memory_0 = memory_0.reshape(num_obj_ptr, 4, 64)    # [num_obj_ptr, 4, 64]
        memory_0 = memory_0.unsqueeze(1)                   # => [num_obj_ptr, 1, 4, 64]
        memory_0 = memory_0.permute(0, 2, 1, 3)            # => [num_obj_ptr, 4, 1, 64]
        memory_0 = memory_0.reshape(num_obj_ptr * 4, 1, 64)# => [num_obj_ptr*4, 1, 64]

        # memory_1 => shape [n, 64, 64, 64].
        # Flatten => (n, 64, 4096) => permute => (n, 4096, 64) => reshape => (n*4096,1,64).
        mem_1_n = memory_1.shape[0]   # n frames
        memory_1 = memory_1.reshape(mem_1_n, 64, 64*64)  # [n, 64, 4096]
        memory_1 = memory_1.permute(0, 2, 1)             # [n, 4096, 64]
        memory_1 = memory_1.reshape(-1, 1, 64)           # [n*4096, 1, 64]

        # Concat memory_1 + memory_0 => shape [n*4096 + num_obj_ptr*4, 1, 64]
        memory = torch.cat((memory_1, memory_0), dim=0)

        # num_obj_ptr_tokens is how many pointer tokens we appended
        num_obj_ptr_tokens = num_obj_ptr * 4

        # Forward memory_attention => fused_hwbc shape [HW,B,C]
        fused_hwbc = self.memory_attention(
            curr=feat_hwbc,
            curr_pos=current_vision_pos_embed,  # [HW,B,C]
            memory=memory,
            memory_pos=memory_pos_embed,        # [some_len,1,64], etc.
            num_obj_ptr_tokens=num_obj_ptr_tokens
        )

        # Reshape back to (B,C,H,W)
        fused_bcHW = fused_hwbc.permute(1, 2, 0).reshape(B, C, H, W)
        return fused_bcHW


class MemEncoder(nn.Module):
    def __init__(self, sam_model: SAM2Base) -> None:
        super().__init__()
        self.model = sam_model
        self.maskmem_tpos_enc = sam_model.maskmem_tpos_enc
        self.feat_sizes = [(256, 256), (128, 128), (64, 64)]
    
    @torch.no_grad()
    def forward(self,
                mask_for_mem: torch.Tensor,  # [1,1,1024,1024]
                pix_feat: torch.Tensor       # [1,256,64,64]
               ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Flatten pix_feat from [B, C, H, W] to [H*W, B, C]
        B, C, H, W = pix_feat.shape
        flattened = pix_feat.view(B, C, H * W).permute(2, 0, 1)

        # Create a placeholder for object_score_logits.
        object_score_logits = torch.zeros(1, 1, device=pix_feat.device)
        
        # Pass a list containing the flattened feature.
        maskmem_features, maskmem_pos_enc = self.model._encode_new_memory(
            current_vision_feats=[flattened],
            feat_sizes=self.feat_sizes,
            pred_masks_high_res=mask_for_mem,
            object_score_logits=object_score_logits,
            is_mask_from_pts=True,
        )
        
        # Suppose we pick the last pos_enc
        maskmem_pos_enc_tensor = maskmem_pos_enc[-1]
        # e.g. shape => [1, 64, H*W], then permute => [H*W,1,64]
        maskmem_pos_enc_tensor = maskmem_pos_enc_tensor.view(1, 64, H * W).permute(2, 0, 1)
        
        return maskmem_features, maskmem_pos_enc_tensor, self.maskmem_tpos_enc
