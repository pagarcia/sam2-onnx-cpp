# src/modules.py
import torch
from torch import nn
from sam2.sam2.modeling.sam2_base import SAM2Base

class ImageEncoder(nn.Module):
    def __init__(self, sam_model: SAM2Base) -> None:
        super().__init__()
        self.model = sam_model
        # Feature map sizes for different levels.
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
        # Convert each level from (HW, B, C) to (B, C, H, W)
        feats = [
            feat.permute(1, 2, 0).view(1, -1, *feat_size)
            for feat, feat_size in zip(vision_feats[::-1], self._bb_feat_sizes[::-1])
        ][::-1]
        image_embeddings = feats[2]       # [1, 256, 64, 64]
        high_res_features1 = feats[0]       # [1, 32, 256, 256]
        high_res_features2 = feats[1]       # [1, 64, 128, 128]
        # For multi-frame memory: get flattened vision features and positional encodings.
        current_vision_feat = vision_feats[-1]   # [4096, 1, 256]
        vision_pos_embed    = vision_pos_embeds[-1]  # [4096, 1, 256]
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
        #    We'll do the final upsample outside ONNX
        #    shape => [1, num_masks, 256, 256]
        pred_mask = low_res_masks

        return obj_ptr, mask_for_mem, pred_mask

class MemAttention(nn.Module):
    """
    Wraps SAM2's memory_attention for multi-frame usage. Typically:
      fused_features = memory_attention(
         curr=current_vision_feat,      # shape [HW,B,C]
         curr_pos=current_vision_pos_embed,  # shape [HW,B,C]
         memory=...,   # object pointers & spatial memories stacked
         memory_pos=...,   # positional encodings for memory
         num_obj_ptr_tokens=...
      )
    and returns fused_features in shape [HW,B,C]. We'll provide a
    friendlier input shape: e.g. [1,256,64,64] for current_vision_feat.
    """

    def __init__(self, sam_model: SAM2Base) -> None:
        super().__init__()
        self.model = sam_model
        self.memory_attention = sam_model.memory_attention
        # If your SAM2 model uses a "no_mem_embed" you want to subtract from feats:
        self.no_mem_embed = sam_model.no_mem_embed

    @torch.no_grad()
    def forward(self,
                current_vision_feat: torch.Tensor,      # e.g. [1,256,64,64]
                current_vision_pos_embed: torch.Tensor, # e.g. [4096,1,256]
                memory_0: torch.Tensor,                 # e.g. [num_obj_ptr,256]
                memory_1: torch.Tensor,                 # e.g. [n,64,64,64]
                memory_pos_embed: torch.Tensor          # e.g. [N,1,64]
               ) -> torch.Tensor:
        """
        Returns a fused feature shaped like [1,256,64,64].

        Typical usage in multi-frame context:
          1) Flatten current_vision_feat from [1,256,64,64] -> [H*W, B=1, C=256].
          2) Subtract no_mem_embed if your SAM2 code does that.
          3) memory_0 is object ptr tokens, memory_1 is spatial memory, etc.
          4) Cat them => memory
          5) Pass memory & memory_pos_embed to self.memory_attention(...)
          6) Reshape result back to [1,256,64,64].
        """

        # 1) Flatten the current_vision_feat from (B,C,H,W) -> (H*W, B, C).
        B, C, H, W = current_vision_feat.shape  # e.g. [1,256,64,64]
        feat_hwbc = current_vision_feat.permute(2, 3, 0, 1).reshape(H*W, B, C)
        # Optionally subtract no_mem_embed:
        feat_hwbc = feat_hwbc - self.no_mem_embed  # if your model requires that

        # 2) Prepare memory_0 (object ptr tokens).
        #    Suppose memory_0 is shape [num_obj_ptr, 256].
        #    If each object pointer is 4 tokens or something, you might do:
        #       memory_0 -> memory_0.view(-1,1,4,64).flatten(0,1), etc.
        #    The code below is just an example. Adapt to your actual design:
        # E.g. if each object pointer is 4 tokens, we do something like:
        num_obj_ptr = memory_0.shape[0]

        memory_0 = memory_0.reshape(-1,1,4,64)
        memory_0 = memory_0.permute(0, 2, 1, 3).flatten(0, 1)
        # -> shape [num_obj_ptr * 4, 1, 64]

        memory_1 = memory_1.view(-1,64,64*64).permute(0,2,1)
        memory_1 = memory_1.reshape(-1,1,64)
        # -> shape [some_spatial_len, 1, 64]

        memory = torch.cat((memory_1, memory_0), dim=0)

        # 5) num_obj_ptr_tokens must tell memory_attention how many tokens are from object ptr:
        #    if we repeated each pointer 4 times, then total obj tokens= (num_obj_ptr*4).
        #    memory_1 had (n*4096) rows, so total memory rows= (n*4096 + num_obj_ptr*4).
        #    Typically we pass that as an int:
        num_obj_ptr_tokens = num_obj_ptr * 4

        # 6) Finally call memory_attention:
        fused_hwbc = self.memory_attention(
            curr=feat_hwbc,  # [HW,B,C]
            curr_pos=current_vision_pos_embed,  # [HW,B,C]
            memory=memory,
            memory_pos=memory_pos_embed,  # [some_len,1,64], etc.
            num_obj_ptr_tokens=num_obj_ptr_tokens
        )
        # The output fused_hwbc is shape [HW,B,C].

        # 7) Reshape it back to [B,C,H,W].
        fused_bcHW = fused_hwbc.permute(1,2,0).view(B, C, H, W)
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
        # Get the shape of the input feature.
        B, C, H, W = pix_feat.shape
        # Flatten pix_feat from [B, C, H, W] to [H*W, B, C]
        flattened = pix_feat.view(B, C, H * W).permute(2, 0, 1)
        
        # Create a placeholder for object_score_logits.
        # For instance, if we have a single score per batch element:
        object_score_logits = torch.zeros(1, 1, device=pix_feat.device)
        
        # Pass a list containing the flattened feature.
        maskmem_features, maskmem_pos_enc = self.model._encode_new_memory(
            current_vision_feats=[flattened],
            feat_sizes=self.feat_sizes,
            pred_masks_high_res=mask_for_mem,
            object_score_logits=object_score_logits,
            is_mask_from_pts=True,
        )
        
        # Since maskmem_pos_enc is a list, we choose (for example) the last tensor.
        maskmem_pos_enc_tensor = maskmem_pos_enc[-1]
        # Reshape it if needed. Here we assume the tensor shape is compatible with [1, 64, H*W].
        maskmem_pos_enc_tensor = maskmem_pos_enc_tensor.view(1, 64, H * W).permute(2, 0, 1)
        
        return maskmem_features, maskmem_pos_enc_tensor, self.maskmem_tpos_enc
