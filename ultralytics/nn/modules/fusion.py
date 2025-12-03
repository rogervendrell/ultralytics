import torch
import torch.nn as nn
import torch.nn.functional as F



class Fusion_pairs(nn.Module):
    def __init__(self, embed_dim, num_heads=1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
        )

    def forward(self, tpe, vpe):
        """
        Apply cross-attention independently per (tpe[i], vpe[i]) pair.
        tpe: [N, embed_dim]
        vpe: [N, embed_dim]
        """
        if tpe is None and vpe is None:
            return None
        
        if tpe is None: return vpe
        if vpe is None: return tpe

        fused, _ = self.cross_attn(
            query=tpe.unsqueeze(0),
            key=vpe.unsqueeze(0),
            value=vpe.unsqueeze(0),
        )
        return fused.squeeze(0)



class Fusion(nn.Module):
    def __init__(self, N, embed_dim, num_heads=1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=False # (N, B, E)
        )
        
        # Create diagonal mask
        self.register_buffer("_diag_mask", torch.full((N, N), float('-inf')))
        self._diag_mask = torch.diag_embed(torch.diag(self._diag_mask), offset=0)

    def forward(self, tpe, vpe_padded):
        """
        Applies masked cross-attention for pair-wise fusion.

        Args:
            tpe: Textual embeddings (Query). Shape: (B, 80, E)
            vpe_padded: Padded visual embeddings (Key/Value). Shape: (B, 80, E)

        Returns:
            fused: Fused embeddings. Shape: (B, 80, E)
        """
        B, N, E = tpe.shape

        query = tpe.permute(1, 0, 2)
        key = vpe_padded.permute(1, 0, 2)
        value = vpe_padded.permute(1, 0, 2)

        # MHA expects a mask of shape (B*H, N, N).
        # Repeat diagonal mask across the batch and heads
        attn_mask_batch = self._diag_mask.unsqueeze(0).repeat(B * self.num_heads, 1, 1)

        fused, _ = self.cross_attn(
            query=query,
            key=key,
            value=value,
            attn_mask=attn_mask_batch
        )
        
        # 4. Permute back to the standard (B, N, E)
        return fused.permute(1, 0, 2)