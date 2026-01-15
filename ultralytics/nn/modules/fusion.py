import torch.nn as nn
import torch.nn.functional as F


class Fusion(nn.Module):
    def __init__(self, embed_dim, num_heads=1, query="tpe"):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
        )
        assert query in ["tpe", "vpe"]
        self.query = query

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

        assert tpe.shape == vpe.shape, \
            f"got mismatching shapes for tpe {tpe.shape} and vpe {vpe.shape}"

        if self.query == "tpe":
            fused, _ = self.cross_attn(
                query=tpe.unsqueeze(0),
                key=vpe.unsqueeze(0),
                value=vpe.unsqueeze(0),
            )
        else:
            fused, _ = self.cross_attn(
                query=vpe.unsqueeze(0),
                key=tpe.unsqueeze(0),
                value=tpe.unsqueeze(0),
            )
            
        return fused.squeeze(0)