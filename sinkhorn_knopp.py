import torch
import numpy as np
import torch.nn.functional as F

# https://github.com/facebookresearch/swav/blob/main/main_swav.py


def shoot_infs(inp_tensor):
    """Replaces inf and nan by zero, then fills inf by maximum of tensor"""
    inp_tensor = torch.nan_to_num(inp_tensor, nan=0.0, posinf=0.0, neginf=0.0)
    
    mask_inf = torch.isinf(inp_tensor)
    if torch.any(mask_inf):
        m = torch.max(inp_tensor[~mask_inf]) if torch.any(~mask_inf) else torch.tensor(1.0)
        inp_tensor[mask_inf] = m
    return inp_tensor


class SinkhornKnopp_imb(torch.nn.Module):
    def __init__(self, args, cls_dist):
        super().__init__()
        self.num_iters = args.num_iters_sk
        self.epsilon = args.epsilon_sk
        self.temperature = args.cos_temp
        self.cls_dist = cls_dist

    @torch.no_grad()
    def iterate(self, Q):
        # Work in float32 for Sinkhorn iterations to avoid precision issues
        Q = Q.float()
        self.cls_dist = self.cls_dist.to(Q.device).float()

        Q = shoot_infs(Q)
        sum_Q = torch.sum(Q)
        Q /= (sum_Q + 1e-12)

        B = Q.shape[1]
        K = Q.shape[0]

        for it in range(self.num_iters):
            sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
            Q /= (sum_of_rows + 1e-12)
            Q = shoot_infs(Q)
            Q *= self.cls_dist

            # normalize each column: total weight per sample must be 1/B
            sum_of_cols = torch.sum(Q, dim=0, keepdim=True)
            Q /= (sum_of_cols + 1e-12)
            Q /= B

        Q *= B  # the colomns must sum to 1 so that Q is an assignment
        
        return Q.t()

    @torch.no_grad()
    def forward(self, embeddings, centroids):
        orig_dtype = embeddings.dtype
        
        last_token_rep = F.normalize(embeddings, p=2, dim=-1).float()
        centroids = F.normalize(centroids, p=2, dim=-1).to(last_token_rep.dtype)

        # Compute cosine similarity
        similarities = torch.matmul(last_token_rep, centroids.T)
        similarities = shoot_infs(similarities)

        # Apply temperature scaling
        similarities = similarities / self.temperature

        # Convert to probability distributions
        pt = F.softmax(similarities, dim=-1)

        # Compute soft assignment weights using sharpening
        pt = torch.log(pt + 1e-12)
        q = pt / (self.epsilon)
        q = torch.exp(q).t()

        # Run sinkhorn iterations in float32 and then cast back
        result = self.iterate(q)
        return result.to(orig_dtype)
