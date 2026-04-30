import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any

from model import NET


class MLP(nn.Module):
    """Simple 2-layer MLP (Python 3.8+ compatible)."""
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: Optional[int] = None, dropout: float = 0.1):
        super().__init__()
        hidden_dim = hidden_dim or out_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class ScorerMLP(nn.Module):
    """MLP that outputs a scalar score for each candidate."""
    def __init__(self, in_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class REXPERTNET_R2(NET):
    r"""R²-DMSA: Query-specific retrieval + candidate reranking (no diffusion).

    This model is built on top of DMSA's strong backbone and training/evaluation pipeline.
    Unlike the earlier "minimal replacement" (swap diffusion output), this method:
      1) Builds a *query-specific* evidence pool by structure recall from recent history
         (match by subject/object and/or relation).
      2) Aggregates evidence via attention to produce an evidence summary vector c.
      3) Uses retrieved entities to expand the candidate set, then applies a reranker that
         scores candidates conditioned on (query, evidence summary).
      4) Adds explicit reranking loss on the candidate set + optional retrieval contrastive loss.

    Implementation detail:
    - We override diffusion()/diffusion2() to provide evidence summary c (no diffusion).
    - We override calculate_nce_loss() so retrieval bias is applied with the correct direction,
      since DMSA forward calls diffusion() and diffusion2() before NCE and would otherwise overwrite it.
    - We override calculate_cand_loss() to perform reranking and to train the reranker.

    Drop-in compatible with the original training scripts:
      - REXPERTNET_R2(num_e, num_rel, num_t, diffu_rec, args)
      - REXPERTNET_R2(num_e, num_rel, num_t, args)
    """

    def __init__(self, num_e, num_rel, num_t, diffu_rec=None, args=None):
        if args is None and diffu_rec is not None:
            # Called as (num_e, num_rel, num_t, args)
            args = diffu_rec
            diffu_rec = None

        super().__init__(num_e, num_rel, num_t, diffu_rec, args)

        d = self.emb_dim
        drop = float(getattr(args, 'dropout', 0.2))

        # Retrieval configuration
        self.retrieval_topk = int(getattr(args, 'retrieval_topk', 8))
        self.retrieval_pool_size = int(getattr(args, 'retrieval_pool_size', 256))
        self.history_pool_max = int(getattr(args, 'history_pool_max', 20000))
        self.retrieval_time_lambda = float(getattr(args, 'retrieval_time_lambda', 0.0))
        self.retrieval_beta = float(getattr(args, 'retrieval_beta', 0.0))  # optional logits bias from retrieved ids

        # Query/key/value projections
        self.rex_q = MLP(3 * d, d, hidden_dim=d, dropout=drop)    # [ent, rel, time] -> d
        self.rex_k = MLP(4 * d, d, hidden_dim=d, dropout=drop)    # [s, r, o, dt] -> d
        self.rex_v = MLP(4 * d, d, hidden_dim=d, dropout=drop)

        # Candidate reranking configuration
        self.rerank_topm = int(getattr(args, 'rerank_topm', 256))              # top-M from DMSA to include
        self.rerank_max_cands = int(getattr(args, 'rerank_max_cands', 512))    # final candidate budget
        self.rerank_eta = float(getattr(args, 'rerank_eta', 1.0))              # strength of reranker delta
        self.lambda_rank = float(getattr(args, 'lambda_rank', 0.5))            # reranking loss weight
        self.lambda_ret = float(getattr(args, 'lambda_ret', 0.0))              # retrieval contrastive loss weight
        self.ret_tau = float(getattr(args, 'ret_tau', 0.07))                   # temperature for contrastive

        self.reranker = ScorerMLP(in_dim=3 * d, hidden_dim=d, dropout=drop)    # [e_cand, q, c] -> scalar
        self.eps = 1e-12

        # Direction pointers (used to pick the correct cache/bias inside overridden losses)
        self._rel_ptr_forward = self.rel_embeds[1:self.num_rel + 1].data_ptr()
        self._rel_ptr_inverse = self.rel_embeds[self.num_rel + 1:].data_ptr()

        # runtime caches (set inside diffusion/diffusion2)
        self._cache_o: Dict[str, Any] = {}
        self._cache_s: Dict[str, Any] = {}

        print("REXPERTNET_R2 Initiated: Query-specific Retrieval + Candidate Reranking (no diffusion).")

    # ---------------------------
    #   Query-specific retrieval
    # ---------------------------
    def _gather_history_events(self, list_diff_data, device):
        """Flatten list_diff_data into a (H,4) tensor and cap to a max length.

        list_diff_data is a list of tensors shaped (B,4) (quadruples).
        """
        if list_diff_data is None or list_diff_data == []:
            return None
        x = torch.cat([t.to(device) for t in list_diff_data], dim=0)
        if x.size(1) > 4:
            x = x[:, :4]
        if x.size(0) > self.history_pool_max:
            x = x[-self.history_pool_max:]
        return x

    def _select_pool(self, quad: torch.Tensor, hist_events: torch.Tensor, direction: str):
        """Structure recall: build a query-specific pool by matching (entity OR relation).

        direction:
          - 'o': predict object for query (s,r,?,t) -> match history by (s==s*) OR (r==r*)
          - 's': predict subject for query (?,r,o,t) -> match history by (o==o*) OR (r==r*)
        Returns:
          pool: (B,S,4)
          pool_valid: (B,S) boolean mask indicating valid recalled events
        """
        device = quad.device
        B = quad.size(0)
        H = hist_events.size(0)
        S = min(self.retrieval_pool_size, H)

        if S <= 0:
            return None, None

        s_q = quad[:, 0].long()
        r_q = quad[:, 1].long()
        o_q = quad[:, 2].long()

        # (B,H) masks
        if direction == 'o':
            m_ent = (hist_events[:, 0].long().unsqueeze(0) == s_q.unsqueeze(1))
        else:
            m_ent = (hist_events[:, 2].long().unsqueeze(0) == o_q.unsqueeze(1))
        m_rel = (hist_events[:, 1].long().unsqueeze(0) == r_q.unsqueeze(1))
        mask = m_ent | m_rel  # (B,H)

        # rank by recency (position index)
        pos = torch.arange(H, device=device, dtype=torch.float32).unsqueeze(0).expand(B, H)
        pos = pos.masked_fill(~mask, -1e9)

        topv, topi = torch.topk(pos, k=S, dim=1)  # (B,S)
        valid = topv > -1e8

        # fallback: if a row has no valid matches, use most recent S events
        if (~valid[:, 0]).any():
            fallback_idx = torch.arange(H - S, H, device=device).long()
            for b in torch.nonzero(~valid[:, 0], as_tuple=False).view(-1).tolist():
                topi[b] = fallback_idx
                valid[b] = torch.ones((S,), device=device, dtype=torch.bool)

        pool = hist_events[topi]  # (B,S,4)
        return pool, valid

    def _time_embed_rel(self, delta_t: torch.Tensor):
        """Relative time embedding using DMSA's time_embeddings table.

        delta_t: (B,S) non-negative
        """
        # time_embeddings was created with length time_max_len + 3 (see model.py)
        idx = (delta_t.long() % self.time_max_len) + 2
        return self.time_embeddings(idx)  # (B,S,d)

    def _retrieve(self, quad: torch.Tensor, hist_events: torch.Tensor, current_h: torch.Tensor,
                  rel_embeds: torch.Tensor, direction: str):
        """Retrieve evidence summary c and save caches for reranking."""
        pool, valid = self._select_pool(quad, hist_events, direction=direction)
        if pool is None:
            B = quad.size(0)
            d = self.emb_dim
            c = torch.zeros((B, d), device=quad.device)
            cache = dict(q=torch.zeros((B, d), device=quad.device),
                         c=c,
                         ent_top=torch.zeros((B, 1), device=quad.device, dtype=torch.long),
                         att=torch.zeros((B, 1), device=quad.device),
                         bias=torch.zeros((B, self.num_e), device=quad.device))
            return c, cache

        B, S, _ = pool.shape
        # pool fields
        hs = pool[:, :, 0].long()
        hr = pool[:, :, 1].long()
        ho = pool[:, :, 2].long()
        ht = pool[:, :, 3].long()

        # Relative time
        tq = quad[:, 3].long().unsqueeze(1)  # (B,1)
        delta_t = (tq - ht).clamp(min=0)     # (B,S)
        t_hist = self._time_embed_rel(delta_t)  # (B,S,d)

        e_s = current_h[hs]  # (B,S,d)
        e_o = current_h[ho]  # (B,S,d)
        r_h = rel_embeds[hr] # (B,S,d) (use the same rel slice as the branch)
        tok = torch.cat([e_s, r_h, e_o, t_hist], dim=-1)  # (B,S,4d)

        # Query embedding
        if direction == 'o':
            q_ent = current_h[quad[:, 0].long()]
        else:
            q_ent = current_h[quad[:, 2].long()]
        q_rel = rel_embeds[quad[:, 1].long()]
        q_time = 0.5 * (self.fact_time[0] + self.fact_time[1])  # (B,d)
        q = self.rex_q(torch.cat([q_ent, q_rel, q_time], dim=1))  # (B,d)

        K = self.rex_k(tok.reshape(B * S, -1)).reshape(B, S, -1)  # (B,S,d)
        V = self.rex_v(tok.reshape(B * S, -1)).reshape(B, S, -1)  # (B,S,d)

        sim = torch.einsum('bd,bsd->bs', q, K) / math.sqrt(self.emb_dim)  # (B,S)

        # optional time decay in attention scores
        if self.retrieval_time_lambda > 0:
            sim = sim - self.retrieval_time_lambda * delta_t.float()

        # attention top-k inside the pool
        k = min(self.retrieval_topk, S)
        topv, topi = torch.topk(sim, k=k, dim=1)  # (B,k)
        att = torch.softmax(topv, dim=1)           # (B,k)

        V_top = torch.gather(V, 1, topi.unsqueeze(-1).expand(-1, -1, self.emb_dim))  # (B,k,d)
        c = torch.sum(att.unsqueeze(-1) * V_top, dim=1)  # (B,d)

        # Retrieved entity ids from top-k evidence
        if direction == 'o':
            ent_ids = torch.gather(ho, 1, topi)  # (B,k)
        else:
            ent_ids = torch.gather(hs, 1, topi)  # (B,k)

        # Optional entity-level bias (stored per direction)
        bias = torch.zeros((B, self.num_e), device=quad.device)
        if self.retrieval_beta > 0:
            bias.scatter_add_(1, ent_ids, att * self.retrieval_beta)

        cache = dict(q=q, c=c, ent_top=ent_ids, att=att, bias=bias)
        return c, cache

    # ---------------------------
    #  Overrides: diffusion hooks
    # ---------------------------
    def diffusion(self, quadruples, list_diff_data, r, o, mode_lk, current_h, rel_embeds):
        """Forward direction (predict object): return evidence summary as diff_out."""
        hist_events = self._gather_history_events(list_diff_data, device=quadruples.device)
        if hist_events is None:
            c = torch.zeros((quadruples.shape[0], self.emb_dim), device=quadruples.device)
            self._cache_o = dict(q=torch.zeros_like(c), c=c,
                                 ent_top=torch.zeros((quadruples.shape[0], 1), device=quadruples.device, dtype=torch.long),
                                 att=torch.zeros((quadruples.shape[0], 1), device=quadruples.device),
                                 bias=torch.zeros((quadruples.shape[0], self.num_e), device=quadruples.device))
            return (c, None)

        c, cache = self._retrieve(quadruples, hist_events, current_h, rel_embeds, direction='o')
        self._cache_o = cache
        return (c, None)

    def diffusion2(self, quadruples, list_diff_data, r, o, mode_lk, current_h, rel_embeds):
        """Inverse direction (predict subject): return evidence summary as diff_out."""
        hist_events = self._gather_history_events(list_diff_data, device=quadruples.device)
        if hist_events is None:
            c = torch.zeros((quadruples.shape[0], self.emb_dim), device=quadruples.device)
            self._cache_s = dict(q=torch.zeros_like(c), c=c,
                                 ent_top=torch.zeros((quadruples.shape[0], 1), device=quadruples.device, dtype=torch.long),
                                 att=torch.zeros((quadruples.shape[0], 1), device=quadruples.device),
                                 bias=torch.zeros((quadruples.shape[0], self.num_e), device=quadruples.device))
            return (c, None)

        c, cache = self._retrieve(quadruples, hist_events, current_h, rel_embeds, direction='s')
        self._cache_s = cache
        return (c, None)

    # ---------------------------
    #  Overrides: losses / scores
    # ---------------------------
    def _pick_cache(self, rel_embeds: torch.Tensor):
        """Pick cache by relation-embedding slice pointer (forward vs inverse)."""
        ptr = rel_embeds.data_ptr()
        if ptr == self._rel_ptr_forward:
            return self._cache_o
        elif ptr == self._rel_ptr_inverse:
            return self._cache_s
        # fallback
        return self._cache_o if self._cache_o else self._cache_s

    def clear_runtime_cache(self):
        """Release step-level retrieval caches to avoid retaining GPU tensors across steps."""
        self._cache_o = {}
        self._cache_s = {}

    def calculate_nce_loss(self, actor1, actor2, r, current_h, rel_embeds, pred_layer, pred_layer2, history_tag,
                           frequency, non_history_tag, fact_time, diff_out):
        """Same as NET.calculate_nce_loss but applies the *correct* retrieval bias per direction."""
        if current_h is not None:
            sub_emb = current_h[actor1]
            obj_emb = current_h
        else:
            sub_emb = self.entity_embeds[actor1]
            obj_emb = self.entity_embeds

        cache = self._pick_cache(rel_embeds)
        bias = cache.get("bias", 0.0) if cache else 0.0

        # diff_out is evidence summary c
        if diff_out is None:
            diff_out = torch.zeros((actor1.shape[0], self.emb_dim), device=sub_emb.device)

        h = pred_layer(self.dropout(torch.cat((sub_emb, rel_embeds[r], diff_out), dim=1)))
        logits = torch.mm(h, obj_emb.transpose(0, 1)) + history_tag + bias
        preds_two = F.softmax(logits, dim=1)
        nce = torch.sum(torch.gather(torch.log(preds_two + self.eps), 1, actor2.view(-1, 1)))
        nce /= -1. * actor2.shape[0]
        if torch.any(torch.isnan(nce)):
            nce = torch.tensor(0.0, requires_grad=True, device=sub_emb.device)
        return nce, preds_two

    def _build_unique_candidates(self, topm_idx: torch.Tensor, ret_idx: torch.Tensor, true_idx: torch.Tensor, include_true: bool = True):
        """Build per-sample unique candidate list with fixed budget (rerank_max_cands).

        Args:
            include_true: only inject the gold entity during training. Must be False for
                validation / test to avoid label leakage.
        """
        device = topm_idx.device
        B = topm_idx.size(0)
        budget = self.rerank_max_cands
        cand_rows = []
        topm_cpu = topm_idx.detach().cpu()
        ret_cpu = ret_idx.detach().cpu() if ret_idx is not None else None
        true_cpu = true_idx.detach().cpu()

        for b in range(B):
            seen = set()
            uniq = []

            # priority: top-M, retrieved, true
            for x in topm_cpu[b].tolist():
                if x not in seen:
                    uniq.append(x); seen.add(x)
                if len(uniq) >= budget:
                    break

            if len(uniq) < budget and ret_cpu is not None:
                for x in ret_cpu[b].tolist():
                    if x not in seen:
                        uniq.append(x); seen.add(x)
                    if len(uniq) >= budget:
                        break

            if include_true:
                t = int(true_cpu[b].item())
                if t not in seen:
                    if len(uniq) >= budget:
                        uniq[-1] = t
                    else:
                        uniq.append(t)
                    seen.add(t)

            # pad if needed (repeat from topm)
            if len(uniq) < budget:
                for x in topm_cpu[b].tolist():
                    if len(uniq) >= budget:
                        break
                    if x not in seen:
                        uniq.append(x); seen.add(x)

            # final pad with 0
            while len(uniq) < budget:
                uniq.append(0)

            cand_rows.append(torch.tensor(uniq, device=device, dtype=torch.long))

        return torch.stack(cand_rows, dim=0)  # (B,C)

    def compute_recall_at_c(self, actor1, actor2, r, rel_embeds, preds, decoder, current_h, diff_out, include_true: bool = False):
        """Compute Recall@C using the natural candidate set (Top-M union retrieved Top-K, then budgeted)."""
        if current_h is None:
            current_h = self.entity_embeds
        score_enhanced_logits = decoder.forward(current_h, actor1, rel_embeds[r], self.fact_time)
        score_enhanced = F.softmax(score_enhanced_logits, dim=1)
        preds_total = self.gamma * score_enhanced + (1 - self.gamma) * preds
        M = min(self.rerank_topm, preds_total.size(1))
        topm_idx = torch.topk(score_enhanced, k=M, dim=1).indices
        cache = self._pick_cache(rel_embeds)
        ret_idx = cache.get("ent_top", None) if cache else None
        cand_ids = self._build_unique_candidates(topm_idx, ret_idx, actor2, include_true=include_true)
        return (cand_ids == actor2.unsqueeze(1)).any(dim=1).float()

    def calculate_cand_loss(self, actor1, actor2, r, rel_embeds, preds, cand_layer, decoder, gru, gate, current_h, frequency, diff_out, include_true: bool = True):
        """Candidate enhancement + retrieval-augmented reranking.

        Returns:
          loss: base DMSA candidate loss + lambda_rank * rerank_loss + lambda_ret * retrieval_loss
          preds_total: final score distribution over all entities (B, num_e) for ranking
        """
        if current_h is None:
            current_h = self.entity_embeds

        B = actor1.shape[0]
        device = actor1.device

        # DMSA enhanced decoder distribution (strong baseline)
        score_enhanced_logits = decoder.forward(current_h, actor1, rel_embeds[r], self.fact_time)  # (B,N) logits
        score_enhanced = F.softmax(score_enhanced_logits, dim=1)  # (B,N)
        preds_total = self.gamma * score_enhanced + (1 - self.gamma) * preds  # probs

        # Original DMSA candidate loss on enhanced decoder (kept for stability)
        loss_base = F.nll_loss(torch.log(score_enhanced + self.eps), actor2) + self.regularization_loss(reg_param=0.01)

        # Reranking on a union candidate set
        cache = self._pick_cache(rel_embeds)
        rerank_loss = torch.tensor(0.0, device=device)
        ret_loss = torch.tensor(0.0, device=device)

        if cache and "q" in cache and "c" in cache and "ent_top" in cache:
            q = cache["q"]           # (B,d)
            c = cache["c"]           # (B,d)
            ret_idx = cache["ent_top"]  # (B,k)

            # top-M candidates from DMSA enhanced decoder
            M = min(self.rerank_topm, preds_total.size(1))
            topm_idx = torch.topk(score_enhanced, k=M, dim=1).indices  # (B,M)

            cand_ids = self._build_unique_candidates(topm_idx, ret_idx, actor2, include_true=include_true)  # (B,C)
            C = cand_ids.size(1)

            e_cand = current_h[cand_ids]  # (B,C,d)
            q_exp = q.unsqueeze(1).expand(-1, C, -1)
            c_exp = c.unsqueeze(1).expand(-1, C, -1)
            feat = torch.cat([e_cand, q_exp, c_exp], dim=-1)  # (B,C,3d)

            delta = self.reranker(feat.reshape(B * C, -1)).reshape(B, C)  # (B,C)

            base_full_logit = torch.log(preds_total + self.eps)  # (B,N)
            base_cand_logit = torch.gather(base_full_logit, 1, cand_ids)  # (B,C)
            cand_logit = base_cand_logit + self.rerank_eta * delta

            # Candidate-set CE loss is only defined when the gold entity is injected
            # into the candidate set (training only). During validation / test we must
            # not use gold-aware candidate construction.
            if include_true:
                target_pos = (cand_ids == actor2.unsqueeze(1)).float()
                target_idx = torch.argmax(target_pos, dim=1)
                rerank_loss = F.cross_entropy(cand_logit, target_idx)

            # Scatter reranker deltas back to full entity space for final ranking distribution.
            # (Unique cand_ids ensures no unintended accumulation across duplicates.)
            full_logit = base_full_logit.clone()
            full_logit.scatter_add_(1, cand_ids, self.rerank_eta * delta)
            preds_total = F.softmax(full_logit, dim=1)

            # Optional retrieval contrastive loss (in-batch negatives)
            if self.lambda_ret > 0:
                e_true = current_h[actor2]  # (B,d)
                sim = torch.mm(q, e_true.t()) / max(self.ret_tau, self.eps)  # (B,B)
                labels = torch.arange(B, device=device)
                ret_loss = F.cross_entropy(sim, labels)

        loss = loss_base + self.lambda_rank * rerank_loss + self.lambda_ret * ret_loss
        if torch.any(torch.isnan(loss)):
            loss = torch.tensor(0.0, requires_grad=True, device=device)

        return loss, preds_total


# Export name used by main.py
REXPERTNET = REXPERTNET_R2
