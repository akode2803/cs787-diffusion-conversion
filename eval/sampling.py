"""Batched generation for both kinds of arm, timed with CUDA synchronisation.

Diffusion: BYOD's official LLaDA sampler (low-confidence remasking, temperature 0, one block), extended to a batch and
to masks anywhere in the answer, which infilling needs. Each row is laid out as [prompt][answer region][padding].
Padding goes at the end and is hidden from attention, so every real token keeps its position and sees exactly what
it would see alone. BYOD's own sampler runs one row with no padding; `gpu_check.py` measures how closely we match it.
The step at which each position was committed is returned, for the decoding-order analysis.

AR: Hugging Face greedy generation with a KV cache and left padding.
"""
import time
import torch
import torch.nn.functional as F


def schedule(mask_count, steps):
    """How many masks to commit at each step. Same rule as BYOD's _llada_transfer_schedule."""
    base, remainder = divmod(int(mask_count), int(steps))
    return [base + int(i < remainder) for i in range(int(steps))]


def _masked_logits(arm, x, real, positions):
    """Logits at the masked positions only: [number of masks, vocabulary]."""
    if arm.llada:
        from diffusion_lm.inference import forward_denoising
        return forward_denoising(arm.session, x, ~real)[positions]
    causal_lm = arm.model.get_base_model()
    dtype = next(p.dtype for p in arm.model.parameters() if p.is_floating_point())
    bias = torch.zeros(x.shape[0], 1, x.shape[1], x.shape[1], device=x.device, dtype=dtype)
    bias.masked_fill_(~real[:, None, None, :], torch.finfo(dtype).min)       # hide padding keys
    hidden = causal_lm.model(input_ids=x, attention_mask=bias, use_cache=False).last_hidden_state
    return causal_lm.get_output_embeddings()(hidden[positions])


@torch.inference_mode()
def diffusion_batch(arm, rows, steps, *, delay_eos=False, ban_eos=False, eot_id=None):
    """rows: token lists where answer positions to fill hold arm.mask_token_id.

    Returns (filled rows, commit step per position with -1 for fixed tokens, seconds).
    delay_eos is BYOD's confidence_eos_eot_inf, ban_eos its logits_eos_inf.
    """
    mask_id, eos_id = arm.mask_token_id, int(arm.tokenizer.eos_token_id)
    pad_id = arm.tokenizer.pad_token_id if arm.tokenizer.pad_token_id is not None else eos_id
    lengths = torch.tensor([len(r) for r in rows], device=arm.device)
    width = int(lengths.max())
    x = torch.full((len(rows), width), pad_id, dtype=torch.long, device=arm.device)
    for i, r in enumerate(rows):
        x[i, :len(r)] = torch.tensor(r, device=arm.device)
    real = torch.arange(width, device=arm.device)[None, :] < lengths[:, None]
    masked = (x == mask_id) & real
    plans = [schedule(int(c), steps) for c in masked.sum(1).tolist()]
    commit = torch.full_like(x, -1)
    torch.cuda.synchronize(); start = time.perf_counter()
    for step in range(steps):
        masked = (x == mask_id) & real
        if not masked.any():
            break
        positions = masked.nonzero(as_tuple=True)
        logits = _masked_logits(arm, x, real, positions)
        if ban_eos:
            logits[:, eos_id] = -torch.inf
        predicted = logits.argmax(-1)
        confidence = F.softmax(logits, dim=-1).gather(-1, predicted[:, None]).squeeze(-1)
        if delay_eos:
            special = predicted == eos_id
            if eot_id is not None:
                special |= predicted == int(eot_id)
            confidence = confidence.masked_fill(special, torch.finfo(confidence.dtype).min)
        full_conf = torch.full(x.shape, -torch.inf, device=x.device, dtype=confidence.dtype)
        full_pred = torch.zeros_like(x)
        full_conf[positions], full_pred[positions] = confidence, predicted
        for i in range(len(rows)):
            count = plans[i][step]
            if count:
                chosen = torch.topk(full_conf[i], k=count).indices
                x[i, chosen] = full_pred[i, chosen]
                commit[i, chosen] = step
    torch.cuda.synchronize()
    seconds = time.perf_counter() - start
    return [x[i, :len(r)].tolist() for i, r in enumerate(rows)], [commit[i, :len(r)].tolist() for i, r in enumerate(rows)], seconds


@torch.inference_mode()
def ar_batch(arm, prompts, max_new_tokens):
    """Greedy generation. Returns (new tokens per prompt, cut at the first end token, seconds)."""
    pad_id = arm.tokenizer.pad_token_id if arm.tokenizer.pad_token_id is not None else arm.tokenizer.eos_token_id
    width = max(len(p) for p in prompts)
    ids = torch.full((len(prompts), width), pad_id, dtype=torch.long)
    attention = torch.zeros_like(ids)
    for i, p in enumerate(prompts):
        ids[i, width - len(p):] = torch.tensor(p)
        attention[i, width - len(p):] = 1
    ids, attention = ids.to(arm.device), attention.to(arm.device)
    torch.cuda.synchronize(); start = time.perf_counter()
    out = arm.model.generate(input_ids=ids, attention_mask=attention, max_new_tokens=max_new_tokens, do_sample=False,
                             use_cache=True, pad_token_id=pad_id, eos_token_id=sorted(arm.eos_ids),
                             temperature=None, top_p=None, top_k=None)
    torch.cuda.synchronize()
    seconds = time.perf_counter() - start
    stops, results = arm.eos_ids, []
    for row in out[:, width:].tolist():
        cut = next((j for j, t in enumerate(row) if t in stops), len(row))
        results.append(row[:cut])
    return results, seconds
