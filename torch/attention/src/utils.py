import torch
import torch.nn as nn
import torch.nn.functional as F
from . import data


@torch.no_grad()
def sample_autoregressive(
    model: nn.Module,
    prompt: str,
    steps: int,
    data_config: data.DatasetConfig,
    device: str = "cpu",
    sample: bool = True,
    top_k: int = 0,  # 0 indicates no top k truncation
    temperature: float = 1.0,
):
    """Autoregressive sampling of provided model.

    model: model to be sampled
    prompt: prompt input, can be empty
    steps: number of tokens to sample after the prompt
    data_config: data configuration for the model, used to pull special tokens and tokenizer
    device: cpu or cuda
    sample: whether deterministic behavior (top-1) or sampling should be used to generate next token
    top_k: truncate logits to the top k before sampling. Note if sampling is
        False then this doesn't affect the output of the function.
    temperature: higher numbers smooths out the softmax function.

    """
    model.eval()
    model.to(device)

    input_tok = [data_config.start_id] + data_config.tokenizer.encode(prompt).ids
    input_tok = torch.tensor(input_tok).unsqueeze(0).to(device)

    for _ in range(steps):
        logits = model(input_tok)  # (batch, seqlen, vocab)
        if isinstance(logits, tuple):
            logits = logits[0]  # min GPT returns a tuple
        logits = logits[0, -1, :]
        probs = F.softmax(logits, dim=-1)

        if top_k == 0:
            k_score = probs
            k_ind = torch.arange(0, logits.shape[-1])
        if top_k > 0:
            k_score, k_ind = torch.topk(probs, k=top_k)

        if sample:
            idx = torch.multinomial(k_score, num_samples=1)
            next_tok = k_ind[idx].unsqueeze(0)
        else:
            next_tok = k_ind[torch.argmax(k_score)]

        input_tok = torch.cat((input_tok, next_tok), dim=1)
        if next_tok[-1] == data_config.end_id:
            break
    toks = input_tok.cpu().numpy().squeeze()
    # topkscore, topkind = torch.topk(probs, k=3, dim=-1)
    return bytes(data_config.tokenizer.decode(toks), encoding="utf-8")
