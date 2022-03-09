import torch
import torch.nn as nn
import logging
from . import data
import os
from tqdm import tqdm

from typing import Callable, Optional


def init_xavier(layer, verbose: bool = False):
    if hasattr(layer, "bias") and layer.bias is not None:
        nn.init.constant_(layer.bias, 0)
    if hasattr(layer, "weight"):
        if isinstance(layer, (nn.Conv2d, nn.Linear)):
            nn.init.xavier_uniform_(layer.weight)
        else:
            if verbose:
                print("skipping initialization of layer:", layer)


def init_ones(layer, verbose: bool = False):
    """For debugging models."""
    if hasattr(layer, "bias") and layer.bias is not None:
        layer.bias = nn.Parameter(torch.zeros_like(layer.bias))
    if hasattr(layer, "weight"):
        if isinstance(layer, (nn.Conv2d, nn.Linear)):
            layer.weight = nn.Parameter(torch.ones_like(layer.weight))
        else:
            if verbose:
                print("skipping initialization of layer:", layer)


def min_gpt_init_weights(module):
    """Copied from https://github.com/karpathy/minGPT/blob/master/mingpt/model.py."""
    if isinstance(module, (nn.Linear, nn.Embedding)):
        module.weight.data.normal_(mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)


@torch.no_grad()
def eval_model(
    model: nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: str,
    loss_fn: Callable,
    model_call: Optional[Callable] = None,
) -> float:
    model.to(device)
    model.eval()
    total_per_token_loss = 0.0
    n_batches = 0
    for i, (inputs, target) in enumerate(test_loader):
        inputs = inputs.to(device)
        target = target.to(device)
        if model_call is None:
            output = model(inputs)
        else:
            output = model_call(model, inputs)
        total_per_token_loss += loss_fn(output.transpose(-1, -2), target).item()
        n_batches += 1
    return total_per_token_loss / float(n_batches)


def train_epoch(
    model,
    loss_histories: dict,
    expt_key: str,
    device: str,
    train_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    data_config: data.DatasetConfig,
    save_every: int = 50,  # number of examples between saves
    eval_every: int = 100,  # number of examples between evals
    save_state_only: bool = False,
    model_call: Optional[Callable[[nn.Module, torch.tensor], torch.tensor]] = None,
    model_dir: Optional[str] = None,
):
    """Minimal train function that saves checkpoints and loss histories."""
    loss_histories[expt_key] = dict()
    loss_histories[expt_key]["train_loss"] = []
    loss_histories[expt_key]["test_loss"] = []
    i = 0

    model.to(device)

    if model_dir is None:
        model_dir = f"./saved_models/{expt_key}"
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    logging.info(f"saving models to {model_dir}")

    loss_fn = nn.CrossEntropyLoss(ignore_index=data_config.pad_id, reduction="mean")
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)

    batch_sz = train_loader.batch_size
    ex_seen = 0
    save_threshold = max(save_every, batch_sz)
    save_threshold_step = save_threshold
    eval_threshold = max(eval_every, batch_sz)
    eval_threshold_step = eval_threshold
    for i, (inputs, target) in tqdm(
        enumerate(train_loader),
        total=len(train_loader),
        position=0,
        leave=True,
    ):
        hist_entry = dict()

        model.train()
        inputs = inputs.to(device)
        target = target.to(device)

        if model_call is None:
            output = model(inputs)
        else:
            output = model_call(model, inputs)

        loss = loss_fn(output.transpose(-1, -2), target)
        loss.requires_grad_()

        model.zero_grad()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)  # doesn't seem necessary yet for our scale
        loss.backward()
        opt.step()
        train_loss = loss.item()
        loss_histories[expt_key]["train_loss"].append((ex_seen, train_loss))

        ex_seen += batch_sz

        if ex_seen >= save_threshold:
            if save_state_only:
                torch.save(model.state_dict(), f"{model_dir}/state_{i}")
            else:
                torch.save(model, f"{model_dir}/model_{i}")
            save_threshold += save_threshold_step

        if ex_seen >= eval_threshold:
            eval_threshold += eval_threshold_step
            test_loss = eval_model(model, test_loader, device, loss_fn, model_call)
            loss_histories[expt_key]["test_loss"].append((ex_seen, test_loss))
            print(
                f"{ex_seen:>12}: training loss {train_loss:>6}, test loss {test_loss:>6}"
            )
    test_loss = eval_model(model, test_loader, device, loss_fn, model_call)
    loss_histories[expt_key]["test_loss"].append((ex_seen, test_loss))
    print(
        f"Final: {ex_seen:>12}: training loss {train_loss:>6}, test loss {test_loss:>6}"
    )
    if save_state_only:
        torch.save(model.state_dict(), f"{model_dir}/state_final")
    else:
        torch.save(model, f"{model_dir}/model_final")
