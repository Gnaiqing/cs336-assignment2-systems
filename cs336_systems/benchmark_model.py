import argparse
import torch
import timeit
import numpy as np
import yaml
import pprint
from cs336_basics.cs336_basics.model import BasicsTransformerLM
from einops import rearrange, einsum, reduce
from utils import merge_args_with_yaml, merge_dictionaries_recursively


def benchmark_time(model, args):
    loss_fn = torch.nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters())
    fw_time = []
    bw_time = []
    for i in range(args.warmup_steps+args.test_steps):
        batch_input = torch.randint(args.vocab_size, size=(args.batch_size, args.context_length), device=args.device)
        batch_target = torch.randint(args.vocab_size, size=(args.batch_size, args.context_length), device=args.device)
        batch_target = rearrange(batch_target, "b s -> (b s)")
        st = timeit.default_timer()
        logits = model(batch_input)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        if i >= args.warmup_steps:
            fw_ed = timeit.default_timer()
            fw_time.append(fw_ed - st)

        if args.include_backward:
            opt.zero_grad()
            loss = loss_fn(rearrange(logits, "b s c -> (b s) c"), batch_target)
            loss.backward()
            opt.step()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            if i >= args.warmup_steps:
                bw_ed = timeit.default_timer()
                bw_time.append(bw_ed - fw_ed)

    fw_time_mean = np.mean(fw_time).item()
    fw_time_std = np.std(fw_time).item()
    bw_time_mean = np.mean(bw_time).item() if args.include_backward else np.nan
    bw_time_std = np.std(bw_time).item() if args.include_backward else np.nan
    return {
        "fw_time_mean": fw_time_mean,
        "fw_time_std": fw_time_std,
        "bw_time_mean": bw_time_mean,
        "bw_time_std": bw_time_std
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # model size
    parser.add_argument('--config', type=str, default=None, help='Specify configuration files.')
    parser.add_argument("--vocab_size", type=int, default=10000)
    parser.add_argument("--context_length", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=12)
    parser.add_argument("--num_heads", type=int, default=12)
    parser.add_argument("--d_model", type=int,default=768)
    parser.add_argument("--d_ff", type=int, default=3072)
    parser.add_argument("--warmup_steps", type=int, default=5)
    parser.add_argument("--test_steps", type=int, default=10)
    parser.add_argument("--include_backward", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    if args.config:
        with open(args.config) as cf_file:
            cfg = yaml.safe_load(cf_file.read())
        args = merge_args_with_yaml(parser, cfg)

    if args.device == "cuda":
        if not torch.cuda.is_available():
            print("Cuda not available. Switching back to CPU")
            args.device = "cpu"

    config = vars(args)

    model = BasicsTransformerLM(vocab_size=args.vocab_size,
                                context_length=args.context_length,
                                d_model=args.d_model,
                                num_layers=args.num_layers,
                                num_heads=args.num_heads,
                                d_ff=args.d_ff,
                                rope_theta=1e4)
    result = benchmark_time(model, args)
    merge_dictionaries_recursively(config, result)
    pprint.pprint(config)








