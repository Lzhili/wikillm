#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
环境自检 + 20 步冒烟训练。
不下载 2GB 数据集，用合成中文文本跑完整链路（建模型 → 训练 → 存档 → 生成），
用来确认设备能跑、精度设置对、显存够、吞吐大概多少。

用法：
  python smoke_test.py                # 默认 20 步
  python smoke_test.py --steps 50
  python smoke_test.py --precision bf16
"""

import argparse
import os
import platform
import sys
import time

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

BASE = (
    "数学是研究数量、结构以及空间等概念及其变化的一门学科，属于形式科学的一种。"
    "数学利用抽象化和逻辑推理，从计数、计算、量度、对物体形状及运动的观察发展而成。"
    "数学家们拓展这些概念，以公式化新的猜想以及从选定的公理及定义中建立起严谨推导出的定理。"
)


def check_deps():
    missing = []
    for mod in ("torch", "transformers", "datasets"):
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)
    if missing:
        print("缺少依赖：" + ", ".join(missing))
        print("请先执行： python -m pip install -r requirements.txt")
        return False
    return True


def build_toy_dataset(tokenizer, ctx=128, n_blocks=240):
    """造定长块，绕开真实数据里'不足 512 就丢弃'的过滤，保证冒烟测试一定有样本。"""
    import datasets

    long_text = BASE * 600
    ids = tokenizer(long_text, add_special_tokens=False)["input_ids"]
    blocks = [ids[i : i + ctx] for i in range(0, len(ids) - ctx + 1, ctx)]
    blocks = (blocks * ((n_blocks // max(len(blocks), 1)) + 1))[:n_blocks]
    return datasets.Dataset.from_dict({"input_ids": blocks})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--context-length", type=int, default=128)
    ap.add_argument("--precision", choices=["fp32", "bf16"], default="fp32")
    ap.add_argument("--out", default="smoke_out")
    args = ap.parse_args()

    if not check_deps():
        return 1

    import torch
    import transformers
    import datasets
    import pretrain as P

    print("=" * 66)
    print(f"python       {platform.python_version()} ({platform.machine()})")
    print(f"torch        {torch.__version__}")
    print(f"transformers {transformers.__version__} · datasets {datasets.__version__}")

    device = P.pick_device()
    print(f"device       {device}")
    if device == "cuda":
        print(f"gpu          {torch.cuda.get_device_name(0)}")
    elif device != "mps":
        print("!! 没有检测到 GPU，训练会退到 CPU（极慢）")

    ok = P.bf16_ok(device)
    print(f"bf16         {'支持' if ok else '不支持'}")
    use_bf16 = args.precision == "bf16" and ok
    if args.precision == "bf16" and not ok:
        print("!! bf16 不可用，自动回退 fp32")

    model_dir = P.ensure_local_model()
    tokenizer = P.load_tokenizer(model_dir)
    print(f"tokenizer    {tokenizer.__class__.__name__} · 词表 {len(tokenizer):,}")

    model, config = P.build_model(tokenizer, model_dir)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model        {n_params / 1e6:.1f}M 参数 · {config.num_hidden_layers} 层 · "
          f"hidden {config.hidden_size}")

    ds = build_toy_dataset(tokenizer, args.context_length)
    print(f"dataset      {len(ds)} 块 × {args.context_length} token（合成数据）")

    ta_args = argparse.Namespace(
        output_dir=args.out,
        batch_size=args.batch_size,
        grad_accum=2,
        lr=5e-4,
        warmup_steps=5,
        log_steps=5,
        save_steps=10,
        save_total_limit=1,
        eval_steps=0,
        num_train_epochs=1.0,
        max_steps=args.steps,
        seed=2333,
        swanlab=False,
    )
    ta = P.make_training_args(ta_args, use_bf16)
    collator = transformers.DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    tokens_per_step = args.batch_size * ta_args.grad_accum * args.context_length
    trainer = P.make_trainer(model, tokenizer, ta, collator, ds, None,
                             [P.SpeedETA(tokens_per_step)])

    print("-" * 66)
    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0
    tps = args.steps * tokens_per_step / max(elapsed, 1e-6)
    print(f"{args.steps} 步训练耗时 {elapsed:.1f}s · 吞吐约 {tps / 1000:.2f}k token/s")

    weight_dir = os.path.join(args.out, "Weight")
    model.save_pretrained(weight_dir)
    tokenizer.save_pretrained(weight_dir)
    print(f"存档         {weight_dir}")

    print("-" * 66)
    P.generate_samples(model, tokenizer, device, prompts=["人工智能", "数学"], max_new_tokens=16)
    print(f"内存占用     约 {P.mps_memory_gb():.2f} GB")

    print("=" * 66)
    if tps > 0:
        full_tokens = 300e6  # 全量 2 epoch 的粗略量级（按实际切块数会浮动）
        print(f"推算：跑完 3 亿 token 约需 {P.fmt_hms(full_tokens / tps)}")
    print("冒烟测试通过：环境、模型、训练、存档、生成链路都正常。")
    print("下一步：把 wiki json 放进 data/ 后执行")
    print("  python pretrain.py --data data/wikipedia-zh-cn-20240820.json --subset 20000")
    return 0


if __name__ == "__main__":
    sys.exit(main())
