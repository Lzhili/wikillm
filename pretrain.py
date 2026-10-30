#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WikiLLM · 从零预训练一个小语言模型（Apple Silicon / CUDA 双平台）
基于 https://docs.swanlab.cn/examples/pretrain_llm.html 改造。

相对原教程的改动（每一条都是为了让它在 M5 MacBook Air 16GB 上跑得完）：
  1. 精度：默认 fp32。MPS 的 bf16 逐算子支持、fp16 易 NaN；--precision bf16 会先自检，不支持自动回退。
  2. batch：32 → 8，梯度累积 8 → 32，等效 batch 仍是 256，梯度行为不变。
  3. eval：测试集默认裁到 2000 条。原版每 500 步扫 14 万条，光评估就能拖垮训练。
  4. 存档：默认每 2000 步、只留 2 份（每份约 500MB），支持 --resume 断点续训。
  5. dataloader：MPS 强制单进程（多进程会卡死），CUDA 自动开 8 进程 + pin_memory。
  6. 新增吞吐/ETA 回调，训练中直接告诉你 token/s 和还要多久。
  7. swanlab 改为可选（--swanlab），默认完全离线，不需要登录。
  8. --subset 先截断原文再分词，试探阶段几分钟就能跑通全流程。

用法：
  # 1) 小规模试跑（5 万篇，约 1-2 小时）
  python pretrain.py --data data/wikipedia-zh-cn-20240820.json --subset 50000

  # 2) 断点续训
  python pretrain.py --data data/... --resume

  # 3) 全量 + 4 进程分词
  python pretrain.py --data data/... --num-proc 4 --save-steps 2000

  # 4) 只准备数据不训练（先看看切块后剩多少）
  python pretrain.py --data data/... --subset 20000 --prepare-only

  # 5) 云 GPU（CUDA 单卡）上跑全量，batch 可以放大回去
  python pretrain.py --data data/... --precision bf16 --batch-size 32 --grad-accum 8 --num-proc 8
"""

import argparse
import math
import os
import sys
import time

# 必须在 import torch 之前设置：让 MPS 缺失的算子自动回落到 CPU，而不是直接报错中断
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
import datasets
import transformers
from transformers import TrainerCallback

DEFAULT_MODEL_ID = "Qwen/Qwen2-0.5B"
DEFAULT_MODEL_DIR = "Qwen2-0.5B"
PROMPTS = ["人工智能", "牛顿", "北京市", "亚洲历史"]


# ---------------------------------------------------------------- 环境 / 设备


def pick_device(prefer: str = "auto") -> str:
    if prefer != "auto":
        return prefer
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def bf16_ok(device: str) -> bool:
    if device == "cuda":
        return torch.cuda.is_bf16_supported()
    if device == "mps":
        return mps_supports_bf16()
    return False


def mps_supports_bf16() -> bool:
    """MPS 上的 bf16 是按算子逐步支持的，这里实测一次矩阵乘。"""
    if not torch.backends.mps.is_available():
        return False
    try:
        a = torch.randn(8, 8, device="mps", dtype=torch.bfloat16)
        val = (a @ a).float().sum().item()
        return math.isfinite(val)
    except Exception:
        return False


def mps_memory_gb() -> float:
    try:
        return torch.mps.driver_allocated_memory() / 1024**3
    except Exception:
        return 0.0


def fmt_hms(seconds: float) -> str:
    seconds = int(max(seconds, 0))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h} 小时 {m} 分"
    if m:
        return f"{m} 分 {s} 秒"
    return f"{s} 秒"


# ---------------------------------------------------------------- 模型


MODEL_FILES = ("config.json", "tokenizer.json", "tokenizer_config.json", "generation_config.json")

# 直连 HTTP 拿文件，绕开 modelscope / huggingface_hub 两个库的列表接口（国内经常卡在这一步）
MIRRORS = (
    "https://modelscope.cn/models/{repo}/resolve/master/{fn}",
    "https://hf-mirror.com/{repo}/resolve/main/{fn}",
    "https://huggingface.co/{repo}/resolve/main/{fn}",
)


def _http_download(url: str, dst: str, timeout: int = 40, attempts: int = 2) -> int:
    import urllib.request

    last = None
    for i in range(attempts):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "wikillm-mps/1.0"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = resp.read()
            with open(dst, "wb") as f:
                f.write(data)
            return len(data)
        except Exception as exc:
            last = exc
            time.sleep(1 + i)
    raise last


def ensure_local_model(model_id: str = DEFAULT_MODEL_ID, model_dir: str = DEFAULT_MODEL_DIR) -> str:
    """只下载 config.json 和 tokenizer 文件（约 7MB），不下 0.5B 的权重。"""
    if os.path.isfile(os.path.join(model_dir, "config.json")):
        return model_dir
    os.makedirs(model_dir, exist_ok=True)
    for fn in MODEL_FILES:
        dst = os.path.join(model_dir, fn)
        if os.path.isfile(dst) and os.path.getsize(dst) > 0:
            continue
        errors = []
        for tpl in MIRRORS:
            url = tpl.format(repo=model_id, fn=fn)
            try:
                size = _http_download(url, dst)
                print(f"[model] 下载 {fn} ({size / 1024:.0f} KB) ← {url.split('/')[2]}")
                break
            except Exception as exc:
                errors.append(f"{url.split('/')[2]}: {exc}")
        else:
            # generation_config.json 之类缺了也能跑，不影响
            if fn == "config.json":
                raise RuntimeError(
                    f"下载 {fn} 失败，请手动把 Qwen/Qwen2-0.5B 的 config.json 与 "
                    f"tokenizer*.json 放到 {model_dir}/\n  " + "\n  ".join(errors)
                )
            print(f"[model] 跳过 {fn}（下载失败，不影响训练）")
    return model_dir


def load_tokenizer(model_dir: str = DEFAULT_MODEL_DIR):
    tok = transformers.AutoTokenizer.from_pretrained(model_dir)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def build_model(tokenizer, model_dir: str = DEFAULT_MODEL_DIR):
    """Qwen2 骨架 + 缩小的尺寸，随机初始化，约 123M 参数。"""
    config = transformers.AutoConfig.from_pretrained(model_dir)
    config.vocab_size = len(tokenizer)
    config.hidden_size = 512
    config.intermediate_size = 2048
    config.num_attention_heads = 8
    config.num_key_value_heads = 2
    config.num_hidden_layers = 12
    try:  # Qwen2-0.5B 的 config 里带 head_dim=64，与 512/8 一致；老版本没有该字段就跳过
        config.head_dim = config.hidden_size // config.num_attention_heads
    except AttributeError:
        pass
    config.tie_word_embeddings = True  # 词表 embedding 占 78M，不共享 lm_head 会变 200M
    config.bos_token_id = tokenizer.bos_token_id
    config.eos_token_id = tokenizer.eos_token_id
    model = transformers.Qwen2ForCausalLM(config)
    return model, config


# ---------------------------------------------------------------- 数据


def prepare_datasets(args, tokenizer):
    if not os.path.isfile(args.data):
        raise FileNotFoundError(
            f"找不到数据文件：{args.data}\n"
            "下载地址（约 1.99GB）：\n"
            "  百度网盘 / huggingface: fjcanyue/wikipedia-zh-cn\n"
            "放到 ./data/ 目录下，文件名 wikipedia-zh-cn-20240820.json"
        )

    print(f"[data] 读取 {args.data}（首次会建 arrow 缓存，磁盘约多占 2-3GB）...")
    raw = datasets.load_dataset("json", data_files=args.data, split="train")
    print(f"[data] 原始 {len(raw):,} 篇")

    if args.subset and args.subset < len(raw):
        raw = raw.select(range(args.subset))
        print(f"[data] --subset 生效，只取前 {len(raw):,} 篇")

    raw = raw.train_test_split(test_size=0.1, seed=args.seed)
    ctx = args.context_length
    cols = raw["train"].column_names

    def tokenize(batch):
        out = tokenizer(
            batch["text"],
            truncation=True,
            max_length=ctx,
            return_overflowing_tokens=True,
            return_length=True,
        )
        # 与原教程一致：只留长度正好等于 ctx 的完整块，尾块丢弃
        return {
            "input_ids": [
                ids for length, ids in zip(out["length"], out["input_ids"]) if length == ctx
            ]
        }

    tok_ds = raw.map(
        tokenize,
        batched=True,
        batch_size=1000,
        remove_columns=cols,
        num_proc=args.num_proc,
        load_from_cache_file=True,
        desc="tokenize",
    )

    train_ds, eval_ds = tok_ds["train"], tok_ds["test"]
    if args.eval_samples and len(eval_ds) > args.eval_samples:
        eval_ds = eval_ds.select(range(args.eval_samples))

    n_train = len(train_ds)
    if n_train == 0:
        raise RuntimeError("切块后一条样本都没有，检查数据文件格式（需要含 text 字段）")
    kept_ratio = n_train / max(len(raw["train"]), 1)
    print(
        f"[data] 训练块 {n_train:,} / 评估块 {len(eval_ds):,} · "
        f"每块 {ctx} token ≈ {n_train * ctx / 1e6:.1f}M token/epoch · "
        f"平均每篇保留 {kept_ratio:.2f} 块"
    )
    if kept_ratio < 0.5:
        print(
            "[data] 注意：平均每篇不足 1 块，说明大量短条目被整条丢弃（原教程的定长过滤），"
            "实际可用数据比文件体积小得多"
        )
    return train_ds, eval_ds


# ---------------------------------------------------------------- 训练


class SpeedETA(TrainerCallback):
    """打印 token/s 与剩余时间，比 loss 曲线更有用。"""

    def __init__(self, tokens_per_step: int):
        self.tokens_per_step = tokens_per_step
        self.t0 = None

    def on_train_begin(self, args, state, control, **kwargs):
        self.t0 = time.time()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not self.t0 or state.global_step <= 0:
            return
        elapsed = time.time() - self.t0
        done_tokens = state.global_step * self.tokens_per_step
        tps = done_tokens / max(elapsed, 1e-6)
        remain = (state.max_steps - state.global_step) * self.tokens_per_step / max(tps, 1.0)
        print(
            f"[{args.device}] {tps / 1000:.2f}k token/s · 已处理 {done_tokens / 1e6:.2f}M token · "
            f"进度 {state.global_step}/{state.max_steps} · 预计还需 {fmt_hms(remain)}"
        )


def make_training_args(args, use_bf16: bool):
    eval_strategy = "steps" if args.eval_steps > 0 else "no"
    common = dict(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        weight_decay=0.1,
        warmup_steps=args.warmup_steps,
        lr_scheduler_type="cosine",
        optim="adamw_torch",
        logging_steps=args.log_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        dataloader_num_workers=getattr(args, "num_workers", 0),
        dataloader_pin_memory=getattr(args, "pin_memory", False),
        seed=args.seed,
        report_to=["swanlab"] if args.swanlab else [],
        fp16=False,
        bf16=use_bf16,
        eval_steps=None if args.eval_steps <= 0 else args.eval_steps,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        save_safetensors=True,
    )
    try:  # transformers >= 4.41
        return transformers.TrainingArguments(eval_strategy=eval_strategy, **common)
    except TypeError:  # 老版本
        return transformers.TrainingArguments(evaluation_strategy=eval_strategy, **common)


def make_trainer(model, tokenizer, ta, collator, train_ds, eval_ds, callbacks):
    kwargs = dict(
        model=model,
        args=ta,
        data_collator=collator,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        callbacks=callbacks,
    )
    try:  # transformers >= 4.46 用 processing_class
        return transformers.Trainer(processing_class=tokenizer, **kwargs)
    except TypeError:
        try:
            return transformers.Trainer(tokenizer=tokenizer, **kwargs)
        except TypeError:  # 更老的版本或参数被移除时，collator 已经足够
            return transformers.Trainer(**kwargs)


# ---------------------------------------------------------------- 生成


def generate_samples(model, tokenizer, device, prompts=PROMPTS, max_new_tokens=40):
    model.eval()
    dev = torch.device(device)
    try:
        pipe = transformers.pipeline(
            "text-generation", model=model, tokenizer=tokenizer, device=dev
        )
    except Exception as exc:
        print(f"[gen] {device} 上建 pipeline 失败（{exc}），回退到 CPU 生成")
        model.to("cpu")
        pipe = transformers.pipeline(
            "text-generation", model=model, tokenizer=tokenizer, device=torch.device("cpu")
        )
    results = []
    for prompt in prompts:
        try:
            text = pipe(
                prompt,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.8,
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.eos_token_id,
                num_return_sequences=1,
            )[0]["generated_text"]
        except Exception as exc:
            text = f"<生成失败: {exc}>"
        print(f"[生成] {text}")
        results.append(text)
    return results


# ---------------------------------------------------------------- 主流程


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="WikiLLM · 从零预训练小模型")
    p.add_argument("--data", default="data/wikipedia-zh-cn-20240820.json", help="wiki json 路径")
    p.add_argument("--subset", type=int, default=0, help="只用前 N 篇原文（0=全量）")
    p.add_argument("--eval-samples", type=int, default=2000, help="评估集最多用多少块（0=全用）")
    p.add_argument("--context-length", type=int, default=512)
    p.add_argument("--precision", choices=["fp32", "bf16"], default="fp32")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--grad-accum", type=int, default=32)
    p.add_argument("--epochs", type=float, default=2.0)
    p.add_argument("--max-steps", type=int, default=-1, help=">0 时覆盖 epochs")
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--warmup-steps", type=int, default=200)
    p.add_argument("--log-steps", type=int, default=20)
    p.add_argument("--save-steps", type=int, default=2000)
    p.add_argument("--eval-steps", type=int, default=2000, help="0=不做评估")
    p.add_argument("--save-total-limit", type=int, default=2)
    p.add_argument("--output-dir", default="WikiLLM")
    p.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    p.add_argument("--num-proc", type=int, default=1, help="分词进程数，全量时建议 4")
    p.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    p.add_argument("--num-workers", type=int, default=-1, help="dataloader 进程数，-1=自动")
    p.add_argument("--seed", type=int, default=2333)
    p.add_argument("--resume", action="store_true", help="从 output-dir 里最后一个 checkpoint 续训")
    p.add_argument("--swanlab", action="store_true", help="启用 SwanLab 在线记录（需先登录）")
    p.add_argument("--prepare-only", action="store_true", help="只准备数据不训练")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    device = pick_device(args.device)
    torch.manual_seed(args.seed)

    if args.num_workers < 0:  # 未指定时按设备给默认值：MPS 必须单进程
        args.num_workers = 0 if device == "mps" else (8 if device == "cuda" else 2)
    args.pin_memory = device == "cuda"

    print("=" * 66)
    print(f"[env] device={device} · torch={torch.__version__} · transformers={transformers.__version__}")
    if device == "cuda":
        print(f"[env] {torch.cuda.get_device_name(0)} · bf16 {'支持' if bf16_ok('cuda') else '不支持'}")
    elif device == "mps":
        print(f"[env] MPS bf16 自检：{'支持' if mps_supports_bf16() else '不支持'}")

    use_bf16 = False
    if args.precision == "bf16":
        if bf16_ok(device):
            use_bf16 = True
            print(f"[env] --precision bf16 生效（{device}）")
        else:
            print(f"[env] {device} 上 bf16 不可用，自动回退 fp32")

    model_dir = ensure_local_model(model_dir=args.model_dir)
    tokenizer = load_tokenizer(model_dir)

    train_ds, eval_ds = prepare_datasets(args, tokenizer)
    if args.prepare_only:
        print("[data] --prepare-only，退出。")
        return 0

    model, config = build_model(tokenizer, model_dir)
    n_params = sum(p.numel() for p in model.parameters())
    n_embed = config.vocab_size * config.hidden_size
    print(
        f"[model] {n_params / 1e6:.1f}M 参数，其中 embedding {n_embed / 1e6:.1f}M "
        f"({n_embed / n_params * 100:.0f}%)，transformer 主干 {(n_params - n_embed) / 1e6:.1f}M"
    )

    collator = transformers.DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    ta = make_training_args(args, use_bf16)

    tokens_per_step = args.batch_size * args.grad_accum * args.context_length
    print(
        f"[train] 等效 batch {args.batch_size}×{args.grad_accum}={args.batch_size * args.grad_accum} 块 "
        f"= {tokens_per_step / 1000:.0f}k token/步"
    )
    callbacks = [SpeedETA(tokens_per_step)]
    if args.swanlab:
        from swanlab.integration.transformers import SwanLabCallback

        callbacks.append(SwanLabCallback())

    trainer = make_trainer(model, tokenizer, ta, collator, train_ds, eval_ds, callbacks)

    ckpt = True if args.resume else None
    if args.resume:
        print(f"[train] 断点续训：{args.output_dir} 下最后一个 checkpoint")

    t0 = time.time()
    trainer.train(resume_from_checkpoint=ckpt)
    total = time.time() - t0
    print(f"[train] 完成，耗时 {fmt_hms(total)}")

    weight_dir = os.path.join(args.output_dir, "Weight")
    model.save_pretrained(weight_dir)
    tokenizer.save_pretrained(weight_dir)
    print(f"[save] 权重已存到 {weight_dir}")

    texts = generate_samples(model, tokenizer, device)
    if args.swanlab:
        import swanlab

        swanlab.log({"Generate": [swanlab.Text(t) for t in texts]})
        swanlab.finish()
    print(f"[mem] 显存/内存占用约 {mps_memory_gb():.2f} GB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
