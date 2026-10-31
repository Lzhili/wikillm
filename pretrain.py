import datasets
import transformers
import swanlab
from swanlab.integration.transformers import SwanLabCallback
import modelscope

def main():
    # using swanlab to save log
    swanlab.init(project="WikiLLM")

    # load dataset
    raw_datasets = datasets.load_dataset(
        "json", data_files="WIKI_CN/wikipedia-zh-cn-20240820.json"
    )

    raw_datasets = raw_datasets["train"].train_test_split(test_size=0.1, seed=2333)
    print("dataset info")
    print(raw_datasets)

    # load tokenizers
    # 因为国内无法直接访问HuggingFace，因此使用魔搭将模型的配置文件和Tokenizer下载下来
    modelscope.AutoConfig.from_pretrained("Qwen/Qwen2-0.5B").save_pretrained(
        "Qwen2-0.5B"
    )
    modelscope.AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B").save_pretrained(
        "Qwen2-0.5B"
    )
    context_length = 512  # use a small context length
    # tokenizer = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        "./Qwen2-0.5B"
    )  # download from local

    # preprocess dataset
    def tokenize(element):
        outputs = tokenizer(
            element["text"], # 输入文本
            truncation=True, # 超过最大长度时自动截断
            max_length=context_length, # 设置最大长度
            return_overflowing_tokens=True,
            return_length=True,
        ) # 字典 其中有包含 input_ids: token ID 序列 {'input_ids': [15496, 11, 4875, 286, 243, 287], 'attention_mask': [1, 1, 1, 1, 1, 1]}
        input_batch = []
        for length, input_ids in zip(outputs["length"], outputs["input_ids"]):
            if length == context_length:
                input_batch.append(input_ids)
        return {"input_ids": input_batch} # {"input_ids": [[101, 2054, ...], [1523, 98, ...], ...]}
                                                        #   ↑ 第1个块(512个ID)  ↑ 第2个块(512个ID)

    tokenized_datasets = raw_datasets.map(
        tokenize, batched=True, remove_columns=raw_datasets["train"].column_names
    ) # 行数不再是原来的维基百科「文章数」，而是切出来的 512-token 块的数量，每条数据就是一个长度为 512 的 token ID 列表
    print("tokenize dataset info")
    print(tokenized_datasets)
    tokenizer.pad_token = tokenizer.eos_token
    data_collator = transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False)

    # prepare a model from scratch
    # 用的Qwen2-0.5B，但是修改模型注意力头数量、模型层数和中间层大小，把模型控制到大概120M参数左右（跟GPT2接近）。
    config = transformers.AutoConfig.from_pretrained(
        "./Qwen2-0.5B",
        # token id 在模型里唯一的用途，就是当作行号去嵌入表里取一行数
        vocab_size=len(tokenizer), # 词嵌入表 行数 = vocab_size = len(tokenizer)，其中len(tokenizer) 代表词表里有多少个 token，而 vocab_size 就是词嵌入表的行数 
        hidden_size=512, # 词嵌入表 列数 = hidden_size = 512
        intermediate_size=2048,
        num_attention_heads=8,
        num_hidden_layers=12,
        max_position_embeddings=context_length,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    model = transformers.Qwen2ForCausalLM(config) # 收到的是配置，没有权重文件 → 所有参数按正态分布随机初始化
    model_size = sum(t.numel() for t in model.parameters())
    print("Model Config:")
    print(config)
    print(f"Model Size: {model_size/1000**2:.1f}M parameters")

    # train
    args = transformers.TrainingArguments(
        output_dir="WikiLLM",
        per_device_train_batch_size=8,  # 每个GPU的训练batch数
        per_device_eval_batch_size=8,  # 每个GPU的测试batch数
        eval_strategy="steps",
        eval_steps=5_00, # # 每 500 步在验证集上跑一遍，相当于10次训练日志评估一次
        logging_steps=50, # # 每 50 步记一次训练loss → 曲线分辨率
        # 含义是显存里只放 8 条，但攒 32 步才更新一次参数。因为 8 条 × 512 token 的激活值刚好塞得下，
        # 而 256 条一次性放进来会爆显存。代价是同样数据量下时间变长（32 次前向才换来 1 次参数更新），换来的是梯度更稳。
        gradient_accumulation_steps=32,  # 梯度累计总数，有效 batch = 8 × 32 = 256 条，每步 token = 256 × 512 = 131,072
        num_train_epochs=2,  # 训练epoch数
        weight_decay=0.1,
        warmup_steps=2_00,
        optim="adamw_torch",  # 优化器使用adamw
        lr_scheduler_type="cosine",  # 学习率衰减策略
        learning_rate=5e-4,  # 基础学习率，
        save_steps=5_00, # 每 500 步存一次盘
        save_total_limit=10,
        bf16=True,  # 开启bf16训练, 对于Amper架构以下的显卡建议替换为fp16=True
    )
    print("Train Args:")
    print(args)
    # enjoy training
    trainer = transformers.Trainer(
        model=model,
        tokenizer=tokenizer,
        args=args,
        data_collator=data_collator,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        callbacks=[SwanLabCallback()],
    )
    # b=8（per_device_train_batch_size）、n=512（context_length）、d=512（hidden_size）、V=151936（词表）
    # 输入是二维的 (b, n)，中间才是 (b, n, d)：embed_tokens 把每个token id整数换成一行 512 维向量，这时才变成 (8, 512, 512)。
    # 后面 12 层从头到尾形状不变——Transformer 层的设计特点就是形状进出一致。
    # 输出logits是 (b, n, V)：lm_head 把 512 维投影到 151936 维，得到每个位置上"词表里每个词的打分"。​最后一维是词表大小，不是 d。
    trainer.train()

    # save model
    model.save_pretrained("./WikiLLM/Weight")  # 保存模型的路径

    # generate
    pipe = transformers.pipeline("text-generation", model=model, tokenizer=tokenizer)
    print("GENERATE:", pipe("人工智能", num_return_sequences=1)[0]["generated_text"])
    prompts = ["牛顿", "北京市", "亚洲历史"]
    examples = []
    for i in range(3):
        # 根据提示词生成数据
        text = pipe(prompts[i], num_return_sequences=1)[0]["generated_text"]
        text = swanlab.Text(text)
        examples.append(text)
    swanlab.log({"Generate": examples})


if __name__ == "__main__":
    main()