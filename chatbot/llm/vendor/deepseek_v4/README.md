# DeepSeek V4 官方 Tokenizer

来源：<https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/tree/60d8d70770c6776ff598c94bb586a859a38244f1>

固定版本：`60d8d70770c6776ff598c94bb586a859a38244f1`。编码器和 tokenizer 文件原样保留；许可证见 `LICENSE`（仅将换行统一为 LF）。

| 文件 | 上游路径 | SHA-256 |
| --- | --- | --- |
| encoding_dsv4.py | encoding/encoding_dsv4.py | bdbd57c132a1b3725042323d02b98b9d1df28e5f388f134399555d041f5055e0 |
| tokenizer.json | tokenizer.json | 8f9f37ca37fdc4f5fd36d5cf4d3b0e8392edb4e894fd10cc0d70b4957c8633cf |

`tokenizer.json` 约 6.4 MB，随代码提供，启动无需访问 Hugging Face 或下载模型权重。
上游同版本 `tokenizer_config.json` 的 `model_max_length` 为 `1048576`。
业务包装器仅允许情绪分析和门控使用的纯文本消息；工具调用、多模态载荷不会被静默忽略。
分别按官方 chat/thinking 模板计数并取较大值，包含消息角色、特殊 token 和生成前缀。
这是固定版本的本地预算计数；服务商更新模板时应同步核对，实际计费用量以 API 返回为准。
