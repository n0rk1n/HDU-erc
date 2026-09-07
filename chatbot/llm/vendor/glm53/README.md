# GLM-5.3 官方分词资源

来源：[zai-org/GLM-5.3](https://huggingface.co/zai-org/GLM-5.3/tree/aca966e4e02791568aa6a4ced368624b3d897f42)，固定版本 `aca966e4e02791568aa6a4ced368624b3d897f42`。

- `tokenizer.json`：官方分词器，SHA-256 `19e773648cb4e65de8660ea6365e10acca112d42a854923df93db4a6f333a82d`，加载时校验。
- `chat_template.jinja`：未经修改的官方消息模板，SHA-256 `3740abcea51c45830cb3ca562084ad5fb2ef53589376f73332e9886f93ade41c`。
- `LICENSE`：上游原始许可证。

`chatbot/llm/glm53_tokens.py` 仅实现该模板的纯文本 system/user/assistant 分支，包括思考强度系统前缀、历史思考内容和生成前缀。分别计算 low/high/max，以及清除/保留历史思考内容的组合，取最大值用于预算。工具、图片和未支持的附加字段会明确报错。服务端的模板差异仍由显式安全余量承担，实际用量以供应商响应为准。

运行时不下载模型或 tokenizer，不依赖 transformers 或远程代码。
