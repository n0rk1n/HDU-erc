# Emotion-Aware Chatbot v1 归档

- 原始提交：`86220ac52df1641fee051bae8e4e04fd09ff6e40`
- 归档日期：2026-09-03
- 包含范围：该提交中所有受 Git 跟踪的项目代码、测试、配置、文档、数据样例与交付物，均按原相对路径冻结在本目录。

本目录是冻结的旧项目快照，仅用于追溯、阅读或单独复现实验。新版 `chatbot/` 不导入这里的任何 Python 模块；不要把本目录加入新版的 `PYTHONPATH`，也不要混用两者的虚拟环境、依赖或运行数据。

## 运行旧项目

在本目录中单独创建环境并运行旧项目，避免影响仓库根目录的新版环境：

```bash
cd archive/emotion-aware-chatbot-v1
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pytest -q
```

旧项目使用本目录自己的 `requirements.txt` 和配置模板。上述命令只说明独立复现边界；旧依赖、外部服务或凭据是否仍可用，需要在隔离环境中另行确认。任何修正或新版功能都应在仓库根目录实现，不应回写到冻结归档。
