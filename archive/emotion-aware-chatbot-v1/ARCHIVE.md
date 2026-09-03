# Emotion-Aware Chatbot v1 archive

- 原始提交：`86220ac52df1641fee051bae8e4e04fd09ff6e40`
- 归档日期：2026-09-03
- 包含范围：该提交中所有受 Git 跟踪的项目代码、测试、配置、文档、数据样例与交付物，均按原相对路径冻结在本目录。

## 运行旧项目

在本目录中单独创建环境并运行旧项目，避免影响仓库根目录的新版环境：

```bash
cd archive/emotion-aware-chatbot-v1
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pytest -q
```

本归档仅供参考、不与新版共享运行时代码。
