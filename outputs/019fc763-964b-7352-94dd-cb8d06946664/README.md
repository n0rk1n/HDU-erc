# EmpatheticDialogues seed64 盲审材料

本目录包含两份顺序使用的工作簿：

1. `emotion_seed64_blind_review_round1.xlsx`
2. `emotion_seed64_alignment_adjudication_round2.xlsx`

## 使用顺序

先填写并保存第一轮文件。在完成第一轮前，不要打开第二轮文件，也不要查阅
`balanced_seed.jsonl`、现有实验报告或模型预测。

第一轮只提供匿名 `audit_id`、`current_input` 和 32 类标签定义；必须填写
`primary_label`、`confidence`、`evidence_quote`，备选标签和备注可选。

第一轮完成后再打开第二轮文件。第二轮按相同 `audit_id` 提供原始 `case_id`、
`original_expected` 和 `source_situation`，用于判断 `clear`、`ambiguous`、
`misaligned` 或 `invalid`。两份文件均不包含模型预测。

## 可复现信息

- 源文件：`data/benchmarks/empathetic_dialogues_v1/release/balanced_seed.jsonl`
- 源文件 SHA-256：`4835ca5a25cae5e239580cef7682210b41837b693645398acb4bf408fde74cb4`
- 随机化标识：`hdu-erc-seed64-blind-review-v1-20260803`
- 样本数：64
- 标签数：32，每类 2 条

原始 benchmark 文件没有被修改。
