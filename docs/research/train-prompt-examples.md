# 训练集示例与提示词调整（2026-09-08）

训练审计工件：`23a4b7f5-a9fe-4b2f-ab60-00dc148a92db`。来源固定为 GoEmotions `5594ac0ee7a13c77eb1e0b98f0f305a2ca65b6c4`，语料集合 `b9106602-3599-4419-97c6-426f6dbd0c39`。

仅检查 train 的 43,410 条原始评论，其中 7,102 条为多标签，1,396 条同时包含 neutral 与其他标签。构建方法时未读取 dev200 真值，test 不用于本轮。

## 训练数据带来的调整

- 标签不均衡：完整保留 28 类及每条原始标注，不把稀有类别删除、合并或伪造扩增。
- 保留多标签及 neutral 共现，提示词明确不做主次排序。
- 训练样本中有仅按短文本难以解释的标注；提示词提醒示例可能有噪声，不能只因主题相似就照抄标签。语义相邻对来自标签含义和人工检查，是方法先验，并非实测混淆矩阵。

## 实际生效的 example 选择

示例来自完整冻结训练库；旧的 56 条示例文件仅为测试夹具，修改它不会影响当前推理，因此本轮改变运行时选择策略。

基线 `ranked` 保留词面 Jaccard 排序前 4 条可用示例。候选 `contrastive-v1` 仍在同一前 50 条中选择：先取最高排名且不重复、满足预算的锚点，再寻找至少一个对应相邻标签（且不含该锚点标签）的正相关示例，提升到第二位置，其余位置沿用原词面排序。找不到合法对照则回退。所有标签保持原样，不生成或改写训练文本。

保存原始排名、实际选择顺序、锚点/对照角色、标签对、完整候选分数、去重与预算原因。多标签样本的一组对照不保证它们在所有其他标签上也不同。

## Prompt 版本

`native-labels-v1` 保留修改前原文；`native-labels-v2` 增加说话者情绪与事件主题区分、否定/讽刺证据约束、多标签保留、相邻标签边界说明（检索策略包含 11 组标签对）及示例独立判断规则。完整规则在 `emotion_lab/taxonomy.py`，每次 run 的 prompt_config_artifact_id 保存完整快照，实际消息另存在 request_artifact_id。

本轮同时调整 prompt 与 example，结果只能解释为组合效果。后续应补“只改 prompt / 只改 example”消融，才可分别归因。200 条属于开发诊断，不是正式 test 结论。

## 训练标签正例数

| 标签 | 正例数 |
|---|---:|
| grief | 77 |
| pride | 111 |
| relief | 153 |
| nervousness | 164 |
| embarrassment | 303 |
| remorse | 545 |
| fear | 596 |
| desire | 641 |
| disgust | 793 |
| excitement | 853 |
| surprise | 1,060 |
| caring | 1,087 |
| realization | 1,110 |
| disappointment | 1,269 |
| sadness | 1,326 |
| confusion | 1,368 |
| joy | 1,452 |
| anger | 1,567 |
| optimism | 1,581 |
| disapproval | 2,022 |
| love | 2,086 |
| curiosity | 2,191 |
| amusement | 2,328 |
| annoyance | 2,470 |
| gratitude | 2,662 |
| approval | 2,939 |
| admiration | 4,130 |
| neutral | 14,219 |

## 实际检查的原始样本（节选）

以下为原始标签的事实记录，不作为人工重新标注的标准答案。

| source_id | 原文 | 完整标签 |
|---|---|---|
| ed2wzm6 | I read on a different post that he died shortly after of internal injuries. | grief, neutral |
| eczqmch | Ehhh I don’t know, I still find it kind of cringey with the “[NAME]” talk and all that stuff. | embarrassment |
| eczbdg4 | Glad to hear it. You deserve your best life without that abuse and negativity. | caring, joy |
| eczbvy8 | I really appreciate you saying that And yes he is the cutest | admiration, gratitude |
| eczsxxo | I'm really worried :( I sent a DM to them, I hope they will be okay...  | disappointment, fear, optimism |
| ed1umha | Classic Trap game... I'm nervous. Hope our boys aren't hungover. | nervousness |
| eczeuct | I think he's a lustful, gluttonous, greedy, lazy, wrathful, envious, and prideful man.  | anger, disgust |
| eczjalj | Thank you so much. Happy New Year to you to. | excitement, gratitude |
| ed1zanc | Thank you for sharing this. May your friend rest easy and you find peace. | gratitude, grief, sadness |
| eczb4bm | TL;DR No more Superbowls for [NAME]. Get ready for another winning season that ends in disappointment. | disappointment |
| eczocm3 | I’m guessing you don’t take bad news lightly. I’m sorry bud :/ | remorse |
| eczdbay | When people don't understand things they invent gods. how little we have changed in all these years. | annoyance |
