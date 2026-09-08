-- 情绪实验数据库 v1；禁止覆盖聊天库。各字段语义详见已批准设计第 5 节。

-- schema_migrations：数据库版本
CREATE TABLE schema_migrations (
  -- 迁移版本号
  version INTEGER PRIMARY KEY,
  -- SHA-256 完整性指纹：migration
  migration_sha256 TEXT NOT NULL UNIQUE,
  -- 迁移应用时间
  applied_at TEXT NOT NULL
);

-- artifacts：完整信息与文件登记
CREATE TABLE artifacts (
  -- 关联记录唯一标识：artifact
  artifact_id TEXT PRIMARY KEY NOT NULL,
  -- SHA-256 完整性指纹：content
  content_sha256 TEXT NOT NULL,
  -- 工件或步骤类型
  kind TEXT NOT NULL,
  -- 内容媒体类型（MIME）
  media_type TEXT NOT NULL,
  -- 保存字节的字符编码或 binary
  encoding TEXT NOT NULL,
  -- 物理存储压缩算法
  compression TEXT NOT NULL,
  -- 压缩前内容字节数
  logical_bytes INTEGER NOT NULL,
  -- 实际存储字节数
  stored_bytes INTEGER NOT NULL,
  -- SHA-256 完整性指纹：storage
  storage_sha256 TEXT NOT NULL,
  -- 内联原始字节
  inline_bytes BLOB,
  -- 工件根目录下的相对路径
  relative_path TEXT,
  -- 凭据脱敏规则版本；无脱敏为 none
  redaction_policy_version TEXT NOT NULL,
  -- 采集层级，区分实际 HTTP 内容与应用生成内容
  capture_level TEXT NOT NULL,
  -- 结构化来源元数据
  metadata_json TEXT NOT NULL CHECK(json_valid(metadata_json)),
  -- 记录创建时间（UTC）
  created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%f000Z','now')),
  CHECK ((inline_bytes IS NOT NULL)+(relative_path IS NOT NULL)=1),
  CHECK (logical_bytes>=0 AND stored_bytes>=0),
  CHECK (compression IN ('none','gzip'))
);

-- dataset_versions：原始数据版本
CREATE TABLE dataset_versions (
  -- 关联记录唯一标识：dataset_version
  dataset_version_id TEXT PRIMARY KEY NOT NULL,
  -- 用户定义名称
  name TEXT NOT NULL,
  -- 源文件与导入规则的联合版本指纹
  version_key TEXT NOT NULL,
  -- 官方来源仓库
  source_repository TEXT NOT NULL,
  -- 固定源提交或发布版本
  source_revision TEXT NOT NULL,
  -- 关联不可变证据工件：source_manifest
  source_manifest_artifact_id TEXT NOT NULL REFERENCES artifacts(artifact_id) ON DELETE RESTRICT,
  -- 关联不可变证据工件：source_bundle
  source_bundle_artifact_id TEXT NOT NULL REFERENCES artifacts(artifact_id) ON DELETE RESTRICT,
  -- SHA-256 完整性指纹：label_manifest
  label_manifest_sha256 TEXT NOT NULL,
  -- 导入程序版本
  importer_version TEXT NOT NULL,
  -- 仅用于去重的文本标准化规则版本
  normalization_version TEXT NOT NULL,
  -- 经过 JSON 校验的结构化内容：expected_counts
  expected_counts_json TEXT NOT NULL CHECK(json_valid(expected_counts_json)),
  -- 经过 JSON 校验的结构化内容：actual_counts
  actual_counts_json TEXT NOT NULL CHECK(json_valid(actual_counts_json)),
  -- 生命周期状态
  status TEXT NOT NULL,
  -- UTC 时间：sealed
  sealed_at TEXT,
  -- 记录创建时间（UTC）
  created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%f000Z','now')),
  UNIQUE(name,version_key),
  CHECK(status IN ('importing','ready','failed'))
);

-- label_definitions：数据集原生标签
CREATE TABLE label_definitions (
  -- 关联记录唯一标识：dataset_version
  dataset_version_id TEXT NOT NULL REFERENCES dataset_versions(dataset_version_id) ON DELETE RESTRICT,
  -- 数据集原生情绪编号
  label_id INTEGER NOT NULL,
  -- 数据集原生标签名称
  label_name TEXT NOT NULL,
  -- 官方标签列表顺序，从零开始
  source_order INTEGER NOT NULL,
  -- 记录创建时间（UTC）
  created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%f000Z','now')),
  PRIMARY KEY(dataset_version_id,label_id),
  UNIQUE(dataset_version_id,label_name),
  UNIQUE(dataset_version_id,source_order),
  CHECK(label_id>=0)
);

-- samples：原始样本
CREATE TABLE samples (
  -- 关联记录唯一标识：sample
  sample_id TEXT PRIMARY KEY NOT NULL,
  -- 关联记录唯一标识：dataset_version
  dataset_version_id TEXT NOT NULL REFERENCES dataset_versions(dataset_version_id) ON DELETE RESTRICT,
  -- 官方原始样本标识
  source_id TEXT NOT NULL,
  -- 官方 train、dev 或 test 划分
  split TEXT NOT NULL,
  -- 源文件行号，从一开始
  source_line INTEGER NOT NULL,
  -- 文本语言
  language TEXT NOT NULL,
  -- 未经改写的原文
  raw_text TEXT NOT NULL,
  -- SHA-256 完整性指纹：text
  text_sha256 TEXT NOT NULL,
  -- SHA-256 完整性指纹：normalized_text
  normalized_text_sha256 TEXT NOT NULL,
  -- SHA-256 完整性指纹：source_record
  source_record_sha256 TEXT NOT NULL,
  -- 结构化来源元数据
  metadata_json TEXT NOT NULL CHECK(json_valid(metadata_json)),
  -- 记录创建时间（UTC）
  created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%f000Z','now')),
  UNIQUE(dataset_version_id,source_id),
  UNIQUE(dataset_version_id,split,source_line),
  UNIQUE(sample_id,dataset_version_id),
  CHECK(split IN ('train','dev','test')),
  CHECK(source_line>0)
);

-- sample_labels：原始多标签标注
CREATE TABLE sample_labels (
  -- 关联记录唯一标识：sample
  sample_id TEXT NOT NULL REFERENCES samples(sample_id) ON DELETE RESTRICT,
  -- 关联记录唯一标识：dataset_version
  dataset_version_id TEXT NOT NULL REFERENCES dataset_versions(dataset_version_id) ON DELETE RESTRICT,
  -- 数据集原生情绪编号
  label_id INTEGER NOT NULL,
  -- 标注来源
  annotation_source TEXT NOT NULL,
  -- 源文件标签顺序，仅表示来源顺序
  source_label_position INTEGER NOT NULL,
  -- 记录创建时间（UTC）
  created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%f000Z','now')),
  PRIMARY KEY(sample_id,label_id),
  FOREIGN KEY(sample_id,dataset_version_id) REFERENCES samples(sample_id,dataset_version_id),
  FOREIGN KEY(dataset_version_id,label_id) REFERENCES label_definitions(dataset_version_id,label_id),
  CHECK(source_label_position>=0)
);

-- sample_sets：冻结的实验样本集合
CREATE TABLE sample_sets (
  -- 关联记录唯一标识：sample_set
  sample_set_id TEXT PRIMARY KEY NOT NULL,
  -- 关联记录唯一标识：dataset_version
  dataset_version_id TEXT NOT NULL REFERENCES dataset_versions(dataset_version_id) ON DELETE RESTRICT,
  -- 用户定义名称
  name TEXT NOT NULL,
  -- 用途
  purpose TEXT NOT NULL,
  -- 集合采用的官方划分
  source_split TEXT NOT NULL,
  -- 关联不可变证据工件：selection_config
  selection_config_artifact_id TEXT NOT NULL REFERENCES artifacts(artifact_id) ON DELETE RESTRICT,
  -- SHA-256 完整性指纹：members
  members_sha256 TEXT NOT NULL,
  -- 冻结集合样本数量
  sample_count INTEGER NOT NULL,
  -- 生命周期状态
  status TEXT NOT NULL,
  -- UTC 时间：frozen
  frozen_at TEXT,
  -- 记录创建时间（UTC）
  created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%f000Z','now')),
  UNIQUE(sample_set_id,dataset_version_id),
  CHECK(status IN ('building','frozen')),
  CHECK(sample_count>=0),
  CHECK(source_split IN ('train','dev','test')),
  CHECK((purpose='corpus' AND source_split='train') OR (purpose IN ('dev_pilot','dev_tuning') AND source_split='dev') OR (purpose='final_test' AND source_split='test') OR purpose='diagnostic')
);

-- sample_set_members：集合中的具体样本
CREATE TABLE sample_set_members (
  -- 关联记录唯一标识：sample_set
  sample_set_id TEXT NOT NULL REFERENCES sample_sets(sample_set_id) ON DELETE RESTRICT,
  -- 关联记录唯一标识：sample
  sample_id TEXT NOT NULL REFERENCES samples(sample_id) ON DELETE RESTRICT,
  -- 关联记录唯一标识：dataset_version
  dataset_version_id TEXT NOT NULL REFERENCES dataset_versions(dataset_version_id) ON DELETE RESTRICT,
  -- 固定执行或集合顺序，从一开始
  ordinal INTEGER NOT NULL,
  -- 样本入选规则说明
  selection_reason TEXT NOT NULL,
  -- 记录创建时间（UTC）
  created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%f000Z','now')),
  PRIMARY KEY(sample_set_id,sample_id),
  UNIQUE(sample_set_id,ordinal),
  CHECK(ordinal>0),
  FOREIGN KEY(sample_id,dataset_version_id) REFERENCES samples(sample_id,dataset_version_id),
  FOREIGN KEY(sample_set_id,dataset_version_id) REFERENCES sample_sets(sample_set_id,dataset_version_id)
);

-- embedding_indexes：共享索引版本
CREATE TABLE embedding_indexes (
  -- 关联记录唯一标识：index
  index_id TEXT PRIMARY KEY NOT NULL,
  -- 关联记录唯一标识：corpus_set
  corpus_set_id TEXT NOT NULL REFERENCES sample_sets(sample_set_id) ON DELETE RESTRICT,
  -- 关联不可变证据工件：index_config
  index_config_artifact_id TEXT NOT NULL REFERENCES artifacts(artifact_id) ON DELETE RESTRICT,
  -- 关联不可变证据工件：model_config
  model_config_artifact_id TEXT NOT NULL REFERENCES artifacts(artifact_id) ON DELETE RESTRICT,
  -- 关联不可变证据工件：build_environment
  build_environment_artifact_id TEXT NOT NULL REFERENCES artifacts(artifact_id) ON DELETE RESTRICT,
  -- 关联不可变证据工件：input_manifest
  input_manifest_artifact_id TEXT NOT NULL REFERENCES artifacts(artifact_id) ON DELETE RESTRICT,
  -- 关联不可变证据工件：index_bundle
  index_bundle_artifact_id TEXT REFERENCES artifacts(artifact_id) ON DELETE RESTRICT,
  -- 向量维度，索引构建完成前可为空
  dimension INTEGER,
  -- 实际向量数值精度
  dtype TEXT,
  -- 向量距离或相似度定义
  metric TEXT NOT NULL,
  -- 语料、模型和索引配置联合指纹
  fingerprint TEXT NOT NULL UNIQUE,
  -- 生命周期状态
  status TEXT NOT NULL,
  -- UTC 时间：started
  started_at TEXT,
  -- UTC 时间：completed
  completed_at TEXT,
  -- 记录创建时间（UTC）
  created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%f000Z','now')),
  CHECK(status IN ('planned','building','ready','failed')),
  CHECK(dimension IS NULL OR dimension>0),
  CHECK(status!='ready' OR (index_bundle_artifact_id IS NOT NULL AND dimension IS NOT NULL AND dtype IS NOT NULL))
);

-- embedding_items：向量与样本映射
CREATE TABLE embedding_items (
  -- 关联记录唯一标识：index
  index_id TEXT NOT NULL REFERENCES embedding_indexes(index_id) ON DELETE RESTRICT,
  -- 关联记录唯一标识：sample
  sample_id TEXT NOT NULL REFERENCES samples(sample_id) ON DELETE RESTRICT,
  -- 样本向量在矩阵中的行号，从零开始
  vector_offset INTEGER NOT NULL,
  -- SHA-256 完整性指纹：input
  input_sha256 TEXT NOT NULL,
  -- SHA-256 完整性指纹：vector
  vector_sha256 TEXT NOT NULL,
  -- 关联记录唯一标识：source_call_attempt
  source_call_attempt_id TEXT REFERENCES call_attempts(call_attempt_id) ON DELETE RESTRICT,
  -- 样本在调用批次中的位置
  batch_offset INTEGER,
  -- 记录创建时间（UTC）
  created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%f000Z','now')),
  PRIMARY KEY(index_id,sample_id),
  UNIQUE(index_id,vector_offset),
  CHECK(vector_offset>=0 AND (batch_offset IS NULL OR batch_offset>=0))
);

-- experiments：比较问题与共同协议
CREATE TABLE experiments (
  -- 关联记录唯一标识：experiment
  experiment_id TEXT PRIMARY KEY NOT NULL,
  -- 用户定义名称
  name TEXT NOT NULL,
  -- 实验要检验的研究问题
  research_question TEXT NOT NULL,
  -- 关联记录唯一标识：dataset_version
  dataset_version_id TEXT NOT NULL REFERENCES dataset_versions(dataset_version_id) ON DELETE RESTRICT,
  -- 关联记录唯一标识：corpus_set
  corpus_set_id TEXT NOT NULL REFERENCES sample_sets(sample_set_id) ON DELETE RESTRICT,
  -- 关联记录唯一标识：evaluation_set
  evaluation_set_id TEXT NOT NULL REFERENCES sample_sets(sample_set_id) ON DELETE RESTRICT,
  -- 关联不可变证据工件：protocol
  protocol_artifact_id TEXT NOT NULL REFERENCES artifacts(artifact_id) ON DELETE RESTRICT,
  -- 共享比较协议分组
  comparison_group TEXT NOT NULL,
  -- 生命周期状态
  status TEXT NOT NULL,
  -- UTC 时间：frozen
  frozen_at TEXT,
  -- 记录创建时间（UTC）
  created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%f000Z','now')),
  CHECK(status IN ('draft','frozen','closed'))
);

-- experiment_runs：具体一次运行
CREATE TABLE experiment_runs (
  -- 关联记录唯一标识：run
  run_id TEXT PRIMARY KEY NOT NULL,
  -- 关联记录唯一标识：experiment
  experiment_id TEXT NOT NULL REFERENCES experiments(experiment_id) ON DELETE RESTRICT,
  -- 本次检索或推理方法
  method_name TEXT NOT NULL,
  -- 关联不可变证据工件：method_config
  method_config_artifact_id TEXT NOT NULL REFERENCES artifacts(artifact_id) ON DELETE RESTRICT,
  -- 关联不可变证据工件：model_config
  model_config_artifact_id TEXT NOT NULL REFERENCES artifacts(artifact_id) ON DELETE RESTRICT,
  -- 关联不可变证据工件：prompt_config
  prompt_config_artifact_id TEXT NOT NULL REFERENCES artifacts(artifact_id) ON DELETE RESTRICT,
  -- 关联不可变证据工件：taxonomy_config
  taxonomy_config_artifact_id TEXT NOT NULL REFERENCES artifacts(artifact_id) ON DELETE RESTRICT,
  -- 关联不可变证据工件：environment
  environment_artifact_id TEXT NOT NULL REFERENCES artifacts(artifact_id) ON DELETE RESTRICT,
  -- 关联不可变证据工件：price_snapshot
  price_snapshot_artifact_id TEXT REFERENCES artifacts(artifact_id) ON DELETE RESTRICT,
  -- 关联记录唯一标识：index
  index_id TEXT REFERENCES embedding_indexes(index_id) ON DELETE RESTRICT,
  -- 运行时 Git 提交
  code_commit TEXT NOT NULL,
  -- 关联不可变证据工件：working_diff
  working_diff_artifact_id TEXT REFERENCES artifacts(artifact_id) ON DELETE RESTRICT,
  -- 固定随机种子
  seed INTEGER NOT NULL,
  -- 独立重复运行编号，从一开始
  repetition_no INTEGER NOT NULL,
  -- SHA-256 完整性指纹：config
  config_sha256 TEXT NOT NULL,
  -- 生命周期状态
  status TEXT NOT NULL,
  -- UTC 时间：started
  started_at TEXT,
  -- UTC 时间：completed
  completed_at TEXT,
  -- 暂停、终止或失败的原因
  stop_reason TEXT,
  -- 记录创建时间（UTC）
  created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%f000Z','now')),
  CHECK(status IN ('created','ready','running','paused','completed','completed_with_errors','failed','cancelled')),
  CHECK(repetition_no>=1)
);

-- run_items：单条样本执行
CREATE TABLE run_items (
  -- 关联记录唯一标识：run_item
  run_item_id TEXT PRIMARY KEY NOT NULL,
  -- 关联记录唯一标识：run
  run_id TEXT NOT NULL REFERENCES experiment_runs(run_id) ON DELETE RESTRICT,
  -- 关联记录唯一标识：sample
  sample_id TEXT NOT NULL REFERENCES samples(sample_id) ON DELETE RESTRICT,
  -- 固定执行或集合顺序，从一开始
  ordinal INTEGER NOT NULL,
  -- SHA-256 完整性指纹：input
  input_sha256 TEXT NOT NULL,
  -- 生命周期状态
  status TEXT NOT NULL,
  -- 关联记录唯一标识：final_prediction
  final_prediction_id TEXT REFERENCES predictions(prediction_id) ON DELETE RESTRICT,
  -- 关联记录唯一标识：worker
  worker_id TEXT,
  -- 防止过期工作者写入的租约令牌
  lease_token TEXT,
  -- UTC 时间：lease_expires
  lease_expires_at TEXT,
  -- UTC 时间：started
  started_at TEXT,
  -- UTC 时间：completed
  completed_at TEXT,
  -- 失败发生的处理阶段
  failure_stage TEXT,
  -- 稳定的失败原因代码
  failure_code TEXT,
  -- 关联不可变证据工件：error
  error_artifact_id TEXT REFERENCES artifacts(artifact_id) ON DELETE RESTRICT,
  -- 记录创建时间（UTC）
  created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%f000Z','now')),
  UNIQUE(run_id,sample_id),
  UNIQUE(run_id,ordinal),
  CHECK(ordinal>0),
  CHECK(status IN ('pending','running','succeeded','failed','cancelled')),
  CHECK(status!='succeeded' OR final_prediction_id IS NOT NULL),
  CHECK(status!='running' OR (lease_token IS NOT NULL AND lease_expires_at IS NOT NULL)),
  FOREIGN KEY(final_prediction_id,run_item_id) REFERENCES predictions(prediction_id,run_item_id)
);

-- execution_steps：可观察的算法步骤
CREATE TABLE execution_steps (
  -- 关联记录唯一标识：step
  step_id TEXT PRIMARY KEY NOT NULL,
  -- 关联记录唯一标识：run_item
  run_item_id TEXT REFERENCES run_items(run_item_id) ON DELETE RESTRICT,
  -- 关联记录唯一标识：index
  index_id TEXT REFERENCES embedding_indexes(index_id) ON DELETE RESTRICT,
  -- 所属样本或索引内唯一的步骤名称
  step_key TEXT NOT NULL,
  -- 算法步骤顺序，从一开始
  step_no INTEGER NOT NULL,
  -- 关联记录唯一标识：parent_step
  parent_step_id TEXT REFERENCES execution_steps(step_id) ON DELETE RESTRICT,
  -- 工件或步骤类型
  kind TEXT NOT NULL,
  -- 步骤实现版本
  implementation_version TEXT NOT NULL,
  -- 关联不可变证据工件：config
  config_artifact_id TEXT NOT NULL REFERENCES artifacts(artifact_id) ON DELETE RESTRICT,
  -- 关联不可变证据工件：input
  input_artifact_id TEXT NOT NULL REFERENCES artifacts(artifact_id) ON DELETE RESTRICT,
  -- 关联不可变证据工件：output
  output_artifact_id TEXT REFERENCES artifacts(artifact_id) ON DELETE RESTRICT,
  -- 生命周期状态
  status TEXT NOT NULL,
  -- UTC 时间：started
  started_at TEXT,
  -- UTC 时间：completed
  completed_at TEXT,
  -- 使用单调时钟测得的耗时（毫秒）
  latency_ms INTEGER CHECK(latency_ms IS NULL OR latency_ms>=0),
  -- 关联不可变证据工件：error
  error_artifact_id TEXT REFERENCES artifacts(artifact_id) ON DELETE RESTRICT,
  -- 是否完整记录本步骤可观察信息，0 或 1
  trace_complete INTEGER NOT NULL,
  -- 记录创建时间（UTC）
  created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%f000Z','now')),
  CHECK((run_item_id IS NOT NULL)+(index_id IS NOT NULL)=1),
  CHECK(step_no>0),
  CHECK(trace_complete IN (0,1)),
  CHECK(status IN ('planned','running','completed','failed')),
  CHECK(kind IN ('preprocess','index_embed','query_embed','retrieve','candidate_predict','rerank','classify','validate'))
);

-- call_attempts：每次调用尝试和费用证据
CREATE TABLE call_attempts (
  -- 关联记录唯一标识：call_attempt
  call_attempt_id TEXT PRIMARY KEY NOT NULL,
  -- 关联记录唯一标识：step
  step_id TEXT NOT NULL REFERENCES execution_steps(step_id) ON DELETE RESTRICT,
  -- 步骤内调用尝试序号，重试递增
  attempt_no INTEGER NOT NULL,
  -- 关联记录唯一标识：retry_of_attempt
  retry_of_attempt_id TEXT REFERENCES call_attempts(call_attempt_id) ON DELETE RESTRICT,
  -- 远端调用 remote 或本地推理 local
  transport_kind TEXT NOT NULL,
  -- 供应商或兼容协议名称
  provider TEXT NOT NULL,
  -- 实际请求的模型名称
  requested_model TEXT NOT NULL,
  -- 供应商响应中的模型名称；未返回为空
  resolved_model TEXT,
  -- 脱敏后的实际调用端点
  endpoint TEXT,
  -- 关联不可变证据工件：request
  request_artifact_id TEXT NOT NULL REFERENCES artifacts(artifact_id) ON DELETE RESTRICT,
  -- SHA-256 完整性指纹：request
  request_sha256 TEXT NOT NULL,
  -- 关联不可变证据工件：response
  response_artifact_id TEXT REFERENCES artifacts(artifact_id) ON DELETE RESTRICT,
  -- 关联不可变证据工件：reasoning
  reasoning_artifact_id TEXT REFERENCES artifacts(artifact_id) ON DELETE RESTRICT,
  -- 关联不可变证据工件：error
  error_artifact_id TEXT REFERENCES artifacts(artifact_id) ON DELETE RESTRICT,
  -- 关联不可变证据工件：response_metadata
  response_metadata_artifact_id TEXT REFERENCES artifacts(artifact_id) ON DELETE RESTRICT,
  -- 关联不可变证据工件：usage
  usage_artifact_id TEXT REFERENCES artifacts(artifact_id) ON DELETE RESTRICT,
  -- 真实 HTTP 状态码；未收到响应为空
  http_status INTEGER,
  -- 关联记录唯一标识：provider_request
  provider_request_id TEXT,
  -- 确实发送给供应商的幂等键；未使用为空
  idempotency_key TEXT,
  -- 供应商返回的结束原因
  finish_reason TEXT,
  -- 生命周期状态
  status TEXT NOT NULL,
  -- UTC 时间：prepared
  prepared_at TEXT NOT NULL,
  -- UTC 时间：dispatch_started
  dispatch_started_at TEXT,
  -- UTC 时间：first_response
  first_response_at TEXT,
  -- UTC 时间：completed
  completed_at TEXT,
  -- 使用单调时钟测得的耗时（毫秒）
  latency_ms INTEGER CHECK(latency_ms IS NULL OR latency_ms>=0),
  -- Token 数量，未提供时为空：input_tokens
  input_tokens INTEGER CHECK(input_tokens IS NULL OR input_tokens>=0),
  -- Token 数量，未提供时为空：output_tokens
  output_tokens INTEGER CHECK(output_tokens IS NULL OR output_tokens>=0),
  -- Token 数量，未提供时为空：cached_input_tokens
  cached_input_tokens INTEGER CHECK(cached_input_tokens IS NULL OR cached_input_tokens>=0),
  -- Token 数量，未提供时为空：reasoning_tokens
  reasoning_tokens INTEGER CHECK(reasoning_tokens IS NULL OR reasoning_tokens>=0),
  -- Token 数量，未提供时为空：total_tokens
  total_tokens INTEGER CHECK(total_tokens IS NULL OR total_tokens>=0),
  -- 用量来源：provider、local_estimate 或 unavailable
  usage_source TEXT NOT NULL,
  -- 未派发、未知、已估算或账单确认
  billing_state TEXT NOT NULL,
  -- 估算费用，单位为百万分之一币种单位；未知为空
  estimated_cost_micros INTEGER,
  -- 费用币种
  currency TEXT,
  -- 关联不可变证据工件：price_snapshot
  price_snapshot_artifact_id TEXT REFERENCES artifacts(artifact_id) ON DELETE RESTRICT,
  -- 关联不可变证据工件：cost_calculation
  cost_calculation_artifact_id TEXT REFERENCES artifacts(artifact_id) ON DELETE RESTRICT,
  -- 记录创建时间（UTC）
  created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%f000Z','now')),
  UNIQUE(step_id,attempt_no),
  CHECK(attempt_no>0),
  CHECK(transport_kind IN ('remote','local')),
  CHECK(status IN ('prepared','dispatched','response_received','transport_error','outcome_unknown','cancelled_before_dispatch')),
  CHECK(status!='response_received' OR response_artifact_id IS NOT NULL),
  CHECK(usage_source IN ('provider','local_estimate','unavailable')),
  CHECK(billing_state IN ('not_dispatched','unknown','estimated','confirmed')),
  CHECK((estimated_cost_micros IS NULL) OR (estimated_cost_micros>=0 AND currency IS NOT NULL))
);

-- predictions：解析后的预测版本
CREATE TABLE predictions (
  -- 关联记录唯一标识：prediction
  prediction_id TEXT PRIMARY KEY NOT NULL,
  -- 关联记录唯一标识：run_item
  run_item_id TEXT NOT NULL REFERENCES run_items(run_item_id) ON DELETE RESTRICT,
  -- 关联记录唯一标识：call_attempt
  call_attempt_id TEXT NOT NULL REFERENCES call_attempts(call_attempt_id) ON DELETE RESTRICT,
  -- 用途
  purpose TEXT NOT NULL,
  -- 输出解析器版本
  parser_version TEXT NOT NULL,
  -- 预测输出结构版本
  output_schema_version TEXT NOT NULL,
  -- SHA-256 完整性指纹：source_response
  source_response_sha256 TEXT NOT NULL,
  -- 关联不可变证据工件：parsed
  parsed_artifact_id TEXT REFERENCES artifacts(artifact_id) ON DELETE RESTRICT,
  -- 关联不可变证据工件：evidence
  evidence_artifact_id TEXT REFERENCES artifacts(artifact_id) ON DELETE RESTRICT,
  -- 关联不可变证据工件：validation
  validation_artifact_id TEXT NOT NULL REFERENCES artifacts(artifact_id) ON DELETE RESTRICT,
  -- 生命周期状态
  status TEXT NOT NULL,
  -- 记录创建时间（UTC）
  created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%f000Z','now')),
  UNIQUE(call_attempt_id,parser_version,output_schema_version),
  UNIQUE(prediction_id,run_item_id),
  CHECK(status IN ('valid','invalid')),
  CHECK(purpose IN ('candidate','final','review'))
);

-- prediction_labels：预测标签集合
CREATE TABLE prediction_labels (
  -- 关联记录唯一标识：prediction
  prediction_id TEXT NOT NULL REFERENCES predictions(prediction_id) ON DELETE RESTRICT,
  -- 关联记录唯一标识：dataset_version
  dataset_version_id TEXT NOT NULL REFERENCES dataset_versions(dataset_version_id) ON DELETE RESTRICT,
  -- 数据集原生情绪编号
  label_id INTEGER NOT NULL,
  -- 标签在原模型输出中的位置，从零开始
  output_position INTEGER NOT NULL,
  -- 模型实际返回的标签分数；未返回为空
  reported_score REAL,
  -- 模型自述分数或对数概率等分数类型
  score_kind TEXT,
  -- 经过 JSON 校验的结构化内容：evidence
  evidence_json TEXT CHECK(json_valid(evidence_json)),
  -- 记录创建时间（UTC）
  created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%f000Z','now')),
  PRIMARY KEY(prediction_id,label_id),
  FOREIGN KEY(dataset_version_id,label_id) REFERENCES label_definitions(dataset_version_id,label_id),
  CHECK(output_position>=0)
);

-- retrieval_items：检索候选与入选详情
CREATE TABLE retrieval_items (
  -- 关联记录唯一标识：retrieval_step
  retrieval_step_id TEXT NOT NULL REFERENCES execution_steps(step_id) ON DELETE RESTRICT,
  -- 关联记录唯一标识：sample
  sample_id TEXT NOT NULL REFERENCES samples(sample_id) ON DELETE RESTRICT,
  -- 候选检索或选择阶段
  stage TEXT NOT NULL,
  -- 进入处理的候选顺序，从一开始
  candidate_rank INTEGER NOT NULL,
  -- 实际计算的相似度，随机检索为空
  similarity_score REAL,
  -- 实际重排得分，未重排为空
  rerank_score REAL,
  -- 经过 JSON 校验的结构化内容：score_components
  score_components_json TEXT NOT NULL CHECK(json_valid(score_components_json)),
  -- 候选、入选或排除
  decision TEXT NOT NULL,
  -- 入选、去重、预算裁剪等决策原因
  reason_code TEXT NOT NULL,
  -- 最终示例顺序，未入选为空
  selected_rank INTEGER,
  -- Token 数量，未提供时为空：example_tokens
  example_tokens INTEGER CHECK(example_tokens IS NULL OR example_tokens>=0),
  -- 记录创建时间（UTC）
  created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%f000Z','now')),
  PRIMARY KEY(retrieval_step_id,stage,sample_id),
  UNIQUE(retrieval_step_id,stage,candidate_rank),
  CHECK(candidate_rank>0),
  CHECK(selected_rank IS NULL OR selected_rank>0),
  CHECK(decision IN ('selected','excluded','candidate'))
);

-- evaluations：一次可复现评分
CREATE TABLE evaluations (
  -- 关联记录唯一标识：evaluation
  evaluation_id TEXT PRIMARY KEY NOT NULL,
  -- 关联记录唯一标识：run
  run_id TEXT NOT NULL REFERENCES experiment_runs(run_id) ON DELETE RESTRICT,
  -- 关联记录唯一标识：sample_set
  sample_set_id TEXT NOT NULL REFERENCES sample_sets(sample_set_id) ON DELETE RESTRICT,
  -- 指标计算程序版本
  scorer_version TEXT NOT NULL,
  -- 关联不可变证据工件：evaluation_config
  evaluation_config_artifact_id TEXT NOT NULL REFERENCES artifacts(artifact_id) ON DELETE RESTRICT,
  -- 关联不可变证据工件：prediction_snapshot
  prediction_snapshot_artifact_id TEXT NOT NULL REFERENCES artifacts(artifact_id) ON DELETE RESTRICT,
  -- SHA-256 完整性指纹：ground_truth_manifest
  ground_truth_manifest_sha256 TEXT NOT NULL,
  -- 关联不可变证据工件：family_mapping
  family_mapping_artifact_id TEXT REFERENCES artifacts(artifact_id) ON DELETE RESTRICT,
  -- 评测范围
  scope TEXT NOT NULL,
  -- 生命周期状态
  status TEXT NOT NULL,
  -- 完整计划评测的样本数
  expected_items INTEGER NOT NULL,
  -- 运行已进入成功或失败状态的样本数
  observed_items INTEGER NOT NULL,
  -- 其中执行失败的样本数
  failed_items INTEGER NOT NULL,
  -- UTC 时间：completed
  completed_at TEXT,
  -- 记录创建时间（UTC）
  created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%f000Z','now')),
  CHECK(status IN ('building','completed','failed')),
  CHECK(scope IN ('official_full','diagnostic_partial','diagnostic_slice')),
  CHECK(expected_items>=observed_items AND observed_items>=failed_items AND failed_items>=0),
  CHECK(scope!='official_full' OR expected_items=observed_items)
);

-- evaluation_items：逐条评分与错误分析
CREATE TABLE evaluation_items (
  -- 关联记录唯一标识：evaluation
  evaluation_id TEXT NOT NULL REFERENCES evaluations(evaluation_id) ON DELETE RESTRICT,
  -- 关联记录唯一标识：run_item
  run_item_id TEXT NOT NULL REFERENCES run_items(run_item_id) ON DELETE RESTRICT,
  -- 关联记录唯一标识：prediction
  prediction_id TEXT REFERENCES predictions(prediction_id) ON DELETE RESTRICT,
  -- 逐条正确、错误、失败或缺失状态
  outcome TEXT NOT NULL,
  -- 预测缺失的原始原因
  missing_reason TEXT,
  -- 正确预测的标签数
  tp INTEGER NOT NULL,
  -- 误报标签数
  fp INTEGER NOT NULL,
  -- 漏报标签数
  fn INTEGER NOT NULL,
  -- 标签集合是否完全一致，0 或 1
  exact_match INTEGER NOT NULL,
  -- 经过 JSON 校验的结构化内容：missing_label_ids
  missing_label_ids_json TEXT NOT NULL CHECK(json_valid(missing_label_ids_json)),
  -- 经过 JSON 校验的结构化内容：extra_label_ids
  extra_label_ids_json TEXT NOT NULL CHECK(json_valid(extra_label_ids_json)),
  -- 经过 JSON 校验的结构化内容：slice_memberships
  slice_memberships_json TEXT NOT NULL CHECK(json_valid(slice_memberships_json)),
  -- 记录创建时间（UTC）
  created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%f000Z','now')),
  PRIMARY KEY(evaluation_id,run_item_id),
  CHECK(tp>=0 AND fp>=0 AND fn>=0),
  CHECK(exact_match IN (0,1)),
  CHECK(outcome IN ('correct','incorrect','failed','missing'))
);

-- comparisons：运行间的配对比较
CREATE TABLE comparisons (
  -- 关联记录唯一标识：comparison
  comparison_id TEXT PRIMARY KEY NOT NULL,
  -- 关联记录唯一标识：baseline_evaluation
  baseline_evaluation_id TEXT NOT NULL REFERENCES evaluations(evaluation_id) ON DELETE RESTRICT,
  -- 关联记录唯一标识：candidate_evaluation
  candidate_evaluation_id TEXT NOT NULL REFERENCES evaluations(evaluation_id) ON DELETE RESTRICT,
  -- 关联不可变证据工件：analysis_config
  analysis_config_artifact_id TEXT NOT NULL REFERENCES artifacts(artifact_id) ON DELETE RESTRICT,
  -- 关联不可变证据工件：paired_members
  paired_members_artifact_id TEXT NOT NULL REFERENCES artifacts(artifact_id) ON DELETE RESTRICT,
  -- 关联不可变证据工件：result
  result_artifact_id TEXT REFERENCES artifacts(artifact_id) ON DELETE RESTRICT,
  -- 生命周期状态
  status TEXT NOT NULL,
  -- UTC 时间：completed
  completed_at TEXT,
  -- 记录创建时间（UTC）
  created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%f000Z','now')),
  CHECK(status IN ('building','completed','failed'))
);

-- metric_values：可查询的指标结果
CREATE TABLE metric_values (
  -- 关联记录唯一标识：metric
  metric_id TEXT PRIMARY KEY NOT NULL,
  -- 关联记录唯一标识：evaluation
  evaluation_id TEXT REFERENCES evaluations(evaluation_id) ON DELETE RESTRICT,
  -- 关联记录唯一标识：comparison
  comparison_id TEXT REFERENCES comparisons(comparison_id) ON DELETE RESTRICT,
  -- 指标名称
  metric_name TEXT NOT NULL,
  -- 总体、标签、family、切片或币种范围
  scope_key TEXT NOT NULL,
  -- 指标数值，未知为空
  value REAL,
  -- 指标计算分子
  numerator REAL,
  -- 指标计算分母
  denominator REAL,
  -- 真值支持数或参与样本数
  support INTEGER,
  -- 置信区间下界
  ci_low REAL,
  -- 置信区间上界
  ci_high REAL,
  -- 未定义或零分母的说明
  undefined_reason TEXT,
  -- 记录创建时间（UTC）
  created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%f000Z','now')),
  CHECK((evaluation_id IS NOT NULL)+(comparison_id IS NOT NULL)=1),
  CHECK(value IS NOT NULL OR undefined_reason IS NOT NULL)
);

-- audit_events：状态与操作事件
CREATE TABLE audit_events (
  -- 递增审计事件编号
  event_id INTEGER PRIMARY KEY AUTOINCREMENT,
  -- 关联记录唯一标识：artifact
  artifact_id TEXT REFERENCES artifacts(artifact_id) ON DELETE RESTRICT,
  -- 关联记录唯一标识：dataset_version
  dataset_version_id TEXT REFERENCES dataset_versions(dataset_version_id) ON DELETE RESTRICT,
  -- 关联记录唯一标识：index
  index_id TEXT REFERENCES embedding_indexes(index_id) ON DELETE RESTRICT,
  -- 关联记录唯一标识：run
  run_id TEXT REFERENCES experiment_runs(run_id) ON DELETE RESTRICT,
  -- 关联记录唯一标识：run_item
  run_item_id TEXT REFERENCES run_items(run_item_id) ON DELETE RESTRICT,
  -- 关联记录唯一标识：step
  step_id TEXT REFERENCES execution_steps(step_id) ON DELETE RESTRICT,
  -- 关联记录唯一标识：call_attempt
  call_attempt_id TEXT REFERENCES call_attempts(call_attempt_id) ON DELETE RESTRICT,
  -- 关联记录唯一标识：evaluation
  evaluation_id TEXT REFERENCES evaluations(evaluation_id) ON DELETE RESTRICT,
  -- 关联记录唯一标识：comparison
  comparison_id TEXT REFERENCES comparisons(comparison_id) ON DELETE RESTRICT,
  -- 追加事件类型
  event_type TEXT NOT NULL,
  -- 状态变更前值
  from_status TEXT,
  -- 状态变更后值
  to_status TEXT,
  -- 操作主体
  actor TEXT NOT NULL,
  -- 关联不可变证据工件：payload
  payload_artifact_id TEXT REFERENCES artifacts(artifact_id) ON DELETE RESTRICT,
  -- 事件发生时间
  occurred_at TEXT NOT NULL,
  -- 记录创建时间（UTC）
  created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%f000Z','now')),
  CHECK((artifact_id IS NOT NULL)+(dataset_version_id IS NOT NULL)+(index_id IS NOT NULL)+(run_id IS NOT NULL)+(run_item_id IS NOT NULL)+(step_id IS NOT NULL)+(call_attempt_id IS NOT NULL)+(evaluation_id IS NOT NULL)+(comparison_id IS NOT NULL)=1)
);

-- 调用明细：工件路径可为空（内联字节位于 artifacts）。
CREATE VIEW v_call_audit AS
SELECT a.*,s.run_item_id,s.index_id,i.run_id,i.sample_id,
 q.relative_path request_path,p.relative_path response_path
FROM call_attempts a JOIN execution_steps s USING(step_id)
LEFT JOIN run_items i USING(run_item_id)
JOIN artifacts q ON q.artifact_id=a.request_artifact_id
LEFT JOIN artifacts p ON p.artifact_id=a.response_artifact_id;
-- 先各自聚合，避免候选、多标签、调用联表造成重复计费。费用按币种分组在调用表查询。
CREATE VIEW v_run_summary AS
WITH items AS (SELECT run_id,COUNT(*) planned,SUM(status='succeeded') succeeded,
 SUM(status='failed') failed FROM run_items GROUP BY run_id),
calls AS (SELECT run_id,COUNT(*) attempts,SUM(total_tokens IS NULL) unknown_usage_calls,
 SUM(estimated_cost_micros IS NULL AND status!='cancelled_before_dispatch') unknown_cost_calls,
 SUM(input_tokens) known_input_tokens,SUM(output_tokens) known_output_tokens
 FROM v_call_audit WHERE run_id IS NOT NULL GROUP BY run_id)
SELECT r.run_id,r.method_name,r.status,i.planned,i.succeeded,i.failed,c.attempts,
 c.unknown_usage_calls,c.unknown_cost_calls,c.known_input_tokens,c.known_output_tokens
FROM experiment_runs r LEFT JOIN items i USING(run_id) LEFT JOIN calls c USING(run_id);
CREATE VIEW v_evaluation_errors AS
SELECT e.*,i.sample_id,s.source_id,s.raw_text,
 (SELECT json_group_array(d.label_name) FROM sample_labels l JOIN label_definitions d
 ON d.dataset_version_id=l.dataset_version_id AND d.label_id=l.label_id WHERE l.sample_id=i.sample_id) true_labels,
 (SELECT json_group_array(d.label_name) FROM prediction_labels l JOIN label_definitions d
 ON d.dataset_version_id=l.dataset_version_id AND d.label_id=l.label_id WHERE l.prediction_id=e.prediction_id) predicted_labels
FROM evaluation_items e JOIN run_items i USING(run_item_id) JOIN samples s USING(sample_id)
WHERE e.outcome!='correct';

CREATE TRIGGER immutable_artifacts_UPDATE BEFORE UPDATE ON artifacts WHEN 1 BEGIN SELECT RAISE(ABORT,'immutable evidence'); END;

CREATE TRIGGER immutable_artifacts_DELETE BEFORE DELETE ON artifacts WHEN 1 BEGIN SELECT RAISE(ABORT,'immutable evidence'); END;

CREATE TRIGGER immutable_predictions_UPDATE BEFORE UPDATE ON predictions WHEN 1 BEGIN SELECT RAISE(ABORT,'immutable evidence'); END;

CREATE TRIGGER immutable_predictions_DELETE BEFORE DELETE ON predictions WHEN 1 BEGIN SELECT RAISE(ABORT,'immutable evidence'); END;

CREATE TRIGGER immutable_prediction_labels_UPDATE BEFORE UPDATE ON prediction_labels WHEN 1 BEGIN SELECT RAISE(ABORT,'immutable evidence'); END;

CREATE TRIGGER immutable_prediction_labels_DELETE BEFORE DELETE ON prediction_labels WHEN 1 BEGIN SELECT RAISE(ABORT,'immutable evidence'); END;

CREATE TRIGGER immutable_audit_events_UPDATE BEFORE UPDATE ON audit_events WHEN 1 BEGIN SELECT RAISE(ABORT,'immutable evidence'); END;

CREATE TRIGGER immutable_audit_events_DELETE BEFORE DELETE ON audit_events WHEN 1 BEGIN SELECT RAISE(ABORT,'immutable evidence'); END;

CREATE TRIGGER immutable_schema_migrations_UPDATE BEFORE UPDATE ON schema_migrations WHEN 1 BEGIN SELECT RAISE(ABORT,'immutable evidence'); END;

CREATE TRIGGER immutable_schema_migrations_DELETE BEFORE DELETE ON schema_migrations WHEN 1 BEGIN SELECT RAISE(ABORT,'immutable evidence'); END;

CREATE TRIGGER sealed_samples_INSERT BEFORE INSERT ON samples WHEN (SELECT status FROM dataset_versions WHERE dataset_version_id=NEW.dataset_version_id)='ready' BEGIN SELECT RAISE(ABORT,'sealed dataset'); END;

CREATE TRIGGER sealed_samples_UPDATE BEFORE UPDATE ON samples WHEN (SELECT status FROM dataset_versions WHERE dataset_version_id=OLD.dataset_version_id)='ready' BEGIN SELECT RAISE(ABORT,'sealed dataset'); END;

CREATE TRIGGER sealed_samples_DELETE BEFORE DELETE ON samples WHEN (SELECT status FROM dataset_versions WHERE dataset_version_id=OLD.dataset_version_id)='ready' BEGIN SELECT RAISE(ABORT,'sealed dataset'); END;

CREATE TRIGGER sealed_sample_labels_INSERT BEFORE INSERT ON sample_labels WHEN (SELECT status FROM dataset_versions WHERE dataset_version_id=NEW.dataset_version_id)='ready' BEGIN SELECT RAISE(ABORT,'sealed dataset'); END;

CREATE TRIGGER sealed_sample_labels_UPDATE BEFORE UPDATE ON sample_labels WHEN (SELECT status FROM dataset_versions WHERE dataset_version_id=OLD.dataset_version_id)='ready' BEGIN SELECT RAISE(ABORT,'sealed dataset'); END;

CREATE TRIGGER sealed_sample_labels_DELETE BEFORE DELETE ON sample_labels WHEN (SELECT status FROM dataset_versions WHERE dataset_version_id=OLD.dataset_version_id)='ready' BEGIN SELECT RAISE(ABORT,'sealed dataset'); END;

CREATE TRIGGER sealed_label_definitions_INSERT BEFORE INSERT ON label_definitions WHEN (SELECT status FROM dataset_versions WHERE dataset_version_id=NEW.dataset_version_id)='ready' BEGIN SELECT RAISE(ABORT,'sealed dataset'); END;

CREATE TRIGGER sealed_label_definitions_UPDATE BEFORE UPDATE ON label_definitions WHEN (SELECT status FROM dataset_versions WHERE dataset_version_id=OLD.dataset_version_id)='ready' BEGIN SELECT RAISE(ABORT,'sealed dataset'); END;

CREATE TRIGGER sealed_label_definitions_DELETE BEFORE DELETE ON label_definitions WHEN (SELECT status FROM dataset_versions WHERE dataset_version_id=OLD.dataset_version_id)='ready' BEGIN SELECT RAISE(ABORT,'sealed dataset'); END;

CREATE TRIGGER frozen_dataset_versions_UPDATE BEFORE UPDATE ON dataset_versions WHEN OLD.status='ready' BEGIN SELECT RAISE(ABORT,'frozen record'); END;

CREATE TRIGGER frozen_dataset_versions_DELETE BEFORE DELETE ON dataset_versions WHEN OLD.status='ready' BEGIN SELECT RAISE(ABORT,'frozen record'); END;

CREATE TRIGGER frozen_sample_sets_UPDATE BEFORE UPDATE ON sample_sets WHEN OLD.status='frozen' BEGIN SELECT RAISE(ABORT,'frozen record'); END;

CREATE TRIGGER frozen_sample_sets_DELETE BEFORE DELETE ON sample_sets WHEN OLD.status='frozen' BEGIN SELECT RAISE(ABORT,'frozen record'); END;

CREATE TRIGGER frozen_evaluations_UPDATE BEFORE UPDATE ON evaluations WHEN OLD.status='completed' BEGIN SELECT RAISE(ABORT,'frozen record'); END;

CREATE TRIGGER frozen_evaluations_DELETE BEFORE DELETE ON evaluations WHEN OLD.status='completed' BEGIN SELECT RAISE(ABORT,'frozen record'); END;

CREATE TRIGGER frozen_comparisons_UPDATE BEFORE UPDATE ON comparisons WHEN OLD.status='completed' BEGIN SELECT RAISE(ABORT,'frozen record'); END;

CREATE TRIGGER frozen_comparisons_DELETE BEFORE DELETE ON comparisons WHEN OLD.status='completed' BEGIN SELECT RAISE(ABORT,'frozen record'); END;

CREATE TRIGGER frozen_sample_set_members_INSERT BEFORE INSERT ON sample_set_members WHEN (SELECT status FROM sample_sets WHERE sample_set_id=NEW.sample_set_id)='frozen' BEGIN SELECT RAISE(ABORT,'frozen parent'); END;

CREATE TRIGGER frozen_sample_set_members_UPDATE BEFORE UPDATE ON sample_set_members WHEN (SELECT status FROM sample_sets WHERE sample_set_id=OLD.sample_set_id)='frozen' BEGIN SELECT RAISE(ABORT,'frozen parent'); END;

CREATE TRIGGER frozen_sample_set_members_DELETE BEFORE DELETE ON sample_set_members WHEN (SELECT status FROM sample_sets WHERE sample_set_id=OLD.sample_set_id)='frozen' BEGIN SELECT RAISE(ABORT,'frozen parent'); END;

CREATE TRIGGER frozen_embedding_items_INSERT BEFORE INSERT ON embedding_items WHEN (SELECT status FROM embedding_indexes WHERE index_id=NEW.index_id)='ready' BEGIN SELECT RAISE(ABORT,'frozen parent'); END;

CREATE TRIGGER frozen_embedding_items_UPDATE BEFORE UPDATE ON embedding_items WHEN (SELECT status FROM embedding_indexes WHERE index_id=OLD.index_id)='ready' BEGIN SELECT RAISE(ABORT,'frozen parent'); END;

CREATE TRIGGER frozen_embedding_items_DELETE BEFORE DELETE ON embedding_items WHEN (SELECT status FROM embedding_indexes WHERE index_id=OLD.index_id)='ready' BEGIN SELECT RAISE(ABORT,'frozen parent'); END;

CREATE TRIGGER frozen_evaluation_items_INSERT BEFORE INSERT ON evaluation_items WHEN (SELECT status FROM evaluations WHERE evaluation_id=NEW.evaluation_id)='completed' BEGIN SELECT RAISE(ABORT,'frozen parent'); END;

CREATE TRIGGER frozen_evaluation_items_UPDATE BEFORE UPDATE ON evaluation_items WHEN (SELECT status FROM evaluations WHERE evaluation_id=OLD.evaluation_id)='completed' BEGIN SELECT RAISE(ABORT,'frozen parent'); END;

CREATE TRIGGER frozen_evaluation_items_DELETE BEFORE DELETE ON evaluation_items WHEN (SELECT status FROM evaluations WHERE evaluation_id=OLD.evaluation_id)='completed' BEGIN SELECT RAISE(ABORT,'frozen parent'); END;

CREATE TRIGGER metric_INSERT BEFORE INSERT ON metric_values WHEN (SELECT status FROM evaluations WHERE evaluation_id=NEW.evaluation_id)='completed' OR (SELECT status FROM comparisons WHERE comparison_id=NEW.comparison_id)='completed' BEGIN SELECT RAISE(ABORT,'frozen metrics'); END;

CREATE TRIGGER metric_UPDATE BEFORE UPDATE ON metric_values WHEN (SELECT status FROM evaluations WHERE evaluation_id=OLD.evaluation_id)='completed' OR (SELECT status FROM comparisons WHERE comparison_id=OLD.comparison_id)='completed' BEGIN SELECT RAISE(ABORT,'frozen metrics'); END;

CREATE TRIGGER metric_DELETE BEFORE DELETE ON metric_values WHEN (SELECT status FROM evaluations WHERE evaluation_id=OLD.evaluation_id)='completed' OR (SELECT status FROM comparisons WHERE comparison_id=OLD.comparison_id)='completed' BEGIN SELECT RAISE(ABORT,'frozen metrics'); END;

CREATE TRIGGER valid_sample_set_members_INSERT BEFORE INSERT ON sample_set_members WHEN (SELECT split FROM samples WHERE sample_id=NEW.sample_id)!=(SELECT source_split FROM sample_sets WHERE sample_set_id=NEW.sample_set_id) BEGIN SELECT RAISE(ABORT,'sample split mismatch'); END;

CREATE TRIGGER valid_sample_set_members_UPDATE BEFORE UPDATE ON sample_set_members WHEN (SELECT split FROM samples WHERE sample_id=NEW.sample_id)!=(SELECT source_split FROM sample_sets WHERE sample_set_id=NEW.sample_set_id) BEGIN SELECT RAISE(ABORT,'sample split mismatch'); END;

CREATE TRIGGER valid_embedding_indexes_INSERT BEFORE INSERT ON embedding_indexes WHEN NOT EXISTS(SELECT 1 FROM sample_sets WHERE sample_set_id=NEW.corpus_set_id AND purpose='corpus' AND status='frozen') BEGIN SELECT RAISE(ABORT,'index requires frozen train corpus'); END;

CREATE TRIGGER valid_embedding_indexes_UPDATE BEFORE UPDATE ON embedding_indexes WHEN NOT EXISTS(SELECT 1 FROM sample_sets WHERE sample_set_id=NEW.corpus_set_id AND purpose='corpus' AND status='frozen') BEGIN SELECT RAISE(ABORT,'index requires frozen train corpus'); END;

CREATE TRIGGER valid_embedding_items_INSERT BEFORE INSERT ON embedding_items WHEN NOT EXISTS(SELECT 1 FROM embedding_indexes i JOIN sample_set_members m ON m.sample_set_id=i.corpus_set_id WHERE i.index_id=NEW.index_id AND m.sample_id=NEW.sample_id) BEGIN SELECT RAISE(ABORT,'index requires corpus member'); END;

CREATE TRIGGER valid_embedding_items_UPDATE BEFORE UPDATE ON embedding_items WHEN NOT EXISTS(SELECT 1 FROM embedding_indexes i JOIN sample_set_members m ON m.sample_set_id=i.corpus_set_id WHERE i.index_id=NEW.index_id AND m.sample_id=NEW.sample_id) BEGIN SELECT RAISE(ABORT,'index requires corpus member'); END;

CREATE TRIGGER valid_experiments_INSERT BEFORE INSERT ON experiments WHEN NOT EXISTS(SELECT 1 FROM sample_sets c JOIN sample_sets e ON e.sample_set_id=NEW.evaluation_set_id WHERE c.sample_set_id=NEW.corpus_set_id AND c.purpose='corpus' AND c.status='frozen' AND e.status='frozen' AND e.purpose!='corpus' AND c.dataset_version_id=NEW.dataset_version_id AND e.dataset_version_id=NEW.dataset_version_id) BEGIN SELECT RAISE(ABORT,'incompatible experiment sets'); END;

CREATE TRIGGER valid_experiments_UPDATE BEFORE UPDATE ON experiments WHEN NOT EXISTS(SELECT 1 FROM sample_sets c JOIN sample_sets e ON e.sample_set_id=NEW.evaluation_set_id WHERE c.sample_set_id=NEW.corpus_set_id AND c.purpose='corpus' AND c.status='frozen' AND e.status='frozen' AND e.purpose!='corpus' AND c.dataset_version_id=NEW.dataset_version_id AND e.dataset_version_id=NEW.dataset_version_id) BEGIN SELECT RAISE(ABORT,'incompatible experiment sets'); END;

CREATE TRIGGER valid_experiment_runs_INSERT BEFORE INSERT ON experiment_runs WHEN (SELECT status FROM experiments WHERE experiment_id=NEW.experiment_id)!='frozen' OR (NEW.index_id IS NOT NULL AND NOT EXISTS(SELECT 1 FROM embedding_indexes i JOIN experiments e ON e.experiment_id=NEW.experiment_id WHERE i.index_id=NEW.index_id AND i.status='ready' AND i.corpus_set_id=e.corpus_set_id)) BEGIN SELECT RAISE(ABORT,'run requires frozen protocol and ready index'); END;

CREATE TRIGGER valid_experiment_runs_UPDATE BEFORE UPDATE ON experiment_runs WHEN (SELECT status FROM experiments WHERE experiment_id=NEW.experiment_id)!='frozen' OR (NEW.index_id IS NOT NULL AND NOT EXISTS(SELECT 1 FROM embedding_indexes i JOIN experiments e ON e.experiment_id=NEW.experiment_id WHERE i.index_id=NEW.index_id AND i.status='ready' AND i.corpus_set_id=e.corpus_set_id)) BEGIN SELECT RAISE(ABORT,'run requires frozen protocol and ready index'); END;

CREATE TRIGGER valid_run_items_INSERT BEFORE INSERT ON run_items WHEN NOT EXISTS(SELECT 1 FROM experiment_runs r JOIN experiments e USING(experiment_id) JOIN sample_set_members m ON m.sample_set_id=e.evaluation_set_id WHERE r.run_id=NEW.run_id AND m.sample_id=NEW.sample_id AND m.ordinal=NEW.ordinal) BEGIN SELECT RAISE(ABORT,'item outside evaluation set'); END;

CREATE TRIGGER valid_run_items_UPDATE BEFORE UPDATE ON run_items WHEN NOT EXISTS(SELECT 1 FROM experiment_runs r JOIN experiments e USING(experiment_id) JOIN sample_set_members m ON m.sample_set_id=e.evaluation_set_id WHERE r.run_id=NEW.run_id AND m.sample_id=NEW.sample_id AND m.ordinal=NEW.ordinal) BEGIN SELECT RAISE(ABORT,'item outside evaluation set'); END;

CREATE TRIGGER valid_execution_steps_INSERT BEFORE INSERT ON execution_steps WHEN NEW.parent_step_id IS NOT NULL AND NOT EXISTS(SELECT 1 FROM execution_steps p WHERE p.step_id=NEW.parent_step_id AND p.run_item_id IS NEW.run_item_id AND p.index_id IS NEW.index_id) BEGIN SELECT RAISE(ABORT,'step parent owner mismatch'); END;

CREATE TRIGGER valid_execution_steps_UPDATE BEFORE UPDATE ON execution_steps WHEN NEW.parent_step_id IS NOT NULL AND NOT EXISTS(SELECT 1 FROM execution_steps p WHERE p.step_id=NEW.parent_step_id AND p.run_item_id IS NEW.run_item_id AND p.index_id IS NEW.index_id) BEGIN SELECT RAISE(ABORT,'step parent owner mismatch'); END;

CREATE TRIGGER valid_call_attempts_INSERT BEFORE INSERT ON call_attempts WHEN NEW.retry_of_attempt_id IS NOT NULL AND NOT EXISTS(SELECT 1 FROM call_attempts p WHERE p.call_attempt_id=NEW.retry_of_attempt_id AND p.step_id=NEW.step_id AND p.attempt_no<NEW.attempt_no) BEGIN SELECT RAISE(ABORT,'retry owner mismatch'); END;

CREATE TRIGGER valid_call_attempts_UPDATE BEFORE UPDATE ON call_attempts WHEN NEW.retry_of_attempt_id IS NOT NULL AND NOT EXISTS(SELECT 1 FROM call_attempts p WHERE p.call_attempt_id=NEW.retry_of_attempt_id AND p.step_id=NEW.step_id AND p.attempt_no<NEW.attempt_no) BEGIN SELECT RAISE(ABORT,'retry owner mismatch'); END;

CREATE TRIGGER valid_predictions_INSERT BEFORE INSERT ON predictions WHEN NOT EXISTS(SELECT 1 FROM call_attempts a JOIN execution_steps s USING(step_id) JOIN artifacts f ON f.artifact_id=a.response_artifact_id WHERE a.call_attempt_id=NEW.call_attempt_id AND s.run_item_id=NEW.run_item_id AND f.content_sha256=NEW.source_response_sha256) BEGIN SELECT RAISE(ABORT,'prediction source mismatch'); END;

CREATE TRIGGER valid_predictions_UPDATE BEFORE UPDATE ON predictions WHEN NOT EXISTS(SELECT 1 FROM call_attempts a JOIN execution_steps s USING(step_id) JOIN artifacts f ON f.artifact_id=a.response_artifact_id WHERE a.call_attempt_id=NEW.call_attempt_id AND s.run_item_id=NEW.run_item_id AND f.content_sha256=NEW.source_response_sha256) BEGIN SELECT RAISE(ABORT,'prediction source mismatch'); END;

CREATE TRIGGER valid_prediction_labels_INSERT BEFORE INSERT ON prediction_labels WHEN NOT EXISTS(SELECT 1 FROM predictions p JOIN run_items i USING(run_item_id) JOIN samples s USING(sample_id) WHERE p.prediction_id=NEW.prediction_id AND p.status='valid' AND s.dataset_version_id=NEW.dataset_version_id) BEGIN SELECT RAISE(ABORT,'prediction label dataset mismatch'); END;

CREATE TRIGGER valid_prediction_labels_UPDATE BEFORE UPDATE ON prediction_labels WHEN NOT EXISTS(SELECT 1 FROM predictions p JOIN run_items i USING(run_item_id) JOIN samples s USING(sample_id) WHERE p.prediction_id=NEW.prediction_id AND p.status='valid' AND s.dataset_version_id=NEW.dataset_version_id) BEGIN SELECT RAISE(ABORT,'prediction label dataset mismatch'); END;

CREATE TRIGGER valid_retrieval_items_INSERT BEFORE INSERT ON retrieval_items WHEN NOT EXISTS(SELECT 1 FROM execution_steps s JOIN run_items i USING(run_item_id) JOIN experiment_runs r USING(run_id) JOIN experiments e USING(experiment_id) JOIN sample_set_members m ON m.sample_set_id=e.corpus_set_id WHERE s.step_id=NEW.retrieval_step_id AND s.kind='retrieve' AND m.sample_id=NEW.sample_id) BEGIN SELECT RAISE(ABORT,'retrieval outside train corpus'); END;

CREATE TRIGGER valid_retrieval_items_UPDATE BEFORE UPDATE ON retrieval_items WHEN NOT EXISTS(SELECT 1 FROM execution_steps s JOIN run_items i USING(run_item_id) JOIN experiment_runs r USING(run_id) JOIN experiments e USING(experiment_id) JOIN sample_set_members m ON m.sample_set_id=e.corpus_set_id WHERE s.step_id=NEW.retrieval_step_id AND s.kind='retrieve' AND m.sample_id=NEW.sample_id) BEGIN SELECT RAISE(ABORT,'retrieval outside train corpus'); END;

CREATE TRIGGER valid_evaluation_items_INSERT BEFORE INSERT ON evaluation_items WHEN NOT EXISTS(SELECT 1 FROM evaluations e JOIN run_items i ON i.run_id=e.run_id JOIN sample_set_members m ON m.sample_set_id=e.sample_set_id AND m.sample_id=i.sample_id WHERE e.evaluation_id=NEW.evaluation_id AND i.run_item_id=NEW.run_item_id) OR (NEW.prediction_id IS NOT NULL AND NOT EXISTS(SELECT 1 FROM predictions p WHERE p.prediction_id=NEW.prediction_id AND p.run_item_id=NEW.run_item_id AND p.status='valid')) BEGIN SELECT RAISE(ABORT,'evaluation owner mismatch'); END;

CREATE TRIGGER valid_evaluation_items_UPDATE BEFORE UPDATE ON evaluation_items WHEN NOT EXISTS(SELECT 1 FROM evaluations e JOIN run_items i ON i.run_id=e.run_id JOIN sample_set_members m ON m.sample_set_id=e.sample_set_id AND m.sample_id=i.sample_id WHERE e.evaluation_id=NEW.evaluation_id AND i.run_item_id=NEW.run_item_id) OR (NEW.prediction_id IS NOT NULL AND NOT EXISTS(SELECT 1 FROM predictions p WHERE p.prediction_id=NEW.prediction_id AND p.run_item_id=NEW.run_item_id AND p.status='valid')) BEGIN SELECT RAISE(ABORT,'evaluation owner mismatch'); END;

CREATE TRIGGER valid_final_prediction BEFORE UPDATE ON run_items WHEN NEW.final_prediction_id IS NOT NULL AND NOT EXISTS(SELECT 1 FROM predictions WHERE prediction_id=NEW.final_prediction_id AND run_item_id=NEW.run_item_id AND status='valid' AND EXISTS(SELECT 1 FROM prediction_labels l WHERE l.prediction_id=NEW.final_prediction_id)) BEGIN SELECT RAISE(ABORT,'invalid final prediction'); END;

CREATE TRIGGER immutable_received_call BEFORE UPDATE ON call_attempts WHEN OLD.status IN ('response_received','transport_error','cancelled_before_dispatch') BEGIN SELECT RAISE(ABORT,'completed call is immutable'); END;

CREATE TRIGGER call_delete BEFORE DELETE ON call_attempts WHEN 1 BEGIN SELECT RAISE(ABORT,'append only call'); END;

CREATE TRIGGER freeze_config_experiment_runs BEFORE UPDATE ON experiment_runs WHEN OLD.status!='created' AND (NEW.run_id IS NOT OLD.run_id OR NEW.experiment_id IS NOT OLD.experiment_id OR NEW.method_name IS NOT OLD.method_name OR NEW.method_config_artifact_id IS NOT OLD.method_config_artifact_id OR NEW.model_config_artifact_id IS NOT OLD.model_config_artifact_id OR NEW.prompt_config_artifact_id IS NOT OLD.prompt_config_artifact_id OR NEW.taxonomy_config_artifact_id IS NOT OLD.taxonomy_config_artifact_id OR NEW.environment_artifact_id IS NOT OLD.environment_artifact_id OR NEW.price_snapshot_artifact_id IS NOT OLD.price_snapshot_artifact_id OR NEW.index_id IS NOT OLD.index_id OR NEW.code_commit IS NOT OLD.code_commit OR NEW.working_diff_artifact_id IS NOT OLD.working_diff_artifact_id OR NEW.seed IS NOT OLD.seed OR NEW.repetition_no IS NOT OLD.repetition_no OR NEW.config_sha256 IS NOT OLD.config_sha256 OR NEW.created_at IS NOT OLD.created_at) BEGIN SELECT RAISE(ABORT,'frozen configuration'); END;

CREATE TRIGGER freeze_config_experiments BEFORE UPDATE ON experiments WHEN OLD.status!='draft' AND (NEW.experiment_id IS NOT OLD.experiment_id OR NEW.name IS NOT OLD.name OR NEW.research_question IS NOT OLD.research_question OR NEW.dataset_version_id IS NOT OLD.dataset_version_id OR NEW.corpus_set_id IS NOT OLD.corpus_set_id OR NEW.evaluation_set_id IS NOT OLD.evaluation_set_id OR NEW.protocol_artifact_id IS NOT OLD.protocol_artifact_id OR NEW.comparison_group IS NOT OLD.comparison_group OR NEW.frozen_at IS NOT OLD.frozen_at OR NEW.created_at IS NOT OLD.created_at) BEGIN SELECT RAISE(ABORT,'frozen configuration'); END;

CREATE TRIGGER freeze_config_run_items BEFORE UPDATE ON run_items WHEN 1 AND (NEW.run_item_id IS NOT OLD.run_item_id OR NEW.run_id IS NOT OLD.run_id OR NEW.sample_id IS NOT OLD.sample_id OR NEW.ordinal IS NOT OLD.ordinal OR NEW.input_sha256 IS NOT OLD.input_sha256 OR NEW.created_at IS NOT OLD.created_at) BEGIN SELECT RAISE(ABORT,'frozen configuration'); END;

CREATE TRIGGER freeze_config_embedding_indexes BEFORE UPDATE ON embedding_indexes WHEN OLD.status='ready' AND (NEW.index_id IS NOT OLD.index_id OR NEW.corpus_set_id IS NOT OLD.corpus_set_id OR NEW.index_config_artifact_id IS NOT OLD.index_config_artifact_id OR NEW.model_config_artifact_id IS NOT OLD.model_config_artifact_id OR NEW.build_environment_artifact_id IS NOT OLD.build_environment_artifact_id OR NEW.input_manifest_artifact_id IS NOT OLD.input_manifest_artifact_id OR NEW.index_bundle_artifact_id IS NOT OLD.index_bundle_artifact_id OR NEW.dimension IS NOT OLD.dimension OR NEW.dtype IS NOT OLD.dtype OR NEW.metric IS NOT OLD.metric OR NEW.fingerprint IS NOT OLD.fingerprint OR NEW.status IS NOT OLD.status OR NEW.started_at IS NOT OLD.started_at OR NEW.completed_at IS NOT OLD.completed_at OR NEW.created_at IS NOT OLD.created_at) BEGIN SELECT RAISE(ABORT,'frozen configuration'); END;
CREATE INDEX ix_samples_0 ON samples(dataset_version_id,split,source_id);
CREATE INDEX ix_samples_1 ON samples(dataset_version_id,normalized_text_sha256);
CREATE INDEX ix_sample_labels_0 ON sample_labels(dataset_version_id,label_id,sample_id);
CREATE INDEX ix_experiment_runs_0 ON experiment_runs(experiment_id,method_name,seed,repetition_no);
CREATE INDEX ix_experiment_runs_1 ON experiment_runs(status,created_at);
CREATE INDEX ix_run_items_0 ON run_items(run_id,status,ordinal);
CREATE INDEX ix_run_items_1 ON run_items(status,lease_expires_at);
CREATE INDEX ix_execution_steps_0 ON execution_steps(run_item_id,kind,step_no);
CREATE INDEX ix_execution_steps_1 ON execution_steps(index_id,kind,step_no);
CREATE INDEX ix_call_attempts_0 ON call_attempts(status,dispatch_started_at);
CREATE INDEX ix_call_attempts_1 ON call_attempts(provider,provider_request_id);
CREATE INDEX ix_predictions_0 ON predictions(run_item_id,purpose,status,created_at);
CREATE INDEX ix_retrieval_items_0 ON retrieval_items(sample_id,decision);
CREATE INDEX ix_evaluation_items_0 ON evaluation_items(evaluation_id,outcome);
CREATE UNIQUE INDEX ux_execution_steps_run_item_id ON execution_steps(run_item_id,step_key) WHERE run_item_id IS NOT NULL;
CREATE UNIQUE INDEX ux_execution_steps_index_id ON execution_steps(index_id,step_key) WHERE index_id IS NOT NULL;
CREATE UNIQUE INDEX ux_metric_values_evaluation_id ON metric_values(evaluation_id,metric_name,scope_key) WHERE evaluation_id IS NOT NULL;
CREATE UNIQUE INDEX ux_metric_values_comparison_id ON metric_values(comparison_id,metric_name,scope_key) WHERE comparison_id IS NOT NULL;
CREATE UNIQUE INDEX ux_retrieval_items_selected_rank ON retrieval_items(selected_rank,retrieval_step_id,stage) WHERE selected_rank IS NOT NULL;
CREATE INDEX ix_events_artifact_id ON audit_events(artifact_id,event_id) WHERE artifact_id IS NOT NULL;
CREATE INDEX ix_events_dataset_version_id ON audit_events(dataset_version_id,event_id) WHERE dataset_version_id IS NOT NULL;
CREATE INDEX ix_events_index_id ON audit_events(index_id,event_id) WHERE index_id IS NOT NULL;
CREATE INDEX ix_events_run_id ON audit_events(run_id,event_id) WHERE run_id IS NOT NULL;
CREATE INDEX ix_events_run_item_id ON audit_events(run_item_id,event_id) WHERE run_item_id IS NOT NULL;
CREATE INDEX ix_events_step_id ON audit_events(step_id,event_id) WHERE step_id IS NOT NULL;
CREATE INDEX ix_events_call_attempt_id ON audit_events(call_attempt_id,event_id) WHERE call_attempt_id IS NOT NULL;
CREATE INDEX ix_events_evaluation_id ON audit_events(evaluation_id,event_id) WHERE evaluation_id IS NOT NULL;
CREATE INDEX ix_events_comparison_id ON audit_events(comparison_id,event_id) WHERE comparison_id IS NOT NULL;

-- 已参与最终结果或评分的标签集合不可再追加，保持历史结果稳定。
CREATE TRIGGER freeze_used_prediction_labels BEFORE INSERT ON prediction_labels
WHEN EXISTS(SELECT 1 FROM run_items WHERE final_prediction_id=NEW.prediction_id)
 OR EXISTS(SELECT 1 FROM evaluation_items WHERE prediction_id=NEW.prediction_id)
BEGIN SELECT RAISE(ABORT,'prediction labels already frozen'); END;
-- 已成功执行项只允许通过新 run 开展独立实验。
CREATE TRIGGER immutable_success_item BEFORE UPDATE ON run_items WHEN OLD.status='succeeded'
BEGIN SELECT RAISE(ABORT,'successful item is immutable'); END;
