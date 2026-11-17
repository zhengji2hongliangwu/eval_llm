# eval_llm

工具调用（tool-call）对话模型的批量评测工具。通过 OpenAI 兼容接口调用被测模型，支持本地 jsonl 与 ModelScope / HuggingFace 数据集，可按 assistant 轮次切分多轮对话逐轮打分，输出 BLEU / ROUGE、工具调用准确率与 SelfCondition 指标，可选上报 SwanLab。

## 目录结构

| 路径 | 说明 |
| --- | --- |
| `eval/eval_run.py` | 主流程：读配置 → 加载数据 → 请求模型 → 打分 → 汇总；入口 `eval_from_data_config` / `eval_from_data_config_swanlab` |
| `eval/llm_client.py` | OpenAI 兼容客户端，同步 + 异步（`OpenAIClient`） |
| `eval/response_messages.py` | 构造请求、并发调用、保存响应（`response_messages` / `response_messages_concurrent`） |
| `eval/cut_messages.py` | 多轮对话切分（`cutmode` / `keepmode`） |
| `eval/tool_call_metrics.py` | 4 项工具调用指标 |
| `eval/self_condition_metric_llm.py` | SelfCondition，调用 judge 模型打分 |
| `eval/metrics/` | BLEU / ROUGE / 通用指标实现 |
| `eval/register_data_custom.py` | 数据集注册与自定义预处理器 |
| `dataset_deal/` | 从 ms-swift 抽出的数据加载层（本地 jsonl、ModelScope、HuggingFace） |
| `eval_example/` | 可直接运行的最小示例（模型配置、数据配置、汇总脚本） |
| `test/` | 调试脚本与 evalscope 对照示例（部分文件为空） |

## 环境

```bash
pip install -r requirements.txt
```

- **不需要安装 ms-swift**：数据加载已内置在 `dataset_deal/`，接口与原 swift 的 `load_dataset` / `register_dataset` 保持一致。
- BLEU / ROUGE 依赖 NLTK 数据，首次运行前执行：

```bash
python test/download_nltk_resources.py
```

## 配置

评测需要两类配置：**数据配置 json** 和 **模型配置 dict**。

### 数据配置

```json
{
  "bench_gesture": {
    "dataset_path": "/workspace/llm/dataset/.../bench_gesture.jsonl",
    "preprocess": "custom",
    "channel": "tool"
  },
  "bench_commensense": {
    "dataset_path": "/workspace/llm/dataset/.../bench_commensense.jsonl",
    "preprocess": "auto",
    "channel": "chat",
    "dataset_sample": 1000
  }
}
```

| 字段 | 说明 |
| --- | --- |
| `dataset_path` | 本地 jsonl 路径，或 `ms::` / `hf::` 开头的在线数据集 ID |
| `preprocess` | `custom`：把 `tool_calls` 拼成 `<tool_call>` 文本、`tool` 角色统一为 `tool_response`、写 `channel`；`auto`：交给 swift 的 `AutoPreprocessor` |
| `channel` | 指标分组维度，常用 `tool` / `chat` / `self_condition` |
| `dataset_sample` | 仅 `auto` 模式使用，抽样条数 |

### 模型配置

```python
llm_config = dict(
    base_url="http://localhost:18008/v1",
    api_key="YOUR_API_KEY",
    model="Qwen3-0.6B-finetune",
    temperature=0.0,
)

# SelfCondition 指标需要 judge 模型
judge_llm_config = dict(
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key="YOUR_API_KEY",
    model="deepseek-v3",
    temperature=0.0,
)

llm_config_list = [llm_config, {...}]   # 传 list 可一次评测多个模型并对比
```

## 快速开始

1. 启动任意 OpenAI 兼容服务（vLLM / SGLang / 云端 API）。
2. 修改 `eval_example/config.py` 里的 `base_url` / `api_key` / `model`。
3. 运行：

```bash
cd eval_example
python main.py
python out_metric_csv.py   # 按 Model / Dataset / Channel 汇总平均分
```

4. 产物：

```text
output/all_metric.csv      # 逐样本指标
output/all_responses.csv   # 完整请求与回复（按 sample_id 与指标表关联）
output/averages_*.csv      # 汇总平均分
```

`main.py` 中常用参数：

| 参数 | 说明 |
| --- | --- |
| `jsonpath` | 数据配置 json 路径 |
| `eval_llm_config` | 单个 dict 或 list |
| `max_per_eval_num` | 每个数据集只取前 N 条（**截断，不是随机抽样**） |
| `cutmode` / `keepmode` | 多轮切分方式，默认 `assistant` / `all` |
| `execution_mode` | `concurrent`（默认）/ `async` / `sequential` |
| `max_workers` / `max_concurrency` | 并发度 |
| `use_swanlab` | 是否上报 SwanLab（需填 `swanlab_*`） |

## 指标

| 指标 | 含义 |
| --- | --- |
| `BLEU-1` ~ `BLEU-4` | 与参考答案的 n-gram 重叠 |
| `Rouge-1/2/L` 的 `-F/-P/-R` | ROUGE 精确率、召回率、F1 |
| `ToolCallUseAccuracy` | 该不该调用工具（调用/不调用的判断是否正确） |
| `ToolFormatAccuracy` | `<tool_call>` 输出格式是否合法 |
| `ToolCategoryNameAccuracy` | 调用的工具类别 / 名称是否正确 |
| `ToolParamValueAccuracy` | 工具参数值是否正确 |
| `SelfCondition` | 用 judge 模型判断回复是否满足前置条件 |

工具指标不需要 judge 模型，纯规则解析；`SelfCondition` 必须提供 `judge_llm_config`。

## 多轮对话切分

长对话默认按 assistant 轮次拆成多条子样本，逐轮构造 prompt 并评测：

| 参数 | 取值 | 说明 |
| --- | --- | --- |
| `cutmode` | `assistant` / `tool_call` | 在每个 assistant 消息处切割，或只在带工具调用的 assistant 处切割 |
| `keepmode` | `first` / `last` / `all` | 只测第一轮 / 只测最后一轮 / 每个切点都生成一条（推荐，覆盖完整对话） |

## 注意事项

1. 示例配置中的 `/workspace/llm/...` 为绝对路径，换机器必须全量替换。
2. `api_key`、`swanlab_token`、`swanlab_lark_webhook_url` 等仍是 `YOUR_*` 占位符，不填会请求失败或无法上报。
3. `eval_example/config_traing_data.json` 是训练侧配置，评测只需 `config_test_*.json`。
4. `test/` 与 `eval/test/` 下部分文件为空或依赖本地路径，属历史调试脚本，不是正式单测。
5. 指标表与响应表通过 `sample_uuid` ↔ `sample_id` 关联，自定义分析时注意保留该列。
