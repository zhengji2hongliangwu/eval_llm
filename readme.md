# llm模型自定义评估代码
1. 数据加载方式与swift对齐，支持alpaca/sharegpt/openaimessages等格式的本地jsonl文件和swift中的modelscope在线数据
2. 支持评估openai格式封装的llm模型http接口测试
3. 支持评估参数有：
BLEU-1	BLEU-2	BLEU-3	BLEU-4	Rouge-1-F	Rouge-1-P	Rouge-1-R	Rouge-2-F	Rouge-2-P	Rouge-2-R	Rouge-L-F	Rouge-L-P	Rouge-L-R	SelfCondition	ToolCallUseAccuracy	ToolCategoryNameAccuracy	ToolFormatAccuracy	ToolParamValueAccuracy
4. 长对话默认自动切分成n个对话测试集测试每个assistant的回复



运行样例参考eval_example

