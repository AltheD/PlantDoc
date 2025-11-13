# data/splits

用于保存训练/验证/测试的划分索引文件，推荐 JSON 或 CSV。

示例：
- `plantdoc_split_seed42.example.json`：演示字段结构，实际使用时复制并改名为不带 `.example` 的文件。

一致性建议：
- 固定随机种子（如 42），并在文件名中注明
- 所有实验共用同一份划分文件，避免数据泄漏


