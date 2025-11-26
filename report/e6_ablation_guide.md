# E6 超参数消融实验指南

## 实验目标

对比不同学习率、优化器、batch size 对 ResNet50 性能的影响，完成指南第 7.2 节“超参数敏感性”要求。

## 实验配置列表

| 实验编号 | 优化器 | 学习率设置 | Batch Size | 说明 |
|---------|--------|-----------|-----------|------|
| E6-1 | AdamW | lr_head=5e-4, lr_backbone=5e-5 | 16 | 较低学习率 |
| E6-2 | AdamW | lr_head=1e-3, lr_backbone=1e-3 | 16 | 统一学习率 |
| E6-3 | SGD | lr=3e-3 (统一) | 16 | SGD vs AdamW 对比 |
| E6-4 | AdamW | lr_head=1e-3, lr_backbone=1e-4 | 8 | Batch size 影响 |

> 所有实验均使用 `--augment basic`（基础增强），与 E3 保持一致。

---

## 使用方法

### 方法一：批量运行所有实验（推荐）

```powershell
# Windows PowerShell
python -m src.training.run_e6_ablation --device cpu --num-workers 0

# 如果使用 GPU
python -m src.training.run_e6_ablation --device cuda
```

**说明：**
- 脚本会按顺序运行 4 个实验，每个实验完成后自动开始下一个
- 每个实验大约需要 30-60 分钟（CPU）或 10-20 分钟（GPU）
- 总时间：约 2-4 小时（CPU）或 40-80 分钟（GPU）

### 方法二：逐个运行单个实验

```powershell
# 只运行 E6-1
python -m src.training.run_e6_ablation --experiment E6-1_adamw_lr5e4 --device cpu --num-workers 0

# 只运行 E6-3 (SGD)
python -m src.training.run_e6_ablation --experiment E6-3_sgd_lr3e3 --device cpu --num-workers 0
```

### 方法三：预览命令（不实际运行）

```powershell
# 查看所有实验的命令
python -m src.training.run_e6_ablation --list

# 预览单个实验的命令
python -m src.training.run_e6_ablation --experiment E6-1_adamw_lr5e4 --dry-run
```

### 方法四：直接使用 resnet50_baseline.py（手动控制）

如果想更精细地控制参数，可以直接调用原始脚本：

```powershell
# E6-1: 较低学习率
python -m src.training.resnet50_baseline ^
    --experiment-name E6-1_adamw_lr5e4 ^
    --augment basic ^
    --optimizer adamw ^
    --lr-head 5e-4 ^
    --lr-backbone 5e-5 ^
    --batch-size 16 ^
    --log-csv outputs/logs/E6-1_adamw_lr5e4.csv ^
    --pred-csv outputs/predictions/E6-1_adamw_lr5e4_test.csv ^
    --device cpu ^
    --num-workers 0

# E6-3: SGD 优化器
python -m src.training.resnet50_baseline ^
    --experiment-name E6-3_sgd_lr3e3 ^
    --augment basic ^
    --optimizer sgd ^
    --lr-head 3e-3 ^
    --lr-backbone 3e-3 ^
    --batch-size 16 ^
    --log-csv outputs/logs/E6-3_sgd_lr3e3.csv ^
    --pred-csv outputs/predictions/E6-3_sgd_lr3e3_test.csv ^
    --device cpu ^
    --num-workers 0
```

---

## 输出文件

每个实验会生成以下文件：

- `outputs/logs/E6-X_*.csv`：训练日志（每个 epoch 的指标）
- `outputs/checkpoints/E6-X_*_best.pt`：最佳模型权重
- `outputs/predictions/E6-X_*_test.csv`：测试集预测结果

---

## 结果汇总建议

实验完成后，可以从日志 CSV 中提取最佳验证集 `val_macro_f1` 和测试集指标，制作对比表格：

```python
import pandas as pd

experiments = [
    "E3_resnet50_basic_aug",  # 基准
    "E6-1_adamw_lr5e4",
    "E6-2_adamw_lr1e3_uniform",
    "E6-3_sgd_lr3e3",
    "E6-4_adamw_batch8",
]

results = []
for exp in experiments:
    log = pd.read_csv(f"outputs/logs/{exp}.csv")
    best_val_f1 = log["val_macro_f1"].max()
    # 从预测文件或日志中提取测试指标
    results.append({"实验": exp, "最佳 Val Macro-F1": best_val_f1})

df = pd.DataFrame(results)
print(df)
```

---

## 注意事项

1. **时间安排**：如果时间紧张，可以优先运行 E6-1 和 E6-3（学习率和优化器对比），这两个对报告最有价值。

2. **资源限制**：如果本地 CPU 太慢，可以考虑：
   - 只运行 1-2 个关键实验
   - 在 Colab 上运行（参考之前的 Colab 教程）

3. **早停机制**：所有实验都启用了早停（连续 3 个 epoch 无提升即停止），所以实际训练时间可能比预期短。

4. **结果记录**：建议在实验过程中记录每个实验的开始/结束时间，方便后续整理。

---

## 故障排查

- **"No module named 'src'"**：确保在项目根目录（`D:\PlantDoc`）运行命令
- **内存不足**：减小 `--batch-size`（例如改为 8）
- **训练中断**：可以单独重新运行失败的实验，使用 `--experiment` 参数

