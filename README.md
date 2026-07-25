# Human Semantic Validation — 标注工具

按论文协议实现:每个 model–method 组合抽 30 个 prediction-change cases(共 540),
两名主标注员独立盲评,分歧由第三人仲裁;界面隐藏生成方法与模型预测;
ImageNet 额外显示 synset 名称、同义词、释义。

纯 Python 标准库(仅导出脚本需要 torch),无需安装任何依赖。

## 1. 准备数据(生成 cases.jsonl)

对每个 (model, method) 组合运行一次,追加写入同一个 study 目录:

```bash
python make_manifest.py \
  --failures ../Latte/results/mnist_lenet5_single/failures_single.pt \
  --dataset MNIST --model lenet5 --method latte \
  --num 30 --seed 0 --out ./study

python make_manifest.py \
  --failures ../Latte/results/cifar10_vgg16_single/failures_single.pt \
  --dataset CIFAR10 --model vgg16 --method latte \
  --num 30 --seed 0 --out ./study

# ImageNet 必须提供类别信息(synset 名称/同义词/释义):
python make_manifest.py \
  --failures ../Latte/results/imagenet_vgg19_single/failures_single.pt \
  --dataset ImageNet --model vgg19 --method latte \
  --num 30 --seed 0 --out ./study \
  --class-info imagenet_class_info.json
```

`--class-info` JSON 格式(可用 NLTK WordNet 生成):

```json
{"0": {"label": "tench", "synonyms": ["Tinca tinca"], "definition": "freshwater dace-like game fish ..."}}
```

**基线方法**的结果如果不是 LATTE 的 `.pt` 格式,自己写导出脚本即可,只要往
`study/cases.jsonl` 追加同样字段的行、把图片放进 `study/images/`:

```json
{"case_id": "随机hex", "dataset": "...", "model": "...", "method": "...",
 "source_image": "images/xxx_a.png", "generated_image": "images/xxx_b.png",
 "gt_class_index": 0, "gt_label": "...", "gt_synonyms": [], "gt_definition": "",
 "pred_class_index": 3}
```

注意 `case_id` 和图片文件名必须是随机的(不能包含 method/model),否则会破坏盲评。

## 2. 主标注阶段(两名标注员)

```bash
python server.py --study ./study --port 8765
```

标注员浏览器打开 `http://<你的IP>:8765`,输入各自的 ID(如 `alice` / `bob`)。

- 每人看到的题目顺序按其 ID 确定性打乱(不同人顺序不同,同一人每次一致);
- 界面只显示原图/生成图 + 源标签信息,**不显示** method、model、模型预测;
- 每次点击立刻写入 `annotations/<ID>.jsonl`,可随时关页面、随时续标;
- 快捷键:`1` 保留 / `2` 未保留 / `←` `→` 翻页;同一题重标以最后一次为准。

先用 `--study ./demo` 给标注员练手(9 个示例题,不计入正式数据)。

## 3. 仲裁阶段(第三名标注员)

两名主标注员**全部标完后**:

```bash
python server.py --study ./study --port 8765 --adjudicate alice bob
```

仲裁人(如 `carol`)登录后只会看到两人有分歧的题目,同样盲评(看不到两人的标签)。

## 4. 统计

```bash
python analyze.py --study ./study --a1 alice --a2 bob --adj carol
```

输出每个 dataset/model/method 组合与总体的:

- 判定为保留源标签的数量与比例;
- 95% Wilson 置信区间;
- Verified DoF(保留案例中不同的 `pred_class_index` 数量);
- Cohen's κ(两名主标注员仲裁前的一致性);

并写出 `study/validation_summary.csv`。

## 文件说明

| 文件 | 作用 |
|---|---|
| `make_manifest.py` | 从 LATTE `failures_*.pt` 抽样导出案例(需 torch) |
| `server.py` | 标注 GUI 服务(标准库,主标注 + 仲裁两种模式) |
| `analyze.py` | Wilson CI / κ / Verified DoF 统计(标准库) |
| `study/cases.jsonl` | 案例清单(含 method/model,**不要发给标注员**) |
| `annotations/*.jsonl` | 每个标注员的判定记录(追加式,含时间戳) |
| `demo/` | 9 个示例题,供练手与检查界面 |
