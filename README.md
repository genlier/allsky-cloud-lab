# Allsky Cloud Lab

全天空图像云分类：PyTorch 迁移学习、批量推理与浏览器交互展示。

## 交互展示

网页支持上传天空照片，自动估算云量，再点击识别按钮获取六种云类型候选得分；也支持掩膜预览、阈值调节与分析 JSON 下载。使用 GitHub Pages，或在项目根目录执行 `python -m http.server 8000 --directory docs` 后打开 `http://localhost:8000`。

原压缩包只保存了两份源码和报告。重新获取标注数据后，分别训练了云类型六分类和云量四分类 ResNet18，并导出为 ONNX。网页使用这两个模型本地识别，另提供可调颜色阈值辅助查看云候选比例。运行方式见 [ONNX Runtime Web 官方文档](https://onnxruntime.ai/docs/get-started/with-javascript/web.html)。

照片只在浏览器中解码、推理，不发送至远端。首次识别会请求本站的两个 ONNX 模型（约 90 MB）与 jsDelivr 的固定版本 ONNX 运行库；网络受限时仍可使用云量阈值分析。模型使用浏览器缓存，耗时依设备性能而变。导出的 JSON 不含照片、原文件名或 EXIF。

## 安装

```sh
conda activate myenv
python -m pip install -r requirements.txt
```

已安装 CUDA 版 PyTorch 时保留现有环境。代码自动选择 CUDA 或 CPU。

## 数据

将自行获得授权的图片放在本地 `data/` 下。仓库包含代码、合成演示和用于网页推理的模型参数，原始照片保持本地。

| 目录 | 用途 | 标签格式 |
| --- | --- | --- |
| `data/cloud_type/` | 云类型监督样本 | `类别_编号.jpg`，类别 1–6 |
| `data/cloud_amount/` | 云量监督样本 | `类别_编号.jpg`，类别 1–4 |
| `data/labeled_raw/` | 可选补充监督样本 | 前两段依次为云类型、云量；使用前请核实 |
| `data/AllskyImagerData/` | 待预测图像，支持年月/日期目录 | 无需标签 |

云类型类别依次为晴空、卷云、层状云、积状云、积雨云、混合云。云量类别依次为无云、少云、多云、阴天。

`peridata_*.csv` 为仪器伴随记录。当前公开 CNN 版本使用图像输入；这些记录的字段单位、设备代码和时间基准需查明后再融合，时间匹配本身不等于获得云类别真值。

原始 Allsky 下载包包含按月嵌套的 ZIP。可先解压一个月的少量图像验证流程：

```sh
python prepare_allsky.py --archive /path/to/AllskyImagerData.zip --month 202303 --max-images 32
```

省略 `--month` 和 `--max-images` 可解压全部月份。脚本以流式方式读取一个月份，并保留选中日期的伴随 CSV，检查路径越界与符号链接。全量解压前请预留足够磁盘空间。原始无标签图像用于推理检查，测试准确率来自独立标注测试集。

## 训练

```sh
python train_cnn.py --data-dir data/cloud_type --task cloud_type --epochs 20 --output outputs/cloud_type.pth
```

补充已标注样本：

```sh
python train_cnn.py --data-dir data/cloud_type --extra-data-dir data/labeled_raw --extra-label-mode raw --task cloud_type --output outputs/cloud_type.pth
```

主样本和补充样本合并后，先按解码像素 SHA-256 去重，再分层划分 70% 训练候选集、30% 测试集；训练候选集内部划出验证集。仅验证集参与最佳轮次选择，最终测试集评估一次。相同像素对应冲突标签会报错，便于人工核查。近重复图、同一时刻的双设备图和相邻时刻图像仍需按日期/观测事件分组评估，当前随机划分只代表这一协议下的效果。

恢复的云量标注存在一组相同像素、不同标签的冲突。此次展示模型使用 `--drop-conflicting` 排除该组所有副本。该参数需要显式启用，默认行为仍为报错。

默认采用 ImageNet 预训练 ResNet18、轻度图像增强、AdamW、余弦学习率与标签平滑；可用 `--model resnet50`。模型越大并不保证小样本精度越高。首次训练会下载公开预训练权重到项目内 `.cache/`。显存不足时减小 `--batch-size`。输出模型、逐轮日志、划分清单及最终评估 JSON 到 `outputs/`。

早期对话中 85.26% 和 92.78% 的结果加入了未去重补充数据且反复查看测试集，存在泄漏和选择偏差；它们未作为项目验证指标发布。这里的指标以重新运行产生的评估文件为准。

## 导出网页模型

```sh
python -m pip install -r requirements-export.txt
python export_onnx.py --weights outputs/cloud_type.pth --output docs/models/cloud_type.onnx
python export_onnx.py --weights outputs/cloud_amount.pth --output docs/models/cloud_amount.onnx
```

网页当前模型均为 ResNet18，输入尺寸 224。重新训练展示模型时使用该配置。导出时清除文档字符串和元数据，并验证 PyTorch 与 ONNX 数值一致性。网页使用 Canvas 缩放，训练使用 Pillow 缩放，两者的像素插值可能产生小幅差异。

## 批量预测

```sh
python predict_cnn.py --weights outputs/cloud_type.pth --data-dir data/AllskyImagerData --output-csv outputs/predictions.csv
```

自动递归扫描。CSV 输出相对路径、类别、最大 softmax 概率及读取状态。损坏图片保留一条错误记录，处理继续。概率反映模型输出，尚未进行概率校准。仅加载可信来源的模型文件。旧纯 state_dict 权重可加 `--model resnet18` 或 `--model resnet50` 指定架构。

## 隐私

原始图片、仪器记录、训练日志、真实预测、权重与缓存均默认忽略。输出 CSV 使用数据根目录的相对路径，仍可能包含采集日期或设备编号，请勿直接公开。公开仓库未包含姓名、学号、邮箱、原提交链接或本机绝对路径。网页仅请求公共依赖与模型资源，无遥测或照片上传。

`legacy/` 保存从压缩包恢复的两份算法源码，默认目录已匿名化。原报告含个人信息，未加入公开仓库。根目录训练代码修正了原评估划分与模型类别数问题。仅 `docs/models/` 下用于展示的 ONNX 网络参数公开，其中没有图像、文件名或个人信息；其他训练权重保持本地。

## 数学方法

ResNet 将图像映射为特征，再由线性层输出类别 logits。softmax 概率为

$$p_k = \frac{\exp(z_k)}{\sum_j\exp(z_j)}.$$

标签平滑使用 $q_k=(1-\varepsilon)\mathbf{1}_{k=y}+\varepsilon/K$，优化交叉熵 $L=-\sum_k q_k\log p_k$。余弦学习率随训练轮次衰减，以降低后期参数更新幅度。

网页阈值演示使用 $D=(R-B)/(R+B+\epsilon)$。在有效圆形区域内，将 $D>t$ 的像素视为云候选；云候选占有效像素的比例用于四档分类。该像素比例受薄云、气溶胶、太阳眩光与镜头投影影响，不等同于标定后的半球云覆盖率。
