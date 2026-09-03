# DefectBench Re-annotation

用于审阅和修正缺陷检测框（BBox）与分割 Mask 的 Web 标注工具。算法结果作为初始标注，专家可移动、缩放、删除、绘制、重置并保存；每一次保存都会生成可追溯的决策记录。

## 功能

- 导入一批原图、算法检测框和算法 Mask。
- 编辑 BBox：移动、缩放、删除、新建、恢复算法初始框。
- 编辑 Mask：画笔添加/擦除、透明度调节、恢复算法初始 Mask。
- 保存专家结果、版本快照与追加式决策轨迹。
- 导出全量最终数据集：已编辑样本使用专家结果，未编辑样本保留算法结果。

## 输入数据集

导入目录必须至少包含：

```text
dataset/
├─ images/                  # 必需，.jpg / .jpeg / .png
├─ detections/              # 必需，检测框 JSON
├─ masks/                   # 必需，算法 Mask（PNG/JPG）
└─ metadata/                # 可选，算法元数据 JSON
```

文件用同一个样本 ID 对应，例如：

```text
images/sample_image.jpg
detections/sample_detection.json
masks/sample_mask.png
metadata/sample_metadata.json
```

检测 JSON 支持 `annotations_in_crop`、`detections`、`annotations` 或 `bboxes` 数组。输出 BBox 的统一坐标格式为 `[x, y, width, height]`。

`unet/`、`sam3/` 和说明文档不是当前网页的必需输入。

## 操作流程

1. 点击 **导入文件夹**，选择数据集根目录；在线模式会复制到服务器的持久化数据目录。
2. 在左侧编辑检测框，在右侧编辑 Mask。回退图标可恢复算法初始结果。
3. 点击 **Save Changes** 保存当前图片。
4. 点击 **导出最终数据集**，生成所有图片的最终结果。

未点击 Save Changes 的修改不会被导出。

## 输出

对每个已保存样本，生成：

```text
expert_labels/<image_stem>.json
expert_masks/<image_stem>_mask.png
revisions/<image_stem>/revision_XXXX.json
decision_events.jsonl
```

完整导出位于 `final_exports/<timestamp>_final/`：

```text
images/       # 完整原图集
labels/       # 每张图一个最终 BBox JSON
masks/        # 每张图一个最终 Mask PNG
metadata/     # 可选元数据副本
manifest.json # 每张图采用 expert 或 algorithm 的来源记录
revisions/ 和 decision_events.jsonl
```

## 本地运行

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python reannotation_app.py
```

访问 `http://127.0.0.1:5010/`。

本机如需直接在原始数据集旁保存，可使用 **本地路径导入**，输入数据集根目录。结果会写入：

```text
<dataset>/annotation_output/
```

## Render 在线部署

本仓库包含 `render.yaml`。在 Render 中从 GitHub 新建 Web Service 并选择该仓库；配置会使用 `/var/data` 挂载的持久化磁盘，上传批次、专家结果与导出数据都会保存在该磁盘中，而不是 Git 仓库中。

GitHub Pages 只能托管静态页面，不能运行本工具的 Flask API、处理上传或保存标注。

### 图片容量说明

代码仓库不保存图片、Mask 或导出数据。小批量数据可经网页文件夹导入并存入持久化磁盘。对于大批量或高分辨率图片，应使用 S3、OSS 或 MinIO 等对象存储，并改为直接上传到对象存储；不要将大量图片提交到 GitHub，也不要依赖临时云磁盘。

## 数据安全

- `decision_events.jsonl`：按时间追加的保存操作日志。
- `revisions/`：按图片保存的版本快照。
- 算法输入 `detections/` 与 `masks/` 不会被覆盖。
- 上传数据与专家结果已在 `.gitignore` 中排除，不应提交到 GitHub。
