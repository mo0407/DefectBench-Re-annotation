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

## 类别体系与 YOLO 映射

本工具的最终标注体系不是只有 4 类，而是由 **4 个主类和 10 个小类**组成。导入 YOLO 检测框时，程序会按下表将 12 个模型类别自动转换为可编辑、可保存的“主类 / 小类”组合；网页右侧的类别下拉框显示的也是这一最终标注体系。

| YOLO 检测类（12） | 程序内映射（主类 / 小类） |
| --- | --- |
| `Concrete_Crack`、`Tile_Crack` | `Crack` / `Linear crack` |
| `Craquelure` | `Crack` / `Map cracking` |
| `Concrete_Spalling`、`Tile_spalling` | `Material_loss` / `Spalling` |
| `Concrete_Delamination`、`Bulging`、`Degraded_Plaster` | `Material_loss` / `Peeling` |
| `Rust_Stain` | `Stain` / `Rust stain` |
| `Water_Stain` | `Stain` / `Leakage stain` |
| `Vegeterian` | `External Fixings` / `Vegetation growth` |
| `Contaminants` | `External Fixings` / `Surface contaminants` |

`Corrosion`（腐蚀）和 `Graffiti`（涂鸦）是最终标注体系中允许专家手工选择的小类，但当前 YOLO 的 12 类检测结果不会自动生成这两类。专家可以在工具中新增或修改检测框后选择它们。

所有保存的检测框都会校验主类与小类的合法组合，以保证导出结果与数据集标注规范一致。当前程序会保留并支持 `Linear crack`、`Map cracking`、`Spalling`、`Peeling` 等细分类；如后续确认最终规范无需区分某些小类，应同步调整程序中的类别体系、YOLO 映射和相关色彩配置。

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

## Linux 启动方式

以下示例适用于 Ubuntu / Debian。应用本身不依赖 Windows 路径；在 Linux 中请使用 `/home/...` 等绝对路径。

```bash
# 1. 安装 Python 与 Git（首次运行时需要）
sudo apt-get update
sudo apt-get install -y python3 python3-venv python3-pip git

# 2. 下载代码并安装依赖
git clone https://github.com/mo0407/DefectBench-Re-annotation.git
cd DefectBench-Re-annotation
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# 3. 可选：指定启动时默认读取的数据集
export DEFECT_BENCH_OPEN_DATASET_ROOT=/home/ubuntu/datasets/defect_batch
export REANNOTATION_PORT=5010

# 4. 启动
python reannotation_app.py
```

浏览器访问 `http://<服务器IP>:5010/`。如服务器启用了防火墙，还需开放端口：

```bash
sudo ufw allow 5010/tcp
```

生产环境建议使用 Gunicorn，而不是 Flask 内置开发服务：

```bash
source .venv/bin/activate
export DEFECT_BENCH_OPEN_DATASET_ROOT=/home/ubuntu/datasets/defect_batch
gunicorn --workers 1 --bind 0.0.0.0:5010 reannotation_app:app
```

在网页中使用 **本地路径导入** 时，填写 Linux 数据集路径，例如 `/home/ubuntu/datasets/defect_batch`。结果会写入该数据集的 `annotation_output/`。如使用 **导入文件夹**，上传批次会保存至 `ANNOTATION_STORAGE_ROOT` 指定目录下的 `imported_datasets/`。

## 在线部署：Render + Cloudflare R2（保留 31 天）

GitHub Pages 只能托管静态页面，不能运行本工具的 Flask API、处理上传或保存标注。要将链接提供给其他人使用，推荐使用 **Render 免费 Web Service + Cloudflare R2 对象存储**：Render 运行网页和 API；R2 保存导入的图片、专家修改、决策轨迹及最终导出结果。

### 1. 创建 R2 存储桶

1. 登录 Cloudflare，打开 **R2**，创建私有 bucket，例如 `defectbench-reannotation`。
2. 在 R2 的 **Manage R2 API Tokens** 创建一个有该 bucket `Object Read & Write` 权限的 API token，保存 Access Key ID 和 Secret Access Key。
3. 复制该账号的 S3 API endpoint，格式类似 `https://<account-id>.r2.cloudflarestorage.com`。
4. 在 bucket 的 **Lifecycle rules** 新建规则：前缀填 `defectbench/datasets/`，对象创建 **31 天后删除**。这会同时清理该批次的原始上传、专家结果、版本轨迹和导出文件。

### 2. 创建 Render 服务

1. 在 Render 选择 **New → Web Service**，连接本 GitHub 仓库。
2. Render 会识别 `render.yaml`；选择 **Free** 计划并创建服务。
3. 在服务的 Environment 页面新增下列环境变量（不要将密钥提交到 GitHub）：

```text
R2_BUCKET=defectbench-reannotation
R2_ENDPOINT_URL=https://<account-id>.r2.cloudflarestorage.com
R2_ACCESS_KEY_ID=<R2 Access Key ID>
R2_SECRET_ACCESS_KEY=<R2 Secret Access Key>
R2_PREFIX=defectbench
R2_RETENTION_DAYS=31
```

4. 点击 **Manual Deploy → Deploy latest commit**。部署成功后，Render 会提供一个 `https://...onrender.com` 链接，可直接发送给其他使用者。

在线模式下请使用 **导入文件夹** 上传数据集；“本地路径导入”只适用于本机运行，线上服务无法访问访问者电脑的磁盘路径。免费 Render 服务闲置后会休眠，首次重新访问可能需要等待启动；由于数据已同步至 R2，重启不会丢失已完成的标注。

### 云端数据结构

R2 中的对象以 `defectbench/datasets/<导入批次 ID>/` 存放。每次保存会立即同步专家标签、Mask、版本快照与 `decision_events.jsonl`；导出最终数据集时，也会同步 `final_exports/`。当前版本是“单个活动批次”模式：新导入的批次会成为网页当前批次，服务重启后会自动恢复最近一次导入的批次。

## 阿里云部署：ECS/轻量应用服务器 + OSS

面向中国大陆用户时，建议将服务器和 OSS Bucket 放在同一地域（如杭州、上海或北京），服务器通过 OSS 内网 Endpoint 访问 Bucket。这样应用与 OSS 之间不产生公网流量费用，且访问延迟更低。

1. 创建私有 OSS Bucket，并在 **数据管理 → 生命周期** 为 `defectbench/datasets/` 设置“最后修改后 31 天删除”。
2. 创建仅授予该 Bucket 读写权限的 RAM 用户或 RAM 角色；不要使用主账号 AccessKey。
3. 购买同地域 ECS 或轻量应用服务器，推荐 Ubuntu 22.04 与 Python 3.11；将本仓库上传或从 GitHub 克隆到服务器。
4. 在服务器配置环境变量后，使用 Dockerfile 或 Gunicorn 启动：

```text
OSS_BUCKET=<bucket-name>
OSS_ENDPOINT=https://oss-<region-id>-internal.aliyuncs.com
OSS_ACCESS_KEY_ID=<RAM AccessKey ID>
OSS_ACCESS_KEY_SECRET=<RAM AccessKey Secret>
OSS_PREFIX=defectbench
R2_RETENTION_DAYS=31
```

```bash
docker build -t defectbench-reannotation .
docker run -d --restart unless-stopped -p 80:8000 \
  --env-file .env defectbench-reannotation
```

配置 `OSS_BUCKET` 后，程序优先使用阿里云 OSS；未配置时仍可兼容原 R2 配置。注意：OSS 的 Python SDK 与 R2 的 boto3 接口不同，不能只替换 Endpoint。中国大陆服务器使用自定义域名对外提供网站时，应按当地要求完成备案。

### 图片容量说明

代码仓库不保存图片、Mask 或导出数据。在线版默认将它们上传到 R2；R2 标准存储每月含 10 GB 免费额度。若单批或一个月内总量超过该额度，应先估算对象存储费用，或缩小批次并及时导出、删除。不要将大量图片提交到 GitHub，也不要依赖 Render 的临时本地磁盘。

## 数据安全

- `decision_events.jsonl`：按时间追加的保存操作日志。
- `revisions/`：按图片保存的版本快照。
- 算法输入 `detections/` 与 `masks/` 不会被覆盖。
- 上传数据与专家结果已在 `.gitignore` 中排除，不应提交到 GitHub。
