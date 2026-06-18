# 数据脱敏系统

一个基于深度学习的智能图像数据脱敏系统，支持人脸、车牌、文本的自动检测与模糊处理。

## 功能特性

- **人脸脱敏**: 使用YOLOv8自动检测并模糊人脸区域
- **车牌脱敏**: 使用YOLOv8检测并模糊车牌号码
- **文本脱敏**: 使用DBNet检测并模糊街牌、标识等敏感文本
- **Web界面**: 现代化的Web用户界面，支持拖拽上传和批量处理
- **多服务器支持**: 支持欧洲、美国、亚洲三个服务器区域，自动使用内置SFTP配置
- **隐私保护**: 本地处理，处理后的数据上传到所选服务器
- **CPU优化**: 专为CPU环境优化，无需GPU即可运行
- **容器化部署**: 支持Docker一键部署
- **后台任务**: 异步处理，支持任务队列和状态查询
- **多语言支持**: 支持中文和英文界面切换

## 快速开始

### 前置要求

- Python 3.9+
- pip 或 Conda
- 或 Docker 24+

### 安装步骤

#### 1. 克隆项目

```bash
git clone http://gitlab.rokibot.com/department_algorithm/ai/tools/data_masking.git
cd data_masking
```

#### 2. 准备模型文件

确保以下模型文件存在：

```bash
# 统一检测模型（人脸+车牌）
models/
└── best.pt                            # 统一检测模型（人脸+车牌）

# 文本检测模型（DBNet）
src/DBNet/weights/
└── best.pt                            # DBNet文本检测模型
```

**模型文件说明**：
- `models/best.pt`：统一检测模型，用于检测和模糊处理人脸和车牌
- `src/DBNet/weights/best.pt`：DBNet文本检测模型，用于检测和模糊处理图像中的文本（街牌、标识等）

#### 3. 配置服务器SFTP信息

编辑 `config/server_regions.yaml` 文件，配置各服务器区域的SFTP连接信息：

```yaml
europe:
  name: "欧洲服务器"
  description: "符合GDPR规范，适合欧洲用户"
  flag: "🇪🇺"
  sftp:
    host: "europe.example.com"
    user: "username"
    password: "password"
```

也可以通过环境变量覆盖配置文件中的 SFTP 值：

```bash
EU_SFTP_HOST=europe.example.com
EU_SFTP_USER=username
EU_SFTP_PASSWORD=password
```

#### 4. 启动服务

```bash
# 安装依赖
pip install -r requirements.txt
# 启动应用
python app.py
```

默认访问地址：`http://127.0.0.1:8000`

#### 5. Docker 部署

```bash
# 构建镜像
docker build -t data-masking:latest .

# 启动容器
docker run --rm -p 8000:8000 \
  -v $(pwd)/logs:/app/logs \
  -v $(pwd)/staging:/app/staging \
  -v $(pwd)/output:/app/output \
  -v $(pwd)/data_masking.db:/app/data_masking.db \
  data-masking:latest
```

容器默认监听 `8000` 端口，访问地址：`http://127.0.0.1:8000`

如果云平台要求注入端口环境变量，可改用：

```bash
docker run --rm -p 8000:8000 \
  -e PORT=8000 \
  -v $(pwd)/logs:/app/logs \
  -v $(pwd)/staging:/app/staging \
  -v $(pwd)/output:/app/output \
  -v $(pwd)/data_masking.db:/app/data_masking.db \
  data-masking:latest python app.py
```

## 使用指南

### Web界面使用

1. **访问系统**: 打开浏览器访问 http://localhost:8000

2. **阅读隐私协议**: 首次访问需要阅读并同意隐私声明

3. **选择服务器区域**: 
   - 欧洲服务器
   - 美国服务器
   - 亚洲服务器
   
   系统会自动使用 `config/server_regions.yaml` 中对应区域的配置，并优先使用环境变量覆盖 SFTP 信息

4. **选择文件或文件夹**: 
   - 支持拖拽上传
   - 支持点击选择文件
   - 支持选择整个文件夹
   - 界面会显示文件统计信息（文件数量和总大小）

5. **开始上传**: 
   - 点击"开始上传"按钮
   - 系统会自动处理图像（人脸、车牌、文本脱敏）
   - 如果所选区域配置了完整 SFTP 信息，处理完成后会上传到对应服务器的 `/mnt/data` 目录
   - 如果所选区域未配置完整 SFTP 信息，系统会自动回退到本机 `/mnt/data` 目录
   - 上传的文件会按时间戳和区域代码组织（格式：`YYYYMMDD_HHMMSS_区域代码/`）
   - 可以查看任务队列状态

## 配置说明

### Docker 运行时目录

容器内以下路径需要可写：

- `/app/logs`：运行日志
- `/app/staging`：上传暂存、解压、中间处理文件
- `/app/output`：本机输出目录
- `/app/data_masking.db`：SQLite 数据库
- `/mnt/data`：当系统走本地落盘模式时的输出目录

建议至少挂载：

```bash
-v $(pwd)/logs:/app/logs \
-v $(pwd)/staging:/app/staging \
-v $(pwd)/output:/app/output
```

如果你希望数据库持久化，也挂载：

```bash
-v $(pwd)/data_masking.db:/app/data_masking.db
```

### Docker 环境变量

Docker 部署时可继续使用现有环境变量覆盖配置，例如：

```bash
PORT=8000
SERVER_PORT=8000
STAGING_ROOT=/app/staging
SQLITE_DB_PATH=/app/data_masking.db
EU_SFTP_HOST=europe.example.com
EU_SFTP_USER=username
EU_SFTP_PASSWORD=password
US_SFTP_HOST=america.example.com
US_SFTP_USER=username
US_SFTP_PASSWORD=password
AS_SFTP_HOST=asia.example.com
AS_SFTP_USER=username
AS_SFTP_PASSWORD=password
```

管理员密码仍然默认从 `config/server_regions.yaml` 读取。

### 配置文件

主要配置入口位于 `config.py`，其中三台区域服务器配置从 `config/server_regions.yaml` 读取，并支持通过环境变量覆盖 SFTP 值。

#### server_regions.yaml 配置

`config/server_regions.yaml` 文件用于配置 Europe / America / Asia 三个区域的服务器信息。示例：

```yaml
europe:
  name: "欧洲服务器"
  description: "符合GDPR规范，适合欧洲用户"
  flag: "🇪🇺"
  sftp:
    host: "europe.example.com"
    user: "username"
    password: "password"
```

**覆盖优先级**：
1. 环境变量（如 `EU_SFTP_HOST` / `EU_SFTP_USER` / `EU_SFTP_PASSWORD`）
2. `config/server_regions.yaml`

如果某区域的 `host/user/password` 不完整，系统会自动回退到本机 `/mnt/data` 存储模式。

#### 文本模糊处理配置

文本检测和模糊处理使用DBNet模型，相关配置通过环境变量或 `config.py` 中的 `ANONYMIZATION` 字典进行设置。

**可配置参数**（通过环境变量）：

```bash
# 文本区域膨胀像素数
# 用于扩大检测到的文本区域，确保完全覆盖文本内容
# 值越大，覆盖范围越大，但可能模糊更多无关区域
# 建议值：4-16
TEXT_DILATE_PX=8

# 文本区域额外填充比例
# 在检测框周围添加额外的填充区域（相对于检测框尺寸的比例）
# 范围：0.0-1.0
# 0.0 表示不添加额外填充
# 建议值：0.0-0.2
TEXT_PAD_RATIO=0.0

# 是否使用填充矩形
# True: 将多边形文本区域转换为带填充的矩形（轴对齐）
# False: 保持原始多边形形状（推荐）
# 注意：矩形模式会放大覆盖区域，可能模糊更多内容
TEXT_USE_PADDED_RECT=false

# DBNet模型路径（可选）
# 默认：src/DBNet/weights/best.pt
DBNET_ROOT=src/DBNet
TEXT_WEIGHTS=src/DBNet/weights/best.pt

# 文本检测输入尺寸
# 推理时的高度，宽度按比例缩放并对齐到32的倍数
# 较大的值可以提高检测精度，但会增加处理时间
# 建议值：640, 960, 1280
TEXT_INPUT_SIZE=960
```

**参数说明**：

1. **TEXT_DILATE_PX** (文本区域膨胀像素):
   - 作用：扩大检测到的文本区域边界，确保完全覆盖文本内容
   - 原理：使用形态学膨胀操作，在文本区域周围添加像素
   - 值越大：覆盖范围越大，但可能模糊更多无关区域
   - 值越小：覆盖更精确，但可能遗漏文本边缘
   - 建议值：8-16（默认8）

2. **TEXT_PAD_RATIO** (文本区域填充比例):
   - 作用：在检测框周围按比例添加额外填充区域
   - 范围：0.0-1.0
   - 0.0：不添加额外填充（默认）
   - 0.1：添加检测框尺寸10%的填充
   - 建议值：0.0-0.2

3. **TEXT_USE_PADDED_RECT** (使用填充矩形):
   - `False`（推荐）：保持原始多边形形状，精确覆盖文本区域
   - `True`：转换为轴对齐矩形，会放大覆盖区域
   - 注意：矩形模式可能模糊更多无关内容

4. **TEXT_INPUT_SIZE** (输入尺寸):
   - 影响检测精度和处理速度
   - 较大值（1280）：更高精度，但处理更慢
   - 较小值（640）：更快处理，但可能降低小文本检测率
   - 建议值：960（平衡精度和速度）

**配置示例**：

```bash
# 精确检测配置（减少误模糊）
TEXT_DILATE_PX=4
TEXT_PAD_RATIO=0.0
TEXT_USE_PADDED_RECT=false
TEXT_INPUT_SIZE=1280

# 高召回率配置（确保不遗漏文本）
TEXT_DILATE_PX=16
TEXT_PAD_RATIO=0.1
TEXT_USE_PADDED_RECT=false
TEXT_INPUT_SIZE=960

# 强隐私保护配置（扩大覆盖范围）
TEXT_DILATE_PX=20
TEXT_PAD_RATIO=0.2
TEXT_USE_PADDED_RECT=true
TEXT_INPUT_SIZE=960
```

**在config.py中配置**：

也可以通过直接修改 `config.py` 文件中的 `ANONYMIZATION` 字典：

```python
ANONYMIZATION = {
    "text_dilate_px": 8,              # 文本区域膨胀像素
    "text_pad_ratio": 0.0,            # 文本区域填充比例
    "text_use_padded_rect": False,    # 是否使用填充矩形
}
```

**注意事项**：
- 修改环境变量或 `config.py` 后需要重启应用才能生效
- `TEXT_DILATE_PX` 值过大会导致模糊过多无关区域
- `TEXT_USE_PADDED_RECT=true` 会显著增加模糊区域，建议谨慎使用
- `TEXT_INPUT_SIZE` 影响处理速度，建议根据实际需求调整

## 项目结构

```
data_masking/
├── app.py                      # FastAPI主应用
├── config.py                   # 系统配置文件
├── translations.py              # 多语言翻译字典（中英文）
├── requirements.txt             # Python依赖
├── README.md                    # 项目说明文档
├── config/
│   └── server_regions.yaml      # 三个区域的SFTP配置
├── src/
│   ├── __init__.py
│   ├── pipeline/               # 脱敏处理管道
│   │   ├── __init__.py
│   │   ├── unified_blurrer.py  # 统一检测器（人脸+车牌）
│   │   └── texts.py            # 文本检测与模糊（DBNet）
│   └── DBNet/                  # DBNet文本检测运行时依赖
│       ├── nets/                # 网络结构定义
│       │   └── nn.py
│       ├── utils/              # 运行时工具函数
│       │   └── util.py
│       ├── weights/            # DBNet模型权重
│       │   └── best.pt         # DBNet最佳模型
│       └── demo/               # 目录保留，可为空
├── models/                     # AI模型文件目录
│   └── best.pt                 # 统一检测模型（人脸+车牌）
├── templates/                  # Web模板文件
│   ├── base.html               # 基础模板
│   ├── index.html              # 主页面
│   ├── privacy.html            # 隐私协议页面
│   └── remote.html             # 远程访问页面
├── static/                     # 静态资源目录（当前可为空）
├── data/                       # 运行时样例数据目录
│   └── carpai.jpeg
├── uploads/                    # 上传文件目录（按区域分类）
├── output/                     # 输出文件目录
├── temp/                       # 临时文件目录（处理会话临时文件）
└── logs/                       # 日志文件目录
    └── app.log                 # 应用日志文件（自动轮转，保留7天）
```

### 自定义模型

1. 将模型文件放入 `models/` 目录
2. 修改对应的pipeline模块
3. 更新 `config.py` 中的模型配置

## 故障排除

### 常见问题

#### 1. 模型文件缺失

**症状**: 启动时提示模型文件不存在

**解决方案**:
```bash
# 检查模型文件
ls -la models/
ls -la /src/DBNet/weights

# 确保以下文件存在：
# - best.pt
```

#### 2. SFTP连接失败

**症状**: 上传失败，提示SFTP连接错误

**解决方案**:

- 检查 `config.py` 中的SFTP配置
- 验证服务器地址、用户名、密码
- 检查网络连接和防火墙设置

#### 3. 日志文件位置

日志文件位置：`logs/app.log`

日志轮转：每天轮转，保留7天

### 数据上传说明

**上传目录**：

处理后的图像文件会自动上传到所选服务器的 `/mnt/data` 目录下。

**目录结构**：

上传的文件会按照以下格式组织：

```
/mnt/data/
└── YYYYMMDD_HHMMSS_区域代码/
    ├── image1.jpg
    ├── image2.jpg
    └── ...
```

**目录命名规则**：
- `YYYYMMDD_HHMMSS`：处理时间戳（年月日_时分秒）
- `区域代码`：根据选择的服务器区域自动添加
  - `EU`：欧洲服务器
  - `US`：美国服务器
  - `AS`：亚洲服务器

**示例**：
- 2025年12月24日 14:30:00 上传到亚洲服务器 → `/mnt/data/20251224_143000_AS/`
- 2025年12月24日 15:45:30 上传到欧洲服务器 → `/mnt/data/20251224_154530_EU/`

**注意事项**：
- 确保所选服务器的 `/mnt/data` 目录存在且有写入权限
- 如果目录不存在，系统会自动创建
- 每个上传任务会创建独立的目录，便于管理和追踪

