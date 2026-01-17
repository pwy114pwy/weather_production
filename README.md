# 城市暴雨洪涝风险预测系统

本项目是一个基于深度学习的城市暴雨洪涝风险预测系统，通过分析历史气象数据（降雨量、温度、风速等），利用LSTM神经网络模型预测未来洪涝风险等级，并提供风险解释和可视化展示。

## 项目特性

- 基于LSTM神经网络的洪涝风险预测模型
- 支持CMA（中国气象局）和NOAA（美国国家海洋和大气管理局）数据源
- 多维度气象数据分析（降雨量、持续时间、风速、温度等）
- 实时风险等级评估（无风险/中等风险/高风险）
- 可视化数据展示和风险趋势预测
- 风险成因解释功能
- 未来天气预测功能

## 技术栈

- **后端**：FastAPI, TensorFlow/Keras, SQLite
- **前端**：Vue.js 3, Pinia, ECharts
- **模型**：LSTM神经网络（Seq2Seq架构）
- **构建工具**：Vite

## 项目结构

```
nature-disaster-prediction/
├── backend/                 # 后端服务
│   ├── app/                # API服务入口
│   │   └── main.py         # 主应用文件
│   └── config/             # 配置文件
│       └── config.py       # 服务配置
├── data/                   # 数据处理模块
│   ├── cma_data_fetcher.py # CMA数据获取器
│   ├── noaa_data_fetcher.py # NOAA数据获取器
│   ├── data_processor.py   # 数据处理器
│   ├── import_to_db.py     # 数据导入数据库
│   ├── raw/                # 原始数据目录
│   │   └── noaa_weather_data.csv # 原始气象数据
│   └── processed/          # 预处理后数据目录
│       └── processed_data.npz # 预处理后数据文件
├── frontend/               # 前端应用
│   ├── src/                # 源码
│   │   ├── components/     # Vue组件
│   │   │   ├── RiskPanel.vue      # 风险面板组件
│   │   │   ├── TrendChart.vue     # 趋势图表组件
│   │   │   └── PredictionPanel.vue # 预测面板组件
│   │   ├── stores/         # Pinia状态管理
│   │   │   └── weather.js  # 天气数据状态
│   │   ├── App.vue         # 主应用组件
│   │   ├── main.js         # 应用入口
│   │   └── style.css       # 样式文件
│   ├── public/             # 静态资源
│   ├── package.json        # 依赖配置
│   └── vite.config.js      # Vite配置
├── models/                 # 机器学习模型
│   ├── lstm_model.py       # 洪涝风险预测模型
│   ├── lstm_weather_model.py # 天气预测模型
│   ├── best_model.keras    # 最佳风险预测模型
│   ├── best_weather_model.keras # 最佳天气预测模型
│   ├── final_model.keras   # 最终风险预测模型
│   └── final_weather_model.keras # 最终天气预测模型
├── train_with_processed_data.py # 模型训练脚本
└── README.md               # 项目说明文档
```

## 环境准备

### 系统要求
- Python 3.8+
- Node.js 16+
- npm 或 yarn

### Python依赖安装

```bash
pip install fastapi uvicorn tensorflow pandas scikit-learn numpy sqlite3 python-multipart
```

### 前端依赖安装

```bash
cd frontend
npm install
```

## 模型训练

### 1. 准备数据
确保 `data/raw/noaa_weather_data.csv` 文件存在，该文件应包含以下列：
- 降雨量
- 持续时间
- 风速
- 平均温度
- 最低温度
- 最高温度
- 洪涝风险标签

### 2. 训练模型
运行以下命令训练两个模型：

```bash
python train_with_processed_data.py
```

此脚本会：
- 加载并预处理气象数据
- 构建洪涝风险预测模型（[LSTMModel](file://d:\nature-disaster-prediction\models\lstm_model.py#L14-L336)）
- 构建天气预测模型（[LSTMWeatherModel](file://d:\nature-disaster-prediction\models\lstm_weather_model.py#L16-L168)）
- 训练两个模型
- 评估模型性能
- 保存训练好的模型到 `models/` 目录

#### 洪涝风险预测模型 ([LSTMModel](file://d:\nature-disaster-prediction\models\lstm_model.py#L14-L336))
- 用途：预测洪涝风险等级（无风险/中等风险/高风险）
- 架构：LSTM + Dense层
- 输入：时间序列气象数据
- 输出：风险等级及概率

#### 天气预测模型 ([LSTMWeatherModel](file://d:\nature-disaster-prediction\models\lstm_weather_model.py#L16-L168))
- 用途：预测未来天气数据
- 架构：Seq2Seq模型，带注意力机制
- 输入：历史气象数据序列
- 输出：未来多天的气象数据预测

## 项目启动

### 1. 启动后端服务

```bash
cd backend/app
uvicorn main:app --reload --port 8000
```

后端服务将在 `http://localhost:8000` 上运行，API文档可在 `http://localhost:8000/docs` 查看。

### 2. 启动前端应用

打开新的终端窗口：

```bash
cd frontend
npm run dev
```

前端应用将在 `http://localhost:5173` 上运行。

### 3. 访问应用

在浏览器中访问 `http://localhost:5173` 即可使用城市暴雨洪涝风险预测系统。

## API接口说明

### 洪涝风险预测
- **POST** `/api/v1/risk_prediction`
- 输入：历史气象数据序列
- 输出：风险等级、风险分数、风险解释、预测时间

### 天气预测
- **POST** `/api/v1/weather_prediction`
- 输入：历史气象数据序列
- 输出：未来7天气象数据、未来7天风险预测

### CMA数据获取
- **GET** `/api/v1/cma_data`
- 参数：city_name（城市名称，默认北京）, hours（历史小时数，默认6小时）

### NOAA数据获取
- **GET** `/api/v1/noaa_data`
- 参数：city_name（城市名称，默认北京）, api_key（API密钥，默认测试密钥）

## 模型架构详解

### 洪涝风险预测模型
- **输入**：时间序列气象数据（降雨量、持续时间、风速、温度等）
- **LSTM层1**：64单元，返回序列
- **Dropout层**：0.2，防止过拟合
- **LSTM层2**：32单元，不返回序列
- **Dropout层**：0.2
- **全连接层**：16单元，ReLU激活
- **输出层**：3单元，Softmax激活（对应3个风险等级）
- **损失函数**：稀疏分类交叉熵
- **优化器**：Adam

### 天气预测模型（Seq2Seq架构）
- **编码器**：3层LSTM（各64单元），提取历史数据特征
- **解码器**：LSTM层，生成未来预测序列
- **注意力机制**：关注重要时间步信息
- **输出**：未来多天的气象特征预测
- **损失函数**：均方误差
- **优化器**：Adam

## 数据处理流程

1. **数据获取**：从CMA或NOAA获取原始气象数据
2. **数据预处理**：标准化特征，处理缺失值
3. **序列构建**：将时间序列数据分割成固定长度的输入窗口
4. **模型训练**：使用LSTM模型学习气象模式
5. **风险预测**：基于模型输出评估洪涝风险
6. **结果展示**：在前端可视化展示预测结果

## 配置文件

- `backend/config/config.py`：服务配置，包括模型路径、序列长度等参数
- `frontend/src/stores/weather.js`：前端状态管理

## 注意事项

1. 确保模型文件 `final_model.keras` 和 `final_weather_model.keras` 存在于 `models/` 目录下
2. 如果模型文件不存在，请先运行模型训练脚本
3. 在生产环境中，请限制CORS允许的域名
4. 确保气象数据质量，数据准确性直接影响预测效果
5. 模型使用归一化数据进行预测，输入数据需要相应预处理

## 扩展建议

- 集成更多气象数据源
- 添加地理信息系统(GIS)数据
- 实现模型在线更新功能
- 添加多城市对比分析功能
- 集成预警通知机制
- 优化模型架构，提高预测精度