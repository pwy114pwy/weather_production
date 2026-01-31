import os
from pathlib import Path

# 后端服务配置


class Settings:
    # 服务基本信息
    PROJECT_NAME = "城市暴雨洪涝风险预测系统"
    VERSION = "1.0.0"
    API_V1_STR = "/api/v1"

    # 项目根目录
    BASE_DIR = Path(__file__).resolve().parent.parent.parent

    # 模型配置
    MODEL_PATH = str(BASE_DIR / "models" / "final_model.keras")
    WEATHER_MODEL_PATH = str(BASE_DIR / "models" / "final_weather_model.keras")
    
    # Transformer模型配置
    TRANSFORMER_MODEL_PATH = str(BASE_DIR / "models" / "final_transformer_model.keras")
    TRANSFORMER_WEATHER_MODEL_PATH = str(BASE_DIR / "models" / "final_transformer_weather_model.keras")
    
    # 默认模型架构选择
    DEFAULT_ARCHITECTURE = "lstm"  # 可选值: "lstm", "transformer"

    # 数据处理配置
    SEQ_LENGTH = 6  # 输入序列长度（时间点数量）
    PRED_LENGTH = 7  # 预测时长（天）

    # CORS配置
    BACKEND_CORS_ORIGINS = [
        "http://localhost:5173",  # Vite开发服务器
        "http://127.0.0.1:5173",
        "http://localhost:8000",  # 后端服务
        "http://127.0.0.1:8000",
        "http://localhost:3000",  # 可能的前端端口
        "http://127.0.0.1:3000"
    ]  # 生产环境应限制为特定域名


# 创建配置实例
settings = Settings()
