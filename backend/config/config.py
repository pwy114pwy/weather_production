# 后端服务配置

class Settings:
    # 服务基本信息
    PROJECT_NAME = "城市暴雨洪涝风险预测系统"
    VERSION = "1.0.0"
    API_V1_STR = "/api/v1"
    
    # 模型配置
    MODEL_PATH = "d:\\nature-disaster-prediction\\models\\final_model.keras"
    WEATHER_MODEL_PATH = "d:\\nature-disaster-prediction\\models\\final_weather_model.keras"
    
    # 数据处理配置
    SEQ_LENGTH = 6  # 输入序列长度（时间点数量）
    PRED_LENGTH = 7  # 预测时长（天）
    
    # CORS配置
    BACKEND_CORS_ORIGINS = ["*"]  # 生产环境应限制为特定域名

# 创建配置实例
settings = Settings()