from models.lstm_weather_model import LSTMWeatherModel
from models.lstm_model import LSTMModel
from data.data_processor import DataProcessor
from backend.config.config import settings
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import numpy as np
import sys
import os
import sqlite3
import pandas as pd
from datetime import datetime

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))


# 初始化数据处理器
# processor = DataProcessor(seq_length=settings.SEQ_LENGTH, pred_length=settings.PRED_LENGTH)

# 初始化模型
model = None
weather_model = None

# 定义请求体模型


class WeatherData(BaseModel):
    # 时间序列数据，每个时间点包含6个气象特征
    # 格式：[[降雨量, 持续时间, 风速, 平均温度, 最低温度, 最高温度], ...]
    data: list = Field(
        ...,
        min_length=3,  # 至少需要3个时间点的数据来预测趋势
        description="历史气象数据序列"
    )

# 定义响应体模型


class RiskPredictionResponse(BaseModel):
    risk_level: str  # 无风险/中等风险/高风险
    risk_score: float  # 风险概率分数
    explanations: list  # 风险成因解释
    prediction_time: str  # 预测时间

# 定义未来天气预测响应模型


class WeatherPredictionResponse(BaseModel):
    future_weather: list  # 未来7天的气象数据预测
    risk_predictions: list  # 未来7天的洪涝风险预测
    message: str  # 响应消息


# 创建FastAPI应用
app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.BACKEND_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 模型加载依赖


def get_model():
    global model
    if model is None:
        # 加载训练好的模型
        try:
            # 获取输入形状
            # (seq_length, num_features)
            input_shape = (settings.SEQ_LENGTH, 6)
            model_instance = LSTMModel(input_shape=input_shape, num_classes=3)
            model_instance.load_model(settings.MODEL_PATH)
            model = model_instance
            return model
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"模型加载失败: {str(e)}")
    return model

# 气象预测模型加载依赖


def get_weather_model():
    global weather_model
    if weather_model is None:
        # 加载训练好的气象预测模型
        try:
            # 获取输入形状
            # (seq_length, num_features)
            input_shape = (settings.SEQ_LENGTH, 6)
            model_instance = LSTMWeatherModel(
                input_shape=input_shape, output_features=6)
            model_instance.load_model(settings.WEATHER_MODEL_PATH)

            # 尝试加载归一化器
            try:
                import joblib
                import os
                
                # 构建归一化器文件路径
                # 首先尝试从训练时保存的处理数据路径加载
                processed_data_path = "../../data/processed/processed_data_scaler.pkl"  # 相对于backend/app的路径
                
                # 如果上述路径不存在，尝试其他可能的路径
                if not os.path.exists(processed_data_path):
                    processed_data_path = os.path.join(os.path.dirname(os.path.dirname(
                        os.path.dirname(os.path.abspath(__file__)))), "data", "processed", "processed_data_scaler.pkl")
                
                if os.path.exists(processed_data_path):
                    scaler = joblib.load(processed_data_path)
                    model_instance.set_scaler(scaler)
                    print(f"归一化器已从 {processed_data_path} 加载到天气预测模型")
                else:
                    print(f"未找到归一化器文件: {processed_data_path}")
                    # 尝试从模型目录加载
                    scaler_path_from_model_dir = os.path.join(os.path.dirname(os.path.dirname(
                        os.path.dirname(os.path.abspath(__file__)))), "models", "processed_data_scaler.pkl")
                    if os.path.exists(scaler_path_from_model_dir):
                        scaler = joblib.load(scaler_path_from_model_dir)
                        model_instance.set_scaler(scaler)
                        print(f"归一化器已从 {scaler_path_from_model_dir} 加载到天气预测模型")
                    else:
                        print(f"在 {scaler_path_from_model_dir} 也未找到归一化器文件")
            except Exception as scaler_error:
                print(f"加载归一化器时出错: {str(scaler_error)}")
                import traceback
                traceback.print_exc()

            weather_model = model_instance
            return model_instance
        except Exception as e:
            # 如果加载失败，返回一个默认的气象预测模型实例（使用趋势预测）
            # (seq_length, num_features)
            input_shape = (settings.SEQ_LENGTH, 6)
            model_instance = LSTMWeatherModel(
                input_shape=input_shape, output_features=6)
            weather_model = model_instance
            return model_instance
    return weather_model

# 根路径


@app.get("/")
def read_root():
    return {
        "项目名称": settings.PROJECT_NAME,
        "版本": settings.VERSION,
        "API文档": "/docs"
    }

# 风险预测接口


@app.post(f"{settings.API_V1_STR}/risk_prediction", response_model=RiskPredictionResponse)
def predict_risk(weather_data: WeatherData, model: LSTMModel = Depends(get_model)):
    try:
        # 验证输入数据格式
        if len(weather_data.data) < 3:  # 至少需要3个时间点来计算趋势
            raise HTTPException(
                status_code=400,
                detail="输入数据必须包含至少3个时间点的气象数据"
            )

        # 转换为numpy数组
        X = np.array(weather_data.data)

        # 检查数据维度
        if X.shape[1] != 6:
            raise HTTPException(
                status_code=400,
                detail="每个时间点必须包含6个气象特征: [降雨量, 持续时间, 风速, 平均温度, 最低温度, 最高温度]"
            )

        # 扩展维度以匹配模型输入要求
        X_scaled = X.reshape(1, X.shape[0], X.shape[1])

        # 预测风险等级
        y_pred, y_pred_prob = model.predict(X_scaled)

        # 获取预测类别和概率
        pred_class = y_pred[0]
        pred_prob = y_pred_prob[0][pred_class]

        # 生成风险解释
        explanation = model.get_risk_explanation(X_scaled, pred_class)

        # 构建响应
        response = {
            "risk_level": explanation["risk_level"],
            "risk_score": float(pred_prob),
            "explanations": explanation["explanations"],
            "prediction_time": "未来1小时"
        }

        return response

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"预测失败: {str(e)}")

# 未来天气预测接口


@app.post(f"{settings.API_V1_STR}/weather_prediction", response_model=WeatherPredictionResponse)
def predict_weather(weather_data: WeatherData,
                    model: LSTMModel = Depends(get_model),
                    weather_model: LSTMWeatherModel = Depends(get_weather_model)):

    try:
        # 验证输入数据格式
        if len(weather_data.data) < 3:  # 至少需要3个时间点来计算趋势
            raise HTTPException(
                status_code=400,
                detail="输入数据必须包含至少3个时间点的气象数据"
            )

        # 转换为numpy数组
        X = np.array(weather_data.data)

        # 检查数据维度
        if X.shape[1] != 6:
            raise HTTPException(
                status_code=400,
                detail="每个时间点必须包含6个气象特征: [降雨量, 持续时间, 风速, 平均温度, 最低温度, 最高温度]"
            )

        # 确保输入序列长度与模型训练时使用的长度一致
        if X.shape[0] > settings.SEQ_LENGTH:
            # 如果输入长度大于模型期望长度，使用最近的 SEQ_LENGTH 个时间点
            X = X[-settings.SEQ_LENGTH:]
        elif X.shape[0] < settings.SEQ_LENGTH:
            # 如果输入长度小于模型期望长度，抛出错误
            raise HTTPException(
                status_code=400,
                detail=f"输入数据必须包含至少 {settings.SEQ_LENGTH} 个时间点的气象数据"
            )

        # 扩展维度以匹配模型输入要求
        X_reshaped = X.reshape(1, X.shape[0], X.shape[1])

        # 预测未来7天的天气数据
        future_weather = weather_model.predict(X_reshaped, days=7)

        # 预测未来7天的洪涝风险
        risk_predictions = []

        # 使用最后的序列作为基础，结合未来预测数据进行风险预测
        base_sequence = X.copy()  # 使用完整的输入序列作为基础

        for i in range(7):
            # 构建当前预测序列
            if i < len(future_weather):
                # 使用原始序列和未来预测数据构建新的序列
                # 为了预测第i天的风险，我们使用最近的SEQ_LENGTH个数据点
                extended_sequence = np.vstack([base_sequence, future_weather[:i+1]])
                
                # 从扩展序列中提取最近的SEQ_LENGTH个数据点
                if len(extended_sequence) >= settings.SEQ_LENGTH:
                    prediction_sequence = extended_sequence[-settings.SEQ_LENGTH:]
                else:
                    # 如果扩展序列不够长，填充重复数据
                    prediction_sequence = extended_sequence
                    while len(prediction_sequence) < settings.SEQ_LENGTH:
                        prediction_sequence = np.vstack([prediction_sequence, prediction_sequence[-1:]])
            else:
                # 防止索引越界
                extended_sequence = np.vstack([base_sequence, future_weather])
                if len(extended_sequence) >= settings.SEQ_LENGTH:
                    prediction_sequence = extended_sequence[-settings.SEQ_LENGTH:]
                else:
                    prediction_sequence = extended_sequence
                    while len(prediction_sequence) < settings.SEQ_LENGTH:
                        prediction_sequence = np.vstack([prediction_sequence, prediction_sequence[-1:]])

            # 预测风险
            prediction_sequence_reshaped = prediction_sequence.reshape(
                1, prediction_sequence.shape[0], prediction_sequence.shape[1])
            y_pred, y_pred_prob = model.predict(prediction_sequence_reshaped)

            # 获取预测类别和概率
            pred_class = y_pred[0]
            pred_prob = y_pred_prob[0][pred_class]

            # 生成风险解释
            explanation = model.get_risk_explanation(
                prediction_sequence_reshaped, pred_class)

            # 保存风险预测结果
            risk_predictions.append({
                "day": i+1,
                "risk_level": explanation["risk_level"],
                "risk_score": float(pred_prob),
                "explanations": explanation["explanations"]
            })

        # 构建响应
        response = {
            "future_weather": future_weather.tolist(),
            "risk_predictions": risk_predictions,
            "message": "未来7天气象数据和洪涝风险预测成功"
        }

        return response

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"预测失败: {str(e)}")

# 获取CMA官方数据接口


@app.get(f"{settings.API_V1_STR}/cma_data")
def get_cma_data(
    city_name: str = "北京",
    hours: int = 6,
    start_time: str = None,
    end_time: str = None,
):
    try:
        # 创建数据处理器
        processor = DataProcessor(
            seq_length=settings.SEQ_LENGTH, pred_length=settings.PRED_LENGTH)

        # 获取CMA数据
        raw_data = processor.load_cma_data(city_name, hours)

        # 构建序列数据（用于前端预测）
        sequences, _ = processor.get_cma_sequences(city_name, hours)

        # latest_sequence已经在前面计算好了

        # 将时间格式化为只包含年月日
        raw_data_df = raw_data.reset_index()
        if '时间' in raw_data_df.columns:
            raw_data_df['时间'] = raw_data_df['时间'].dt.strftime('%Y-%m-%d')
        elif 'time' in raw_data_df.columns:
            raw_data_df['time'] = raw_data_df['time'].dt.strftime('%Y-%m-%d')

        return {
            "city_name": city_name,
            "data_timestamp": datetime.now().strftime("%Y-%m-%d"),
            "raw_data": raw_data_df.to_dict(orient="records"),
            "normalized_data": latest_sequence,
            "message": "数据获取成功"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取CMA数据失败: {str(e)}")

# 获取NOAA官方数据接口（从本地数据库获取）


@app.get(f"{settings.API_V1_STR}/noaa_data")
def get_noaa_data(
    city_name: str = "北京",
    api_key: str = "sOUtGIPKTiBikrgafJueBImLZuVyvFGC",  # 保持兼容性，实际不再使用
    start_time: str = None,
    end_time: str = None,
):
    try:
        # 获取数据库文件路径
        db_path = os.path.join(os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__)))), "data", "weather_data.db")

        # 连接到SQLite数据库
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # 构建查询语句
        query = "SELECT * FROM noaa_weather_data"

        # 添加城市过滤条件（根据数据库结构，实际可能需要调整）
        # 注意：这里假设数据库中没有城市字段，所以暂时不添加城市过滤
        # 如果数据库中有城市字段，可以添加：WHERE city = ?

        # 添加时间过滤条件
        if start_time and end_time:
            query += " WHERE 时间 BETWEEN ? AND ?"
            cursor.execute(query, (start_time, end_time))
        else:
            cursor.execute(query)

        # 获取查询结果
        rows = cursor.fetchall()

        # 关闭数据库连接
        conn.close()

        # 如果没有数据，返回模拟数据
        if not rows:
            # 返回模拟数据
            latest_sequence = [
                [0.2, 0.1, 0.7, 0.3, 0.5, 0.4],
                [0.3, 0.2, 0.8, 0.2, 0.4, 0.3],
                [0.4, 0.3, 0.85, 0.15, 0.35, 0.25],
                [0.5, 0.4, 0.9, 0.1, 0.3, 0.2],
                [0.6, 0.5, 0.95, 0.05, 0.25, 0.15],
                [0.7, 0.6, 0.98, 0.02, 0.2, 0.1]
            ]
            return {
                "city_name": city_name,
                "data_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "raw_data": [],
                "normalized_data": latest_sequence,
                "message": "数据库中没有符合条件的数据，返回模拟数据"
            }

        # 将查询结果转换为DataFrame
        columns = ['id', '时间', '降雨量', '持续时间',
                   '风速', '最低温度', '最高温度', '平均温度', '洪涝风险标签']
        df = pd.DataFrame(rows, columns=columns)

        # 设置时间索引
        df['时间'] = pd.to_datetime(df['时间'])
        df.set_index('时间', inplace=True)
        df.sort_index(inplace=True)

        # 删除id列
        df.drop('id', axis=1, inplace=True)

        # 转换为 datetime 对象
        start = datetime.strptime(start_time, "%Y-%m-%d")
        end = datetime.strptime(end_time, "%Y-%m-%d")

        # 计算天数差
        day_len = (end - start).days

        print("111", len(df))
        # 创建数据处理器
        processor = DataProcessor(seq_length=len(
            df), pred_length=settings.PRED_LENGTH)

        # 预处理数据
        processed_df = processor.preprocess(df)

        # 构建序列数据（用于前端预测）
        # 使用整个时间范围的数据作为一个序列
        feature_cols = ['降雨量', '持续时间', '风速', '平均温度', '最低温度', '最高温度']
        latest_sequence = processed_df[feature_cols].values.tolist()
        sequences = [np.array(latest_sequence)]

        # 转换为前端需要的格式
        if len(sequences) > 0:
            # 获取最新的一个序列
            latest_sequence = sequences[-1].tolist()
        else:
            # 如果没有足够数据，返回模拟数据
            latest_sequence = [
                [0.2, 0.1, 0.7, 0.3, 0.5, 0.4],
                [0.3, 0.2, 0.8, 0.2, 0.4, 0.3],
                [0.4, 0.3, 0.85, 0.15, 0.35, 0.25],
                [0.5, 0.4, 0.9, 0.1, 0.3, 0.2],
                [0.6, 0.5, 0.95, 0.05, 0.25, 0.15],
                [0.7, 0.6, 0.98, 0.02, 0.2, 0.1]
            ]

        # 准备响应数据 - 将时间格式化为只包含年月日
        df.reset_index(inplace=True)
        df['时间'] = df['时间'].dt.strftime('%Y-%m-%d')
        raw_data_dict = df.to_dict(orient="records")
        print(raw_data_dict)
        return {
            "city_name": city_name,
            "data_timestamp": datetime.now().strftime("%Y-%m-%d"),
            "raw_data": raw_data_dict,
            "normalized_data": latest_sequence,
            "message": "数据获取成功（来自本地数据库）"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取NOAA数据失败: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
