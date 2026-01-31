import os
import sys

import numpy as np

from models.lstm_model import LSTMModel

# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.dirname(__file__)))


def train_model_with_processed_data(data_file_path):
    """使用已预处理的数据文件训练模型

    Args:
        data_file_path: 预处理后的数据文件路径 (.npz格式)
    """
    print(f"加载预处理数据文件: {data_file_path}")

    try:
        # 从数据处理模块加载数据和归一化器
        from data.data_processor import DataProcessor
        processor = DataProcessor()
        data = processor.load_processed_data(data_file_path)

        # 提取数据
        X_train = data['X_train']
        y_train = data['y_train']
        X_val = data['X_val']
        y_val = data['y_val']
        X_test = data['X_test']
        y_test = data['y_test']
        scaler = data['scaler']  # 获取归一化器

        # 检查是否有weather_targets数据（用于Seq2Seq模型训练）
        has_weather_targets = 'y_weather_train' in data and 'y_weather_val' in data

        if has_weather_targets:
            y_weather_train = data['y_weather_train']
            y_weather_val = data['y_weather_val']
            y_weather_test = data.get('y_weather_test')

            print("数据加载成功！")
            print(
                f"训练集形状: X={X_train.shape}, y={y_train.shape}, y_weather={y_weather_train.shape}")
            print(
                f"验证集形状: X={X_val.shape}, y={y_val.shape}, y_weather={y_weather_val.shape}")
        else:
            print("数据加载成功！")
            print(f"训练集形状: X={X_train.shape}, y={y_train.shape}")
            print(f"验证集形状: X={X_val.shape}, y={y_val.shape}")
            print(f"测试集形状: X={X_test.shape}, y={y_test.shape}")

        # 检查数据维度是否正确
        if len(X_train.shape) != 3:
            raise ValueError("训练数据格式错误，应为3维数组: (样本数, 序列长度, 特征数)")

        # 初始化模型
        input_shape = (X_train.shape[1], X_train.shape[2])  # (序列长度, 特征数)
        num_classes = len(np.unique(np.concatenate(
            [y_train, y_val, y_test])))  # 自动检测类别数量

        print(f"输入形状: {input_shape}, 类别数量: {num_classes}")

        model = LSTMModel(input_shape=input_shape, num_classes=num_classes)

        # 训练模型
        print("\n开始训练模型...")
        history = model.train(X_train, y_train, X_val,
                              y_val, epochs=50, batch_size=32)

        # 如果有weather_targets数据，同时训练天气预测模型
        if has_weather_targets:
            print("\n开始训练天气预测模型...")
            from models.lstm_weather_model import LSTMWeatherModel

            # 天气预测模型输入形状与风险预测模型相同
            weather_model = LSTMWeatherModel(
                input_shape=input_shape, output_features=X_train.shape[2])

            # 设置归一化器到模型（如果存在）
            if scaler is not None:
                weather_model.set_scaler(scaler)

            # 训练天气预测模型
            weather_history = weather_model.train(
                X_train, y_weather_train,
                X_val, y_weather_val,
                epochs=50, batch_size=32
            )

            # 保存天气预测模型
            base_dir = os.path.abspath(os.path.dirname(__file__))
            model_dir = os.path.join(base_dir, "models")
            os.makedirs(model_dir, exist_ok=True)
            weather_model_path = os.path.join(
                model_dir, "final_weather_model.keras")
            weather_model.save_model(weather_model_path)
            print(f"天气预测模型已保存到: {weather_model_path}")

        # 评估模型
        print("\n开始评估模型...")
        evaluation_results = model.evaluate(X_test, y_test)

        # 保存模型（使用os.path.join构建安全的路径）
        base_dir = os.path.abspath(os.path.dirname(__file__))
        model_dir = os.path.join(base_dir, "models")
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, "final_model.keras")
        model.save_model(model_path)
        print(f"\n模型已保存到: {model_path}")

        # 测试预测功能
        print("\n测试预测功能...")
        if len(X_test) > 0:
            sample = X_test[0:1]  # 取一个样本
            pred_class, pred_prob = model.predict(sample)
            explanation = model.get_risk_explanation(sample, pred_class[0])
            print(f"预测类别: {pred_class[0]}, 预测概率: {pred_prob[0]}")
            print(f"风险解释: {explanation}")

        return True, "模型训练完成！"

    except Exception as e:
        print(f"训练过程中发生错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return False, str(e)


if __name__ == "__main__":
    # 默认数据文件路径（使用原始字符串避免转义问题）
    default_data_path = r"D:\nature-disaster-prediction\data\processed\processed_data.npz"

    # 可以通过命令行参数指定数据文件路径
    data_path = default_data_path
    if len(sys.argv) > 1:
        data_path = sys.argv[1]

    print("使用已预处理数据训练LSTM模型")
    print(f"数据文件: {data_path}")
    print("=" * 60)

    success, message = train_model_with_processed_data(data_path)

    if success:
        print("\n✅ 训练成功完成！")
        print(message)
    else:
        print(f"\n❌ 训练失败: {message}")
