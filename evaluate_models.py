import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import sys
import os
from pathlib import Path
# 在evaluate_models.py文件的开头添加以下代码
import matplotlib.pyplot as plt
import matplotlib as mpl

# 设置中文字体支持
mpl.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
mpl.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.lstm_weather_model import LSTMWeatherModel
from models.transformer_weather_model import TransformerWeatherModel
from data.data_processor import DataProcessor


def mean_absolute_percentage_error(y_true, y_pred):
    """计算MAPE（平均绝对百分比误差）"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    # 避免除以零
    non_zero_mask = np.abs(y_true) > 1e-8
    if np.sum(non_zero_mask) == 0:
        return 0.0
    return np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100


def calculate_metrics(y_true, y_pred):
    """计算各种评估指标"""
    metrics = {
        'mae': mean_absolute_error(y_true, y_pred),
        'mse': mean_squared_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'r2': r2_score(y_true, y_pred),
        'mape': mean_absolute_percentage_error(y_true, y_pred)
    }
    return metrics


def evaluate_models(num_samples=10, days=7):
    """评估并对比两个模型的性能"""
    print("=" * 80)
    print("开始评估LSTM和Transformer天气预测模型")
    print("=" * 80)
    
    # 1. 加载测试数据
    print("\n[1/6] 加载测试数据...")
    processor = DataProcessor()
    
    # 加载处理后的数据
    BASE_DIR = Path(__file__).resolve().parent
    processed_data_path = BASE_DIR / "data" / "processed" / "processed_data.npz"
    
    if not processed_data_path.exists():
        print(f"错误: 找不到处理后的数据文件 {processed_data_path}")
        print("请先运行数据预处理脚本生成数据文件")
        return None
    
    # 使用numpy直接加载数据
    data = np.load(str(processed_data_path))
    X_test = data['X_test']
    
    # 检查是否有y_weather_test
    if 'y_weather_test' in data.files:
        y_weather_test = data['y_weather_test']
    else:
        y_weather_test = None
    
    # 加载scaler
    scaler_path = str(processed_data_path).replace('.npz', '_scaler.pkl')
    import joblib
    try:
        scaler = joblib.load(scaler_path)
    except FileNotFoundError:
        scaler = None
    
    print(f"测试集形状: X_test={X_test.shape}")
    if y_weather_test is not None:
        print(f"真实天气标签形状: y_weather_test={y_weather_test.shape}")
        # 检查y_weather_test的形状是否符合预期
        expected_shape = (X_test.shape[0], days, 6)
        if y_weather_test.shape != expected_shape:
            print(f"警告: y_weather_test形状不符合预期")
            print(f"  预期形状: {expected_shape}")
            print(f"  实际形状: {y_weather_test.shape}")
            # 尝试调整形状
            if len(y_weather_test.shape) == 2:
                print("  尝试将2D数组重塑为3D数组...")
                y_weather_test = y_weather_test.reshape(-1, days, 6)
                print(f"  调整后形状: {y_weather_test.shape}")
    else:
        print("警告: 测试集中没有天气标签数据，将无法计算准确度指标")
        return None
    
    # 限制评估样本数量（避免计算时间过长）
    if num_samples > len(X_test):
        num_samples = len(X_test)
        print(f"警告: 请求的样本数({num_samples})超过测试集大小，使用全部测试集")
    
    X_test_subset = X_test[:num_samples]
    
    # 确保y_weather_test的形状正确
    if y_weather_test is not None:
        if len(y_weather_test.shape) == 2:
            print(f"y_weather_test是2D数组，形状为{y_weather_test.shape}")
            print(f"这意味着测试集只包含最后一天的数据，而不是{days}天的数据")
            print(f"将只比较最后一天的预测结果")
            # 保持2D形状，只比较最后一天
            y_weather_test_subset = y_weather_test[:num_samples]
        elif len(y_weather_test.shape) == 3:
            if len(y_weather_test) < num_samples:
                num_samples = len(y_weather_test)
                print(f"警告: 天气标签数量({len(y_weather_test)})少于请求的样本数，调整为{num_samples}")
            y_weather_test_subset = y_weather_test[:num_samples]
        else:
            print(f"警告: y_weather_test形状异常: {y_weather_test.shape}")
            y_weather_test_subset = None
    else:
        y_weather_test_subset = None
    
    print(f"评估样本数: {num_samples}")
    
    # 2. 加载LSTM模型
    print("\n[2/6] 加载LSTM模型...")
    lstm_model_path = BASE_DIR / "models" / "best_weather_model.keras"
    
    if not lstm_model_path.exists():
        print(f"警告: 找不到LSTM模型文件 {lstm_model_path}")
        print("将跳过LSTM模型的评估")
        lstm_model = None
    else:
        lstm_model = LSTMWeatherModel(
            input_shape=(X_test.shape[1], X_test.shape[2]),
            output_features=6,
            pred_days=days
        )
        lstm_model.load_model(str(lstm_model_path))
        lstm_model.set_scaler(scaler)
        print(f"LSTM模型已加载: {lstm_model_path}")
    
    # 3. 加载Transformer模型
    print("\n[3/6] 加载Transformer模型...")
    transformer_model_path = BASE_DIR / "models" / "best_transformer_weather_model.keras"
    
    if not transformer_model_path.exists():
        print(f"警告: 找不到Transformer模型文件 {transformer_model_path}")
        print("将跳过Transformer模型的评估")
        transformer_model = None
    else:
        transformer_model = TransformerWeatherModel(
            input_shape=(X_test.shape[1], X_test.shape[2]),
            output_features=6,
            pred_days=days
        )
        transformer_model.load_model(str(transformer_model_path))
        transformer_model.set_scaler(scaler)
        print(f"Transformer模型已加载: {transformer_model_path}")
    
    if lstm_model is None and transformer_model is None:
        print("错误: 没有可用的模型进行评估")
        return None
    
    # 4. 进行预测
    print("\n[4/6] 进行预测...")
    
    lstm_predictions = []
    transformer_predictions = []
    
    # 对每个样本进行预测
    for i in range(num_samples):
        sample = X_test_subset[i:i+1]
        
        # LSTM预测
        if lstm_model is not None:
            lstm_pred = lstm_model.predict(sample, days=days)
            lstm_predictions.append(lstm_pred)
        
        # Transformer预测
        if transformer_model is not None:
            transformer_pred = transformer_model.predict(sample, days=days)
            transformer_predictions.append(transformer_pred)
        
        # 显示进度
        if (i + 1) % max(1, num_samples // 10) == 0:
            print(f"  已完成 {i + 1}/{num_samples} 样本的预测")
    
    # 转换为numpy数组
    if lstm_model is not None:
        lstm_predictions = np.array(lstm_predictions)
        print(f"LSTM预测结果形状: {lstm_predictions.shape}")
    
    if transformer_model is not None:
        transformer_predictions = np.array(transformer_predictions)
        print(f"Transformer预测结果形状: {transformer_predictions.shape}")
    
    # 5. 计算评估指标
    print("\n[5/6] 计算评估指标...")
    
    metrics = {}
    feature_names = ['降雨量', '持续时间', '风速', '平均温度', '最低温度', '最高温度']
    
    # 计算整体指标
    if lstm_model is not None:
        # 根据y_weather_test的形状选择比较方式
        if len(y_weather_test_subset.shape) == 3 and y_weather_test_subset.shape[1] == 1:
            # 只比较最后一天的预测结果（测试集只包含1天）
            lstm_pred_last_day = lstm_predictions[:, -1, :]
            y_true_flat = y_weather_test_subset[:, 0, :]
            lstm_pred_flat = lstm_pred_last_day
        elif len(y_weather_test_subset.shape) == 2:
            # 只比较最后一天的预测结果（2D数组）
            lstm_pred_last_day = lstm_predictions[:, -1, :]
            y_true_flat = y_weather_test_subset
            lstm_pred_flat = lstm_pred_last_day
        else:
            # 比较所有天的预测结果
            y_true_flat = y_weather_test_subset.reshape(-1, y_weather_test_subset.shape[-1])
            lstm_pred_flat = lstm_predictions.reshape(-1, lstm_predictions.shape[-1])
        
        # 检查形状是否匹配
        if y_true_flat.shape != lstm_pred_flat.shape:
            print(f"警告: LSTM预测数据形状不匹配")
            print(f"  真实值形状: {y_true_flat.shape}")
            print(f"  预测值形状: {lstm_pred_flat.shape}")
            # 调整形状
            min_samples = min(y_true_flat.shape[0], lstm_pred_flat.shape[0])
            y_true_flat = y_true_flat[:min_samples]
            lstm_pred_flat = lstm_pred_flat[:min_samples]
        
        metrics['lstm'] = calculate_metrics(y_true_flat, lstm_pred_flat)
    
    if transformer_model is not None:
        # 根据y_weather_test的形状选择比较方式
        if len(y_weather_test_subset.shape) == 3 and y_weather_test_subset.shape[1] == 1:
            # 只比较最后一天的预测结果（测试集只包含1天）
            transformer_pred_last_day = transformer_predictions[:, -1, :]
            y_true_flat = y_weather_test_subset[:, 0, :]
            transformer_pred_flat = transformer_pred_last_day
        elif len(y_weather_test_subset.shape) == 2:
            # 只比较最后一天的预测结果（2D数组）
            transformer_pred_last_day = transformer_predictions[:, -1, :]
            y_true_flat = y_weather_test_subset
            transformer_pred_flat = transformer_pred_last_day
        else:
            # 比较所有天的预测结果
            y_true_flat = y_weather_test_subset.reshape(-1, y_weather_test_subset.shape[-1])
            transformer_pred_flat = transformer_predictions.reshape(-1, transformer_predictions.shape[-1])
        
        # 检查形状是否匹配
        if y_true_flat.shape != transformer_pred_flat.shape:
            print(f"警告: Transformer预测数据形状不匹配")
            print(f"  真实值形状: {y_true_flat.shape}")
            print(f"  预测值形状: {transformer_pred_flat.shape}")
            # 调整形状
            min_samples = min(y_true_flat.shape[0], transformer_pred_flat.shape[0])
            y_true_flat = y_true_flat[:min_samples]
            transformer_pred_flat = transformer_pred_flat[:min_samples]
        
        metrics['transformer'] = calculate_metrics(y_true_flat, transformer_pred_flat)
    
    # 计算每个特征的指标
    metrics['lstm_per_feature'] = {}
    metrics['transformer_per_feature'] = {}
    
    for feature_idx, feature_name in enumerate(feature_names):
        if lstm_model is not None:
            if len(y_weather_test_subset.shape) == 3 and y_weather_test_subset.shape[1] == 1:
                # 只比较最后一天的预测结果（测试集只包含1天）
                y_true_feature = y_weather_test_subset[:, 0, feature_idx]
                lstm_pred_feature = lstm_predictions[:, -1, feature_idx]
            elif len(y_weather_test_subset.shape) == 2:
                # 只比较最后一天的预测结果（2D数组）
                y_true_feature = y_weather_test_subset[:, feature_idx]
                lstm_pred_feature = lstm_predictions[:, -1, feature_idx]
            else:
                # 比较所有天的预测结果
                y_true_feature = y_weather_test_subset[:, :, feature_idx].flatten()
                lstm_pred_feature = lstm_predictions[:, :, feature_idx].flatten()
            
            # 检查形状是否匹配
            if y_true_feature.shape != lstm_pred_feature.shape:
                print(f"警告: {feature_name} LSTM特征数据形状不匹配")
                print(f"  真实值形状: {y_true_feature.shape}")
                print(f"  预测值形状: {lstm_pred_feature.shape}")
                # 调整形状
                min_samples = min(y_true_feature.shape[0], lstm_pred_feature.shape[0])
                y_true_feature = y_true_feature[:min_samples]
                lstm_pred_feature = lstm_pred_feature[:min_samples]
            
            metrics['lstm_per_feature'][feature_name] = calculate_metrics(y_true_feature, lstm_pred_feature)
        
        if transformer_model is not None:
            if len(y_weather_test_subset.shape) == 3 and y_weather_test_subset.shape[1] == 1:
                # 只比较最后一天的预测结果（测试集只包含1天）
                y_true_feature = y_weather_test_subset[:, 0, feature_idx]
                transformer_pred_feature = transformer_predictions[:, -1, feature_idx]
            elif len(y_weather_test_subset.shape) == 2:
                # 只比较最后一天的预测结果（2D数组）
                y_true_feature = y_weather_test_subset[:, feature_idx]
                transformer_pred_feature = transformer_predictions[:, -1, feature_idx]
            else:
                # 比较所有天的预测结果
                y_true_feature = y_weather_test_subset[:, :, feature_idx].flatten()
                transformer_pred_feature = transformer_predictions[:, :, feature_idx].flatten()
            
            # 检查形状是否匹配
            if y_true_feature.shape != transformer_pred_feature.shape:
                print(f"警告: {feature_name} Transformer特征数据形状不匹配")
                print(f"  真实值形状: {y_true_feature.shape}")
                print(f"  预测值形状: {transformer_pred_feature.shape}")
                # 调整形状
                min_samples = min(y_true_feature.shape[0], transformer_pred_feature.shape[0])
                y_true_feature = y_true_feature[:min_samples]
                transformer_pred_feature = transformer_pred_feature[:min_samples]
            
            metrics['transformer_per_feature'][feature_name] = calculate_metrics(y_true_feature, transformer_pred_feature)
    
    # 6. 打印评估结果
    print("\n[6/6] 生成评估报告...")
    print("\n" + "=" * 80)
    print("模型性能对比（整体指标）")
    print("=" * 80)
    print(f"{'指标':<15}", end="")
    if lstm_model is not None:
        print(f"{'LSTM':<20}", end="")
    if transformer_model is not None:
        print(f"{'Transformer':<20}", end="")
    print()
    print("-" * 80)
    
    for metric in ['mae', 'mse', 'rmse', 'r2', 'mape']:
        metric_name = {
            'mae': 'MAE (平均绝对误差)',
            'mse': 'MSE (均方误差)',
            'rmse': 'RMSE (均方根误差)',
            'r2': 'R² (决定系数)',
            'mape': 'MAPE (%) (平均绝对百分比误差)'
        }[metric]
        
        print(f"{metric_name:<15}", end="")
        
        if lstm_model is not None:
            print(f"{metrics['lstm'][metric]:<20.4f}", end="")
        
        if transformer_model is not None:
            print(f"{metrics['transformer'][metric]:<20.4f}", end="")
        
        print()
    
    print("=" * 80)
    
    # 打印每个特征的指标
    print("\n各特征预测性能对比:")
    print("-" * 80)
    
    for feature_name in feature_names:
        print(f"\n{feature_name}:")
        print(f"  {'指标':<15}", end="")
        if lstm_model is not None:
            print(f"{'LSTM':<20}", end="")
        if transformer_model is not None:
            print(f"{'Transformer':<20}", end="")
        print()
        
        for metric in ['mae', 'rmse', 'r2', 'mape']:
            print(f"  {metric.upper():<15}", end="")
            
            if lstm_model is not None:
                print(f"{metrics['lstm_per_feature'][feature_name][metric]:<20.4f}", end="")
            
            if transformer_model is not None:
                print(f"{metrics['transformer_per_feature'][feature_name][metric]:<20.4f}", end="")
            
            print()
    
    # 7. 生成可视化对比图
    print("\n生成可视化对比图...")
    
    # 调试信息：打印y_weather_test_subset的形状
    print(f"调试: y_weather_test_subset.shape = {y_weather_test_subset.shape}")
    print(f"调试: len(y_weather_test_subset.shape) = {len(y_weather_test_subset.shape)}")
    
    # 选择前3个样本进行可视化
    num_vis_samples = min(3, num_samples)
    
    fig, axes = plt.subplots(num_vis_samples, 6, figsize=(18, 4 * num_vis_samples))
    if num_vis_samples == 1:
        axes = axes.reshape(1, -1)
    
    for sample_idx in range(num_vis_samples):
        for feature_idx, feature_name in enumerate(feature_names):
            ax = axes[sample_idx, feature_idx]
            
            # 根据y_weather_test的形状选择显示方式
            if len(y_weather_test_subset.shape) == 3 and y_weather_test_subset.shape[1] == 1:
                # 只显示最后一天的真实值（测试集只包含1天）
                y_true_single = y_weather_test_subset[sample_idx, 0, feature_idx]
                ax.scatter([days - 1], [y_true_single], label='真实值', marker='o', s=100, color='red', zorder=5)
            elif len(y_weather_test_subset.shape) == 2:
                # 只显示最后一天的真实值（2D数组）
                y_true_single = y_weather_test_subset[sample_idx, feature_idx]
                ax.scatter([days - 1], [y_true_single], label='真实值', marker='o', s=100, color='red', zorder=5)
            else:
                # 显示所有天的真实值
                ax.plot(range(days), y_weather_test_subset[sample_idx, :, feature_idx], 
                        label='真实值', marker='o', linewidth=2, markersize=6)
            
            # 绘制LSTM预测
            if lstm_model is not None:
                ax.plot(range(days), lstm_predictions[sample_idx, :, feature_idx], 
                        label='LSTM预测', marker='s', linewidth=2, markersize=6, linestyle='--')
            
            # 绘制Transformer预测
            if transformer_model is not None:
                ax.plot(range(days), transformer_predictions[sample_idx, :, feature_idx], 
                        label='Transformer预测', marker='^', linewidth=2, markersize=6, linestyle='-.')
            
            ax.set_title(f'{feature_name} (样本{sample_idx + 1})', fontsize=10, fontweight='bold')
            ax.set_xlabel('预测天数', fontsize=9)
            ax.set_ylabel(feature_name, fontsize=9)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = BASE_DIR / "model_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"预测对比图已保存为: {output_path}")
    
    # 8. 生成柱状图对比
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for feature_idx, feature_name in enumerate(feature_names):
        ax = axes[feature_idx]
        
        # 准备数据
        x = np.arange(len(['MAE', 'RMSE', 'MAPE']))
        width = 0.35
        
        if lstm_model is not None:
            lstm_values = [
                metrics['lstm_per_feature'][feature_name]['mae'],
                metrics['lstm_per_feature'][feature_name]['rmse'],
                metrics['lstm_per_feature'][feature_name]['mape']
            ]
            ax.bar(x - width/2, lstm_values, width, label='LSTM', alpha=0.8)
        
        if transformer_model is not None:
            transformer_values = [
                metrics['transformer_per_feature'][feature_name]['mae'],
                metrics['transformer_per_feature'][feature_name]['rmse'],
                metrics['transformer_per_feature'][feature_name]['mape']
            ]
            ax.bar(x + width/2, transformer_values, width, label='Transformer', alpha=0.8)
        
        ax.set_title(f'{feature_name} - 误差对比', fontsize=11, fontweight='bold')
        ax.set_ylabel('误差值', fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(['MAE', 'RMSE', 'MAPE'])
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_path = BASE_DIR / "model_metrics_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"指标对比图已保存为: {output_path}")
    
    # 9. 保存评估报告
    report_path = BASE_DIR / "model_evaluation_report.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("LSTM和Transformer天气预测模型评估报告\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"评估时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"评估样本数: {num_samples}\n")
        f.write(f"预测天数: {days}\n\n")
        
        f.write("=" * 80 + "\n")
        f.write("整体性能对比\n")
        f.write("=" * 80 + "\n")
        f.write(f"{'指标':<20}")
        if lstm_model is not None:
            f.write(f"{'LSTM':<20}")
        if transformer_model is not None:
            f.write(f"{'Transformer':<20}")
        f.write("\n")
        f.write("-" * 80 + "\n")
        
        for metric in ['mae', 'mse', 'rmse', 'r2', 'mape']:
            metric_name = {
                'mae': 'MAE (平均绝对误差)',
                'mse': 'MSE (均方误差)',
                'rmse': 'RMSE (均方根误差)',
                'r2': 'R² (决定系数)',
                'mape': 'MAPE (%) (平均绝对百分比误差)'
            }[metric]
            
            f.write(f"{metric_name:<20}")
            
            if lstm_model is not None:
                f.write(f"{metrics['lstm'][metric]:<20.4f}")
            
            if transformer_model is not None:
                f.write(f"{metrics['transformer'][metric]:<20.4f}")
            
            f.write("\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("各特征预测性能对比\n")
        f.write("=" * 80 + "\n\n")
        
        for feature_name in feature_names:
            f.write(f"{feature_name}:\n")
            f.write(f"  {'指标':<15}")
            if lstm_model is not None:
                f.write(f"{'LSTM':<20}")
            if transformer_model is not None:
                f.write(f"{'Transformer':<20}")
            f.write("\n")
            
            for metric in ['mae', 'rmse', 'r2', 'mape']:
                f.write(f"  {metric.upper():<15}")
                
                if lstm_model is not None:
                    f.write(f"{metrics['lstm_per_feature'][feature_name][metric]:<20.4f}")
                
                if transformer_model is not None:
                    f.write(f"{metrics['transformer_per_feature'][feature_name][metric]:<20.4f}")
                
                f.write("\n")
            f.write("\n")
        
        f.write("=" * 80 + "\n")
        f.write("结论与建议\n")
        f.write("=" * 80 + "\n")
        
        # 比较模型性能
        if lstm_model is not None and transformer_model is not None:
            lstm_mae = metrics['lstm']['mae']
            transformer_mae = metrics['transformer']['mae']
            
            if lstm_mae < transformer_mae:
                f.write(f"基于MAE指标，LSTM模型表现更好（LSTM: {lstm_mae:.4f} vs Transformer: {transformer_mae:.4f}）\n")
                f.write("建议：优先使用LSTM模型进行天气预测。\n")
            else:
                f.write(f"基于MAE指标，Transformer模型表现更好（Transformer: {transformer_mae:.4f} vs LSTM: {lstm_mae:.4f}）\n")
                f.write("建议：优先使用Transformer模型进行天气预测。\n")
            
            lstm_r2 = metrics['lstm']['r2']
            transformer_r2 = metrics['transformer']['r2']
            
            f.write(f"\n基于R²指标：\n")
            f.write(f"- LSTM模型解释了 {lstm_r2*100:.2f}% 的数据变异\n")
            f.write(f"- Transformer模型解释了 {transformer_r2*100:.2f}% 的数据变异\n")
        elif lstm_model is not None:
            f.write("仅LSTM模型可用，建议使用LSTM模型进行天气预测。\n")
        elif transformer_model is not None:
            f.write("仅Transformer模型可用，建议使用Transformer模型进行天气预测。\n")
    
    print(f"评估报告已保存为: {report_path}")
    
    print("\n" + "=" * 80)
    print("评估完成！")
    print("=" * 80)
    
    return metrics


if __name__ == "__main__":
    # 评估模型（使用前10个测试样本）
    metrics = evaluate_models(num_samples=10, days=7)
    
    if metrics is not None:
        print("\n评估成功完成！请查看生成的图表和报告文件。")
    else:
        print("\n评估失败，请检查错误信息。")
