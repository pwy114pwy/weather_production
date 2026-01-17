import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# 动态导入，支持直接运行和作为模块导入
try:
    from .cma_data_fetcher import CMAMeteorologicalDataFetcher
    from .noaa_data_fetcher import NOAAMeteorologicalDataFetcher
except ImportError:
    # 直接运行时使用绝对导入
    import sys
    import os
    sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
    from data.cma_data_fetcher import CMAMeteorologicalDataFetcher
    from data.noaa_data_fetcher import NOAAMeteorologicalDataFetcher

class DataProcessor:
    def __init__(self, seq_length=7, pred_length=7):
        self.seq_length = seq_length  # 输入序列长度（天）
        self.pred_length = pred_length  # 预测未来时长（天）
        self.scaler = MinMaxScaler()
        
    def load_data(self, file_path):
        """加载原始气象数据"""
        df = pd.read_csv(file_path, parse_dates=['时间'], index_col='时间')
        return df
    
    def preprocess(self, df):
        """数据预处理：归一化特征"""
        # 选择气象特征列（使用NOAA返回的5个特征）
        feature_cols = ['降雨量', '持续时间', '风速', '平均温度', '最低温度', '最高温度']
        target_col = '洪涝风险标签'
        
        # 分离特征和标签
        features = df[feature_cols]
        targets = df[target_col]
        
        # 归一化特征
        scaled_features = self.scaler.fit_transform(features)
        
        # 重建DataFrame
        scaled_df = pd.DataFrame(scaled_features, columns=feature_cols, index=df.index)
        scaled_df[target_col] = targets
        
        return scaled_df
    
    def create_sequences(self, df):
        """按时间窗口构建序列数据"""
        feature_cols = ['降雨量', '持续时间', '风速', '平均温度', '最低温度', '最高温度']
        target_col = '洪涝风险标签'
        
        sequences = []
        targets = []
        weather_targets = []  # 用于存储未来多步的气象特征
        
        # 确保数据按时间排序
        df = df.sort_index()
        
        # 构建序列
        for i in range(len(df) - self.seq_length - self.pred_length + 1):
            # 输入序列：过去seq_length小时的数据
            seq_features = df[feature_cols].iloc[i:i+self.seq_length].values
            # 目标：未来pred_length小时后的风险标签
            seq_target = df[target_col].iloc[i+self.seq_length+self.pred_length-1]
            
            # 未来多步的气象特征（用于Seq2Seq模型训练）
            future_weather = df[feature_cols].iloc[i+self.seq_length:i+self.seq_length+self.pred_length].values
            
            sequences.append(seq_features)
            targets.append(seq_target)
            weather_targets.append(future_weather)
        
        # 转换为numpy数组
        sequences = np.array(sequences)
        targets = np.array(targets)
        weather_targets = np.array(weather_targets)
        
        return sequences, targets, weather_targets
    
    def split_data(self, sequences, targets, weather_targets=None, train_ratio=0.7, val_ratio=0.15):
        """按时间划分数据集，避免数据泄漏"""
        total_samples = len(sequences)
        train_size = int(total_samples * train_ratio)
        val_size = int(total_samples * val_ratio)
        
        # 按时间顺序划分：先训练，再验证，最后测试
        X_train = sequences[:train_size]
        y_train = targets[:train_size]
        
        X_val = sequences[train_size:train_size+val_size]
        y_val = targets[train_size:train_size+val_size]
        
        X_test = sequences[train_size+val_size:]
        y_test = targets[train_size+val_size:]
        
        # 如果有weather_targets，也进行划分
        if weather_targets is not None:
            y_weather_train = weather_targets[:train_size]
            y_weather_val = weather_targets[train_size:train_size+val_size]
            y_weather_test = weather_targets[train_size+val_size:]
            
            return X_train, y_train, y_weather_train, X_val, y_val, y_weather_val, X_test, y_test, y_weather_test
        
        return X_train, y_train, X_val, y_val, X_test, y_test
    
    def save_processed_data(self, data_dict, save_path):
        """保存处理后的数据"""
        np.savez(save_path, **data_dict)
    
    def load_processed_data(self, load_path):
        """加载处理后的数据"""
        data = np.load(load_path)
        return {
            'X_train': data['X_train'],
            'y_train': data['y_train'],
            'X_val': data['X_val'],
            'y_val': data['y_val'],
            'X_test': data['X_test'],
            'y_test': data['y_test']
        }

    def load_cma_data(self, city_name, hours=24):
        """
        从国家气象中心获取数据并预处理
        
        Args:
            city_name: 城市名称
            api_key: CMA API密钥
            hours: 要获取的历史小时数
            
        Returns:
            预处理后的数据(DataFrame)
        """
        print(city_name)
        print(hours)
        # 创建CMA数据获取器
        fetcher = CMAMeteorologicalDataFetcher()
        
        # 获取原始数据
        raw_df = fetcher.fetch_weather_data(city_name, hours)
        
        # 预处理数据
        processed_df = self.preprocess(raw_df)
        
        return processed_df
    
    def get_cma_sequences(self, city_name, hours=24):
        """
        从国家气象中心获取数据并构建序列
        
        Args:
            city_name: 城市名称
            api_key: CMA API密钥
            hours: 要获取的历史小时数
            
        Returns:
            序列数据和目标标签
        """
        # 获取预处理后的数据
        processed_df = self.load_cma_data(city_name, hours)
        
        # 构建序列
        sequences, targets = self.create_sequences(processed_df)
        
        return sequences, targets
    
    def load_noaa_data(self, city_name, api_key, start_time=None, end_time=None):
        """
        从NOAA获取数据并预处理
        
        Args:
            city_name: 城市名称
            api_key: NOAA API密钥
            start_time: 开始时间
            end_time: 结束时间
            
        Returns:
            预处理后的数据(DataFrame)
        """
        # 创建NOAA数据获取器
        fetcher = NOAAMeteorologicalDataFetcher(api_key)
        
        # 获取原始数据
        raw_df = fetcher.fetch_weather_data(city_name, start_time, end_time)
        # print(raw_df.head())
        
        # 预处理数据
        processed_df = self.preprocess(raw_df)
        # print(processed_df.head())
        
        return {
            'raw_df': raw_df,
            'processed_df': processed_df
        }
    
    def get_noaa_sequences(self, city_name, api_key, start_time=None, end_time=None):
        """
        从NOAA获取数据并构建序列
        
        Args:
            city_name: 城市名称
            api_key: NOAA API密钥
            start_time: 开始时间
            end_time: 结束时间
            
        Returns:
            序列数据和目标标签
        """
        # 获取预处理后的数据
        raw_df = self.load_noaa_data(city_name, api_key, start_time, end_time)['raw_df']
        
        # 构建序列
        sequences, targets = self.create_sequences(raw_df)
        
        return sequences, targets

# 示例用法
if __name__ == "__main__":
    # 初始化数据处理器
    processor = DataProcessor(seq_length=6, pred_length=1)
    
    # 加载数据（假设已经有原始数据文件）
    df = processor.load_data("d:\\nature-disaster-prediction\\data\\raw\\noaa_weather_data.csv")
    
    # 预处理数据
    scaled_df = processor.preprocess(df)
    
    # 创建序列数据
    sequences, targets, weather_targets = processor.create_sequences(scaled_df)
    
    # 划分数据集
    X_train, y_train, y_weather_train, X_val, y_val, y_weather_val, X_test, y_test, y_weather_test = processor.split_data(sequences, targets, weather_targets)
    
    # 保存处理后的数据
    data_dict = {
        'X_train': X_train,
        'y_train': y_train,
        'y_weather_train': y_weather_train,
        'X_val': X_val,
        'y_val': y_val,
        'y_weather_val': y_weather_val,
        'X_test': X_test,
        'y_test': y_test,
        'y_weather_test': y_weather_test
    }
    processor.save_processed_data(data_dict, "d:\\nature-disaster-prediction\\data\\processed\\processed_data.npz")
    
    print("数据处理完成！")
    print(f"训练集: {X_train.shape}, 验证集: {X_val.shape}, 测试集: {X_test.shape}")