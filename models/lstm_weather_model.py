from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import (
    LSTM, Dense, Dropout, Input, Attention,
    Concatenate, RepeatVector, TimeDistributed
)
from tensorflow.keras.models import Model
import tensorflow as tf
import numpy as np
import sys
import os
# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class LSTMWeatherModel:
    def __init__(self, input_shape, output_features=6, pred_days=7):
        self.input_shape = input_shape  # (seq_length, num_features)
        self.output_features = output_features  # 预测的特征数量
        self.pred_days = pred_days  # 预测未来天数
        self.model = self._build_seq2seq_model()  # 使用Seq2Seq模型
        self.scaler = None
        self.feature_names = ['降雨量', '持续时间', '风速', '平均温度', '最低温度', '最高温度']  # 特征名称，用于反归一化时参考

    def _build_seq2seq_model(self):
        """构建带注意力机制的Seq2Seq模型"""
        # 编码器部分
        encoder_inputs = Input(shape=self.input_shape)

        # 编码器LSTM层
        encoder_lstm1 = LSTM(units=64, return_sequences=True,
                             activation='relu')(encoder_inputs)
        encoder_lstm1 = Dropout(0.2)(encoder_lstm1)

        encoder_lstm2 = LSTM(units=64, return_sequences=True,
                             activation='relu')(encoder_lstm1)
        encoder_lstm2 = Dropout(0.2)(encoder_lstm2)

        encoder_lstm3 = LSTM(units=64, return_sequences=True,
                             activation='relu')(encoder_lstm2)
        encoder_outputs = Dropout(0.2)(encoder_lstm3)

        # 解码器部分 - 使用编码器的最后状态作为初始状态
        encoder_last_h = encoder_outputs[:, -1, :]  # 获取编码器最后一个时间步的输出

        # 重复编码器的最后状态以匹配预测天数
        decoder_input = RepeatVector(self.pred_days)(encoder_last_h)

        # 解码器LSTM层
        decoder_lstm = LSTM(units=64, return_sequences=True,
                            activation='relu')(decoder_input)
        decoder_lstm = Dropout(0.2)(decoder_lstm)

        # 注意力层
        attention_layer = Attention()([decoder_lstm, encoder_outputs])

        # 合并注意力输出和解码器输出
        merged = Concatenate(axis=-1)([decoder_lstm, attention_layer])

        # 合并注意力输出和解码器输出
        merged = Concatenate(axis=-1)([decoder_lstm, attention_layer])

        # 时间分布式全连接层，用于预测每个时间步的输出
        outputs = TimeDistributed(
            Dense(self.output_features, activation='linear'))(merged)

        # 构建完整模型
        model = Model(inputs=encoder_inputs, outputs=outputs)

        # 编译模型
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mean_squared_error',
            metrics=['mae']
        )

        return model

    def train(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
        """训练气象预测模型"""
        # 定义回调函数
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10,
                          restore_best_weights=True),
            ModelCheckpoint(
                filepath='d:\\nature-disaster-prediction\\models\\best_weather_model.keras',
                monitor='val_loss',
                mode='min',
                save_best_only=True
            )
        ]

        # 训练模型
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )

        return history

    def predict(self, X, days=7):
        """预测未来指定天数的气象数据"""
        try:
            # 直接预测未来所有天的数据，避免递归预测的误差累积
            future_predictions = self.model.predict(X)

            # 确保预测结果是 (days, output_features) 形状
            if len(future_predictions.shape) == 3:
                future_predictions = future_predictions[0]  # 取第一个样本

            # 限制预测天数
            future_predictions = future_predictions[:days]

            # 如果有归一化器，尝试反归一化
            if self.scaler is not None:
                try:
                    # 将预测结果重塑为适合反归一化的形状
                    original_shape = future_predictions.shape
                    reshaped_preds = future_predictions.reshape(-1, original_shape[-1])
                    
                    # 检查scaler是否适用于当前数据形状
                    if reshaped_preds.shape[1] == self.scaler.n_features_in_:
                        future_predictions = self.scaler.inverse_transform(reshaped_preds).reshape(original_shape)
                    else:
                        print(f"警告: Scaler特征数({self.scaler.n_features_in_})与预测数据特征数({reshaped_preds.shape[1]})不匹配")
                except Exception as scaler_error:
                    print(f"反归一化失败: {str(scaler_error)}，返回归一化后的预测结果")
                    # 反归一化失败时仍返回原始预测结果

            return future_predictions
        except Exception as e:
            print(f"预测失败，使用备用预测方法: {str(e)}")
            # 备用方案：如果Seq2Seq模型预测失败，使用基于趋势的简单预测
            return self._trend_based_prediction(X, days)

    def _trend_based_prediction(self, X, days=7):
        """基于历史趋势的备用预测方法"""
        future_predictions = []
        current_sequence = X[0]  # 取第一个样本

        # 对每个特征分别计算趋势
        for _ in range(days):
            next_step = []

            for feature_idx in range(self.output_features):
                feature_values = current_sequence[:, feature_idx]

                # 使用简单的移动平均和趋势外推
                if len(feature_values) >= 3:
                    # 计算最近3个值的平均值和趋势
                    avg = feature_values[-3:].mean()
                    # 计算趋势斜率
                    if len(feature_values) >= 5:
                        slope = (feature_values[-1] - feature_values[-5]) / 4
                    else:
                        slope = 0

                    # 预测下一个值 = 平均值 + 趋势斜率
                    next_value = avg + slope
                else:
                    # 如果数据不足，使用最后一个值
                    next_value = feature_values[-1]

                # 确保值在合理范围内
                next_value = max(0, min(1, next_value))
                next_step.append(next_value)

            future_predictions.append(next_step)

            # 更新序列，将新预测添加到末尾，去掉最前面的时间步
            next_step_reshaped = np.array(next_step).reshape(1, -1)
            current_sequence = np.concatenate(
                [current_sequence[1:], next_step_reshaped], axis=0)

        return np.array(future_predictions)

    def save_model(self, file_path):
        """保存模型"""
        self.model.save(file_path)

    def load_model(self, file_path):
        """加载模型"""
        self.model = tf.keras.models.load_model(file_path)

    def set_scaler(self, scaler):
        """设置数据归一化器"""
        self.scaler = scaler

    def get_scaler(self):
        """获取数据归一化器"""
        return self.scaler
