from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import (
    Input, Dropout, Dense, TimeDistributed, RepeatVector,
    Layer, GlobalAveragePooling1D
)
from tensorflow.keras.models import Model
import tensorflow as tf
from tensorflow import keras
import numpy as np
import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@tf.keras.utils.register_keras_serializable()
class PositionalEncoding(Layer):
    """位置编码层"""
    def __init__(self, name=None, **kwargs):
        super(PositionalEncoding, self).__init__(name=name, **kwargs)
    
    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        seq_length = tf.shape(inputs)[1]
        feature_dim = tf.shape(inputs)[2]
        
        # 创建位置编码
        position = tf.range(seq_length, dtype=tf.float32)[:, tf.newaxis]
        
        # 计算位置编码的维度
        max_dim = tf.cast(feature_dim, tf.int32)
        even_dim = tf.cast(tf.floor(tf.cast(max_dim, tf.float32) / 2.0), tf.int32)
        
        # 确保所有操作都使用float32类型
        feature_dim_float = tf.cast(max_dim, tf.float32)
        div_term = tf.exp(tf.range(0, even_dim * 2, 2, dtype=tf.float32) * (-tf.cast(tf.math.log(10000.0), tf.float32) / feature_dim_float))
        
        # 计算正弦和余弦编码
        sin_encoding = tf.sin(position * div_term)
        cos_encoding = tf.cos(position * div_term)
        
        # 创建位置编码矩阵
        # 扩展sin和cos编码到特征维度
        sin_encoding = tf.tile(sin_encoding, [1, 2])[:, :max_dim]
        cos_encoding = tf.concat([tf.zeros_like(cos_encoding), cos_encoding], axis=1)[:, :max_dim]
        
        # 交替放置sin和cos编码
        pos_encoding = sin_encoding + cos_encoding
        
        # 扩展维度并广播到batch_size
        pos_encoding = tf.tile(pos_encoding[tf.newaxis, :, :], [batch_size, 1, 1])
        
        return inputs + pos_encoding
    
    def get_config(self):
        config = super(PositionalEncoding, self).get_config()
        return config


@tf.keras.utils.register_keras_serializable()
class MultiHeadAttention(Layer):
    """多头注意力层"""
    def __init__(self, d_model, num_heads, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.d_model = d_model
        
        assert d_model % self.num_heads == 0
        
        self.depth = d_model // self.num_heads
        
        self.wq = Dense(d_model)
        self.wk = Dense(d_model)
        self.wv = Dense(d_model)
        
        self.dense = Dense(d_model)
    
    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])
    
    def call(self, v, k, q, mask=None):
        batch_size = tf.shape(q)[0]
        
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)
        
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)
        
        matmul_qk = tf.matmul(q, k, transpose_b=True)
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
        
        if mask is not None:
            scaled_attention_logits += (mask * -1e9)
        
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        output = tf.matmul(attention_weights, v)
        output = tf.transpose(output, perm=[0, 2, 1, 3])
        output = tf.reshape(output, (batch_size, -1, self.d_model))
        output = self.dense(output)
        
        return output
    
    def get_config(self):
        config = super(MultiHeadAttention, self).get_config()
        config.update({
            'd_model': self.d_model,
            'num_heads': self.num_heads
        })
        return config


@tf.keras.utils.register_keras_serializable()
class TransformerEncoderLayer(Layer):
    """Transformer编码器层"""
    def __init__(self, d_model, num_heads, ff_dim, dropout=0.1, **kwargs):
        super(TransformerEncoderLayer, self).__init__(**kwargs)
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = tf.keras.Sequential([
            Dense(ff_dim, activation='relu'),
            Dense(d_model)
        ])
        
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout = dropout
    
    def call(self, x, training=True):
        attn_output = self.mha(x, x, x)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)
        
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)
        
        return out2
    
    def get_config(self):
        config = super(TransformerEncoderLayer, self).get_config()
        config.update({
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'ff_dim': self.ff_dim,
            'dropout': self.dropout
        })
        return config


@tf.keras.utils.register_keras_serializable()
class TransformerDecoderLayer(Layer):
    """Transformer解码器层"""
    def __init__(self, d_model, num_heads, ff_dim, dropout=0.1, **kwargs):
        super(TransformerDecoderLayer, self).__init__(**kwargs)
        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)
        
        self.ffn = tf.keras.Sequential([
            Dense(ff_dim, activation='relu'),
            Dense(d_model)
        ])
        
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.dropout3 = Dropout(dropout)
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout = dropout
        self.built = False
    
    def build(self, input_shape):
        """构建层"""
        # 确保所有子层都被构建
        if not self.built:
            # 构建多头注意力层
            self.mha1.build([input_shape, input_shape, input_shape])
            self.mha2.build([input_shape, input_shape, input_shape])
            
            # 构建前馈网络
            self.ffn.build(input_shape)
            
            # 构建层归一化
            self.layernorm1.build(input_shape)
            self.layernorm2.build(input_shape)
            self.layernorm3.build(input_shape)
            
            # 构建 dropout 层
            self.dropout1.build(input_shape)
            self.dropout2.build(input_shape)
            self.dropout3.build(input_shape)
            
            self.built = True
    
    def call(self, x, enc_output, training=True, look_ahead_mask=None):
        # 确保层已构建
        if not self.built:
            self.build(x.shape)
            
        attn1 = self.mha1(x, x, x, look_ahead_mask)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)
        
        attn2 = self.mha2(enc_output, enc_output, out1)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1)
        
        ffn_output = self.ffn(out2)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + out2)
        
        return out3
    
    def get_config(self):
        config = super(TransformerDecoderLayer, self).get_config()
        config.update({
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'ff_dim': self.ff_dim,
            'dropout': self.dropout
        })
        return config


class TransformerWeatherModel:
    def __init__(self, input_shape, output_features=6, pred_days=7):
        self.input_shape = input_shape  # (seq_length, num_features)
        self.output_features = output_features  # 预测的特征数量
        self.pred_days = pred_days  # 预测未来天数
        self.model = self._build_seq2seq_model()  # 使用Seq2Seq模型
        self.scaler = None
        self.feature_names = ['降雨量', '持续时间', '风速',
                              '平均温度', '最低温度', '最高温度']  # 特征名称，用于反归一化时参考

    def _build_seq2seq_model(self):
        """构建带注意力机制的Seq2Seq Transformer模型"""
        # 编码器部分
        encoder_inputs = Input(shape=self.input_shape)
        
        # 位置编码
        encoder_embedding = PositionalEncoding()(encoder_inputs)
        
        # 将输入特征维度映射到d_model
        encoder_embedding = Dense(64, activation='relu')(encoder_embedding)
        
        # 编码器Transformer层
        encoder_outputs = TransformerEncoderLayer(
            d_model=64,
            num_heads=4,
            ff_dim=128,
            dropout=0.2
        )(encoder_embedding)
        
        encoder_outputs = TransformerEncoderLayer(
            d_model=64,
            num_heads=4,
            ff_dim=128,
            dropout=0.2
        )(encoder_outputs)
        
        # 解码器部分 - 使用pred_days参数来设置预测天数
        decoder_inputs = Input(shape=(self.pred_days, self.output_features))
        
        # 位置编码
        decoder_embedding = PositionalEncoding()(decoder_inputs)
        
        # 将输入特征维度映射到d_model
        decoder_embedding = Dense(64, activation='relu')(decoder_embedding)
        
        # 创建前瞻掩码,防止解码器看到未来的位置
        def create_look_ahead_mask(size):
            mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
            return mask  # (seq_len, seq_len)
        
        look_ahead_mask = create_look_ahead_mask(self.pred_days)
        
        # 解码器Transformer层
        decoder_outputs = TransformerDecoderLayer(
            d_model=64,
            num_heads=4,
            ff_dim=128,
            dropout=0.2
        )(decoder_embedding, encoder_outputs, look_ahead_mask=look_ahead_mask)
        
        decoder_outputs = TransformerDecoderLayer(
            d_model=64,
            num_heads=4,
            ff_dim=128,
            dropout=0.2
        )(decoder_outputs, encoder_outputs, look_ahead_mask=look_ahead_mask)
        
        # 时间分布式全连接层，用于预测每个时间步的输出
        outputs = TimeDistributed(
            Dense(self.output_features, activation='linear')
        )(decoder_outputs)

        # 构建完整模型
        model = Model(inputs=[encoder_inputs, decoder_inputs], outputs=outputs)

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
        import os
        from pathlib import Path

        # 获取项目根目录
        BASE_DIR = Path(__file__).resolve().parent.parent
        best_weather_model_path = str(
            BASE_DIR / "models" / "best_transformer_weather_model.keras")

        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10,
                          restore_best_weights=True),
            ModelCheckpoint(
                filepath=best_weather_model_path,
                monitor='val_loss',
                mode='min',
                save_best_only=True
            )
        ]

        # 为解码器创建输入（使用移位的目标序列作为解码器输入）
        # 对于训练，解码器输入是目标序列的移位版本（移除最后一个元素，添加一个起始标记）
        # 这里简化处理，使用目标序列的前n-1个元素作为输入
        decoder_input_train = np.zeros_like(y_train)
        decoder_input_val = np.zeros_like(y_val)
        
        if y_train.shape[1] > 1:
            decoder_input_train[:, 1:, :] = y_train[:, :-1, :]
            decoder_input_val[:, 1:, :] = y_val[:, :-1, :]
        
        # 训练模型
        history = self.model.fit(
            [X_train, decoder_input_train], y_train,
            validation_data=([X_val, decoder_input_val], y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )

        return history

    def predict(self, X, days=7):
        """预测未来指定天数的气象数据"""
        try:
            # 直接使用递归预测方法生成多天结果
            # 这样可以避免解码器输入形状不匹配的问题
            future_predictions = self._recursive_prediction(X, days)

            # 如果有归一化器，尝试反归一化
            if self.scaler is not None:
                try:
                    # 将预测结果重塑为适合反归一化的形状
                    original_shape = future_predictions.shape
                    reshaped_preds = future_predictions.reshape(
                        -1, original_shape[-1])

                    # 检查scaler是否适用于当前数据形状
                    if reshaped_preds.shape[1] == self.scaler.n_features_in_:
                        future_predictions = self.scaler.inverse_transform(
                            reshaped_preds).reshape(original_shape)
                    else:
                        print(
                            f"警告: Scaler特征数({self.scaler.n_features_in_})与预测数据特征数({reshaped_preds.shape[1]})不匹配")
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

        future_predictions = np.array(future_predictions)
        
        # 如果有归一化器，尝试反归一化
        if self.scaler is not None:
            try:
                # 将预测结果重塑为适合反归一化的形状
                original_shape = future_predictions.shape
                reshaped_preds = future_predictions.reshape(
                    -1, original_shape[-1])

                # 检查scaler是否适用于当前数据形状
                if reshaped_preds.shape[1] == self.scaler.n_features_in_:
                    future_predictions = self.scaler.inverse_transform(
                        reshaped_preds).reshape(original_shape)
                else:
                    print(
                        f"警告: Scaler特征数({self.scaler.n_features_in_})与预测数据特征数({reshaped_preds.shape[1]})不匹配")
            except Exception as scaler_error:
                print(f"备用预测反归一化失败: {str(scaler_error)}，返回归一化后的预测结果")
                # 反归一化失败时仍返回原始预测结果
        
        return future_predictions

    def _recursive_prediction(self, X, days=7):
        """递归预测方法：当模型只被训练为预测1天时，通过递归预测生成多天结果"""
        future_predictions = []
        current_input = X.copy()
        
        for day in range(days):
            # 为解码器创建初始输入（全零序列）- 使用模型期望的形状
            batch_size = current_input.shape[0]
            decoder_input = np.zeros((batch_size, 1, self.output_features))
            
            # 预测下一天的天气数据
            next_day_pred = self.model.predict([current_input, decoder_input])
            
            # 确保预测结果是 (1, output_features) 形状
            if len(next_day_pred.shape) == 3:
                next_day_pred = next_day_pred[0]  # 取第一个样本
            
            # 只取第一天的预测结果
            next_day_pred = next_day_pred[:1]
            
            # 添加到预测结果列表
            future_predictions.append(next_day_pred[0])
            
            # 更新输入序列：移除最早的时间步，添加新的预测结果
            if current_input.shape[1] > 1:
                # 如果当前输入序列长度大于1，移除最早的时间步
                current_input = np.concatenate([current_input[:, 1:, :], 
                                              next_day_pred.reshape(1, 1, -1)], axis=1)
            else:
                # 如果当前输入序列长度等于1，直接替换
                current_input = next_day_pred.reshape(1, 1, -1)
        
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
