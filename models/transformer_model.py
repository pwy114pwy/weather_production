from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Dense, Dropout, Input, Layer, GlobalAveragePooling1D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
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


class TransformerModel:
    def __init__(self, input_shape, num_classes=3):
        self.input_shape = input_shape  # (seq_length, num_features)
        self.num_classes = num_classes  # 0=无, 1=中, 2=高
        self.model = self._build_model()

    def _build_model(self):
        """构建Transformer模型"""
        inputs = Input(shape=self.input_shape)
        
        # 位置编码
        x = PositionalEncoding()(inputs)
        
        # 将输入特征维度映射到d_model
        x = Dense(64, activation='relu')(x)
        
        # Transformer编码器层
        x = TransformerEncoderLayer(
            d_model=64,
            num_heads=4,
            ff_dim=128,
            dropout=0.2
        )(x)
        
        x = TransformerEncoderLayer(
            d_model=64,
            num_heads=4,
            ff_dim=128,
            dropout=0.2
        )(x)
        
        # 全局池化
        x = GlobalAveragePooling1D()(x)
        
        # 全连接层
        x = Dense(units=16, activation='relu')(x)
        x = Dropout(0.2)(x)
        
        # 输出层
        outputs = Dense(units=self.num_classes, activation='softmax')(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model

    def train(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
        """训练模型"""
        # 计算类别权重，处理类别不平衡
        class_counts = np.bincount(y_train.astype(int))
        # 由于灾害预测中漏判代价高，使用class_weight平衡类别
        class_weights = {0: 1.0, 1: 2.0, 2: 3.0}  # 高风险类别权重更高

        # 定义回调函数
        import os
        from pathlib import Path

        # 获取项目根目录
        BASE_DIR = Path(__file__).resolve().parent.parent
        best_model_path = str(BASE_DIR / "models" / "best_transformer_model.keras")

        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10,
                          restore_best_weights=True),
            ModelCheckpoint(
                filepath=best_model_path,
                monitor='val_accuracy',
                mode='max',
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
            class_weight=class_weights,
            verbose=1
        )

        return history

    def evaluate(self, X_test, y_test):
        """评估模型性能"""
        # 预测概率
        y_pred_prob = self.model.predict(X_test)

        # 预测类别
        y_pred = np.argmax(y_pred_prob, axis=1)

        # 计算评估指标
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # 计算AUC（需要至少2个类别）
        auc = None
        try:
            unique_classes = np.unique(y_test)
            if len(unique_classes) >= 2:
                auc = roc_auc_score(y_test, y_pred_prob,
                                    multi_class='ovr', average='weighted')
            else:
                print(f"警告：只有{len(unique_classes)}个类别，无法计算AUC")
        except ValueError as e:
            print(f"计算AUC时出错：{str(e)}")

        # 混淆矩阵
        cm = confusion_matrix(y_test, y_pred)

        # 打印评估结果
        print("模型评估结果：")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")  # 重点关注
        print(f"F1-score: {f1:.4f}")
        if auc is not None:
            print(f"AUC: {auc:.4f}")
        print("混淆矩阵：")
        print(cm)

        evaluation_results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': cm,
            'y_pred': y_pred,
            'y_pred_prob': y_pred_prob
        }

        # 只有当AUC计算成功时才添加到结果中
        if auc is not None:
            evaluation_results['auc'] = auc

        return evaluation_results

    def save_model(self, file_path):
        """保存模型"""
        self.model.save(file_path)

    def load_model(self, file_path):
        """加载模型"""
        self.model = tf.keras.models.load_model(file_path)

    def predict(self, X):
        """预测风险等级"""
        y_pred_prob = self.model.predict(X)
        y_pred = np.argmax(y_pred_prob, axis=1)
        return y_pred, y_pred_prob

    def predict_future(self, X, days=7):
        """预测未来7天的气象数据（时间序列预测）"""
        # 确保输入是3D数组 (samples, seq_length, features)
        if len(X.shape) == 2:
            X = np.expand_dims(X, axis=0)

        future_predictions = []
        current_sequence = X[0].copy()  # 使用第一个样本进行预测

        # 递归预测未来days天
        for _ in range(days):
            # 预测下一个时间步的特征值（这里使用自定义的预测逻辑）
            # 注意：当前模型是分类模型，需要修改为回归模型才能直接预测特征值
            # 这里简化处理，使用基于历史数据的趋势预测

            # 计算历史数据的趋势特征
            trend_prediction = self._calculate_trend_prediction(
                current_sequence)
            future_predictions.append(trend_prediction)

            # 更新序列，将新预测添加到末尾，去掉最前面的时间步
            current_sequence = np.vstack(
                [current_sequence[1:], trend_prediction])

        return np.array(future_predictions)

    def _calculate_trend_prediction(self, sequence):
        """基于历史序列计算趋势预测值"""
        # 对每个特征分别计算趋势
        num_features = sequence.shape[1]
        prediction = []

        for feature_idx in range(num_features):
            feature_values = sequence[:, feature_idx]

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

                # 确保值在合理范围内（归一化后应该在0-1之间）
                next_value = max(0, min(1, next_value))
            else:
                # 如果数据不足，使用最后一个值
                next_value = feature_values[-1]

            prediction.append(next_value)

        return np.array(prediction)

    def get_risk_explanation(self, X, pred_class):
        """生成风险解释"""
        # 计算输入序列的统计特征
        seq = X[0]  # 假设X是单个样本

        # 提取特征值（注意需要反归一化）
        # 这里简化处理，直接使用归一化后的值进行判断
        rainfall = seq[:, 0].mean()  # 平均降雨量
        max_rainfall = seq[:, 0].max()  # 最大降雨量
        wind_speed = seq[:, 2].mean()  # 平均风速
        avg_temp = seq[:, 3].mean()  # 平均温度

        # 根据预测类别生成不同的解释
        risk_levels = {0: "无风险", 1: "中等风险", 2: "高风险"}

        # 生成风险解释
        explanations = []

        if pred_class == 0:  # 无风险
            # 检测是否有轻微风险因素，但不足以构成风险
            if max_rainfall > 0.3:  # 轻微降雨
                explanations.append("存在轻微降雨，但强度和持续时间均在安全范围内")
            elif wind_speed > 0.5:  # 中等风速
                explanations.append("风速适中，无明显降雨，发生洪涝概率低")
            else:
                explanations.append("气象条件良好，无明显洪涝风险因素")
        else:  # 有风险（中等或高）
            if max_rainfall > 0.7:  # 假设0.7是归一化后的高值
                explanations.append("单位时间降雨强度高")

            if rainfall > 0.5:  # 假设0.5是归一化后的中高值
                explanations.append("连续降雨时间过长")

            if wind_speed > 0.6:  # 假设0.6是归一化后的高风速
                explanations.append("风速较大，可能加剧洪涝风险")

            if not explanations:
                explanations.append("气象条件综合分析存在洪涝风险")

        return {
            "risk_level": risk_levels[pred_class],
            "explanations": explanations
        }


# 示例用法
if __name__ == "__main__":
    import numpy as np
    from data.data_processor import DataProcessor

    # 初始化数据处理器
    processor = DataProcessor(seq_length=6, pred_length=1)

    # 使用CSV文件训练模型
    print("从CSV文件加载气象数据...")
    from pathlib import Path
    BASE_DIR = Path(__file__).resolve().parent.parent
    csv_file_path = str(BASE_DIR / "data" / "raw" /
                        "noaa_weather_data.csv")  # CSV文件路径

    # 加载CSV数据
    df = processor.load_data(csv_file_path)

    # 检查是否包含风险标签
    if '洪涝风险标签' not in df.columns:
        print("\nCSV文件中缺少风险标签，开始生成...")
        # 生成风险标签（基于降雨量和其他气象特征）
        # 规则：
        # - 降雨量 > 0.6 且 持续时间 > 0.5 → 高风险 (2)
        # - 降雨量 > 0.4 或 持续时间 > 0.3 → 中等风险 (1)
        # - 其他情况 → 无风险 (0)
        risk_labels = np.zeros(len(df))

        for i in range(len(df)):
            row = df.iloc[i]
            rainfall = row['降雨量']
            duration = row['持续时间']

            if rainfall > 0.6 and duration > 0.5:
                risk_labels[i] = 2  # 高风险
            elif rainfall > 0.4 or duration > 0.3:
                risk_labels[i] = 1  # 中等风险
            else:
                risk_labels[i] = 0  # 无风险

        # 更新DataFrame
        df['洪涝风险标签'] = risk_labels

    # 预处理数据
    print("\n预处理数据...")
    scaled_df = processor.preprocess(df)

    # 统计标签分布
    # 从预处理后的DataFrame中获取风险标签
    risk_labels = scaled_df['洪涝风险标签']
    unique, counts = np.unique(risk_labels, return_counts=True)
    print(f"\n风险标签分布:")
    for label, count in zip(unique, counts):
        label_names = {0: "无风险", 1: "中等风险", 2: "高风险"}
        print(
            f"{label_names[label]}: {count} ({count/len(risk_labels)*100:.1f}%)")

    # 构建序列数据
    print("\n构建时间序列数据...")
    sequences, targets = processor.create_sequences(scaled_df)

    # 划分数据集
    print("\n划分训练集、验证集和测试集...")
    X_train, y_train, X_val, y_val, X_test, y_test = processor.split_data(
        sequences, targets)

    # 打印数据形状
    print(f"训练集形状: X={X_train.shape}, y={y_train.shape}")
    print(f"验证集形状: X={X_val.shape}, y={y_val.shape}")
    print(f"测试集形状: X={X_test.shape}, y={y_test.shape}")

    # 初始化模型
    # (seq_length, num_features)
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = TransformerModel(input_shape=input_shape, num_classes=3)

    # 训练模型
    print("\n开始训练模型...")
    history = model.train(X_train, y_train, X_val, y_val,
                          epochs=50, batch_size=32)

    # 评估模型
    print("\n开始评估模型...")
    evaluation_results = model.evaluate(X_test, y_test)

    # 保存模型
    final_model_path = str(BASE_DIR / "models" / "final_transformer_model.keras")
    model.save_model(final_model_path)
    print("\n模型已保存！")

    # 测试预测和解释功能
    print("\n测试预测和解释功能...")
    sample = X_test[0:1]  # 取一个样本
    pred_class, pred_prob = model.predict(sample)
    explanation = model.get_risk_explanation(sample, pred_class[0])
    print(f"预测类别: {pred_class[0]}, 预测概率: {pred_prob[0]}")
    print(f"风险解释: {explanation}")
