<template>
  <div class="prediction-panel">
    <h3>未来天气预测</h3>
    <div v-if="!weatherStore.isDataLoaded" class="no-data">
      请先获取气象数据
    </div>
    <div v-else>
      <div class="prediction-controls">
        <button @click="predictWeather" :disabled="weatherStore.isPredicting" class="predict-button">
          {{ weatherStore.isPredicting ? '预测中...' : '预测未来7天天气' }}
        </button>
        <button @click="clearPrediction" :disabled="!weatherStore.getPredictionResults" class="clear-button">
          清除预测结果
        </button>
      </div>

      <div v-if="weatherStore.getPredictionResults" class="prediction-results">
        <div class="result-header">
          <h4>预测结果</h4>
          <!-- <span class="prediction-time">预测时间: {{ new Date().toLocaleString() }}</span> -->
        </div>

        <div class="risk-overview">
          <div class="risk-item">
            <span class="label">风险等级:</span>
            <span class="value risk-level" :class="weatherStore.getPredictionResults.risk_level">
              {{ weatherStore.getPredictionResults.risk_level }}
            </span>
          </div>
          <div class="risk-item">
            <span class="label">风险分数:</span>
            <span class="value">{{ (weatherStore.getPredictionResults.risk_score * 100).toFixed(2) }}%</span>
          </div>
        </div>

        <div class="risk-explanations">
          <h5>风险解释:</h5>
          <ul>
            <li v-for="(explanation, index) in weatherStore.getPredictionResults.explanations" :key="index">
              {{ explanation }}
            </li>
          </ul>
        </div>

        <!-- 未来7天预测趋势图 -->
        <div class="prediction-chart">
          <h5>未来7天气象趋势预测</h5>
          
          <div ref="predictionChartRef" class="chart"></div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, watch, nextTick } from 'vue';
import * as echarts from 'echarts';
import axios from 'axios';
import { useWeatherStore } from '../stores/weather';

// 获取weather store
const weatherStore = useWeatherStore();

// 预测图表引用
const predictionChartRef = ref(null);
let predictionChartInstance = null;

// 预测天气
const predictWeather = async () => {
  try {
    // 设置预测状态
    weatherStore.setIsPredicting(true);

    // 获取归一化数据
    const normalizedData = weatherStore.getWeatherData;

    // 发送风险预测请求
    const riskResponse = await axios.post('http://localhost:8000/api/v1/risk_prediction', {
    data: normalizedData
    });
  

    // 发送未来天气预测请求
    const weatherResponse = await axios.post('http://localhost:8000/api/v1/weather_prediction', {
     data: normalizedData
    });

    // 保存预测结果
    const predictionResults = {
      ...riskResponse.data,
      future_predictions: weatherResponse.data.future_weather
    };
    weatherStore.setPredictionResults(predictionResults);

    // 初始化预测趋势图
    nextTick(() => {
      initPredictionChart();
    });

  } catch (error) {
    console.error('预测失败:', error);
    alert('预测失败，请稍后重试');
  } finally {
    // 重置预测状态
    weatherStore.setIsPredicting(false);
  }
};

// 清除预测结果
const clearPrediction = () => {
  weatherStore.clearPredictionResults();
  if (predictionChartInstance) {
    predictionChartInstance.dispose();
    predictionChartInstance = null;
  }
};

// 初始化预测趋势图
const initPredictionChart = () => {
  if (!predictionChartRef.value || !weatherStore.getPredictionResults?.future_predictions) return;

  // 创建ECharts实例
  predictionChartInstance = echarts.init(predictionChartRef.value);

  // 生成未来7天的日期（基于用户选择的截止日期）
  const futureDates = [];
  const endDate = weatherStore.getEndTime ? new Date(weatherStore.getEndTime) : new Date();
  for (let i = 1; i <= 7; i++) {
    const date = new Date(endDate);
    date.setDate(endDate.getDate() + i);
    futureDates.push(date.toLocaleDateString());
  }

  // 获取后端返回的预测数据
  const futurePredictions = weatherStore.getPredictionResults.future_predictions;

  // 将归一化数据转换为原始数据范围（使用更科学的反归一化方法）
  // 这里假设数据已经被模型内部正确反归一化，否则使用估计的范围
  const featureData = {
    '降雨量': futurePredictions.map(item => item[0]),  // 保持原样，如果数据已经反归一化
    '风速': futurePredictions.map(item => item[2]),  // 保持原样，如果数据已经反归一化
    '平均温度': futurePredictions.map(item => item[3]),  // 保持原样，如果数据已经反归一化
    // 注意：以下特征仅用于图表展示，风险预测不依赖它们
    '持续时间': futurePredictions.map(item => item[1]),
    '最低温度': futurePredictions.map(item => item[4]),
    '最高温度': futurePredictions.map(item => item[5])
  };

  // 配置图表
  const option = {
    title: {
      text: '未来7天气象趋势预测',
      left: 'center'
    },
    tooltip: {
      trigger: 'axis',
      formatter: function (params) {
        let result = `${params[0].axisValue}<br/>`;
        params.forEach(param => {
          result += `${param.seriesName}: ${param.value.toFixed(2)}<br/>`;
        });
        return result;
      }
    },
    legend: {
      data: ['降雨量', '风速', '平均温度'],
      top: '10%'
    },
    grid: {
      left: '3%',
      right: '4%',
      bottom: '3%',
      containLabel: true
    },
    xAxis: {
      type: 'category',
      boundaryGap: false,
      data: futureDates
    },
    yAxis: {
      type: 'value',
      name: '数值',
      axisLabel: {
        formatter: '{value}'
      }
    },
    series: [
      {
        name: '降雨量',
        type: 'line',
        smooth: true,
        data: featureData['降雨量'],
        lineStyle: {
          color: '#1890ff'
        },
        areaStyle: {
          opacity: 0.2,
          color: '#1890ff'
        }
      },
      {
        name: '风速',
        type: 'line',
        smooth: true,
        data: featureData['风速'],
        lineStyle: {
          color: '#52c41a'
        },
        areaStyle: {
          opacity: 0.2,
          color: '#52c41a'
        }
      },
      {
        name: '平均温度',
        type: 'line',
        smooth: true,
        data: featureData['平均温度'],
        lineStyle: {
          color: '#f5222d'
        },
        areaStyle: {
          opacity: 0.2,
          color: '#f5222d'
        }
      }
    ]
  };

  // 设置图表配置
  predictionChartInstance.setOption(option);
};

// 监听窗口大小变化
const handleResize = () => {
  if (predictionChartInstance) {
    predictionChartInstance.resize();
  }
};

// 组件挂载时初始化
onMounted(() => {
  window.addEventListener('resize', handleResize);
});

// 监听预测结果变化
watch(() => weatherStore.getPredictionResults, () => {
  if (weatherStore.getPredictionResults) {
    nextTick(() => {
      initPredictionChart();
    });
  }
}, { deep: true });
</script>

<style scoped>
.prediction-panel {
  padding: 20px;
  background-color: #f9f9f9;
  border-radius: 8px;
  box-shadow: 0 2px 12px rgba(0, 0, 0, 0.1);
  margin-top: 20px;
}

h3 {
  margin-top: 0;
  color: #333;
  text-align: center;
  border-bottom: 1px solid #e8e8e8;
  padding-bottom: 10px;
}

.no-data {
  text-align: center;
  color: #999;
  padding: 40px 0;
  background-color: white;
  border-radius: 4px;
  border: 1px solid #e8e8e8;
}

.prediction-controls {
  display: flex;
  gap: 10px;
  margin-bottom: 20px;
  justify-content: center;
  flex-wrap: wrap;
}

.predict-button,
.clear-button {
  padding: 10px 20px;
  font-size: 14px;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  transition: background-color 0.3s;
  min-width: 150px;
}

.predict-button {
  background-color: #1890ff;
  color: white;
}

.predict-button:hover:not(:disabled) {
  background-color: #40a9ff;
}

.clear-button {
  background-color: #f5222d;
  color: white;
}

.clear-button:hover:not(:disabled) {
  background-color: #ff4d4f;
}

.predict-button:disabled,
.clear-button:disabled {
  background-color: #d9d9d9;
  cursor: not-allowed;
  color: #999;
}

.prediction-results {
  background-color: white;
  padding: 20px;
  border-radius: 4px;
  border: 1px solid #e8e8e8;
}

.result-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 20px;
  border-bottom: 1px solid #e8e8e8;
  padding-bottom: 10px;
}

.result-header h4 {
  margin: 0;
  color: #333;
}

.prediction-time {
  font-size: 12px;
  color: #999;
}

.risk-overview {
  display: flex;
  gap: 20px;
  margin-bottom: 20px;
  flex-wrap: wrap;
}

.risk-item {
  display: flex;
  align-items: center;
  font-size: 16px;
}

.label {
  font-weight: bold;
  color: #333;
  margin-right: 10px;
}

.value {
  color: #666;
}

.risk-level {
  padding: 5px 10px;
  border-radius: 4px;
  font-weight: bold;
}

.无风险 {
  background-color: #e6f7ff;
  color: #1890ff;
}

.中等风险 {
  background-color: #fff7e6;
  color: #faad14;
}

.高风险 {
  background-color: #fff2f0;
  color: #f5222d;
}

.risk-explanations {
  margin-bottom: 20px;
}

.risk-explanations h5 {
  margin: 0 0 10px 0;
  color: #333;
}

.risk-explanations ul {
  margin: 0;
  padding-left: 20px;
  color: #666;
}

.risk-explanations li {
  margin-bottom: 5px;
}

.prediction-chart {
  margin-top: 20px;
}

.prediction-chart h5 {
  margin: 0 0 15px 0;
  color: #333;
}

.chart {
  width: 100%;
  height: 400px;
  background-color: #fafafa;
  border-radius: 4px;
  border: 1px solid #e8e8e8;
}

@media (max-width: 768px) {
  .risk-overview {
    flex-direction: column;
    gap: 10px;
  }

  .prediction-controls {
    flex-direction: column;
  }

  .predict-button,
  .clear-button {
    width: 100%;
  }

  .chart {
    height: 250px;
  }
}
</style>