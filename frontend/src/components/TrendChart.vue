<template>
  <div class="trend-chart-container">
    <h3>气象数据趋势图</h3>
    <div v-if="!weatherStore.isDataLoaded" class="no-data">
      请先获取气象数据
    </div>
    <div v-else>
      <div class="chart-header">
        <div class="data-info">
          <span>{{ weatherStore.getSelectedCity }} - {{ weatherStore.getDataSource }}</span>
          <!-- <span class="timestamp">{{ weatherStore.getDataTimestamp }}</span> -->
        </div>
        <div class="feature-selector">
          <label>选择特征:</label>
          <select v-model="selectedFeature">
            <option value="0">降雨量</option>
            <option value="1">持续时间</option>
            <option value="2">风速</option>
            <option value="3">平均温度</option>
            <option value="4">最低温度</option>
            <option value="5">最高温度</option>
          </select>
        </div>
      </div>
      <div ref="chartRef" class="chart"></div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, watch, nextTick } from 'vue';
import * as echarts from 'echarts';
import { useWeatherStore } from '../stores/weather';

// 获取weather store
const weatherStore = useWeatherStore();

// 图表引用
const chartRef = ref(null);
let chartInstance = null;

// 选择的特征
const selectedFeature = ref('0');

// 特征名称映射
const featureNames = {
  '0': '降雨量',
  '1': '持续时间',
  '2': '风速',
  '3': '平均温度',
  '4': '最低温度',
  '5': '最高温度'
};

// 初始化图表
const initChart = () => {
  if (!chartRef.value) return;

  // 创建ECharts实例
  chartInstance = echarts.init(chartRef.value);

  // 配置图表
  const option = {
    title: {
      text: featureNames[selectedFeature.value] + '趋势图',
      left: 'center'
    },
    tooltip: {
      trigger: 'axis',
      formatter: function (params) {
        const featureName = featureNames[selectedFeature.value];
        const unit = getFeatureUnit();
        return `${params[0].axisValue}<br/>${featureName}: ${params[0].value.toFixed(2)}${unit}`;
      }
    },
    grid: {
      left: '3%',
      right: '4%',
      bottom: '3%',
      containLabel: true
    },
    xAxis: {
      type: 'category',
      boundaryGap: selectedFeature.value === '0' ? true : false, // 降雨量使用间隔，其他使用连续
      data: getXAxisData()
    },
    yAxis: {
      type: 'value',
      name: featureNames[selectedFeature.value] + getFeatureUnit(),
      nameTextStyle: {
        padding: [0, 0, 0, 40]
      },
      splitLine: {
        lineStyle: {
          type: 'dashed'
        }
      }
    },
    series: [
      {
        name: featureNames[selectedFeature.value],
        type: selectedFeature.value === '0' ? 'bar' : 'line', // 降雨量使用柱状图，其他使用折线图
        smooth: selectedFeature.value !== '0', // 折线图使用平滑曲线
        data: getChartData(),
        lineStyle: {
          width: selectedFeature.value !== '0' ? 3 : 1
        },
        itemStyle: {
          color: selectedFeature.value === '0' ? {
            type: 'linear',
            x: 0,
            y: 0,
            x2: 0,
            y2: 1,
            colorStops: [{
              offset: 0, color: '#1890ff' // 顶部颜色
            }, {
              offset: 1, color: '#b3d8ff' // 底部颜色
            }]
          } : getFeatureColor()
        },
        areaStyle: {
          opacity: selectedFeature.value !== '0' ? 0.3 : 0,
          color: selectedFeature.value !== '0' ? getFeatureColor() : undefined
        },
        emphasis: {
          itemStyle: {
            shadowBlur: 10,
            shadowOffsetX: 0,
            shadowColor: 'rgba(0, 0, 0, 0.5)'
          }
        }
      }
    ]
  };

  // 设置图表配置
  chartInstance.setOption(option);
};

// 获取X轴数据
const getXAxisData = () => {
  const data = weatherStore.getRawWeatherData;
  return data.map(item => item.时间);
};

// 获取图表数据
const getChartData = () => {
  const data = weatherStore.getRawWeatherData;
  const featureMap = {
    '0': '降雨量',
    '1': '持续时间',
    '2': '风速',
    '3': '平均温度',
    '4': '最低温度',
    '5': '最高温度'
  };
  const featureName = featureMap[selectedFeature.value];
  console.log(data);

  return data.map(item => item[featureName]);
};

// 获取特征颜色
const getFeatureColor = () => {
  const colors = {
    '0': '#1890ff',  // 降雨量 - 蓝色
    '1': '#52c41a',  // 持续时间 - 绿色
    '2': '#faad14',  // 风速 - 黄色
    '3': '#f5222d',  // 平均温度 - 红色
    '4': '#722ed1',  // 最低温度 - 紫色
    '5': '#eb2f96'   // 最高温度 - 粉色
  };
  return colors[selectedFeature.value] || '#1890ff';
};

// 获取特征单位
const getFeatureUnit = () => {
  const units = {
    '0': ' mm',  // 降雨量 - 毫米
    '1': ' min',  // 持续时间 - 分钟
    '2': ' m/s',  // 风速 - 米/秒
    '3': ' °C',  // 平均温度 - 摄氏度
    '4': ' °C',  // 最低温度 - 摄氏度
    '5': ' °C'   // 最高温度 - 摄氏度
  };
  return units[selectedFeature.value] || '';
};

// 更新图表
const updateChart = () => {
  if (!chartInstance) return;

  chartInstance.setOption({
    title: {
      text: featureNames[selectedFeature.value] + '趋势图'
    },
    tooltip: {
      formatter: function (params) {
        const featureName = featureNames[selectedFeature.value];
        const unit = getFeatureUnit();
        return `${params[0].axisValue}<br/>${featureName}: ${params[0].value.toFixed(2)}${unit}`;
      }
    },
    xAxis: {
      boundaryGap: selectedFeature.value === '0' ? true : false // 降雨量使用间隔，其他使用连续
    },
    yAxis: {
      name: featureNames[selectedFeature.value] + getFeatureUnit(),
      nameTextStyle: {
        padding: [0, 0, 0, 40]
      }
    },
    series: [
      {
        name: featureNames[selectedFeature.value],
        type: selectedFeature.value === '0' ? 'bar' : 'line', // 降雨量使用柱状图，其他使用折线图
        smooth: selectedFeature.value !== '0', // 折线图使用平滑曲线
        data: getChartData(),
        lineStyle: {
          width: selectedFeature.value !== '0' ? 3 : 1
        },
        itemStyle: {
          color: selectedFeature.value === '0' ? {
            type: 'linear',
            x: 0,
            y: 0,
            x2: 0,
            y2: 1,
            colorStops: [{
              offset: 0, color: '#1890ff' // 顶部颜色
            }, {
              offset: 1, color: '#b3d8ff' // 底部颜色
            }]
          } : getFeatureColor()
        },
        areaStyle: {
          opacity: selectedFeature.value !== '0' ? 0.3 : 0,
          color: selectedFeature.value !== '0' ? getFeatureColor() : undefined
        }
      }
    ]
  });
};

// 监听特征选择变化
watch(selectedFeature, () => {
  updateChart();
});

// 监听数据变化
watch(() => weatherStore.getWeatherData, () => {
  if (weatherStore.isDataLoaded) {
    nextTick(() => {
      if (chartInstance) {
        chartInstance.dispose();
      }
      initChart();
    });
  }
}, { deep: true });

// 监听窗口大小变化
const handleResize = () => {
  if (chartInstance) {
    chartInstance.resize();
  }
};

// 组件挂载时初始化图表
onMounted(() => {
  if (weatherStore.isDataLoaded) {
    nextTick(() => {
      initChart();
    });
  }
  window.addEventListener('resize', handleResize);
});
</script>

<style scoped>
.trend-chart-container {
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

.chart-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 15px;
  flex-wrap: wrap;
  gap: 10px;
}

.data-info {
  display: flex;
  flex-direction: column;
  font-size: 14px;
  color: #666;
}

.timestamp {
  font-size: 12px;
  color: #999;
}

.feature-selector {
  display: flex;
  align-items: center;
  gap: 8px;
}

.feature-selector label {
  font-size: 14px;
  color: #666;
}

.feature-selector select {
  padding: 6px 12px;
  border: 1px solid #d9d9d9;
  border-radius: 4px;
  font-size: 14px;
  background-color: white;
  cursor: pointer;
}

.feature-selector select:focus {
  outline: none;
  border-color: #1890ff;
  box-shadow: 0 0 0 2px rgba(24, 144, 255, 0.2);
}

.chart {
  width: 100%;
  height: 400px;
  background-color: white;
  border-radius: 4px;
  border: 1px solid #e8e8e8;
}

@media (max-width: 768px) {
  .chart-header {
    flex-direction: column;
    align-items: flex-start;
  }

  .chart {
    height: 300px;
  }
}
</style>