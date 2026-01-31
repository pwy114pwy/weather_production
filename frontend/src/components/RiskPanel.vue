<template>
  <div class="risk-panel">
    <h3>风险预测系统</h3>

    <!-- 数据源选择 -->
    <div class="data-source-selector">
      <h4>选择数据源</h4>
      <div class="selector-group">
        <!-- <label>
          <input type="radio" v-model="selectedDataSource" value="cma" />
          国家气象中心(CMA)
        </label> -->
        <label>
          <input type="radio" v-model="selectedDataSource" value="noaa" />
          NOAA(美国国家海洋和大气管理局)
        </label>
      </div>
    </div>

    <!-- 城市选择 -->
    <div class="city-selector">
      <h4>选择城市</h4>
      <select v-model="selectedCity">
        <option value="纽约">纽约</option>
        <option value="新加坡">新加坡</option>
        <option value="哥伦比亚">哥伦比亚</option>
        <option value="夏威夷">夏威夷</option>
        <option value="孟买">孟买</option>
        <option value="北京">北京</option>
        <option value="上海">上海</option>
        <option value="广州">广州</option>
        <option value="深圳">深圳</option>
        <option value="杭州">杭州</option>
        <option value="南京">南京</option>
        <option value="武汉">武汉</option>
        <option value="成都">成都</option>
        <option value="重庆">重庆</option>
        <option value="西安">西安</option>
      </select>
    </div>

    <!-- 时间范围选择 -->
    <div class="time-range-selector">
      <h4>选择时间范围</h4>
      <div class="time-selector-group">
        <div class="time-input-wrapper">
          <label>开始时间：</label>
          <input type="datetime-local" v-model="startTime" @change="handleStartTimeChange" />
        </div>
        <div class="time-input-wrapper">
          <label>结束时间：</label>
          <input type="datetime-local" v-model="endTime" />
        </div>
      </div>
      <!-- <p class="hint">注：选择开始时间后，结束时间会自动设置为7天后</p> -->
    </div>

    <!-- API密钥输入 -->
    <div class="api-key-input">
      <h4>API密钥（可选）</h4>
      <input type="text" v-model="apiKey" placeholder="输入API密钥" />
      <p class="hint">注：留空将使用模拟数据</p>
    </div>

    <!-- 数据获取按钮 -->
    <div class="data-fetch-button">
      <button @click="fetchOfficialData" :disabled="isFetching">
        {{ isFetching ? '获取中...' : '获取官方数据' }}
      </button>
    </div>
  </div>
</template>

<script setup>
import axios from 'axios';
import { ref } from 'vue';
import { useWeatherStore } from '../stores/weather';

// 定义事件
const emit = defineEmits(['risk-updated', 'weather-data-loaded']);

// 获取weather store
const weatherStore = useWeatherStore();

// 响应式数据
const isFetching = ref(false);
const dataSourceInfo = ref('');

// 数据源选择
const selectedDataSource = ref('noaa'); // 默认使用CMA数据源

// 城市选择
const selectedCity = ref('纽约'); // 默认选择北京

// API密钥
const apiKey = ref(''); // API密钥，可选

// 时间范围
const startTime = ref('2024-01-01'); // 开始时间
const endTime = ref('2024-03-01'); // 结束时间

// 处理开始时间变化
const handleStartTimeChange = () => {
  // if (startTime.value) {
  //   const start = new Date(startTime.value);
  //   const end = new Date(start.getTime() + 2 * 30 * 24 * 60 * 60 * 1000);
  //   endTime.value = end.toISOString().slice(0, 16); // 格式化为YYYY-MM-DDTHH:MM
  // }
};


// 获取官方数据
const fetchOfficialData = async () => {
  try {
    isFetching.value = true;
    let apiUrl = '';
    let sourceName = '';

    // 根据选择的数据源构建API URL
    if (selectedDataSource.value === 'cma') {
      apiUrl = `http://localhost:8000/api/v1/cma_data`;
      sourceName = '国家气象中心(CMA)';
    } else if (selectedDataSource.value === 'noaa') {
      apiUrl = `http://localhost:8000/api/v1/noaa_data`;
      sourceName = 'NOAA(美国国家海洋和大气管理局)';
    }

    // 发送API请求
    const response = await axios.get(apiUrl, {
      params: {
        city_name: selectedCity.value,
        start_time: startTime.value.slice(0, 10),
        end_time: endTime.value.slice(0, 10),
        api_key: apiKey.value || 'sOUtGIPKTiBikrgafJueBImLZuVyvFGC'
      }
    });

    // 更新气象数据到Pinia store
    weatherStore.setWeatherData(response.data.normalized_data);
    weatherStore.setRawWeatherData(response.data.raw_data);
    weatherStore.setDataSource(sourceName);
    weatherStore.setSelectedCity(selectedCity.value);
    weatherStore.setDataTimestamp(response.data.data_timestamp);
    weatherStore.setTimeRange(startTime.value, endTime.value);

    // 更新数据源信息
    dataSourceInfo.value = `已获取${selectedCity.value}的${sourceName}最新数据 (${response.data.data_timestamp})`;
    console.log(weatherStore.getWeatherData)
    alert('数据获取成功！');

  } catch (error) {
    console.error('获取官方数据失败:', error);
    alert('获取官方数据失败，请检查API服务是否正常运行或API密钥是否正确');
  } finally {
    isFetching.value = false;
  }
};
</script>

<style scoped>
.risk-panel {
  padding: 20px;
  background-color: #f9f9f9;
  border-radius: 8px;
  box-shadow: 0 2px 12px rgba(0, 0, 0, 0.1);
  margin-top: 20px;
}

.risk-panel h3 {
  margin-top: 0;
  color: #333;
  text-align: center;
  border-bottom: 1px solid #e8e8e8;
  padding-bottom: 10px;
}

.risk-panel h4 {
  margin-top: 20px;
  margin-bottom: 10px;
  color: #555;
  font-size: 14px;
}

/* 数据源选择器 */
.data-source-selector,
.city-selector,
.api-key-input {
  background-color: white;
  padding: 15px;
  border-radius: 4px;
  margin-bottom: 15px;
  border: 1px solid #e8e8e8;
}

.selector-group {
  display: flex;
  gap: 15px;
  flex-wrap: wrap;
}

.selector-group label {
  display: flex;
  align-items: center;
  gap: 5px;
  cursor: pointer;
  font-size: 14px;
  color: #666;
}

/* 城市选择器 */
.city-selector select {
  width: 100%;
  padding: 8px 12px;
  border: 1px solid #d9d9d9;
  border-radius: 4px;
  font-size: 14px;
  background-color: white;
  cursor: pointer;
}

.city-selector select:focus {
  outline: none;
  border-color: #1890ff;
  box-shadow: 0 0 0 2px rgba(24, 144, 255, 0.2);
}

/* API密钥输入 */
.api-key-input input {
  width: 100%;
  padding: 8px 12px;
  border: 1px solid #d9d9d9;
  border-radius: 4px;
  font-size: 14px;
  margin-bottom: 5px;
}

.api-key-input input:focus {
  outline: none;
  border-color: #1890ff;
  box-shadow: 0 0 0 2px rgba(24, 144, 255, 0.2);
}

/* 时间范围选择器 */
.time-range-selector {
  background-color: white;
  padding: 15px;
  border-radius: 4px;
  margin-bottom: 15px;
  border: 1px solid #e8e8e8;
}

.time-selector-group {
  display: flex;
  gap: 15px;
  flex-wrap: wrap;
}

.time-input-wrapper {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 10px;
  flex: 1;
  min-width: 250px;
}

.time-input-wrapper label {
  font-size: 14px;
  color: #666;
  min-width: 80px;
}

.time-input-wrapper input[type="datetime-local"] {
  flex: 1;
  padding: 8px 12px;
  border: 1px solid #d9d9d9;
  border-radius: 4px;
  font-size: 14px;
  background-color: white;
  cursor: pointer;
}

.time-input-wrapper input[type="datetime-local"]:focus {
  outline: none;
  border-color: #1890ff;
  box-shadow: 0 0 0 2px rgba(24, 144, 255, 0.2);
}

.hint {
  font-size: 12px;
  color: #999;
  margin: 0;
}

/* 按钮样式 */
.data-fetch-button,
.prediction-button {
  text-align: center;
  margin: 15px 0;
}

button {
  padding: 10px 20px;
  font-size: 16px;
  background-color: #1890ff;
  color: white;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  transition: background-color 0.3s;
  width: 100%;
}

button:hover {
  background-color: #40a9ff;
}

button:disabled {
  background-color: #d9d9d9;
  cursor: not-allowed;
}

/* 风险结果 */
.risk-result {
  background-color: white;
  padding: 15px;
  border-radius: 4px;
  margin-bottom: 15px;
  border: 1px solid #e8e8e8;
}

.risk-result>div {
  margin-bottom: 15px;
  font-size: 16px;
}

.label {
  font-weight: bold;
  color: #333;
}

.value {
  margin-left: 10px;
  color: #666;
}

.risk-level .value {
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
  margin-top: 20px;
}

.risk-explanations h4 {
  margin-bottom: 10px;
  color: #333;
}

.risk-explanations ul {
  list-style-type: disc;
  padding-left: 20px;
  color: #666;
}

.risk-explanations li {
  margin-bottom: 5px;
}

.no-data {
  text-align: center;
  color: #999;
  padding: 40px 0;
  background-color: white;
  border-radius: 4px;
  border: 1px solid #e8e8e8;
}

/* 数据来源信息 */
.data-info {
  text-align: center;
  font-size: 12px;
  color: #999;
  margin-top: 15px;
  padding-top: 10px;
  border-top: 1px dashed #e8e8e8;
}
</style>