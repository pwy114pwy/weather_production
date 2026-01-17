<script setup>
import { ref } from 'vue'
import RiskPanel from './components/RiskPanel.vue'
import TrendChart from './components/TrendChart.vue'
import PredictionPanel from './components/PredictionPanel.vue'

</script>

<template>
  <div class="app-container">
    <div class="app-header">
      <h1>城市暴雨洪涝风险预测系统</h1>
    </div>
    <div class="app-main">
      <div class="left-section">
        <TrendChart />
        <PredictionPanel />
      </div>
      <div class="right-section">
        <RiskPanel />
      </div>
    </div>
    <div class="app-footer">
      <p>© 2026 城市暴雨洪涝风险预测系统</p>
    </div>
  </div>
</template>

<style scoped>
.app-container {
  display: flex;
  flex-direction: column;
  min-height: 100vh;
}

.app-header {
  background-color: #1890ff;
  color: white;
  padding: 20px;
  text-align: center;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
}

.app-header h1 {
  margin: 0;
  font-size: 24px;
}

.app-main {
  display: flex;
  flex: 1;
  padding: 20px;
  gap: 20px;
  flex-wrap: wrap;
}

.left-section {
  flex: 1;
  min-width: 600px;
  display: flex;
  flex-direction: column;
  gap: 20px;
}

.right-section {
  width: 350px;
  min-width: 300px;
}

.app-footer {
  background-color: #f0f0f0;
  text-align: center;
  padding: 10px;
  color: #666;
  font-size: 14px;
}

@media (max-width: 768px) {
  .app-main {
    flex-direction: column;
  }

  .left-section {
    min-width: auto;
  }

  .right-section {
    width: 100%;
  }
}
</style>