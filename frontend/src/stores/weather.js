import { defineStore } from 'pinia'

export const useWeatherStore = defineStore('weather', {
  state: () => ({
    weatherData: [],
    rawWeatherData: [],
    weatherDataLoaded: false,
    dataSource: '',
    selectedCity: '',
    dataTimestamp: '',
    startTime: '',
    endTime: '',
    predictionResults: null,
    isPredicting: false
  }),

  getters: {
    isDataLoaded: (state) => state.weatherDataLoaded,
    getWeatherData: (state) => state.weatherData,
    getRawWeatherData: (state) => state.rawWeatherData,
    getDataSource: (state) => state.dataSource,
    getSelectedCity: (state) => state.selectedCity,
    getDataTimestamp: (state) => state.dataTimestamp,
    getStartTime: (state) => state.startTime,
    getEndTime: (state) => state.endTime,
    getPredictionResults: (state) => state.predictionResults,
    isPredicting: (state) => state.isPredicting
  },

  actions: {
    setWeatherData(data) {
      this.weatherData = data
      this.weatherDataLoaded = true
    },
    setRawWeatherData(data) {
      this.rawWeatherData = data
    },
    setDataSource(source) {
      this.dataSource = source
    },
    setSelectedCity(city) {
      this.selectedCity = city
    },
    setDataTimestamp(timestamp) {
      this.dataTimestamp = timestamp
    },
    setTimeRange(startTime, endTime) {
      this.startTime = startTime
      this.endTime = endTime
    },
    setPredictionResults(results) {
      this.predictionResults = results
    },
    setIsPredicting(isPredicting) {
      this.isPredicting = isPredicting
    },
    clearWeatherData() {
      this.weatherData = []
      this.rawWeatherData = []
      this.weatherDataLoaded = false
    },
    clearPredictionResults() {
      this.predictionResults = null
    }
  }
})
