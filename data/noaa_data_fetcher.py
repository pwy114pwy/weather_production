import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os


class NOAAMeteorologicalDataFetcher:
    """
    NOAA(美国国家海洋和大气管理局)数据获取器
    用于从NOAA Climate Data API获取气象数据
    """

    def __init__(self, api_key="sOUtGIPKTiBikrgafJueBImLZuVyvFGC"):
        """
        初始化NOAA数据获取器

        Args:
            api_key: NOAA Climate Data API密钥
        """
        self.api_key = api_key
        self.base_url = "https://www.ncdc.noaa.gov/cdo-web/api/v2"
        self.data_format = "json"

    def generate_risk_labels(self, df):
        """根据气象特征生成洪涝风险标签

        Args:
            df: 包含气象数据的DataFrame

        Returns:
            包含风险标签的DataFrame
        """
        # 避免链式赋值问题，先复制DataFrame
        df = df.copy()

        # 创建风险标签列
        df['洪涝风险标签'] = 0

        # 生成风险标签的规则（可根据实际情况调整）
        for i in range(len(df)):
            # 综合考虑降雨量和持续时间
            if df['降雨量'].iloc[i] > 20 and df['持续时间'].iloc[i] > 30:
                df.loc[df.index[i], '洪涝风险标签'] = 2  # 高风险
            elif df['降雨量'].iloc[i] > 10 or (df['风速'].iloc[i] > 8 and df['降雨量'].iloc[i] > 5):
                df.loc[df.index[i], '洪涝风险标签'] = 1  # 中等风险
            else:
                df.loc[df.index[i], '洪涝风险标签'] = 0  # 无风险

        return df

    def get_station_id(self, city_name):
        """
        根据城市名称获取NOAA气象站ID
        注：实际应用中需从NOAA获取完整的气象站列表

        Args:
            city_name: 城市名称（如"Beijing"、"New York"）

        Returns:
            气象站ID
        """
        station_mapping = {
            "北京": "CHM00054511",  # 北京站
            "上海": "CHM00058362",  # 上海站
            "广州": "CHM00059287",  # 广州站
            "深圳": "CHM00059493",  # 深圳站
            "杭州": "CHM00058457",  # 杭州站
            "南京": "CHM00058238",  # 南京站
            "武汉": "CHM00057494",  # 武汉站
            "成都": "CHM00056294",  # 成都站
            "重庆": "CHM00057516",  # 重庆站
            "西安": "CHM00057036",  # 西安站
            "纽约": "USW00094728",  # 纽约站
            "伦敦": "UKM00003772",  # 伦敦站
            "东京": "JA000047759",  # 东京站
            "孟买": "INM00043003",   # 孟买站
            "夏威夷": "USW00022521",   # 夏威夷站
            "哥伦比亚": "COM00080370",   # 哥伦比亚站
            "新加坡": "SGM00061666"   # 新加坡站

        }

        return station_mapping.get(city_name, "CHM00054511")  # 默认返回北京站

    def fetch_weather_data(self, city_name, start_date, end_date):
        """        
        Args:
            city_name: 城市名称            
        Returns:
            包含气象数据的DataFrame
        """
        # 获取气象站ID
        station_id = self.get_station_id(city_name)

        # 构建API请求
        endpoint = f"{self.base_url}/data"
        headers = {
            "token": self.api_key,
            "Content-Type": "application/json"
        }

        # 计算日期范围
        start_date = start_date
        end_date = end_date

        # 格式化日期字符串（GHCND需要YYYY-MM-DD格式）
        # start_date_str = start_date.strftime("%Y-%m-%d")
        # end_date_str = end_date.strftime("%Y-%m-%d")

        params = {
            "datasetid": "GHCND",  # 全球历史气候网络每日数据集
            "stationid": f"GHCND:{station_id}",
            "startdate": start_date,
            "enddate": end_date,
            "datatypeid": ["PRCP", "TMIN", "TMAX", "AWND"],
            "limit": 1000,
            "units": "metric"
        }
        print(params)
        print(headers)
        try:
            # 发送API请求
            print(1)
            response = requests.get(
                endpoint, headers=headers, params=params, timeout=30)
            print(response.text)
            response.raise_for_status()

            # 解析响应数据
            data = response.json()
            print(data)
            # 转换为DataFrame
            if "results" in data:
                df = pd.DataFrame(data["results"])

                # 数据预处理
                df = self._process_noaa_data(df)
                df = self.generate_risk_labels(df)  # 生成风险标签
                return df
            else:
                print("NOAA API返回空结果")
                # 返回模拟数据
                return self._get_mock_data(city_name)

        except requests.exceptions.RequestException as e:
            print(f"获取NOAA数据失败: {str(e)}")
            # 返回模拟数据
            df = self._get_mock_data(city_name)
            return self.generate_risk_labels(df)  # 生成风险标签
        except ValueError as e:
            print(f"解析NOAA数据失败: {str(e)}")
            # 返回模拟数据
            df = self._get_mock_data(city_name)
            return self.generate_risk_labels(df)  # 生成风险标签

    def _process_noaa_data(self, df):
        """
        处理NOAA返回的原始数据

        Args:
            df: NOAA API返回的原始数据

        Returns:
            处理后的DataFrame
        """
        # 处理微量降水标记（T）
        for i, row in df.iterrows():
            if row["datatype"] == "PRCP" and row["attributes"] and "T" in row["attributes"]:
                # T表示微量降水，设置为0.1毫米
                df.at[i, "value"] = 0.1

        # 转换日期时间
        df["datetime"] = pd.to_datetime(df["date"])

        # 转换为宽格式
        df_wide = df.pivot(
            index="datetime", columns="datatype", values="value")

        # 重命名列以匹配系统需求
        column_mapping = {
            "PRCP": "降雨量",   # 降水量（毫米）
            "AWND": "风速",     # 平均风速（米/秒）
            "TMIN": "最低温度",  # 最低温度（摄氏度）
            "TMAX": "最高温度"   # 最高温度（摄氏度）
        }

        # 重命名列
        df_wide = df_wide.rename(columns=column_mapping)

        # 计算平均温度（如果没有TAVG数据）
        if "TAVG" in df.columns:
            # 如果原始数据中有TAVG，直接重命名
            df_wide = df_wide.rename(columns={"TAVG": "平均温度"})
        elif "最低温度" in df_wide.columns and "最高温度" in df_wide.columns:
            # 否则使用(TMIN + TMAX) / 2计算平均温度
            df_wide["平均温度"] = (df_wide["最低温度"] + df_wide["最高温度"]) / 2
            print("使用(TMIN + TMAX) / 2计算平均温度")
        else:
            # 如果都没有，使用默认值
            df_wide["平均温度"] = 20.0
            print("警告：无法计算平均温度，使用默认值20°C")

        # 计算降雨持续时间（简化处理）
        if "降雨量" in df_wide.columns:
            df_wide["持续时间"] = df_wide["降雨量"].apply(
                lambda x: 10 if x > 0 else 0)
        else:
            df_wide["降雨量"] = 0
            df_wide["持续时间"] = 0

        # 检查风速列是否存在，否则设置默认值
        if "风速" not in df_wide.columns:
            df_wide["风速"] = 0

        # 检查温度列是否存在，否则设置默认值
        if "最低温度" not in df_wide.columns:
            df_wide["最低温度"] = 20.0
        if "最高温度" not in df_wide.columns:
            df_wide["最高温度"] = 20.0
        if "平均温度" not in df_wide.columns:
            df_wide["平均温度"] = 20.0

        # 向前填充和向后填充处理缺失值
        df_wide = df_wide.ffill()
        df_wide = df_wide.bfill()
        df_wide = df_wide.fillna(0)

        # 只保留需要的列（使用NOAA返回的5个特征）
        required_columns = ["降雨量", "持续时间", "风速", "最低温度", "最高温度", "平均温度"]
        df_wide = df_wide[required_columns]

        # 设置时间索引
        df_wide.index.name = "时间"

        return df_wide

    def _get_mock_data(self, city_name):
        """
        生成模拟数据（当API调用失败时使用）

        Args:
            city_name: 城市名称
            hours: 要生成的历史小时数

        Returns:
            包含模拟气象数据的DataFrame
        """
        # 生成时间序列
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=6)
        dates = pd.date_range(start=start_time, end=end_time, freq='h')

        # 生成模拟数据
        np.random.seed(42)
        n_samples = len(dates)

        rainfall = np.random.exponential(scale=5, size=n_samples)
        duration = np.random.randint(0, 60, size=n_samples)
        wind_speed = np.random.uniform(0, 15, size=n_samples)
        avg_temp = np.random.uniform(5, 35, size=n_samples)  # 平均温度
        min_temp = avg_temp - np.random.uniform(2, 8, size=n_samples)  # 最低温度
        max_temp = avg_temp + np.random.uniform(2, 8, size=n_samples)  # 最高温度

        # 创建DataFrame
        df = pd.DataFrame({
            '时间': dates,
            '降雨量': rainfall,
            '持续时间': duration,
            '风速': wind_speed,
            '平均温度': avg_temp,
            '最低温度': min_temp,
            '最高温度': max_temp
        })

        df.set_index('时间', inplace=True)

        return df

    def save_raw_data(self, df, file_path):
        """
        保存原始气象数据到CSV文件

        Args:
            df: 包含气象数据的DataFrame
            file_path: 保存路径
        """
        # 创建目录（如果不存在）
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        # 保存到CSV
        df.to_csv(file_path)
        print(f"原始数据已保存到: {file_path}")


# 示例用法
if __name__ == "__main__":
    # 初始化数据获取器
    api_key = "sOUtGIPKTiBikrgafJueBImLZuVyvFGC"  # 替换为实际API密钥
    fetcher = NOAAMeteorologicalDataFetcher(api_key)
    start_time = '2024-01-01'
    end_time = '2024-09-01'

    # 获取数据
    df = fetcher.fetch_weather_data("纽约", start_time, end_time)

    # 保存原始数据（使用双反斜杠或原始字符串）
    fetcher.save_raw_data(
        df, "d:\\nature-disaster-prediction\\data\\raw\\noaa_weather_data.csv")

    print("\n获取的气象数据:")
    print(df.head())
