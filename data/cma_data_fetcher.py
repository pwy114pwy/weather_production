import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os


class CMAMeteorologicalDataFetcher:
    """
    国家气象中心(CMA)数据获取器
    用于从中国气象局开放平台获取气象数据
    """

    def __init__(self):
        """
        初始化CMA数据获取器

        Args:
            api_key: 中国气象局开放平台API密钥
        """
        self.base_url = "http://api.data.cma.cn:8090/api"
        self.data_format = "json"

    def get_station_id(self, city_name):
        """
        根据城市名称获取气象站ID
        注：实际应用中需从CMA获取完整的气象站列表

        Args:
            city_name: 城市名称（如"北京"、"上海"）

        Returns:
            气象站ID
        """
        station_mapping = {
            "北京": "54511",  # 北京站
            "上海": "58362",  # 上海站
            "广州": "59287",  # 广州站
            "深圳": "59493",  # 深圳站
            "杭州": "58457",  # 杭州站
            "南京": "58238",  # 南京站
            "武汉": "57494",  # 武汉站
            "成都": "56294",  # 成都站
            "重庆": "57516",  # 重庆站
            "西安": "57036"   # 西安站
        }

        return station_mapping.get(city_name, "54511")  # 默认返回北京站

    def fetch_weather_data(self, city_name, hours=6):
        print(1)
        """
        获取指定城市过去hours小时的气象数据
        
        Args:
            city_name: 城市名称
            hours: 要获取的历史小时数
            
        Returns:
            包含气象数据的DataFrame
        """
        # 获取气象站ID
        station_id = self.get_station_id(city_name)

        # 以当前时间为基准
        now = datetime.now()

        # CMA 推荐：避免请求“最新1小时”，往前错开
        end_time = now - timedelta(days=1)
        start_time = now - timedelta(days=6)

        # CMA 要求的时间格式
        start_str = start_time.strftime("%Y%m%d%H%M%S")
        end_str = end_time.strftime("%Y%m%d%H%M%S")

        time_range = f"[{start_str},{end_str}]"

        # elements = "PRE_3h,RHU,WIN_S_Avg_2mi,PRS"
        elements = "PRE_1h,RHU,WIN_S_Avg_2mi,PRS"
        # 构建API请求参数
        params = {
            "userId": "768447306516RAbgV",  # 替换为实际用户ID
            "pwd": "27Qh1JL",  # 替换为实际密码
            "dataFormat": self.data_format,
            "interfaceId": "getSurfEleByTimeRangeAndStaID",
            "dataCode": "SURF_CHN_MUL_HOR_3H",
            "timeRange": time_range,
            "staIDs": station_id,
            "elements": elements,  # CMA API的气象要素代码
        }
        print(params)
        try:
            # 发送API请求
            response = requests.get(self.base_url, params=params, timeout=30)
            response.raise_for_status()

            # 解析响应数据
            data = response.json()
            # print(data)
            # 转换为DataFrame
            weather_records = []

            if "DS" in data and "SurfEle" in data["DS"]:
                for record in data["DS"]["SurfEle"]:
                    # 解析时间
                    obs_time_str = record.get("OBS_TIME") or record.get("datetime")
                    if obs_time_str:
                        obs_time = datetime.strptime(
                            obs_time_str, "%Y-%m-%d %H:%M:%S")

                        # 构建记录
                        weather_record = {
                            "时间": obs_time,
                            # 降水量(1小时)
                            "降雨量": float(record.get("PRE_1h", record.get("PRE", 0.0))),
                            # 相对湿度
                            "湿度": float(record.get("RHU_Hour", record.get("RHU", 0.0))),
                            # 2分钟平均风速
                            "风速": float(record.get("WIN_S_Avg_2mi", record.get("WIN_S", 0.0))),
                            # 海平面气压
                            "气压": float(record.get("PRS_Sea", record.get("PRS", 0.0))),
                            "洪涝风险标签": 0  # 初始化为无风险
                        }

                        weather_records.append(weather_record)

            # 创建DataFrame
            df = pd.DataFrame(weather_records)

            if not df.empty:
                # 设置时间索引并排序
                df.set_index("时间", inplace=True)
                df.sort_index(inplace=True)

            return df

        except requests.exceptions.RequestException as e:
            print(f"获取CMA数据失败: {str(e)}")
            # 如果API调用失败，返回模拟数据
            return self._get_mock_data(city_name, hours)

    def _get_mock_data(self, city_name, hours=6):
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
        start_time = end_time - timedelta(hours=hours)
        dates = pd.date_range(start=start_time, end=end_time, freq='h')

        # 生成模拟数据
        np.random.seed(42)
        n_samples = len(dates)

        rainfall = np.random.exponential(scale=5, size=n_samples)
        duration = np.random.randint(0, 60, size=n_samples)
        humidity = np.random.randint(50, 100, size=n_samples)
        wind_speed = np.random.uniform(0, 15, size=n_samples)
        pressure = np.random.uniform(990, 1020, size=n_samples)

        # 创建DataFrame
        df = pd.DataFrame({
            '时间': dates,
            '降雨量': rainfall,
            '持续时间': duration,
            '湿度': humidity,
            '风速': wind_speed,
            '气压': pressure,
            '洪涝风险标签': np.zeros(n_samples)
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
    api_key = "your_cma_api_key"  # 替换为实际API密钥
    fetcher = CMAMeteorologicalDataFetcher(api_key)

    # 获取北京过去6小时的气象数据
    df = fetcher.fetch_weather_data("北京", hours=6)

    # 保存原始数据
    fetcher.save_raw_data(
        df, "d:\\nature-disaster-prediction\\data\\raw\\cma_weather_data.csv")

    print("\n获取的气象数据:")
    print(df)
