"""
POI数据处理模块
"""
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import requests
import logging

logger = logging.getLogger(__name__)

class POIProcessor:
    """POI数据处理器"""
    
    def __init__(self, config):
        self.config = config
        self.api_key = config.get("api_keys", {}).get("gaode")
        
    def fetch_poi_data(self, region, poi_types=None):
        """从高德API获取POI数据"""
        if poi_types is None:
            poi_types = self.config["poi"]["categories"]
        
        base_url = "https://restapi.amap.com/v3/place/text"
        
        all_pois = []
        for poi_type in poi_types:
            params = {
                "key": self.api_key,
                "keywords": "",
                "types": poi_type,
                "city": region,
                "offset": 50,
                "page": 1,
                "extensions": "all"
            }
            
            try:
                response = requests.get(base_url, params=params)
                data = response.json()
                
                if data["status"] == "1":
                    for poi in data["pois"]:
                        all_pois.append({
                            "id": poi["id"],
                            "name": poi["name"],
                            "type": poi["type"],
                            "type_code": poi["typecode"],
                            "address": poi["address"],
                            "lon": float(poi["location"].split(",")[0]),
                            "lat": float(poi["location"].split(",")[1]),
                            "geometry": Point(float(poi["location"].split(",")[0]), 
                                            float(poi["location"].split(",")[1]))
                        })
            except Exception as e:
                logger.error(f"获取POI类型 {poi_type} 失败: {e}")
        
        return gpd.GeoDataFrame(all_pois, crs="EPSG:4326")
    
    def clean_poi_data(self, poi_gdf):
        """清洗POI数据"""
        # 去除重复
        poi_gdf = poi_gdf.drop_duplicates(subset=["id"])
        
        # 去除无效坐标
        poi_gdf = poi_gdf[
            (poi_gdf["lon"] >= self.config["region"]["bounds"]["min_lon"]) &
            (poi_gdf["lon"] <= self.config["region"]["bounds"]["max_lon"]) &
            (poi_gdf["lat"] >= self.config["region"]["bounds"]["min_lat"]) &
            (poi_gdf["lat"] <= self.config["region"]["bounds"]["max_lat"])
        ]
        
        # 分类编码
        poi_type_mapping = self.config["poi"]["type_mapping"]
        poi_gdf["category"] = poi_gdf["type_code"].map(poi_type_mapping)
        
        return poi_gdf
    
    def process(self, region):
        """处理POI数据完整流程"""
        logger.info(f"开始处理{region}的POI数据")
        
        # 获取数据
        raw_pois = self.fetch_poi_data(region)
        
        # 清洗数据
        clean_pois = self.clean_poi_data(raw_pois)
        
        # 保存数据
        output_path = f"data/processed/cleaned/{region}_pois.geojson"
        clean_pois.to_file(output_path, driver="GeoJSON")
        
        logger.info(f"POI数据处理完成，共{len(clean_pois)}条记录")
        return clean_pois