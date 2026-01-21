"""
空间特征提取模块
"""
import numpy as np
import geopandas as gpd
import pandas as pd
from shapely.geometry import Polygon
import logging

logger = logging.getLogger(__name__)

class SpatialFeatureExtractor:
    """空间特征提取器"""
    
    def __init__(self, config):
        self.config = config
        self.cell_size = config["features"].get("grid_size", 500)  # 500米网格
        
    def create_grid(self, bounds):
        """创建分析网格"""
        min_lon, min_lat, max_lon, max_lat = bounds
        
        # 计算网格数量
        lon_cells = int((max_lon - min_lon) * 111320 / self.cell_size)  # 1度≈111.32km
        lat_cells = int((max_lat - min_lat) * 111320 / self.cell_size)
        
        grids = []
        for i in range(lon_cells):
            for j in range(lat_cells):
                cell_min_lon = min_lon + i * (self.cell_size / 111320)
                cell_min_lat = min_lat + j * (self.cell_size / 111320)
                cell_max_lon = cell_min_lon + (self.cell_size / 111320)
                cell_max_lat = cell_min_lat + (self.cell_size / 111320)
                
                polygon = Polygon([
                    (cell_min_lon, cell_min_lat),
                    (cell_max_lon, cell_min_lat),
                    (cell_max_lon, cell_max_lat),
                    (cell_min_lon, cell_max_lat)
                ])
                
                grids.append({
                    "grid_id": f"G{i:03d}_{j:03d}",
                    "geometry": polygon
                })
        
        return gpd.GeoDataFrame(grids, crs="EPSG:4326")
    
    def extract_poi_features(self, grid_gdf, poi_gdf):
        """提取POI相关特征"""
        # 空间连接
        joined = gpd.sjoin(grid_gdf, poi_gdf, how="left", predicate="contains")
        
        # 按网格聚合
        features = []
        for grid_id in grid_gdf["grid_id"]:
            grid_pois = joined[joined["grid_id"] == grid_id]
            
            # 基本统计
            total_pois = len(grid_pois)
            
            # 类别分布
            if total_pois > 0:
                category_counts = grid_pois["category"].value_counts().to_dict()
                
                # 多样性指数（香农熵）
                proportions = np.array(list(category_counts.values())) / total_pois
                diversity = -np.sum(proportions * np.log(proportions + 1e-10))
            else:
                category_counts = {}
                diversity = 0
            
            features.append({
                "grid_id": grid_id,
                "poi_count": total_pois,
                "poi_diversity": diversity,
                **{f"poi_{cat}": category_counts.get(cat, 0) 
                   for cat in ["餐饮", "购物", "交通", "教育", "医疗", "办公"]}
            })
        
        return pd.DataFrame(features)
    
    def extract_all_features(self, poi_data, ndvi_data, road_data):
        """提取所有特征"""
        logger.info("开始提取空间特征")
        
        # 创建网格
        bounds = self.config["region"]["bounds"].values()
        grid_gdf = self.create_grid(bounds)
        
        # 提取POI特征
        poi_features = self.extract_poi_features(grid_gdf, poi_data)
        
        # 提取NDVI特征（简化示例）
        ndvi_features = self._extract_ndvi_features(grid_gdf, ndvi_data)
        
        # 提取路网特征
        road_features = self._extract_road_features(grid_gdf, road_data)
        
        # 合并所有特征
        features_df = grid_gdf.merge(poi_features, on="grid_id")
        features_df = features_df.merge(ndvi_features, on="grid_id", how="left")
        features_df = features_df.merge(road_features, on="grid_id", how="left")
        
        # 填充缺失值
        features_df = features_df.fillna(0)
        
        logger.info(f"特征提取完成，共{len(features_df)}个网格")
        return features_df
    
    def _extract_ndvi_features(self, grid_gdf, ndvi_data):
        """提取NDVI特征"""
        # 简化实现 - 实际需要栅格数据处理
        features = []
        for _, row in grid_gdf.iterrows():
            # 这里简化处理，实际应该计算网格内的NDVI统计值
            features.append({
                "grid_id": row["grid_id"],
                "ndvi_mean": np.random.uniform(0, 0.8),  # 示例
                "ndvi_std": np.random.uniform(0, 0.1)
            })
        return pd.DataFrame(features)
    
    def _extract_road_features(self, grid_gdf, road_data):
        """提取路网特征"""
        features = []
        for _, row in grid_gdf.iterrows():
            # 计算路网密度
            intersection_count = np.random.randint(0, 5)  # 示例
            road_length = np.random.uniform(0, 1000)  # 示例
            
            features.append({
                "grid_id": row["grid_id"],
                "road_density": road_length / (self.cell_size ** 2),
                "intersection_count": intersection_count
            })
        return pd.DataFrame(features)