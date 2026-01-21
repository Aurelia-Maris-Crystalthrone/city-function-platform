"""
多源数据融合模块
"""
import geopandas as gpd
import pandas as pd
import numpy as np
import logging
from shapely.geometry import Polygon
from pyproj import CRS, Transformer

logger = logging.getLogger(__name__)

class DataIntegrator:
    """多源数据融合器"""
    
    def __init__(self, config):
        self.config = config
        
    def integrate_spatial_data(self, poi_data, road_data, ndvi_data):
        """融合空间数据到统一网格"""
        logger.info("开始多源数据融合")
        
        # 创建统一网格
        bounds = self.config["region"]["bounds"]
        grid_gdf = self._create_uniform_grid(bounds)
        
        # 融合POI数据
        grid_gdf = self._integrate_poi_to_grid(grid_gdf, poi_data)
        
        # 融合路网数据
        grid_gdf = self._integrate_road_to_grid(grid_gdf, road_data)
        
        # 融合NDVI数据
        grid_gdf = self._integrate_raster_to_grid(grid_gdf, ndvi_data)
        
        # 计算综合特征
        grid_gdf = self._calculate_composite_features(grid_gdf)
        
        logger.info(f"数据融合完成，网格数量: {len(grid_gdf)}")
        return grid_gdf
    
    def _create_uniform_grid(self, bounds, cell_size=500):
        """创建统一网格"""
        min_lon, min_lat = bounds["min_lon"], bounds["min_lat"]
        max_lon, max_lat = bounds["max_lon"], bounds["max_lat"]
        
        # 转换为投影坐标系进行计算
        wgs84 = CRS("EPSG:4326")
        utm_zone = self._get_utm_zone((min_lon + max_lon) / 2, (min_lat + max_lat) / 2)
        utm_crs = CRS(f"EPSG:{utm_zone}")
        
        transformer = Transformer.from_crs(wgs84, utm_crs, always_xy=True)
        
        # 转换边界
        min_x, min_y = transformer.transform(min_lon, min_lat)
        max_x, max_y = transformer.transform(max_lon, max_lat)
        
        # 计算网格数量
        x_cells = int((max_x - min_x) / cell_size)
        y_cells = int((max_y - min_y) / cell_size)
        
        grids = []
        for i in range(x_cells):
            for j in range(y_cells):
                # 在UTM坐标系中创建网格
                cell_min_x = min_x + i * cell_size
                cell_min_y = min_y + j * cell_size
                cell_max_x = cell_min_x + cell_size
                cell_max_y = cell_min_y + cell_size
                
                # 创建多边形（在UTM坐标系中）
                polygon_utm = Polygon([
                    (cell_min_x, cell_min_y),
                    (cell_max_x, cell_min_y),
                    (cell_max_x, cell_max_y),
                    (cell_min_x, cell_max_y)
                ])
                
                # 转换回WGS84
                transformer_rev = Transformer.from_crs(utm_crs, wgs84, always_xy=True)
                polygon_wgs84 = self._transform_polygon(polygon_utm, transformer_rev)
                
                grids.append({
                    "grid_id": f"G_{i:03d}_{j:03d}",
                    "geometry": polygon_wgs84,
                    "cell_size": cell_size,
                    "utm_x": cell_min_x,
                    "utm_y": cell_min_y
                })
        
        grid_gdf = gpd.GeoDataFrame(grids, crs="EPSG:4326")
        return grid_gdf
    
    def _get_utm_zone(self, lon, lat):
        """计算UTM带号"""
        zone = int((lon + 180) / 6) + 1
        if lat >= 0:
            return 32600 + zone  # 北半球
        else:
            return 32700 + zone  # 南半球
    
    def _transform_polygon(self, polygon, transformer):
        """转换多边形坐标系"""
        coords = list(polygon.exterior.coords)
        transformed_coords = [transformer.transform(x, y) for x, y in coords]
        return Polygon(transformed_coords)
    
    def _integrate_poi_to_grid(self, grid_gdf, poi_gdf):
        """将POI数据融合到网格"""
        # 空间连接
        poi_in_grid = gpd.sjoin(poi_gdf, grid_gdf, how="left", predicate="within")
        
        # 按网格聚合
        poi_stats = poi_in_grid.groupby("grid_id").agg({
            "category": ["count", lambda x: x.nunique()],
            "name": "count"
        }).reset_index()
        
        poi_stats.columns = ["grid_id", "poi_count", "poi_category_count", "total_pois"]
        
        # 合并到网格
        grid_gdf = grid_gdf.merge(poi_stats, on="grid_id", how="left")
        
        # 填充缺失值
        grid_gdf["poi_count"] = grid_gdf["poi_count"].fillna(0)
        grid_gdf["poi_category_count"] = grid_gdf["poi_category_count"].fillna(0)
        
        return grid_gdf
    
    def _integrate_road_to_grid(self, grid_gdf, road_edges):
        """将路网数据融合到网格"""
        if road_edges is None or len(road_edges) == 0:
            grid_gdf["road_length"] = 0
            grid_gdf["road_density"] = 0
            return grid_gdf
        
        # 计算每个网格内的道路长度
        road_lengths = []
        
        for _, grid in grid_gdf.iterrows():
            # 查找与网格相交的道路
            intersected_roads = road_edges[road_edges.intersects(grid["geometry"])]
            
            if len(intersected_roads) > 0:
                # 计算相交部分的长度
                total_length = 0
                for _, road in intersected_roads.iterrows():
                    intersection = road["geometry"].intersection(grid["geometry"])
                    if not intersection.is_empty:
                        total_length += intersection.length
                
                road_lengths.append(total_length)
            else:
                road_lengths.append(0)
        
        grid_gdf["road_length"] = road_lengths
        grid_gdf["road_density"] = grid_gdf["road_length"] / (500 * 500)  # 500m网格
        
        return grid_gdf
    
    def _integrate_raster_to_grid(self, grid_gdf, ndvi_data):
        """将栅格数据融合到网格"""
        if ndvi_data is None:
            grid_gdf["ndvi_mean"] = np.random.uniform(0, 0.8, len(grid_gdf))
            grid_gdf["ndvi_std"] = np.random.uniform(0, 0.2, len(grid_gdf))
            return grid_gdf
        
        # 简化实现：为每个网格分配随机NDVI值
        # 实际实现应该从栅格数据中提取
        
        ndvi_means = []
        ndvi_stds = []
        
        for _, grid in grid_gdf.iterrows():
            # 示例：基于位置生成NDVI值
            centroid = grid["geometry"].centroid
            distance_to_center = np.sqrt((centroid.x - 113.35)**2 + (centroid.y - 23.14)**2)
            
            # 中心区域NDVI较低，边缘较高
            ndvi_mean = 0.7 - distance_to_center * 2
            ndvi_mean = max(0, min(0.8, ndvi_mean))
            
            ndvi_means.append(ndvi_mean)
            ndvi_stds.append(np.random.uniform(0, 0.1))
        
        grid_gdf["ndvi_mean"] = ndvi_means
        grid_gdf["ndvi_std"] = ndvi_stds
        
        return grid_gdf
    
    def _calculate_composite_features(self, grid_gdf):
        """计算综合特征"""
        # 功能混合度指数
        if "poi_category_count" in grid_gdf.columns:
            grid_gdf["function_mix_index"] = grid_gdf["poi_category_count"] / \
                                            grid_gdf["poi_count"].replace(0, 1)
        else:
            grid_gdf["function_mix_index"] = 0
        
        # 城市活力指数（简化）
        grid_gdf["urban_vitality"] = (
            grid_gdf["poi_count"] * 0.5 +
            grid_gdf["road_density"] * 0.3 +
            grid_gdf["ndvi_mean"] * 0.2
        )
        
        # 功能区潜力评分
        grid_gdf["development_potential"] = (
            (grid_gdf["poi_count"] / grid_gdf["poi_count"].max() if grid_gdf["poi_count"].max() > 0 else 0) * 0.4 +
            grid_gdf["road_density"] * 0.4 +
            (1 - grid_gdf["ndvi_mean"]) * 0.2  # 植被少的地方开发潜力高
        )
        
        # 标准化
        for col in ["urban_vitality", "development_potential"]:
            if grid_gdf[col].max() > grid_gdf[col].min():
                grid_gdf[f"{col}_norm"] = (grid_gdf[col] - grid_gdf[col].min()) / \
                                         (grid_gdf[col].max() - grid_gdf[col].min())
            else:
                grid_gdf[f"{col}_norm"] = 0.5
        
        return grid_gdf
    
    def save_integrated_data(self, integrated_gdf, output_path="data/processed/integrated_data.geojson"):
        """保存融合数据"""
        integrated_gdf.to_file(output_path, driver="GeoJSON")
        logger.info(f"融合数据已保存: {output_path}")
        
        # 同时保存为CSV用于分析
        csv_path = output_path.replace(".geojson", ".csv")
        integrated_gdf.drop(columns=['geometry']).to_csv(csv_path, index=False)
        
        logger.info(f"CSV格式数据已保存: {csv_path}")
        return output_path, csv_path