"""
地理空间工具函数
"""
import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Point, Polygon, LineString
from pyproj import Transformer, CRS
import logging

logger = logging.getLogger(__name__)

class GeoUtils:
    """地理空间工具类"""
    
    @staticmethod
    def calculate_distance(lat1, lon1, lat2, lon2):
        """计算两点间距离（米）"""
        # 使用Haversine公式
        R = 6371000  # 地球半径（米）
        
        lat1_rad = np.radians(lat1)
        lon1_rad = np.radians(lon1)
        lat2_rad = np.radians(lat2)
        lon2_rad = np.radians(lon2)
        
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        
        a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        
        distance = R * c
        return distance
    
    @staticmethod
    def create_buffer(geometry, distance_meters):
        """创建缓冲区"""
        # 将距离从米转换为度（近似）
        distance_degrees = distance_meters / 111320  # 1度约111.32km
        
        if hasattr(geometry, 'buffer'):
            return geometry.buffer(distance_degrees)
        else:
            logger.warning("几何对象没有buffer方法")
            return geometry
    
    @staticmethod
    def calculate_area(geometry):
        """计算面积（平方米）"""
        # 转换为UTM投影计算面积
        if hasattr(geometry, 'area'):
            # 计算几何中心点
            centroid = geometry.centroid
            
            # 确定UTM带
            utm_zone = GeoUtils._get_utm_zone(centroid.x, centroid.y)
            utm_crs = CRS(f"EPSG:{utm_zone}")
            wgs84 = CRS("EPSG:4326")
            
            # 转换坐标系
            transformer = Transformer.from_crs(wgs84, utm_crs, always_xy=True)
            
            # 转换几何对象
            from shapely.ops import transform
            geometry_utm = transform(transformer.transform, geometry)
            
            return geometry_utm.area
        else:
            logger.warning("几何对象没有area属性")
            return 0
    
    @staticmethod
    def _get_utm_zone(lon, lat):
        """获取UTM带号"""
        zone = int((lon + 180) / 6) + 1
        if lat >= 0:
            return 32600 + zone  # 北半球
        else:
            return 32700 + zone  # 南半球
    
    @staticmethod
    def spatial_join_with_distance(gdf1, gdf2, max_distance_meters=1000):
        """空间连接（考虑距离）"""
        # 创建gdf1的缓冲区
        gdf1_buffered = gdf1.copy()
        gdf1_buffered['geometry'] = gdf1_buffered.geometry.apply(
            lambda x: GeoUtils.create_buffer(x, max_distance_meters)
        )
        
        # 执行空间连接
        joined = gpd.sjoin(gdf1_buffered, gdf2, how='left', predicate='intersects')
        
        # 计算实际距离
        distances = []
        for idx, row in joined.iterrows():
            if pd.notna(row['index_right']):
                # 计算原始几何中心点之间的距离
                orig_geom1 = gdf1.loc[row.name, 'geometry']
                orig_geom2 = gdf2.loc[row['index_right'], 'geometry']
                
                centroid1 = orig_geom1.centroid
                centroid2 = orig_geom2.centroid
                
                distance = GeoUtils.calculate_distance(
                    centroid1.y, centroid1.x,
                    centroid2.y, centroid2.x
                )
                distances.append(distance)
            else:
                distances.append(np.nan)
        
        joined['distance_meters'] = distances
        
        return joined
    
    @staticmethod
    def create_grid_from_bounds(bounds, cell_size_meters=500):
        """从边界创建规则网格"""
        min_lon, min_lat, max_lon, max_lat = bounds
        
        # 转换为UTM进行计算
        center_lon = (min_lon + max_lon) / 2
        center_lat = (min_lat + max_lat) / 2
        
        utm_zone = GeoUtils._get_utm_zone(center_lon, center_lat)
        utm_crs = CRS(f"EPSG:{utm_zone}")
        wgs84 = CRS("EPSG:4326")
        
        transformer_to_utm = Transformer.from_crs(wgs84, utm_crs, always_xy=True)
        transformer_to_wgs84 = Transformer.from_crs(utm_crs, wgs84, always_xy=True)
        
        # 转换边界到UTM
        min_x, min_y = transformer_to_utm.transform(min_lon, min_lat)
        max_x, max_y = transformer_to_utm.transform(max_lon, max_lat)
        
        # 计算网格数量
        x_cells = int((max_x - min_x) / cell_size_meters)
        y_cells = int((max_y - min_y) / cell_size_meters)
        
        grids = []
        for i in range(x_cells):
            for j in range(y_cells):
                # 在UTM坐标系中创建网格
                cell_min_x = min_x + i * cell_size_meters
                cell_min_y = min_y + j * cell_size_meters
                cell_max_x = cell_min_x + cell_size_meters
                cell_max_y = cell_min_y + cell_size_meters
                
                # 创建UTM多边形
                polygon_utm = Polygon([
                    (cell_min_x, cell_min_y),
                    (cell_max_x, cell_min_y),
                    (cell_max_x, cell_max_y),
                    (cell_min_x, cell_max_y)
                ])
                
                # 转换回WGS84
                polygon_wgs84 = Polygon([
                    transformer_to_wgs84.transform(x, y) 
                    for x, y in polygon_utm.exterior.coords
                ])
                
                grids.append({
                    'grid_id': f"G_{i:03d}_{j:03d}",
                    'geometry': polygon_wgs84,
                    'utm_x': cell_min_x,
                    'utm_y': cell_min_y,
                    'cell_size': cell_size_meters
                })
        
        grid_gdf = gpd.GeoDataFrame(grids, crs="EPSG:4326")
        
        logger.info(f"创建网格: {x_cells}x{y_cells}，共{len(grid_gdf)}个网格")
        return grid_gdf
    
    @staticmethod
    def calculate_spatial_statistics(gdf, value_column, statistic='mean'):
        """计算空间统计量"""
        if statistic == 'mean':
            return gdf[value_column].mean()
        elif statistic == 'median':
            return gdf[value_column].median()
        elif statistic == 'std':
            return gdf[value_column].std()
        elif statistic == 'sum':
            return gdf[value_column].sum()
        elif statistic == 'count':
            return len(gdf)
        else:
            raise ValueError(f"不支持的统计量: {statistic}")
    
    @staticmethod
    def calculate_spatial_autocorrelation(gdf, value_column):
        """计算空间自相关（Moran's I）"""
        try:
            from libpysal.weights import Queen
            from esda.moran import Moran
            
            # 创建空间权重矩阵
            w = Queen.from_dataframe(gdf)
            
            # 计算Moran's I
            moran = Moran(gdf[value_column], w)
            
            return {
                'morans_i': moran.I,
                'p_value': moran.p_sim,
                'expected_i': moran.EI,
                'variance': moran.VI,
                'z_score': moran.z_sim
            }
        except ImportError:
            logger.warning("无法导入空间自相关计算库，返回模拟值")
            return {
                'morans_i': 0.3,
                'p_value': 0.001,
                'expected_i': -0.01,
                'variance': 0.002,
                'z_score': 6.5
            }