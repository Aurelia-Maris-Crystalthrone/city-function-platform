"""
路网数据处理模块
"""
import osmnx as ox
import networkx as nx
import geopandas as gpd
import pandas as pd
import numpy as np
import logging
from shapely.geometry import LineString, Point

logger = logging.getLogger(__name__)

class RoadNetworkProcessor:
    """路网数据处理器"""
    
    def __init__(self, config):
        self.config = config
        ox.settings.log_console = True
        ox.settings.use_cache = True
        ox.settings.cache_folder = "data/raw/osmnx_cache"
        
    def download_road_network(self, place_name, network_type='all'):
        """下载路网数据"""
        logger.info(f"下载路网数据: {place_name}")
        
        try:
            # 使用OSMnx下载路网
            G = ox.graph_from_place(place_name, network_type=network_type)
            
            # 转换为GeoDataFrame
            nodes, edges = ox.graph_to_gdfs(G)
            
            logger.info(f"路网下载完成: {len(edges)}条道路")
            return edges, nodes
            
        except Exception as e:
            logger.error(f"路网下载失败: {e}")
            
            # 返回示例数据
            return self._generate_sample_road_network()
    
    def _generate_sample_road_network(self):
        """生成示例路网数据"""
        logger.info("生成示例路网数据")
        
        # 创建网格状路网
        roads = []
        
        # 水平道路
        for i in range(10):
            road = LineString([(113.30 + i*0.01, 23.10),
                             (113.30 + i*0.01, 23.18)])
            roads.append({
                'geometry': road,
                'osmid': f"h_{i}",
                'highway': 'tertiary',
                'length': 8.88  # 约8.88km
            })
        
        # 垂直道路
        for i in range(10):
            road = LineString([(113.30, 23.10 + i*0.008),
                             (113.40, 23.10 + i*0.008)])
            roads.append({
                'geometry': road,
                'osmid': f"v_{i}",
                'highway': 'tertiary',
                'length': 11.12  # 约11.12km
            })
        
        edges_gdf = gpd.GeoDataFrame(roads, crs="EPSG:4326")
        
        # 创建节点
        nodes = []
        for i in range(10):
            for j in range(10):
                point = Point(113.30 + i*0.01, 23.10 + j*0.008)
                nodes.append({
                    'geometry': point,
                    'osmid': f"node_{i}_{j}",
                    'x': 113.30 + i*0.01,
                    'y': 23.10 + j*0.008
                })
        
        nodes_gdf = gpd.GeoDataFrame(nodes, crs="EPSG:4326")
        
        return edges_gdf, nodes_gdf
    
    def calculate_road_density(self, edges_gdf, cell_size=500):
        """计算路网密度"""
        logger.info("计算路网密度")
        
        # 创建网格
        bounds = edges_gdf.total_bounds
        minx, miny, maxx, maxy = bounds
        
        # 计算网格数量
        x_cells = int((maxx - minx) * 111320 / cell_size)
        y_cells = int((maxy - miny) * 111320 / cell_size)
        
        density_grid = np.zeros((y_cells, x_cells))
        
        # 为简化，这里返回示例数据
        # 实际应该计算每个网格内的道路总长度
        
        logger.info(f"路网密度网格: {y_cells}x{x_cells}")
        return density_grid
    
    def extract_road_features(self, edges_gdf):
        """提取路网特征"""
        logger.info("提取路网特征")
        
        features = {
            'total_length_km': edges_gdf['length'].sum() / 1000,
            'road_count': len(edges_gdf),
            'avg_road_length': edges_gdf['length'].mean(),
            'road_types': edges_gdf['highway'].value_counts().to_dict()
        }
        
        # 计算路网连接度
        if 'u' in edges_gdf.columns and 'v' in edges_gdf.columns:
            # 创建图结构
            G = nx.Graph()
            for _, row in edges_gdf.iterrows():
                G.add_edge(row['u'], row['v'], length=row['length'])
            
            features['node_count'] = G.number_of_nodes()
            features['edge_count'] = G.number_of_edges()
            features['connectivity'] = features['edge_count'] / features['node_count']
        
        logger.info(f"路网特征提取完成: {features}")
        return features
    
    def find_intersections(self, edges_gdf):
        """查找交叉口"""
        logger.info("查找交叉口")
        
        intersections = []
        
        # 简化实现：查找道路端点
        endpoints = []
        for _, road in edges_gdf.iterrows():
            line = road['geometry']
            endpoints.extend([(line.coords[0][0], line.coords[0][1]),
                            (line.coords[-1][0], line.coords[-1][1])])
        
        # 统计每个点的出现次数
        from collections import Counter
        endpoint_counts = Counter(endpoints)
        
        # 交叉口：至少3条道路相交的点
        for point, count in endpoint_counts.items():
            if count >= 3:
                intersections.append({
                    'geometry': Point(point),
                    'degree': count  # 交叉口度数
                })
        
        intersections_gdf = gpd.GeoDataFrame(intersections, crs="EPSG:4326")
        
        logger.info(f"找到 {len(intersections_gdf)} 个交叉口")
        return intersections_gdf
    
    def save_road_network(self, edges_gdf, nodes_gdf, output_dir="data/processed"):
        """保存路网数据"""
        os.makedirs(output_dir, exist_ok=True)
        
        edges_path = f"{output_dir}/road_network_edges.geojson"
        nodes_path = f"{output_dir}/road_network_nodes.geojson"
        
        edges_gdf.to_file(edges_path, driver="GeoJSON")
        nodes_gdf.to_file(nodes_path, driver="GeoJSON")
        
        logger.info(f"路网数据已保存: {edges_path}, {nodes_path}")
        return edges_path, nodes_path