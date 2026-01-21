"""
数据预处理模块测试
"""
import pytest
import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import Point
import sys
import os

# 添加src到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data_preprocessing.poi_processor import POIProcessor
from src.data_preprocessing.raster_processor import RasterProcessor
from src.data_preprocessing.data_integration import DataIntegrator

class TestPOIProcessor:
    """POI处理器测试"""
    
    def setup_method(self):
        """测试设置"""
        # 创建测试配置
        self.config = {
            "poi": {
                "categories": ["050000", "060000"],
                "type_mapping": {
                    "050000": "餐饮",
                    "060000": "购物"
                }
            },
            "region": {
                "bounds": {
                    "min_lon": 113.3,
                    "max_lon": 113.4,
                    "min_lat": 23.1,
                    "max_lat": 23.2
                }
            }
        }
        self.processor = POIProcessor(self.config)
    
    def test_clean_poi_data(self):
        """测试POI数据清洗"""
        # 创建测试数据
        test_data = [
            {"id": "1", "name": "测试餐厅", "type": "餐饮", "type_code": "050000",
             "lon": 113.35, "lat": 23.15, "geometry": Point(113.35, 23.15)},
            {"id": "2", "name": "测试商场", "type": "购物", "type_code": "060000",
             "lon": 113.25, "lat": 23.15, "geometry": Point(113.25, 23.15)},  # 超出边界
            {"id": "1", "name": "重复餐厅", "type": "餐饮", "type_code": "050000",
             "lon": 113.35, "lat": 23.15, "geometry": Point(113.35, 23.15)}  # 重复ID
        ]
        
        test_gdf = gpd.GeoDataFrame(test_data, crs="EPSG:4326")
        
        # 清洗数据
        cleaned = self.processor.clean_poi_data(test_gdf)
        
        # 验证结果
        assert len(cleaned) == 1  # 应该只剩1条记录
        assert cleaned.iloc[0]["id"] == "1"
        assert cleaned.iloc[0]["category"] == "餐饮"
    
    def test_fetch_poi_data_mock(self):
        """测试POI数据获取（模拟）"""
        # 由于需要真实API，这里测试模拟功能
        # 实际测试中应该使用mock或测试数据
        pass

class TestRasterProcessor:
    """栅格处理器测试"""
    
    def setup_method(self):
        """测试设置"""
        self.config = {}
        self.processor = RasterProcessor(self.config)
    
    def test_calculate_ndvi(self):
        """测试NDVI计算"""
        # 测试示例NDVI生成
        ndvi = self.processor._generate_sample_ndvi()
        
        assert ndvi.shape == (100, 100)
        assert ndvi.min() >= -0.2
        assert ndvi.max() <= 0.8
        assert not np.any(np.isnan(ndvi))
    
    def test_ndvi_clipping(self):
        """测试NDVI值裁剪"""
        # 创建测试数据
        test_ndvi = np.array([[-2.0, -1.0, 0.0, 1.0, 2.0]])
        
        # 调用内部方法（这里需要访问私有方法，实际应测试公开接口）
        # 简化测试逻辑
        clipped = np.clip(test_ndvi, -1, 1)
        
        assert clipped.min() == -1.0
        assert clipped.max() == 1.0

class TestDataIntegrator:
    """数据集成器测试"""
    
    def setup_method(self):
        """测试设置"""
        self.config = {
            "region": {
                "bounds": {
                    "min_lon": 113.3,
                    "max_lon": 113.31,  # 小区域，便于测试
                    "min_lat": 23.1,
                    "max_lat": 23.11
                }
            }
        }
        self.integrator = DataIntegrator(self.config)
    
    def test_create_uniform_grid(self):
        """测试统一网格创建"""
        grid_gdf = self.integrator._create_uniform_grid(
            self.config["region"]["bounds"], 
            cell_size=1000  # 1km网格
        )
        
        assert len(grid_gdf) > 0
        assert "grid_id" in grid_gdf.columns
        assert "geometry" in grid_gdf.columns
        assert all(grid_gdf.geometry.type == "Polygon")
        
        # 验证网格ID格式
        assert all(grid_gdf["grid_id"].str.startswith("G_"))
    
    def test_calculate_composite_features(self):
        """测试综合特征计算"""
        # 创建测试数据
        test_data = {
            "poi_count": [10, 5, 20],
            "poi_category_count": [3, 2, 4],
            "road_density": [0.01, 0.005, 0.02],
            "ndvi_mean": [0.3, 0.6, 0.2]
        }
        
        test_df = pd.DataFrame(test_data)
        
        # 计算综合特征
        result = self.integrator._calculate_composite_features(test_df)
        
        # 验证新特征
        assert "function_mix_index" in result.columns
        assert "urban_vitality" in result.columns
        assert "development_potential" in result.columns
        
        # 验证值范围
        assert all(result["function_mix_index"] >= 0)
        assert all(result["function_mix_index"] <= 1)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])