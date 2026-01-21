"""
城市功能区动态分析平台 - 主程序
"""
import argparse
import logging
from pathlib import Path
import yaml

from src.data_preprocessing.poi_processor import POIProcessor
from src.data_preprocessing.raster_processor import RasterProcessor
from src.feature_engineering.spatial_features import SpatialFeatureExtractor
from src.models.clustering import FunctionZoneClustering
from src.visualization.map_visualizer import MapVisualizer

# 配置日志
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CityFunctionPlatform:
    """城市功能区分析平台主类"""
    
    def __init__(self, config_path="config/settings.yaml"):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # 初始化组件
        self.poi_processor = POIProcessor(self.config)
        self.raster_processor = RasterProcessor(self.config)
        self.feature_extractor = SpatialFeatureExtractor(self.config)
        self.clustering_model = FunctionZoneClustering(self.config)
        self.visualizer = MapVisualizer(self.config)
        
    def run_pipeline(self, region="guangzhou_tianhe"):
        """运行完整分析流程"""
        logger.info(f"开始分析区域: {region}")
        
        # 1. 数据预处理
        logger.info("步骤1: 数据预处理")
        poi_data = self.poi_processor.process(region)
        ndvi_data = self.raster_processor.calculate_ndvi(region)
        road_data = self.poi_processor.extract_road_network(region)
        
        # 2. 特征工程
        logger.info("步骤2: 特征工程")
        features = self.feature_extractor.extract_all_features(
            poi_data, ndvi_data, road_data
        )
        
        # 3. 功能区识别
        logger.info("步骤3: 功能区识别")
        zones = self.clustering_model.predict(features)
        
        # 4. 可视化
        logger.info("步骤4: 可视化展示")
        html_path = self.visualizer.create_interactive_map(
            zones, features, output_path=f"output/{region}_zones.html"
        )
        
        logger.info(f"分析完成! 结果保存至: {html_path}")
        return zones, html_path

def main():
    parser = argparse.ArgumentParser(description="城市功能区动态分析平台")
    parser.add_argument("--region", default="guangzhou_tianhe", 
                       help="分析区域")
    parser.add_argument("--config", default="config/settings.yaml",
                       help="配置文件路径")
    args = parser.parse_args()
    
    platform = CityFunctionPlatform(args.config)
    zones, html_path = platform.run_pipeline(args.region)
    
    # 启动Web服务预览结果
    import webbrowser
    webbrowser.open(f"file://{Path(html_path).absolute()}")

if __name__ == "__main__":
    main()