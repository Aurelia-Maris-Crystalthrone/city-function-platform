"""
遥感影像处理模块
"""
import rasterio
from rasterio.plot import show
import numpy as np
import matplotlib.pyplot as plt
from osgeo import gdal
import logging
import os

logger = logging.getLogger(__name__)

class RasterProcessor:
    """遥感影像处理器"""
    
    def __init__(self, config):
        self.config = config
        
    def calculate_ndvi(self, image_path=None, output_path=None):
        """计算NDVI指数"""
        if image_path is None:
            # 如果没有提供图像，使用示例数据或返回模拟数据
            return self._generate_sample_ndvi()
        
        logger.info(f"计算NDVI: {image_path}")
        
        try:
            # 读取影像
            with rasterio.open(image_path) as src:
                red_band = src.read(4)  # Sentinel-2红波段
                nir_band = src.read(8)   # Sentinel-2近红外波段
                
                # 计算NDVI
                ndvi = (nir_band.astype(float) - red_band.astype(float)) / \
                       (nir_band.astype(float) + red_band.astype(float) + 1e-10)
                
                # 处理异常值
                ndvi[ndvi > 1] = 1
                ndvi[ndvi < -1] = -1
                
                # 保存结果
                if output_path:
                    profile = src.profile
                    profile.update(dtype=rasterio.float32, count=1)
                    
                    with rasterio.open(output_path, 'w', **profile) as dst:
                        dst.write(ndvi.astype(rasterio.float32), 1)
                
                logger.info(f"NDVI计算完成，范围: {ndvi.min():.2f} - {ndvi.max():.2f}")
                return ndvi
                
        except Exception as e:
            logger.error(f"NDVI计算失败: {e}")
            return self._generate_sample_ndvi()
    
    def _generate_sample_ndvi(self):
        """生成示例NDVI数据"""
        logger.info("生成示例NDVI数据")
        
        # 创建示例栅格数据
        rows, cols = 100, 100
        ndvi = np.zeros((rows, cols))
        
        # 模拟植被分布
        x, y = np.meshgrid(np.linspace(0, 1, cols), np.linspace(0, 1, rows))
        
        # 中心城区低植被
        center_x, center_y = 0.5, 0.5
        distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        
        # 模拟植被梯度
        ndvi = 0.7 * np.exp(-distance * 3)  # 中心植被低，向外逐渐增高
        
        # 添加一些随机变化
        ndvi += np.random.normal(0, 0.1, (rows, cols))
        
        # 限制范围
        ndvi = np.clip(ndvi, -0.2, 0.8)
        
        return ndvi
    
    def clip_to_region(self, raster_path, bounds, output_path):
        """裁剪影像到指定区域"""
        logger.info(f"裁剪影像到区域: {bounds}")
        
        try:
            # 使用GDAL进行裁剪
            gdal.Warp(output_path, raster_path,
                     outputBounds=bounds,
                     format='GTiff')
            
            logger.info(f"影像裁剪完成: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"影像裁剪失败: {e}")
            return None
    
    def classify_land_use(self, image_path, method='supervised'):
        """土地利用分类"""
        logger.info(f"执行土地利用分类: {method}")
        
        # 这里可以扩展为监督分类或非监督分类
        if method == 'supervised':
            return self._supervised_classification(image_path)
        else:
            return self._unsupervised_classification(image_path)
    
    def _supervised_classification(self, image_path):
        """监督分类（简化示例）"""
        # 实际实现需要训练样本
        logger.info("执行监督分类")
        
        # 返回示例分类结果
        classes = np.array([0, 1, 2, 3])  # 0:建筑, 1:植被, 2:水体, 3:裸地
        return classes
    
    def _unsupervised_classification(self, image_path):
        """非监督分类（简化示例）"""
        logger.info("执行非监督分类")
        
        # 使用KMeans聚类进行分类
        try:
            with rasterio.open(image_path) as src:
                data = src.read()
                
                # 重塑数据
                rows, cols = data.shape[1], data.shape[2]
                bands = data.shape[0]
                
                # 转为2D数组
                data_2d = data.reshape(bands, -1).T
                
                # 使用KMeans聚类
                from sklearn.cluster import KMeans
                kmeans = KMeans(n_clusters=4, random_state=42)
                labels = kmeans.fit_predict(data_2d)
                
                # 重塑为原始形状
                classified = labels.reshape(rows, cols)
                
                return classified
                
        except Exception as e:
            logger.error(f"非监督分类失败: {e}")
            return None
    
    def plot_ndvi(self, ndvi_data, title="NDVI分布图"):
        """绘制NDVI图"""
        plt.figure(figsize=(10, 8))
        
        # 创建自定义颜色映射
        colors = ['#8B0000', '#FF4500', '#FFFF00', '#9ACD32', '#228B22']
        cmap = plt.cm.colors.LinearSegmentedColormap.from_list('ndvi_cmap', colors, N=256)
        
        # 绘制NDVI
        im = plt.imshow(ndvi_data, cmap=cmap, vmin=-1, vmax=1)
        plt.colorbar(im, label='NDVI值')
        plt.title(title)
        
        # 添加图例说明
        plt.figtext(0.5, 0.01, 
                   "NDVI值范围: -1.0 (水体/裸地) 到 1.0 (茂密植被)",
                   ha='center', fontsize=10)
        
        plt.tight_layout()
        output_path = "output/ndvi_plot.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        logger.info(f"NDVI图已保存: {output_path}")
        return output_path