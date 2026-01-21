"""
功能区聚类模型
"""
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import logging

logger = logging.getLogger(__name__)

class FunctionZoneClustering:
    """功能区聚类模型"""
    
    def __init__(self, config):
        self.config = config
        self.model_type = config["model"]["clustering"]["algorithm"]
        self.n_clusters = config["model"]["clustering"].get("n_clusters", 6)
        
    def preprocess_features(self, features_df):
        """预处理特征"""
        # 选择特征列
        feature_cols = [
            "poi_count", "poi_diversity",
            "poi_餐饮", "poi_购物", "poi_交通", "poi_教育", "poi_医疗", "poi_办公",
            "ndvi_mean", "road_density"
        ]
        
        # 提取特征矩阵
        X = features_df[feature_cols].values
        
        # 标准化
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        return X_scaled, scaler
    
    def cluster_dbscan(self, X, eps=0.5, min_samples=5):
        """DBSCAN聚类"""
        clustering = DBSCAN(eps=eps, min_samples=min_samples)
        labels = clustering.fit_predict(X)
        
        # 将噪声点(-1)单独作为一类
        labels[labels == -1] = np.max(labels) + 1 if np.max(labels) >= 0 else 0
        
        return labels
    
    def cluster_kmeans(self, X, n_clusters=6):
        """KMeans聚类"""
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(X)
        return labels
    
    def predict(self, features_df):
        """预测功能区类型"""
        logger.info("开始功能区聚类分析")
        
        # 预处理特征
        X, scaler = self.preprocess_features(features_df)
        
        # 选择聚类算法
        if self.model_type == "dbscan":
            labels = self.cluster_dbscan(X)
        elif self.model_type == "kmeans":
            labels = self.cluster_kmeans(X, self.n_clusters)
        else:
            raise ValueError(f"不支持的聚类算法: {self.model_type}")
        
        # 计算轮廓系数
        if len(np.unique(labels)) > 1:
            score = silhouette_score(X, labels)
            logger.info(f"聚类轮廓系数: {score:.3f}")
        
        # 添加聚类标签
        features_df["zone_label"] = labels
        
        # 解释聚类结果
        features_df["zone_type"] = features_df["zone_label"].map(
            self._interpret_clusters(features_df)
        )
        
        logger.info(f"聚类完成，共识别出{len(np.unique(labels))}类功能区")
        return features_df
    
    def _interpret_clusters(self, features_df):
        """解释聚类结果，为每个类别命名"""
        # 分析每个簇的特征均值
        cluster_stats = features_df.groupby("zone_label").mean()
        
        zone_names = {}
        for label, stats in cluster_stats.iterrows():
            # 根据特征判断功能区类型
            if stats["poi_购物"] > stats["poi_餐饮"]:
                zone_type = "商业区"
            elif stats["poi_办公"] > stats.mean():
                zone_type = "办公区"
            elif stats["ndvi_mean"] > 0.6:
                zone_type = "绿地/公园"
            elif stats["poi_教育"] > stats.mean():
                zone_type = "文教区"
            elif stats["poi_医疗"] > stats.mean():
                zone_type = "医疗区"
            elif stats["poi_count"] < 5:
                zone_type = "居住区"
            else:
                zone_type = "混合功能区"
            
            zone_names[label] = zone_type
        
        return zone_names