"""
变化检测模块
"""
import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
import logging

logger = logging.getLogger(__name__)

class ChangeDetector:
    """变化检测器"""
    
    def __init__(self, config):
        self.config = config
        
    def detect_changes(self, data_t1, data_t2, method='difference'):
        """检测变化"""
        logger.info(f"使用{method}方法检测变化")
        
        if method == 'difference':
            return self._detect_by_difference(data_t1, data_t2)
        elif method == 'ratio':
            return self._detect_by_ratio(data_t1, data_t2)
        elif method == 'anomaly':
            return self._detect_by_anomaly(data_t1, data_t2)
        elif method == 'clustering':
            return self._detect_by_clustering(data_t1, data_t2)
        else:
            raise ValueError(f"不支持的变化检测方法: {method}")
    
    def _detect_by_difference(self, data_t1, data_t2):
        """基于差值检测变化"""
        # 确保数据对齐
        common_ids = set(data_t1['grid_id']).intersection(set(data_t2['grid_id']))
        
        changes = []
        for grid_id in common_ids:
            t1_row = data_t1[data_t1['grid_id'] == grid_id].iloc[0]
            t2_row = data_t2[data_t2['grid_id'] == grid_id].iloc[0]
            
            change_record = {'grid_id': grid_id}
            
            # 计算各指标的变化量
            for col in ['poi_count', 'road_density', 'ndvi_mean', 'urban_vitality']:
                if col in t1_row and col in t2_row:
                    change = t2_row[col] - t1_row[col]
                    change_record[f'{col}_change'] = change
                    change_record[f'{col}_change_abs'] = abs(change)
            
            # 计算综合变化得分
            change_columns = [c for c in change_record.keys() if 'change_abs' in c]
            if change_columns:
                change_record['change_score'] = sum(change_record[c] for c in change_columns) / len(change_columns)
            else:
                change_record['change_score'] = 0
            
            changes.append(change_record)
        
        changes_df = pd.DataFrame(changes)
        
        # 识别显著变化
        threshold = changes_df['change_score'].quantile(0.75)  # 上四分位数作为阈值
        changes_df['is_significant'] = changes_df['change_score'] > threshold
        
        logger.info(f"检测到 {changes_df['is_significant'].sum()} 个显著变化网格")
        return changes_df
    
    def _detect_by_ratio(self, data_t1, data_t2):
        """基于比率检测变化"""
        changes_df = self._detect_by_difference(data_t1, data_t2)
        
        # 计算变化比率
        for grid_id in changes_df['grid_id']:
            t1_row = data_t1[data_t1['grid_id'] == grid_id].iloc[0]
            t2_row = data_t2[data_t2['grid_id'] == grid_id].iloc[0]
            
            for col in ['poi_count', 'road_density', 'ndvi_mean']:
                if col in t1_row and col in t2_row and t1_row[col] != 0:
                    change_ratio = (t2_row[col] - t1_row[col]) / t1_row[col]
                    changes_df.loc[changes_df['grid_id'] == grid_id, f'{col}_change_ratio'] = change_ratio
        
        # 识别异常比率
        for col in ['poi_count_change_ratio', 'ndvi_mean_change_ratio']:
            if col in changes_df.columns:
                changes_df[f'{col}_anomaly'] = abs(changes_df[col]) > 0.5  # 50%变化视为异常
        
        return changes_df
    
    def _detect_by_anomaly(self, data_t1, data_t2):
        """基于异常检测的变化检测"""
        # 计算变化特征
        changes_df = self._detect_by_difference(data_t1, data_t2)
        
        if len(changes_df) < 10:
            logger.warning("数据量太少，无法进行异常检测")
            changes_df['is_anomaly'] = False
            return changes_df
        
        # 提取变化特征
        feature_cols = [c for c in changes_df.columns if 'change' in c and 'ratio' not in c]
        
        if len(feature_cols) == 0:
            logger.warning("没有找到变化特征列")
            changes_df['is_anomaly'] = False
            return changes_df
        
        X = changes_df[feature_cols].fillna(0).values
        
        # 使用Isolation Forest检测异常
        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        anomalies = iso_forest.fit_predict(X)
        
        # -1表示异常，1表示正常
        changes_df['is_anomaly'] = anomalies == -1
        
        logger.info(f"检测到 {changes_df['is_anomaly'].sum()} 个异常变化")
        return changes_df
    
    def _detect_by_clustering(self, data_t1, data_t2):
        """基于聚类的变化检测"""
        # 合并两期数据
        merged = pd.merge(data_t1, data_t2, on='grid_id', suffixes=('_t1', '_t2'))
        
        # 计算变化特征
        feature_cols = []
        for col in ['poi_count', 'ndvi_mean', 'urban_vitality']:
            if f'{col}_t1' in merged.columns and f'{col}_t2' in merged.columns:
                merged[f'{col}_change'] = merged[f'{col}_t2'] - merged[f'{col}_t1']
                feature_cols.append(f'{col}_change')
        
        if len(feature_cols) == 0:
            logger.warning("没有足够的变化特征进行聚类")
            merged['change_cluster'] = 0
            merged['is_changed'] = False
            return merged
        
        # 使用DBSCAN聚类
        X = merged[feature_cols].fillna(0).values
        
        # 标准化
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # 聚类
        clustering = DBSCAN(eps=0.5, min_samples=5)
        clusters = clustering.fit_predict(X_scaled)
        
        merged['change_cluster'] = clusters
        
        # 将噪声点(-1)和最大的簇(假设为未变化)以外的点视为变化
        cluster_counts = pd.Series(clusters).value_counts()
        if len(cluster_counts) > 1:
            largest_cluster = cluster_counts.idxmax()
            merged['is_changed'] = clusters != largest_cluster
        else:
            merged['is_changed'] = clusters == -1  # 所有点都在一个簇中，只有噪声点视为变化
        
        logger.info(f"聚类检测到 {merged['is_changed'].sum()} 个变化网格")
        return merged
    
    def classify_change_types(self, changes_df):
        """分类变化类型"""
        logger.info("分类变化类型")
        
        change_types = []
        
        for _, row in changes_df.iterrows():
            change_type = self._determine_change_type(row)
            change_types.append(change_type)
        
        changes_df['change_type'] = change_types
        
        # 统计各类变化数量
        type_counts = changes_df['change_type'].value_counts()
        logger.info(f"变化类型分布:\n{type_counts}")
        
        return changes_df
    
    def _determine_change_type(self, row):
        """确定单个网格的变化类型"""
        # 获取变化值
        poi_change = row.get('poi_count_change', 0)
        ndvi_change = row.get('ndvi_mean_change', 0)
        road_change = row.get('road_density_change', 0)
        
        # 判断主要变化方向
        changes = {
            'development': 0,
            'decline': 0,
            'greening': 0,
            'urbanization': 0,
            'regression': 0
        }
        
        # POI数量变化
        if poi_change > 2:
            changes['development'] += 1
        elif poi_change < -2:
            changes['decline'] += 1
        
        # 植被变化
        if ndvi_change > 0.1:
            changes['greening'] += 1
        elif ndvi_change < -0.1:
            changes['urbanization'] += 1
        
        # 路网变化
        if road_change > 0.001:
            changes['development'] += 1
        
        # 找出最大的变化类型
        max_type = max(changes, key=changes.get)
        
        if changes[max_type] == 0:
            return 'stable'
        
        # 如果有多个类型得分相同
        max_score = changes[max_type]
        tied_types = [t for t, s in changes.items() if s == max_score]
        
        if len(tied_types) > 1:
            return 'mixed_' + '_'.join(tied_types)
        else:
            return max_type
    
    def generate_change_report(self, changes_df, output_path="output/change_report.md"):
        """生成变化报告"""
        logger.info("生成变化报告")
        
        report_lines = [
            "# 城市功能区变化检测报告\n",
            f"**生成时间**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
            f"**分析网格数量**: {len(changes_df)}\n",
            f"**显著变化网格**: {changes_df['is_significant'].sum() if 'is_significant' in changes_df.columns else 'N/A'}\n",
            f"**异常变化网格**: {changes_df['is_anomaly'].sum() if 'is_anomaly' in changes_df.columns else 'N/A'}\n",
            "\n## 变化类型分布\n"
        ]
        
        if 'change_type' in changes_df.columns:
            type_stats = changes_df['change_type'].value_counts()
            for change_type, count in type_stats.items():
                percentage = count / len(changes_df) * 100
                report_lines.append(f"- **{change_type}**: {count}个网格 ({percentage:.1f}%)")
        
        report_lines.extend([
            "\n## 主要变化区域\n",
            "| 网格ID | 变化类型 | POI变化 | NDVI变化 | 路网密度变化 |",
            "|--------|----------|---------|----------|--------------|"
        ])
        
        # 显示变化最显著的10个网格
        if 'change_score' in changes_df.columns:
            top_changes = changes_df.nlargest(10, 'change_score')
            for _, row in top_changes.iterrows():
                report_lines.append(
                    f"| {row['grid_id']} | "
                    f"{row.get('change_type', 'N/A')} | "
                    f"{row.get('poi_count_change', 0):+.1f} | "
                    f"{row.get('ndvi_mean_change', 0):+.3f} | "
                    f"{row.get('road_density_change', 0):+.5f} |"
                )
        
        report_content = "\n".join(report_lines)
        
        # 保存报告
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        logger.info(f"变化报告已保存: {output_path}")
        return report_content