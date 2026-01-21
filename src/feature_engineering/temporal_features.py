"""
时序特征提取模块
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class TemporalFeatureExtractor:
    """时序特征提取器"""
    
    def __init__(self, config):
        self.config = config
        
    def extract_time_series_features(self, data_df, time_column='timestamp'):
        """提取时间序列特征"""
        logger.info("提取时间序列特征")
        
        if time_column not in data_df.columns:
            logger.warning(f"时间列 {time_column} 不存在，生成模拟时间数据")
            data_df = self._add_simulated_timestamps(data_df, time_column)
        
        # 转换为时间类型
        data_df[time_column] = pd.to_datetime(data_df[time_column])
        
        # 提取时间特征
        temporal_features = []
        
        for _, group in data_df.groupby('grid_id'):
            if len(group) > 1:
                features = self._extract_grid_temporal_features(group, time_column)
                features['grid_id'] = group['grid_id'].iloc[0]
                temporal_features.append(features)
        
        temporal_df = pd.DataFrame(temporal_features)
        
        logger.info(f"时序特征提取完成，共{len(temporal_df)}个网格")
        return temporal_df
    
    def _add_simulated_timestamps(self, data_df, time_column):
        """添加模拟时间戳"""
        # 创建24小时的时间序列
        base_time = datetime(2024, 1, 1)
        timestamps = []
        
        for i in range(len(data_df)):
            # 模拟不同时间点的数据
            hour = i % 24
            timestamp = base_time + timedelta(hours=hour)
            timestamps.append(timestamp)
        
        data_df[time_column] = timestamps
        return data_df
    
    def _extract_grid_temporal_features(self, grid_data, time_column):
        """提取单个网格的时序特征"""
        # 按时间排序
        grid_data = grid_data.sort_values(time_column)
        
        # 提取小时特征
        grid_data['hour'] = grid_data[time_column].dt.hour
        
        features = {
            'temporal_range_days': (grid_data[time_column].max() - 
                                   grid_data[time_column].min()).days,
            'data_points_count': len(grid_data),
            'avg_daily_frequency': len(grid_data) / max(1, features['temporal_range_days'])
        }
        
        # 日变化特征
        if len(grid_data) > 0:
            hourly_counts = grid_data.groupby('hour').size()
            
            # 峰值时段
            peak_hour = hourly_counts.idxmax() if not hourly_counts.empty else -1
            features['peak_hour'] = peak_hour
            
            # 日变化幅度
            if len(hourly_counts) > 1:
                features['daily_variation'] = hourly_counts.max() / hourly_counts.min()
            else:
                features['daily_variation'] = 1
            
            # 时段分类
            features['time_pattern'] = self._classify_time_pattern(hourly_counts)
        
        # 趋势特征
        if len(grid_data) > 5:
            # 计算简单线性趋势
            x = np.arange(len(grid_data))
            y = grid_data['poi_count'].values if 'poi_count' in grid_data.columns else np.ones(len(grid_data))
            
            # 线性回归
            coeffs = np.polyfit(x, y, 1)
            features['trend_slope'] = coeffs[0]
            features['trend_intercept'] = coeffs[1]
            
            # 判断趋势方向
            if abs(features['trend_slope']) < 0.01:
                features['trend_direction'] = 'stable'
            elif features['trend_slope'] > 0:
                features['trend_direction'] = 'increasing'
            else:
                features['trend_direction'] = 'decreasing'
        
        return features
    
    def _classify_time_pattern(self, hourly_counts):
        """分类时间模式"""
        if len(hourly_counts) == 0:
            return 'unknown'
        
        # 计算各时段的活动量
        morning = hourly_counts.reindex(range(6, 12), fill_value=0).sum()
        afternoon = hourly_counts.reindex(range(12, 18), fill_value=0).sum()
        evening = hourly_counts.reindex(range(18, 24), fill_value=0).sum()
        night = hourly_counts.reindex(range(0, 6), fill_value=0).sum()
        
        total = morning + afternoon + evening + night
        
        if total == 0:
            return 'inactive'
        
        # 计算比例
        morning_ratio = morning / total
        afternoon_ratio = afternoon / total
        evening_ratio = evening / total
        
        # 判断主要活动时段
        ratios = [morning_ratio, afternoon_ratio, evening_ratio]
        max_ratio = max(ratios)
        
        if max_ratio > 0.4:
            if max_ratio == morning_ratio:
                return 'morning_peak'
            elif max_ratio == afternoon_ratio:
                return 'afternoon_peak'
            else:
                return 'evening_peak'
        elif sum(r > 0.25 for r in ratios) >= 2:
            return 'multi_peak'
        else:
            return 'balanced'
    
    def calculate_change_features(self, data_t1, data_t2, id_column='grid_id'):
        """计算变化特征"""
        logger.info("计算变化特征")
        
        # 确保两个数据集有相同的网格
        common_ids = set(data_t1[id_column]).intersection(set(data_t2[id_column]))
        
        change_features = []
        
        for grid_id in common_ids:
            t1_data = data_t1[data_t1[id_column] == grid_id].iloc[0]
            t2_data = data_t2[data_t2[id_column] == grid_id].iloc[0]
            
            features = {
                'grid_id': grid_id,
                'poi_count_change': t2_data.get('poi_count', 0) - t1_data.get('poi_count', 0),
                'poi_count_change_ratio': (t2_data.get('poi_count', 0) - t1_data.get('poi_count', 0)) / 
                                         max(1, t1_data.get('poi_count', 0))
            }
            
            # 计算其他指标的变化
            for col in ['road_density', 'ndvi_mean', 'urban_vitality']:
                if col in t1_data and col in t2_data:
                    features[f'{col}_change'] = t2_data[col] - t1_data[col]
                    if t1_data[col] != 0:
                        features[f'{col}_change_ratio'] = (t2_data[col] - t1_data[col]) / t1_data[col]
                    else:
                        features[f'{col}_change_ratio'] = 0
            
            # 判断变化类型
            features['change_type'] = self._classify_change_type(features)
            
            change_features.append(features)
        
        change_df = pd.DataFrame(change_features)
        
        logger.info(f"变化特征计算完成，共{len(change_df)}个网格")
        return change_df
    
    def _classify_change_type(self, features):
        """分类变化类型"""
        poi_change = features.get('poi_count_change_ratio', 0)
        ndvi_change = features.get('ndvi_mean_change', 0)
        
        if abs(poi_change) < 0.1 and abs(ndvi_change) < 0.05:
            return 'stable'
        elif poi_change > 0.2:
            return 'development'
        elif poi_change < -0.2:
            return 'decline'
        elif ndvi_change > 0.1:
            return 'greening'
        elif ndvi_change < -0.1:
            return 'degradation'
        else:
            return 'mixed'