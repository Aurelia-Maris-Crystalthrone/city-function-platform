"""
趋势预测模块
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from prophet import Prophet
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class TrendPredictor:
    """趋势预测器"""
    
    def __init__(self, config):
        self.config = config
        self.models = {}
        
    def predict_future_trends(self, historical_data, target_column='poi_count', 
                             periods=12, freq='M', method='prophet'):
        """预测未来趋势"""
        logger.info(f"使用{method}方法预测{target_column}的未来趋势")
        
        if method == 'prophet':
            return self._predict_with_prophet(historical_data, target_column, periods, freq)
        elif method == 'random_forest':
            return self._predict_with_random_forest(historical_data, target_column, periods)
        elif method == 'linear':
            return self._predict_with_linear(historical_data, target_column, periods)
        else:
            raise ValueError(f"不支持的预测方法: {method}")
    
    def _predict_with_prophet(self, data, target_column, periods, freq):
        """使用Prophet进行时间序列预测"""
        try:
            # 准备Prophet数据格式
            prophet_data = pd.DataFrame({
                'ds': pd.to_datetime(data['timestamp']),
                'y': data[target_column]
            })
            
            # 创建并训练模型
            model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=False,
                changepoint_prior_scale=0.05
            )
            
            model.fit(prophet_data)
            
            # 生成未来日期
            future = model.make_future_dataframe(periods=periods, freq=freq)
            
            # 预测
            forecast = model.predict(future)
            
            # 提取预测结果
            predictions = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(periods)
            
            logger.info(f"Prophet预测完成，预测范围: {predictions['ds'].min()} 到 {predictions['ds'].max()}")
            
            return {
                'model': model,
                'forecast': forecast,
                'predictions': predictions,
                'components': model.plot_components(forecast)
            }
            
        except Exception as e:
            logger.error(f"Prophet预测失败: {e}")
            return self._predict_with_linear(data, target_column, periods)
    
    def _predict_with_random_forest(self, data, target_column, periods):
        """使用随机森林进行预测"""
        # 创建时序特征
        data_with_features = self._create_time_features(data)
        
        if len(data_with_features) < 10:
            logger.warning("数据量太少，使用简单线性预测")
            return self._predict_with_linear(data, target_column, periods)
        
        # 准备特征和目标
        feature_cols = [c for c in data_with_features.columns if c not in ['timestamp', target_column]]
        
        if not feature_cols:
            logger.warning("没有足够特征，使用简单线性预测")
            return self._predict_with_linear(data, target_column, periods)
        
        X = data_with_features[feature_cols]
        y = data_with_features[target_column]
        
        # 训练模型
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        
        # 生成未来特征
        last_timestamp = data['timestamp'].max()
        future_features = []
        
        for i in range(1, periods + 1):
            future_date = last_timestamp + timedelta(days=30 * i)  # 假设每月
        
        # 预测
        future_X = self._create_future_features(last_timestamp, periods, feature_cols)
        predictions = model.predict(future_X)
        
        # 创建预测结果
        future_dates = [last_timestamp + timedelta(days=30 * i) for i in range(1, periods + 1)]
        
        predictions_df = pd.DataFrame({
            'ds': future_dates,
            'yhat': predictions,
            'yhat_lower': predictions * 0.9,  # 估计下限
            'yhat_upper': predictions * 1.1   # 估计上限
        })
        
        logger.info(f"随机森林预测完成，预测{periods}期")
        
        return {
            'model': model,
            'predictions': predictions_df,
            'feature_importance': pd.DataFrame({
                'feature': feature_cols,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
        }
    
    def _predict_with_linear(self, data, target_column, periods):
        """使用线性回归进行预测"""
        # 准备数据
        data = data.sort_values('timestamp')
        data['time_index'] = np.arange(len(data))
        
        X = data[['time_index']]
        y = data[target_column]
        
        # 训练模型
        model = LinearRegression()
        model.fit(X, y)
        
        # 生成未来时间索引
        future_indices = np.arange(len(data), len(data) + periods).reshape(-1, 1)
        
        # 预测
        predictions = model.predict(future_indices)
        
        # 计算置信区间
        y_pred = model.predict(X)
        residuals = y - y_pred
        std_residuals = np.std(residuals)
        
        # 创建预测结果
        last_timestamp = data['timestamp'].max()
        future_dates = [last_timestamp + timedelta(days=30 * i) for i in range(1, periods + 1)]
        
        predictions_df = pd.DataFrame({
            'ds': future_dates,
            'yhat': predictions,
            'yhat_lower': predictions - 1.96 * std_residuals,
            'yhat_upper': predictions + 1.96 * std_residuals
        })
        
        logger.info(f"线性回归预测完成，斜率: {model.coef_[0]:.3f}")
        
        return {
            'model': model,
            'predictions': predictions_df,
            'r_squared': model.score(X, y)
        }
    
    def _create_time_features(self, data):
        """创建时间特征"""
        df = data.copy()
        
        # 确保时间列是datetime类型
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # 提取时间特征
        df['year'] = df['timestamp'].dt.year
        df['month'] = df['timestamp'].dt.month
        df['quarter'] = df['timestamp'].dt.quarter
        df['day_of_year'] = df['timestamp'].dt.dayofyear
        df['day_of_month'] = df['timestamp'].dt.day
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        
        # 创建滞后特征
        for lag in [1, 2, 3, 6, 12]:
            df[f'lag_{lag}'] = df['poi_count'].shift(lag)
        
        # 创建滚动统计特征
        for window in [3, 6, 12]:
            df[f'rolling_mean_{window}'] = df['poi_count'].rolling(window=window, min_periods=1).mean()
            df[f'rolling_std_{window}'] = df['poi_count'].rolling(window=window, min_periods=1).std()
        
        # 填充缺失值
        df = df.fillna(method='bfill').fillna(method='ffill')
        
        return df
    
    def _create_future_features(self, last_timestamp, periods, feature_cols):
        """创建未来时间特征"""
        future_dates = [last_timestamp + timedelta(days=30 * i) for i in range(1, periods + 1)]
        
        future_data = []
        for i, date in enumerate(future_dates):
            features = {
                'year': date.year,
                'month': date.month,
                'quarter': (date.month - 1) // 3 + 1,
                'day_of_year': date.timetuple().tm_yday,
                'day_of_month': date.day,
                'day_of_week': date.weekday(),
                'time_index': i + 1
            }
            
            # 添加其他特征（这里简化处理）
            for col in feature_cols:
                if col not in features:
                    features[col] = 0  # 默认值
            
            future_data.append(features)
        
        return pd.DataFrame(future_data)[feature_cols]
    
    def predict_zones_development(self, current_zones, historical_changes, years=5):
        """预测功能区发展"""
        logger.info(f"预测未来{years}年功能区发展")
        
        predictions = []
        
        for zone_type in current_zones['zone_type'].unique():
            zone_data = current_zones[current_zones['zone_type'] == zone_type]
            
            # 计算当前状态
            current_state = {
                'zone_type': zone_type,
                'grid_count': len(zone_data),
                'avg_poi_count': zone_data['poi_count'].mean(),
                'avg_ndvi': zone_data['ndvi_mean'].mean(),
                'avg_vitality': zone_data['urban_vitality'].mean() if 'urban_vitality' in zone_data.columns else 0
            }
            
            # 基于历史变化预测
            if historical_changes is not None and len(historical_changes) > 0:
                # 查找该类区域的历史变化
                zone_changes = historical_changes[
                    historical_changes['zone_type'] == zone_type
                ] if 'zone_type' in historical_changes.columns else historical_changes
                
                if len(zone_changes) > 0:
                    # 计算平均变化率
                    growth_rate = self._calculate_growth_rate(zone_changes)
                    
                    # 预测未来
                    future_state = current_state.copy()
                    for key in ['avg_poi_count', 'avg_ndvi', 'avg_vitality']:
                        if key in current_state:
                            future_state[f'future_{key}'] = current_state[key] * (1 + growth_rate.get(key, 0)) ** years
                else:
                    future_state = current_state
            else:
                future_state = current_state
            
            # 添加发展建议
            future_state['development_potential'] = self._assess_development_potential(current_state)
            future_state['recommendation'] = self._generate_recommendation(current_state, future_state)
            
            predictions.append(future_state)
        
        predictions_df = pd.DataFrame(predictions)
        
        logger.info(f"功能区发展预测完成，共{len(predictions_df)}类区域")
        return predictions_df
    
    def _calculate_growth_rate(self, changes_data):
        """计算增长率"""
        growth_rates = {}
        
        for col in ['poi_count', 'ndvi_mean', 'urban_vitality']:
            if col in changes_data.columns:
                changes = changes_data[col].dropna()
                if len(changes) > 1:
                    growth_rates[col] = changes.mean() / changes.std() if changes.std() > 0 else 0
        
        return growth_rates
    
    def _assess_development_potential(self, current_state):
        """评估发展潜力"""
        score = 0
        
        # POI密度高，发展潜力低（趋于饱和）
        if current_state['avg_poi_count'] < 10:
            score += 0.3
        elif current_state['avg_poi_count'] < 20:
            score += 0.2
        else:
            score += 0.1
        
        # 植被指数低，发展潜力高
        if current_state['avg_ndvi'] < 0.3:
            score += 0.4
        elif current_state['avg_ndvi'] < 0.6:
            score += 0.2
        else:
            score += 0.1
        
        # 城市活力低，发展潜力高
        if 'avg_vitality' in current_state:
            if current_state['avg_vitality'] < 0.3:
                score += 0.3
            elif current_state['avg_vitality'] < 0.6:
                score += 0.2
            else:
                score += 0.1
        
        return min(1.0, score)
    
    def _generate_recommendation(self, current_state, future_state):
        """生成发展建议"""
        zone_type = current_state['zone_type']
        
        recommendations = {
            '商业区': '建议加强商业设施配套，提升商业活力',
            '办公区': '建议优化办公环境，增加配套设施',
            '居住区': '建议完善生活服务设施，提升居住品质',
            '绿地/公园': '建议加强生态保护，提升绿地质量',
            '文教区': '建议优化教育资源配置，提升文化氛围',
            '医疗区': '建议完善医疗设施，提升服务质量',
            '混合功能区': '建议优化功能布局，提升综合效益'
        }
        
        base_recommendation = recommendations.get(zone_type, '建议加强规划管理，优化功能布局')
        
        # 根据发展潜力调整建议
        if 'development_potential' in future_state:
            if future_state['development_potential'] > 0.7:
                base_recommendation += "，该区域发展潜力较大，建议优先考虑"
            elif future_state['development_potential'] < 0.3:
                base_recommendation += "，该区域发展趋于饱和，建议优化提升"
        
        return base_recommendation
    
    def generate_prediction_report(self, predictions, output_path="output/prediction_report.md"):
        """生成预测报告"""
        logger.info("生成预测报告")
        
        report_lines = [
            "# 城市功能区发展趋势预测报告\n",
            f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
            f"**预测年限**: {predictions.get('prediction_years', 5)}年\n",
            "\n## 各功能区发展预测\n",
            "| 功能区类型 | 当前网格数 | 预测未来网格数 | 发展潜力 | 主要建议 |",
            "|------------|------------|----------------|----------|----------|"
        ]
        
        if isinstance(predictions, pd.DataFrame):
            for _, row in predictions.iterrows():
                report_lines.append(
                    f"| {row.get('zone_type', 'N/A')} | "
                    f"{row.get('grid_count', 0)} | "
                    f"{row.get('future_grid_count', row.get('grid_count', 0)):.0f} | "
                    f"{row.get('development_potential', 0):.2f} | "
                    f"{row.get('recommendation', '')} |"
                )
        
        report_lines.extend([
            "\n## 主要预测结论\n",
            "1. **发展趋势**: 根据历史数据和当前状态预测未来发展趋势",
            "2. **重点关注**: 识别出具有较高发展潜力的功能区",
            "3. **规划建议**: 为各类功能区提供针对性的发展建议",
            "\n## 数据说明\n",
            "- 预测基于历史变化趋势和当前状态",
            "- 发展潜力评分范围0-1，分值越高发展潜力越大",
            "- 建议根据区域特点和城市发展战略制定"
        ])
        
        report_content = "\n".join(report_lines)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        logger.info(f"预测报告已保存: {output_path}")
        return report_content