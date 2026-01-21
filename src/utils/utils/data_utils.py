"""
数据工具函数
"""
import pandas as pd
import numpy as np
import json
import yaml
from datetime import datetime, timedelta
import logging
import os

logger = logging.getLogger(__name__)

class DataUtils:
    """数据工具类"""
    
    @staticmethod
    def load_data(file_path, file_type=None):
        """加载数据文件"""
        if file_type is None:
            # 从文件扩展名推断类型
            ext = os.path.splitext(file_path)[1].lower()
            if ext in ['.csv', '.txt']:
                file_type = 'csv'
            elif ext in ['.json', '.geojson']:
                file_type = 'json'
            elif ext in ['.xlsx', '.xls']:
                file_type = 'excel'
            elif ext in ['.shp']:
                file_type = 'shapefile'
            elif ext in ['.yaml', '.yml']:
                file_type = 'yaml'
            elif ext in ['.parquet']:
                file_type = 'parquet'
            else:
                raise ValueError(f"不支持的文件类型: {ext}")
        
        logger.info(f"加载{file_type}文件: {file_path}")
        
        try:
            if file_type == 'csv':
                return pd.read_csv(file_path, encoding='utf-8')
            elif file_type == 'json':
                with open(file_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            elif file_type == 'geojson':
                import geopandas as gpd
                return gpd.read_file(file_path)
            elif file_type == 'excel':
                return pd.read_excel(file_path)
            elif file_type == 'shapefile':
                import geopandas as gpd
                return gpd.read_file(file_path)
            elif file_type == 'yaml':
                with open(file_path, 'r', encoding='utf-8') as f:
                    return yaml.safe_load(f)
            elif file_type == 'parquet':
                return pd.read_parquet(file_path)
            else:
                raise ValueError(f"不支持的文件类型: {file_type}")
        except Exception as e:
            logger.error(f"加载文件失败: {e}")
            raise
    
    @staticmethod
    def save_data(data, file_path, file_type=None):
        """保存数据文件"""
        if file_type is None:
            ext = os.path.splitext(file_path)[1].lower()
            if ext in ['.csv', '.txt']:
                file_type = 'csv'
            elif ext in ['.json', '.geojson']:
                file_type = 'json'
            elif ext in ['.xlsx', '.xls']:
                file_type = 'excel'
            elif ext in ['.yaml', '.yml']:
                file_type = 'yaml'
            elif ext in ['.parquet']:
                file_type = 'parquet'
            else:
                raise ValueError(f"不支持的文件类型: {ext}")
        
        logger.info(f"保存{file_type}文件: {file_path}")
        
        # 确保目录存在
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        
        try:
            if file_type == 'csv':
                data.to_csv(file_path, index=False, encoding='utf-8')
            elif file_type == 'json':
                if isinstance(data, pd.DataFrame):
                    data.to_json(file_path, orient='records', force_ascii=False)
                else:
                    with open(file_path, 'w', encoding='utf-8') as f:
                        json.dump(data, f, ensure_ascii=False, indent=2)
            elif file_type == 'geojson':
                data.to_file(file_path, driver='GeoJSON', encoding='utf-8')
            elif file_type == 'excel':
                data.to_excel(file_path, index=False)
            elif file_type == 'yaml':
                with open(file_path, 'w', encoding='utf-8') as f:
                    yaml.dump(data, f, allow_unicode=True)
            elif file_type == 'parquet':
                data.to_parquet(file_path)
            else:
                raise ValueError(f"不支持的文件类型: {file_type}")
            
            logger.info(f"文件保存成功: {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"保存文件失败: {e}")
            return False
    
    @staticmethod
    def clean_missing_data(df, strategy='drop', fill_value=None):
        """处理缺失数据"""
        logger.info(f"处理缺失数据，策略: {strategy}")
        
        if strategy == 'drop':
            # 删除包含缺失值的行
            cleaned_df = df.dropna()
            logger.info(f"删除缺失值后，剩余{len(cleaned_df)}行")
            
        elif strategy == 'fill':
            # 填充缺失值
            if fill_value is None:
                # 数值列用中位数，分类列用众数
                cleaned_df = df.copy()
                for col in df.columns:
                    if pd.api.types.is_numeric_dtype(df[col]):
                        cleaned_df[col] = df[col].fillna(df[col].median())
                    else:
                        cleaned_df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown')
            else:
                cleaned_df = df.fillna(fill_value)
            
            logger.info(f"填充缺失值完成")
            
        elif strategy == 'interpolate':
            # 插值
            cleaned_df = df.interpolate(method='linear', limit_direction='forward')
            logger.info(f"插值完成")
            
        else:
            raise ValueError(f"不支持的缺失数据处理策略: {strategy}")
        
        return cleaned_df
    
    @staticmethod
    def detect_outliers(df, column, method='iqr', threshold=1.5):
        """检测异常值"""
        logger.info(f"检测{column}列的异常值，方法: {method}")
        
        if method == 'iqr':
            # IQR方法
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            
            outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
            
        elif method == 'zscore':
            # Z-score方法
            from scipy import stats
            z_scores = np.abs(stats.zscore(df[column].fillna(df[column].mean())))
            outliers = df[z_scores > threshold]
            
        elif method == 'percentile':
            # 百分位数方法
            lower_bound = df[column].quantile(0.01)
            upper_bound = df[column].quantile(0.99)
            outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
            
        else:
            raise ValueError(f"不支持的异常值检测方法: {method}")
        
        logger.info(f"检测到{len(outliers)}个异常值")
        return outliers
    
    @staticmethod
    def normalize_data(df, columns=None, method='minmax'):
        """数据归一化"""
        logger.info(f"数据归一化，方法: {method}")
        
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        normalized_df = df.copy()
        
        for col in columns:
            if col not in df.columns:
                continue
            
            if method == 'minmax':
                min_val = df[col].min()
                max_val = df[col].max()
                
                if max_val > min_val:
                    normalized_df[col] = (df[col] - min_val) / (max_val - min_val)
                else:
                    normalized_df[col] = 0.5
            
            elif method == 'zscore':
                mean_val = df[col].mean()
                std_val = df[col].std()
                
                if std_val > 0:
                    normalized_df[col] = (df[col] - mean_val) / std_val
                else:
                    normalized_df[col] = 0
            
            elif method == 'robust':
                from sklearn.preprocessing import RobustScaler
                scaler = RobustScaler()
                normalized_df[col] = scaler.fit_transform(df[[col]].values.reshape(-1, 1))
            
            else:
                raise ValueError(f"不支持的归一化方法: {method}")
        
        logger.info(f"数据归一化完成，处理列: {columns}")
        return normalized_df
    
    @staticmethod
    def create_time_series(start_date, end_date, freq='D'):
        """创建时间序列"""
        date_range = pd.date_range(start=start_date, end=end_date, freq=freq)
        
        logger.info(f"创建时间序列: {start_date} 到 {end_date}，频率: {freq}，共{len(date_range)}个时间点")
        return date_range
    
    @staticmethod
    def split_train_test(df, test_size=0.2, random_state=42):
        """划分训练集和测试集"""
        from sklearn.model_selection import train_test_split
        
        train_df, test_df = train_test_split(
            df, test_size=test_size, random_state=random_state, shuffle=True
        )
        
        logger.info(f"数据集划分: 训练集{len(train_df)}行，测试集{len(test_df)}行")
        return train_df, test_df
    
    @staticmethod
    def calculate_statistics(df):
        """计算基本统计量"""
        stats = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'missing_values': df.isnull().sum().sum(),
            'duplicate_rows': df.duplicated().sum()
        }
        
        # 数值列统计
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            numeric_stats = df[numeric_cols].describe().T
            stats['numeric_statistics'] = numeric_stats.to_dict()
        
        # 分类列统计
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            categorical_stats = {}
            for col in categorical_cols:
                value_counts = df[col].value_counts().head(10).to_dict()
                categorical_stats[col] = {
                    'unique_values': df[col].nunique(),
                    'top_values': value_counts
                }
            stats['categorical_statistics'] = categorical_stats
        
        return stats
    
    @staticmethod
    def generate_sample_data(sample_size=1000, random_seed=42):
        """生成示例数据"""
        np.random.seed(random_seed)
        
        # 创建示例数据
        sample_df = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=sample_size, freq='H'),
            'grid_id': [f'G_{i:03d}' for i in range(sample_size)],
            'zone_type': np.random.choice(['商业区', '居住区', '办公区', '绿地', '混合区'], sample_size),
            'poi_count': np.random.poisson(10, sample_size) + np.random.randint(0, 20, sample_size),
            'ndvi_mean': np.random.uniform(0, 0.8, sample_size),
            'road_density': np.random.exponential(0.01, sample_size),
            'temperature': np.random.normal(25, 5, sample_size),
            'precipitation': np.random.exponential(5, sample_size),
            'latitude': np.random.uniform(23.1, 23.2, sample_size),
            'longitude': np.random.uniform(113.3, 113.4, sample_size)
        })
        
        logger.info(f"生成{sample_size}行示例数据")
        return sample_df