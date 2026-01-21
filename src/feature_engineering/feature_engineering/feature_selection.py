"""
特征选择模块
"""
import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import logging

logger = logging.getLogger(__name__)

class FeatureSelector:
    """特征选择器"""
    
    def __init__(self, config):
        self.config = config
        
    def select_features(self, X, y=None, method='kbest', n_features=10):
        """特征选择"""
        logger.info(f"使用{method}方法选择{n_features}个特征")
        
        if method == 'kbest':
            return self._select_kbest(X, y, n_features)
        elif method == 'mutual_info':
            return self._select_mutual_info(X, y, n_features)
        elif method == 'pca':
            return self._select_pca(X, n_features)
        elif method == 'correlation':
            return self._select_by_correlation(X, n_features)
        else:
            logger.warning(f"未知的特征选择方法: {method}，返回所有特征")
            return X.columns.tolist()
    
    def _select_kbest(self, X, y, n_features):
        """使用SelectKBest选择特征"""
        if y is None:
            logger.warning("目标变量y为None，无法使用SelectKBest")
            return X.columns[:n_features].tolist()
        
        # 确保y是分类变量
        if len(np.unique(y)) < 2:
            logger.warning("目标变量类别少于2，使用方差选择")
            return self._select_by_variance(X, n_features)
        
        selector = SelectKBest(score_func=f_classif, k=n_features)
        selector.fit(X, y)
        
        selected_indices = selector.get_support(indices=True)
        selected_features = X.columns[selected_indices].tolist()
        
        # 输出特征得分
        scores = pd.DataFrame({
            'feature': X.columns,
            'score': selector.scores_
        }).sort_values('score', ascending=False)
        
        logger.info(f"KBest特征选择得分前10:\n{scores.head(10)}")
        
        return selected_features
    
    def _select_mutual_info(self, X, y, n_features):
        """使用互信息选择特征"""
        if y is None:
            logger.warning("目标变量y为None，无法使用互信息")
            return X.columns[:n_features].tolist()
        
        selector = SelectKBest(score_func=mutual_info_classif, k=n_features)
        selector.fit(X, y)
        
        selected_indices = selector.get_support(indices=True)
        selected_features = X.columns[selected_indices].tolist()
        
        return selected_features
    
    def _select_pca(self, X, n_components):
        """使用PCA降维"""
        # 标准化数据
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # 应用PCA
        pca = PCA(n_components=min(n_components, X_scaled.shape[1]))
        X_pca = pca.fit_transform(X_scaled)
        
        # 解释方差
        explained_variance = pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance)
        
        logger.info(f"PCA解释方差: {explained_variance}")
        logger.info(f"PCA累计解释方差: {cumulative_variance}")
        
        # 返回主成分
        pca_columns = [f'PC{i+1}' for i in range(X_pca.shape[1])]
        X_pca_df = pd.DataFrame(X_pca, columns=pca_columns, index=X.index)
        
        return X_pca_df
    
    def _select_by_correlation(self, X, n_features, threshold=0.8):
        """基于相关性选择特征"""
        # 计算相关性矩阵
        corr_matrix = X.corr().abs()
        
        # 选择特征
        selected_features = []
        feature_scores = {}
        
        for feature in X.columns:
            # 计算该特征与其他已选特征的平均相关性
            if selected_features:
                avg_corr = corr_matrix.loc[feature, selected_features].mean()
            else:
                avg_corr = 0
            
            # 特征得分 = 1 - 平均相关性（避免多重共线性）
            feature_scores[feature] = 1 - avg_corr
        
        # 按得分排序
        sorted_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)
        
        # 选择前n个特征
        selected_features = [feature for feature, _ in sorted_features[:n_features]]
        
        logger.info(f"基于相关性选择的特征: {selected_features}")
        return selected_features
    
    def _select_by_variance(self, X, n_features, threshold=0.0):
        """基于方差选择特征"""
        variances = X.var()
        
        # 选择方差大于阈值的特征
        high_variance_features = variances[variances > threshold].index.tolist()
        
        if len(high_variance_features) > n_features:
            # 选择方差最大的n个特征
            selected_features = variances.nlargest(n_features).index.tolist()
        else:
            selected_features = high_variance_features
        
        logger.info(f"基于方差选择的特征: {selected_features}")
        return selected_features
    
    def create_feature_importance_plot(self, X, y, model=None, output_path="output/feature_importance.png"):
        """创建特征重要性图"""
        import matplotlib.pyplot as plt
        
        if model is None:
            # 使用随机森林计算特征重要性
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X, y)
        
        # 获取特征重要性
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_[0])
        else:
            logger.warning("模型没有特征重要性属性")
            return None
        
        # 创建DataFrame
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        # 绘制图形
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(feature_importance)), feature_importance['importance'])
        plt.yticks(range(len(feature_importance)), feature_importance['feature'])
        plt.xlabel('特征重要性')
        plt.title('特征重要性排名')
        plt.tight_layout()
        
        plt.savefig(output_path, dpi=300)
        plt.show()
        
        logger.info(f"特征重要性图已保存: {output_path}")
        return feature_importance