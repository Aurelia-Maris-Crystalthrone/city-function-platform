"""
功能区分类模型
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import logging
import joblib

logger = logging.getLogger(__name__)

class FunctionZoneClassifier:
    """功能区分类器"""
    
    def __init__(self, config):
        self.config = config
        self.model_type = config["model"]["classification"]["algorithm"]
        self.model = None
        self.feature_columns = None
        
    def train(self, X, y, test_size=0.2, random_state=42):
        """训练分类模型"""
        logger.info(f"训练{self.model_type}分类模型")
        
        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        self.feature_columns = X.columns.tolist()
        
        # 选择模型
        if self.model_type == "random_forest":
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=random_state,
                n_jobs=-1
            )
        elif self.model_type == "svm":
            self.model = SVC(
                kernel='rbf',
                C=1.0,
                probability=True,
                random_state=random_state
            )
        elif self.model_type == "mlp":
            self.model = MLPClassifier(
                hidden_layer_sizes=(100, 50),
                activation='relu',
                solver='adam',
                max_iter=500,
                random_state=random_state
            )
        else:
            raise ValueError(f"不支持的分类算法: {self.model_type}")
        
        # 训练模型
        self.model.fit(X_train, y_train)
        
        # 评估模型
        train_score = self.model.score(X_train, y_train)
        test_score = self.model.score(X_test, y_test)
        
        logger.info(f"训练集准确率: {train_score:.3f}")
        logger.info(f"测试集准确率: {test_score:.3f}")
        
        # 交叉验证
        cv_scores = cross_val_score(self.model, X, y, cv=5, n_jobs=-1)
        logger.info(f"交叉验证平均准确率: {cv_scores.mean():.3f} (±{cv_scores.std():.3f})")
        
        # 详细评估报告
        y_pred = self.model.predict(X_test)
        report = classification_report(y_test, y_pred)
        logger.info(f"分类报告:\n{report}")
        
        return {
            'model': self.model,
            'train_score': train_score,
            'test_score': test_score,
            'cv_scores': cv_scores,
            'classification_report': report
        }
    
    def predict(self, X):
        """预测功能区类型"""
        if self.model is None:
            raise ValueError("模型尚未训练，请先调用train方法")
        
        logger.info("进行功能区类型预测")
        
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)
        
        # 获取类别标签
        if hasattr(self.model, 'classes_'):
            class_labels = self.model.classes_
        else:
            class_labels = np.unique(predictions)
        
        # 创建结果DataFrame
        results = pd.DataFrame({
            'predicted_class': predictions,
            'confidence': probabilities.max(axis=1)
        })
        
        # 添加每个类别的概率
        for i, class_label in enumerate(class_labels):
            results[f'prob_{class_label}'] = probabilities[:, i]
        
        logger.info(f"预测完成，共{len(results)}条预测")
        return results
    
    def interpret_results(self, X, y_true=None):
        """解释模型结果"""
        logger.info("解释模型结果")
        
        if self.model is None:
            raise ValueError("模型尚未训练")
        
        # 特征重要性（如果模型支持）
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            importance_df = pd.DataFrame({
                'feature': self.feature_columns,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            logger.info(f"特征重要性:\n{importance_df.head(10)}")
            
            # 可视化
            self._plot_feature_importance(importance_df)
        
        # 混淆矩阵（如果有真实标签）
        if y_true is not None:
            y_pred = self.model.predict(X)
            cm = confusion_matrix(y_true, y_pred)
            
            logger.info(f"混淆矩阵:\n{cm}")
            self._plot_confusion_matrix(cm, np.unique(y_true))
        
        return importance_df if hasattr(self.model, 'feature_importances_') else None
    
    def _plot_feature_importance(self, importance_df, top_n=20):
        """绘制特征重要性图"""
        import matplotlib.pyplot as plt
        
        top_features = importance_df.head(top_n)
        
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('特征重要性')
        plt.title(f'Top {top_n} 特征重要性排名')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        
        output_path = "output/feature_importance.png"
        plt.savefig(output_path, dpi=300)
        plt.show()
        
        logger.info(f"特征重要性图已保存: {output_path}")
    
    def _plot_confusion_matrix(self, cm, classes):
        """绘制混淆矩阵"""
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=classes, yticklabels=classes)
        plt.xlabel('预测标签')
        plt.ylabel('真实标签')
        plt.title('混淆矩阵')
        
        output_path = "output/confusion_matrix.png"
        plt.savefig(output_path, dpi=300)
        plt.show()
        
        logger.info(f"混淆矩阵图已保存: {output_path}")
    
    def save_model(self, filepath="models/function_zone_classifier.pkl"):
        """保存模型"""
        if self.model is None:
            raise ValueError("没有训练好的模型可以保存")
        
        # 保存模型
        joblib.dump({
            'model': self.model,
            'feature_columns': self.feature_columns,
            'config': self.config
        }, filepath)
        
        logger.info(f"模型已保存: {filepath}")
        return filepath
    
    def load_model(self, filepath):
        """加载模型"""
        data = joblib.load(filepath)
        self.model = data['model']
        self.feature_columns = data['feature_columns']
        
        logger.info(f"模型已加载: {filepath}")
        return self