"""
时序动画模块
"""
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import logging
import json

logger = logging.getLogger(__name__)

class TimeSeriesAnimator:
    """时序动画生成器"""
    
    def __init__(self, config):
        self.config = config
        
    def create_animation(self, time_series_data, x_column='timestamp', 
                        y_column='poi_count', group_column='zone_type',
                        title="时序变化动画"):
        """创建时序动画"""
        logger.info(f"创建{title}动画")
        
        if time_series_data is None or len(time_series_data) == 0:
            logger.warning("时序数据为空")
            return None
        
        # 确保时间列是datetime类型
        time_series_data[x_column] = pd.to_datetime(time_series_data[x_column])
        
        # 创建动画
        fig = px.scatter(
            time_series_data,
            x=x_column,
            y=y_column,
            color=group_column,
            size=y_column if y_column != 'poi_count' else None,
            animation_frame=x_column if len(time_series_data[x_column].unique()) > 1 else None,
            hover_name=group_column,
            title=title,
            labels={x_column: '时间', y_column: '数值', group_column: '分组'}
        )
        
        # 添加趋势线
        if len(time_series_data[x_column].unique()) > 5:
            for group in time_series_data[group_column].unique():
                group_data = time_series_data[time_series_data[group_column] == group]
                if len(group_data) > 2:
                    # 添加趋势线
                    z = np.polyfit(range(len(group_data)), group_data[y_column], 1)
                    p = np.poly1d(z)
                    
                    fig.add_trace(go.Scatter(
                        x=group_data[x_column],
                        y=p(range(len(group_data))),
                        mode='lines',
                        name=f'{group}趋势线',
                        line=dict(dash='dash'),
                        showlegend=True
                    ))
        
        # 更新布局
        fig.update_layout(
            xaxis_title="时间",
            yaxis_title=y_column,
            hovermode='x unified',
            legend_title=group_column,
            font=dict(size=12)
        )
        
        # 更新动画设置
        if 'animation_frame' in fig.layout:
            fig.layout.updatemenus = [{
                'buttons': [
                    {
                        'args': [None, {'frame': {'duration': 500, 'redraw': True},
                                       'fromcurrent': True, 'transition': {'duration': 300}}],
                        'label': '播放',
                        'method': 'animate'
                    },
                    {
                        'args': [[None], {'frame': {'duration': 0, 'redraw': True},
                                         'mode': 'immediate',
                                         'transition': {'duration': 0}}],
                        'label': '暂停',
                        'method': 'animate'
                    }
                ],
                'direction': 'left',
                'pad': {'r': 10, 't': 87},
                'showactive': False,
                'type': 'buttons',
                'x': 0.1,
                'xanchor': 'right',
                'y': 0,
                'yanchor': 'top'
            }]
        
        return fig
    
    def create_heatmap_animation(self, spatial_time_series, time_column='timestamp',
                                value_column='poi_count', title="时空热力图动画"):
        """创建时空热力图动画"""
        logger.info(f"创建{title}动画")
        
        if spatial_time_series is None:
            logger.warning("时空数据为空")
            return None
        
        # 准备数据
        spatial_time_series[time_column] = pd.to_datetime(spatial_time_series[time_column])
        
        # 创建动画热力图
        fig = px.density_heatmap(
            spatial_time_series,
            x='lon',
            y='lat',
            z=value_column,
            animation_frame=time_column,
            title=title,
            labels={'lon': '经度', 'lat': '纬度', value_column: '数值'},
            color_continuous_scale='Viridis'
        )
        
        # 更新布局
        fig.update_layout(
            xaxis_title="经度",
            yaxis_title="纬度",
            font=dict(size=12)
        )
        
        return fig
    
    def create_comparison_animation(self, data_before, data_after, 
                                  time_column='timestamp', value_column='poi_count',
                                  title="前后对比动画"):
        """创建前后对比动画"""
        logger.info(f"创建{title}对比动画")
        
        # 添加时期标签
        data_before['period'] = '改造前'
        data_after['period'] = '改造后'
        
        combined_data = pd.concat([data_before, data_after], ignore_index=True)
        
        # 创建对比动画
        fig = px.bar(
            combined_data,
            x=time_column,
            y=value_column,
            color='period',
            barmode='group',
            animation_frame=time_column,
            title=title,
            labels={time_column: '时间', value_column: '数值', 'period': '时期'}
        )
        
        # 更新布局
        fig.update_layout(
            xaxis_title="时间",
            yaxis_title=value_column,
            font=dict(size=12),
            legend_title="时期"
        )
        
        return fig
    
    def create_zone_evolution_animation(self, zones_series, time_column='year',
                                      title="功能区演变动画"):
        """创建功能区演变动画"""
        logger.info(f"创建{title}动画")
        
        if zones_series is None or len(zones_series) == 0:
            logger.warning("功能区序列数据为空")
            return None
        
        # 统计每年的功能区分布
        evolution_data = []
        
        for time_point in zones_series[time_column].unique():
            time_data = zones_series[zones_series[time_column] == time_point]
            
            zone_counts = time_data['zone_type'].value_counts().reset_index()
            zone_counts.columns = ['zone_type', 'count']
            zone_counts[time_column] = time_point
            
            evolution_data.append(zone_counts)
        
        if not evolution_data:
            return None
        
        evolution_df = pd.concat(evolution_data, ignore_index=True)
        
        # 创建堆叠面积图动画
        fig = px.area(
            evolution_df,
            x=time_column,
            y='count',
            color='zone_type',
            animation_frame=time_column,
            title=title,
            labels={time_column: '时间', 'count': '网格数量', 'zone_type': '功能区类型'}
        )
        
        # 更新布局
        fig.update_layout(
            xaxis_title="时间",
            yaxis_title="网格数量",
            font=dict(size=12),
            legend_title="功能区类型"
        )
        
        return fig
    
    def save_animation(self, fig, output_path, format='html', width=1200, height=800):
        """保存动画"""
        logger.info(f"保存动画到: {output_path}")
        
        if fig is None:
            logger.warning("没有可保存的图形")
            return None
        
        if format == 'html':
            fig.write_html(output_path)
        elif format == 'png':
            # 保存第一帧为图片
            fig.write_image(output_path, width=width, height=height)
        elif format == 'gif':
            # 需要额外的库来生成GIF
            logger.warning("GIF格式需要额外配置，保存为HTML")
            fig.write_html(output_path.replace('.gif', '.html'))
        else:
            logger.warning(f"不支持的格式: {format}，保存为HTML")
            fig.write_html(output_path)
        
        logger.info(f"动画已保存: {output_path}")
        return output_path
    
    def create_animation_report(self, animations_data, output_path="output/animation_report.md"):
        """创建动画报告"""
        logger.info("创建动画报告")
        
        report_lines = [
            "# 城市功能区动态分析动画报告\n",
            f"**生成时间**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
            "\n## 动画内容概览\n"
        ]
        
        for anim_name, anim_info in animations_data.items():
            report_lines.append(f"### {anim_info['title']}")
            report_lines.append(f"- **描述**: {anim_info.get('description', '')}")
            report_lines.append(f"- **时间范围**: {anim_info.get('time_range', 'N/A')}")
            report_lines.append(f"- **数据量**: {anim_info.get('data_points', 0)}个数据点")
            report_lines.append(f"- **保存路径**: `{anim_info.get('output_path', 'N/A')}`")
            report_lines.append("")
        
        report_lines.extend([
            "\n## 动画使用说明\n",
            "1. **HTML动画**: 使用浏览器打开，可通过按钮控制播放",
            "2. **交互功能**: 支持鼠标悬停查看详细信息",
            "3. **时间控制**: 可调整播放速度，暂停查看特定时刻",
            "\n## 数据分析要点\n",
            "1. **时序趋势**: 观察各指标随时间的变化趋势",
            "2. **空间分布**: 分析指标的空间分布特征",
            "3. **演变规律**: 总结功能区演变的一般规律",
            "4. **异常检测**: 识别异常变化点和区域"
        ])
        
        report_content = "\n".join(report_lines)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        logger.info(f"动画报告已保存: {output_path}")
        return report_content