"""
地图可视化模块
"""
import folium
from folium import plugins
import geopandas as gpd
import branca.colormap as cm
import logging

logger = logging.getLogger(__name__)

class MapVisualizer:
    """地图可视化器"""
    
    def __init__(self, config):
        self.config = config
        
    def create_interactive_map(self, zones_data, features_data=None, output_path="output/zones_map.html"):
        """创建交互式地图"""
        logger.info("创建交互式地图")
        
        # 计算中心点
        centroid = zones_data.geometry.centroid.iloc[0]
        center = [centroid.y, centroid.x]
        
        # 创建基础地图
        m = folium.Map(location=center, zoom_start=13, 
                      tiles="CartoDB positron")
        
        # 添加功能区图层
        zones_layer = self._create_zones_layer(zones_data)
        zones_layer.add_to(m)
        
        # 添加POI点（示例）
        if features_data is not None:
            poi_layer = self._create_poi_layer(features_data)
            poi_layer.add_to(m)
        
        # 添加图层控制
        folium.LayerControl().add_to(m)
        
        # 添加全屏控件
        plugins.Fullscreen().add_to(m)
        
        # 添加比例尺
        plugins.ScaleBar().add_to(m)
        
        # 保存地图
        m.save(output_path)
        logger.info(f"地图已保存至: {output_path}")
        
        return output_path
    
    def _create_zones_layer(self, zones_data):
        """创建功能区图层"""
        # 定义颜色映射
        zone_types = zones_data["zone_type"].unique()
        colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", 
                 "#FFEAA7", "#DDA0DD", "#98D8C8"]
        
        color_map = {zone: colors[i % len(colors)] 
                    for i, zone in enumerate(zone_types)}
        
        # 创建GeoJson图层
        def style_function(feature):
            zone_type = feature["properties"]["zone_type"]
            return {
                "fillColor": color_map.get(zone_type, "#CCCCCC"),
                "color": "#000000",
                "weight": 1,
                "fillOpacity": 0.6
            }
        
        def highlight_function(feature):
            return {
                "weight": 3,
                "color": "#FF0000",
                "fillOpacity": 0.8
            }
        
        # 添加交互
        zones_geojson = folium.GeoJson(
            zones_data,
            name="功能区分布",
            style_function=style_function,
            highlight_function=highlight_function,
            tooltip=folium.GeoJsonTooltip(
                fields=["zone_type", "poi_count", "poi_diversity"],
                aliases=["功能区类型", "POI数量", "多样性指数"],
                localize=True
            ),
            popup=folium.GeoJsonPopup(
                fields=["zone_type", "poi_count", "ndvi_mean", "road_density"],
                aliases=["类型", "POI数", "植被指数", "路网密度"],
                localize=True
            )
        )
        
        return zones_geojson
    
    def _create_poi_layer(self, features_data):
        """创建POI点图层"""
        # 示例：创建热点图
        # 实际应该从原始POI数据创建
        
        heat_data = []
        for _, row in features_data.iterrows():
            if row["poi_count"] > 0:
                centroid = row["geometry"].centroid
                heat_data.append([centroid.y, centroid.x, row["poi_count"]])
        
        heat_layer = plugins.HeatMap(heat_data, name="POI热点", radius=15)
        return heat_layer
    
    def create_change_animation(self, zones_t1, zones_t2, output_path="output/change_animation.html"):
        """创建变化动画"""
        logger.info("创建变化动画")
        
        # 计算变化
        changes = self._calculate_changes(zones_t1, zones_t2)
        
        # 创建时间序列地图
        # 这里简化实现，实际应该使用TimeSliderChoropleth
        
        m = folium.Map(location=[23.13, 113.26], zoom_start=12)
        
        # 添加两期数据
        folium.GeoJson(
            zones_t1,
            name="2023年功能区",
            style_function=lambda x: {"fillColor": "#4ECDC4", "fillOpacity": 0.6}
        ).add_to(m)
        
        folium.GeoJson(
            zones_t2,
            name="2024年功能区",
            style_function=lambda x: {"fillColor": "#FF6B6B", "fillOpacity": 0.6}
        ).add_to(m)
        
        # 保存
        m.save(output_path)
        return output_path
    
    def _calculate_changes(self, zones_t1, zones_t2):
        """计算变化区域"""
        # 简化实现
        return {"changed_cells": len(zones_t1) - len(zones_t1.merge(zones_t2, on="grid_id"))}