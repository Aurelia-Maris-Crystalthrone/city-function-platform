"""
交互式仪表板模块
"""
import dash
from dash import dcc, html, Input, Output
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import geopandas as gpd
import logging

logger = logging.getLogger(__name__)

class InteractiveDashboard:
    """交互式仪表板"""
    
    def __init__(self, config):
        self.config = config
        self.app = dash.Dash(__name__)
        self.data = None
        self.zones_data = None
        
    def load_data(self, zones_path, features_path=None):
        """加载数据"""
        logger.info("加载数据用于仪表板")
        
        # 加载功能区数据
        self.zones_data = gpd.read_file(zones_path)
        
        # 加载特征数据（如果有）
        if features_path:
            self.features_data = pd.read_csv(features_path)
        else:
            self.features_data = None
        
        logger.info(f"数据加载完成: {len(self.zones_data)}个功能区网格")
        return self
    
    def create_layout(self):
        """创建仪表板布局"""
        self.app.layout = html.Div([
            # 标题
            html.H1("城市功能区动态分析平台", 
                   style={'textAlign': 'center', 'color': '#2c3e50'}),
            
            # 控制面板
            html.Div([
                html.Div([
                    html.Label("选择可视化图层:"),
                    dcc.Dropdown(
                        id='layer-selector',
                        options=[
                            {'label': '功能区分布', 'value': 'zones'},
                            {'label': 'POI密度', 'value': 'poi_density'},
                            {'label': '植被指数', 'value': 'ndvi'},
                            {'label': '路网密度', 'value': 'road_density'},
                            {'label': '城市活力', 'value': 'vitality'}
                        ],
                        value='zones',
                        style={'width': '200px'}
                    )
                ], style={'display': 'inline-block', 'marginRight': '20px'}),
                
                html.Div([
                    html.Label("功能区类型筛选:"),
                    dcc.Dropdown(
                        id='zone-filter',
                        multi=True,
                        style={'width': '300px'}
                    )
                ], style={'display': 'inline-block'})
            ], style={'padding': '20px', 'backgroundColor': '#f8f9fa'}),
            
            # 地图和图表区域
            html.Div([
                # 地图
                html.Div([
                    dcc.Graph(id='main-map', style={'height': '600px'})
                ], style={'width': '70%', 'display': 'inline-block'}),
                
                # 统计图表
                html.Div([
                    dcc.Tabs([
                        dcc.Tab(label='功能区统计', children=[
                            dcc.Graph(id='zone-stats-chart')
                        ]),
                        dcc.Tab(label='特征分布', children=[
                            dcc.Graph(id='feature-distribution')
                        ]),
                        dcc.Tab(label='变化趋势', children=[
                            dcc.Graph(id='change-trend')
                        ])
                    ])
                ], style={'width': '30%', 'display': 'inline-block', 'verticalAlign': 'top'})
            ]),
            
            # 数据表格
            html.Div([
                html.H3("详细数据"),
                html.Div(id='data-table-container')
            ], style={'padding': '20px'}),
            
            # 隐藏的数据存储
            dcc.Store(id='zones-data-store'),
            dcc.Store(id='features-data-store')
        ])
        
        return self.app
    
    def register_callbacks(self):
        """注册回调函数"""
        @self.app.callback(
            Output('zone-filter', 'options'),
            Input('zones-data-store', 'data')
        )
        def update_zone_filter_options(data):
            if data:
                df = pd.DataFrame(data)
                zone_types = df['zone_type'].unique()
                return [{'label': zone, 'value': zone} for zone in zone_types]
            return []
        
        @self.app.callback(
            Output('main-map', 'figure'),
            [Input('layer-selector', 'value'),
             Input('zone-filter', 'value'),
             Input('zones-data-store', 'data')]
        )
        def update_map(selected_layer, selected_zones, data):
            if not data:
                return go.Figure()
            
            df = pd.DataFrame(data)
            
            # 筛选数据
            if selected_zones:
                df = df[df['zone_type'].isin(selected_zones)]
            
            # 创建地图
            if selected_layer == 'zones':
                fig = self._create_zones_map(df)
            elif selected_layer == 'poi_density':
                fig = self._create_choropleth_map(df, 'poi_count', 'POI密度')
            elif selected_layer == 'ndvi':
                fig = self._create_choropleth_map(df, 'ndvi_mean', '植被指数(NDVI)')
            elif selected_layer == 'road_density':
                fig = self._create_choropleth_map(df, 'road_density', '路网密度')
            elif selected_layer == 'vitality':
                if 'urban_vitality' in df.columns:
                    fig = self._create_choropleth_map(df, 'urban_vitality', '城市活力指数')
                else:
                    fig = self._create_zones_map(df)
            
            fig.update_layout(
                mapbox_style="carto-positron",
                mapbox_zoom=12,
                mapbox_center={"lat": df['centroid_lat'].mean(), 
                             "lon": df['centroid_lon'].mean()}
            )
            
            return fig
        
        @self.app.callback(
            Output('zone-stats-chart', 'figure'),
            Input('zones-data-store', 'data')
        )
        def update_zone_stats(data):
            if not data:
                return go.Figure()
            
            df = pd.DataFrame(data)
            
            # 统计各功能区数量
            zone_counts = df['zone_type'].value_counts().reset_index()
            zone_counts.columns = ['zone_type', 'count']
            
            fig = px.pie(zone_counts, values='count', names='zone_type',
                        title='功能区类型分布',
                        hole=0.3)
            
            return fig
        
        @self.app.callback(
            Output('data-table-container', 'children'),
            [Input('main-map', 'clickData'),
             Input('zones-data-store', 'data')]
        )
        def update_data_table(click_data, data):
            if not data or not click_data:
                return html.Div("点击地图上的区域查看详细数据")
            
            df = pd.DataFrame(data)
            point = click_data['points'][0]
            
            # 根据点击的位置查找对应的网格
            # 这里简化处理，实际应该根据坐标查找
            
            # 显示前10行数据
            table = html.Table([
                html.Thead(
                    html.Tr([html.Th(col) for col in df.columns[:6]])
                ),
                html.Tbody([
                    html.Tr([
                        html.Td(df.iloc[i][col]) for col in df.columns[:6]
                    ]) for i in range(min(10, len(df)))
                ])
            ], style={'width': '100%'})
            
            return table
        
        logger.info("回调函数注册完成")
        return self
    
    def _create_zones_map(self, df):
        """创建功能区分布图"""
        # 为每种功能区分配颜色
        zone_types = df['zone_type'].unique()
        colors = px.colors.qualitative.Set3
        
        color_map = {}
        for i, zone_type in enumerate(zone_types):
            color_map[zone_type] = colors[i % len(colors)]
        
        df['color'] = df['zone_type'].map(color_map)
        
        # 创建散点图（简化，实际应该显示多边形）
        fig = px.scatter_mapbox(
            df,
            lat='centroid_lat',
            lon='centroid_lon',
            color='zone_type',
            hover_data=['zone_type', 'poi_count', 'ndvi_mean', 'road_density'],
            title='城市功能区分布',
            color_discrete_map=color_map
        )
        
        return fig
    
    def _create_choropleth_map(self, df, value_column, title):
        """创建等值区域图"""
        fig = px.choropleth_mapbox(
            df,
            geojson=self._geojson_from_gdf(df),
            locations='grid_id',
            color=value_column,
            hover_data=['zone_type', value_column],
            title=title,
            mapbox_style="carto-positron",
            center={"lat": df['centroid_lat'].mean(), 
                   "lon": df['centroid_lon'].mean()},
            zoom=12,
            opacity=0.5
        )
        
        return fig
    
    def _geojson_from_gdf(self, gdf):
        """从GeoDataFrame创建GeoJSON"""
        # 这里需要实际实现几何数据的转换
        # 简化返回
        return {"type": "FeatureCollection", "features": []}
    
    def run_server(self, debug=True, port=8050):
        """运行服务器"""
        logger.info(f"启动仪表板服务器，端口: {port}")
        
        # 准备数据
        if self.zones_data is not None:
            # 计算中心点
            self.zones_data['centroid_lon'] = self.zones_data.geometry.centroid.x
            self.zones_data['centroid_lat'] = self.zones_data.geometry.centroid.y
            
            # 转换为JSON
            zones_json = self.zones_data.drop(columns=['geometry']).to_dict('records')
            
            # 存储到dcc.Store
            self.app.layout.children[-2].data = zones_json
        
        self.app.run_server(debug=debug, port=port)
        return self