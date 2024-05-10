import os
from dash import Dash, html, dcc, Input, Output, State, dash_table
from dash.dash_table.Format import Format, Scheme, Trim

import plotly.express as px
import plotly.graph_objects as go

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import pickle
import SimpleITK as sitk

from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import classification_report

import argparse

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

def main(app, args):

    mount_point = args.mount_point

    csv_path = args.csv

    df = pd.read_csv(csv_path)


    @app.callback(
        Output('studies-img', 'figure'),
        Input('studies-img', 'figure'))
    def studies_img(fig):

        fig_img = go.Figure()

        df_real = df.query('source == "Real - C1 random sample"')

        fig = go.Scatter(x=df_real['tsne_x'], y=df_real['tsne_y'], mode='markers', opacity=0.3, showlegend=False)

        fig_img.add_trace(fig)



        df_simu = df.query('source != "Real - C1 random sample"')
        # fig_simu = px.scatter(df_simu, x="tsne_x", y="tsne_y",
        #             hover_data=[df_simu.index],
        #             color='source')
        
        # Create a color map based on unique sources
        

        color_map = {"Real - Juan's sample": 'blue', 'Simulated': 'red'}  # Define more colors if there are more categories

        # Map the 'source' column to actual colors
        colors = df_simu['source'].map(color_map)

        fig = go.Scatter(x=df_simu['tsne_x'], y=df_simu['tsne_y'], marker=dict(color=colors), mode='markers')
        fig_img.add_trace(fig)
        
        # for trace in fig_simu.data:
        #     fig_img.add_trace(trace)

        
        
        fig_img.update_layout(
            title="Combined Scatter Plot with Real and Simulated Data",
            xaxis_title="t-SNE X",
            yaxis_title="t-SNE Y",
            width=1024,
            height=1024

        )

        return fig_img


    @app.callback(
        Output('study-img', 'figure'),   
        Input('studies-img', 'clickData'))
    def update_img(dict_points):
        
        fig_img = go.Figure()
        img_path = ""
        idx = -1
        
        if dict_points is not None and dict_points["points"] is not None and len(dict_points["points"]) > 0:
            
            idx = dict_points["points"][0]["pointIndex"]
            
            img_path = os.path.join(mount_point, df.loc[idx]["img_path"])
            print("Reading:", img_path)
            img_np = sitk.GetArrayFromImage(sitk.ReadImage(img_path))
            img_np = (img_np - np.min(img_np)) / (np.max(img_np) - np.min(img_np)) * 255
            img_np = img_np.astype(np.ubyte)
            print(np.min(img_np), np.max(img_np))

            # img_path_grad = os.path.join(grad_cam_path, img_path)

            fig_img.add_trace(go.Heatmap(z=np.flip(img_np, axis=0), colorscale='gray'))

            # if os.path.exists(img_path_grad):

            #     img_np_grad = sitk.GetArrayFromImage(sitk.ReadImage(img_path_grad)).astype(np.ubyte)
            
            #     fig_img.add_trace(go.Heatmap(z=np.flip(img_np_grad, axis=0), colorscale='jet', opacity=0.3, showlegend=False))

            fig_img.update_layout(
                width=1024,
                height=1024
            )

        return fig_img

    app.layout = html.Div(children=[
        html.H1(children='Simu - Web Analysis App'),
        html.Div([
            html.Div([
                html.Div(
                    [
                    # dcc.Dropdown([df['source'].drop_duplicates(), 'Simulated'], id='colorby-dropdown'),
                    # dcc.RangeSlider(0, 1, 0.01, value=[.1, 1], id='score-range-slider', marks={ 0: {'label': '0'}, 1: {'label': '1'}}, tooltip={"placement": "bottom", "always_visible": True}),
                    # dcc.RangeSlider(ga_boe_range[0], ga_boe_range[1], 1, value=[ga_boe_range[0], ga_boe_range[1]], id='ga-range-slider', marks=ga_boe_range_marks, tooltip={"placement": "bottom", "always_visible": True}),
                    # dcc.Graph(id='studies-img-clusters'),
                    dcc.Graph(id='studies-img')],
                    className='six columns'
                ),
                html.Div(
                    [
                        dcc.Graph(id='study-img'),
                        # dcc.Slider(0, 1500, 10, value=450, id='img-size', marks={ 0: {'label': '0'}, 1500: {'label': '1500'}}),
                        # html.Div([
                        #     html.Div(html.H3('', id='study-index'), className='two columns'),
                        #     html.Div(html.H3('id:', id='study-id'), className='ten columns')                    
                        # ], className='row'),
                    ],
                    className='six columns'
                    )
            ], className='row')
        ])
    ])

if __name__ == '__main__':
    

    parser = argparse.ArgumentParser(description='Simulated data app')
    parser.add_argument('--csv', type=str, help='Path to csv file', required=True)
    parser.add_argument('--mount_point', type=str, help='Mount point for data', required=True)
    args = parser.parse_args()


    app = Dash(__name__, external_stylesheets=external_stylesheets)

    main(app, args)

    app.run_server(debug=True, port=8787)