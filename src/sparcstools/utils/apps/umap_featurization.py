import pandas as pd
from dash import Dash, html, dcc, Input, Output, callback
from jupyter_dash import JupyterDash
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.subplots as sp


from sparcstools.dataset_manipulation import get_indexes
import h5py
import os
import sys



path_csv = 'test_umap_Florian.csv'
project_location = "/fs/pool/pool-mann-dvp/02_Users/FAS/OperaPhenix/AATD/Florian_EXP230602_SytoxG_panCadh_2C1__2023-06-02T14_59_34-Measurement1/Datasets/cyto2/AATD_Aachen-test-SytoxG_panCadh_2C1/"
channel_of_interest = "AADT"
n_masks = 1

data_umap = pd.read_csv(path_csv, index_col = 0)
df = pd.read_csv(path_csv)
df.slidename = df.slidename.astype('category')


#define path to stylesheet

app = JupyterDash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

#read dataset for plotting umap

def plot_umap(df, variable_color):
    fig = px.scatter(data_frame = df,
                     x="UMAP1",
                     y="UMAP2",
                     color = variable_color,
                     hover_name="cell_id",
                     custom_data = "cell_id",
                )
    fig.update_layout(clickmode='event+select',
                  plot_bgcolor='white'
                  )
    
    fig.update_xaxes(
        mirror=True,
        ticks='outside',
        showline=True,
        linecolor='black',
        showgrid=False
    )

    fig.update_yaxes(
        mirror=True,
        ticks='outside',
        showline=True,
        linecolor='black',
        showgrid=False
    )
    
    return(fig)

def create_image_data_all(cellID, project_location):

    hf_index = get_indexes(project_location, cellID)

    with h5py.File(f"{project_location}/extraction/data/single_cells.h5","r") as hf:
        img = hf.get("single_cell_data")[hf_index]
    
    if n_masks == 1:
        fig = sp.make_subplots(rows=1, cols=4, horizontal_spacing=0, subplot_titles=("Mask", "Nucleus", "Membrane", channel_of_interest))    
        traces_to_add = [go.Heatmap(z=img[0][1], colorscale='viridis', showscale = False),
                        go.Heatmap(z=img[0][2], colorscale='viridis', showscale = False),
                        go.Heatmap(z=img[0][3], colorscale='viridis', showscale = False),
                        go.Heatmap(z=img[0][4], colorscale='viridis', showscale = False)]
    elif n_masks ==2:
        fig = sp.make_subplots(rows=1, cols=5, horizontal_spacing=0, subplot_titles=("Mask_Nuc","Mask_Cyto", "Nucleus", "Membrane", channel_of_interest))    
        traces_to_add = [go.Heatmap(z=img[0][0], colorscale='viridis', showscale = False),
                        go.Heatmap(z=img[0][1], colorscale='viridis', showscale = False),
                        go.Heatmap(z=img[0][2], colorscale='viridis', showscale = False),
                        go.Heatmap(z=img[0][3], colorscale='viridis', showscale = False),
                        go.Heatmap(z=img[0][4], colorscale='viridis', showscale = False)]
        
    # Get the Express fig broken down as traces and add the traces to the proper plot within in the subplot
    for i, trace in enumerate(traces_to_add):
        fig.add_trace(trace, row=1, col=i+1)

    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.layout.coloraxis.showscale = False
    fig.update_traces(hoverinfo='skip', hovertemplate = None)
    fig.update_layout(yaxis_scaleanchor="x", yaxis2_scaleanchor="x2", yaxis3_scaleanchor="x3", yaxis4_scaleanchor="x4")
    
    return fig

def create_image_data(cellID, project_location, colorscale = "viridis", channel = 4):

    hf_index = get_indexes(project_location, cellID)

    with h5py.File(f"{project_location}/extraction/data/single_cells.h5","r") as hf:
        img = hf.get("single_cell_data")[hf_index]
    
    fig = go.Figure(go.Heatmap(z=img[0][channel], colorscale=colorscale, showscale = False))
    
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.layout.coloraxis.showscale = False
    fig.update_traces(hoverinfo='skip', hovertemplate = None)
    fig.update_layout(yaxis_scaleanchor="x")

    return fig

#inititialize plot
fig = plot_umap(df, "slidename")

#get colnames for plotting
col_names = df.columns.unique().tolist()
col_names.remove("UMAP1")
col_names.remove("UMAP2")
col_names.remove("cell_id")
print(col_names)

empty_fig_all = create_image_data_all([159], project_location)
empty_fig_single = create_image_data([159], project_location)

app.layout = html.Div([
    dbc.Row([
        dbc.Col(
            dbc.Row(
                [
                    dbc.Row([
                        html.H3("Featurization of Images via UMAP", className='text-center')
                        ],
                        justify="center", align="center", className="header-50"
                    ),
                    
                    dbc.Row([
                        dbc.Col(
                            html.Label(['Color UMAP by'], style={'font-weight': 'bold', "text-align": "center"}),
                            width="auto"
                        ),
                        dbc.Col(
                            dcc.Dropdown(
                                col_names,
                                "slidename",
                                id="select_var_color"
                            ),
                            width="4"
                        )
                        
                    ]
                    ),
                    
                    dcc.Graph(figure = fig,
                         id='UMAP_Scatter',
                         hoverData={'points': [{'customdata': 'CellID'}]},
                         style = {"height":"90vh"}
                    )       
                ], 
                justify="center", style = {"height":"100vh"},
            )
        ),
        dbc.Col(
            dbc.Row(
                [
                    dbc.Row(
                        [
                            html.H3("Visualization of single-cell images", className='text-center'),
                        ],
                        justify="center", align="center", className="header-50"
                    ),
                    dbc.Row(
                        [
                            dcc.Graph(
                                figure = empty_fig_all, 
                                id='image_cells_all',
                                style = {'height': '30vh', "valign":"top"}
                            ),
                        ], justify = "center",
                    ),
                    
                    dbc.Row(
                        [
                            dbc.Col(
                                dcc.Graph(
                                    figure = empty_fig_single,
                                    id='image_cells_single',
                                    style={'height': '70vh', "valign":"top"}
                                ),
                            ),
                            dbc.Col(
                                [
                                    dbc.Row(
                                        [
                                            dbc.Col(
                                                html.Label(['Colormap to Show Image'], style={'font-weight': 'bold', "text-align": "center"}),
                                                width="auto"
                                            ),
                                            dbc.Col(
                                                dcc.Dropdown(
                                                    ["viridis", "Greys_r", "Blues_r", "Reds_r", "Greens_r", "Oranges_r"],
                                                    "viridis",
                                                    id="select_colormap"
                                                ),
                                                width="4"
                                            )
                                        ]
                                    ),
                                    dbc.Row(
                                        [
                                            dbc.Col(
                                                html.Label(['Channel to Show'], style={'font-weight': 'bold', "text-align": "center"}),
                                                width="auto"
                                            ),
                                            dbc.Col(
                                                dcc.Dropdown(
                                                    options= [
                                                               {'label': 0, 'value': 0},
                                                               {'label': 1, 'value': 2},
                                                               {'label': 2, 'value': 3},
                                                               {'label': 3, 'value': 4},
                                                            ] if n_masks == 1 else [
                                                               {'label': 0, 'value': 0},
                                                               {'label': 1, 'value': 1},
                                                               {'label': 2, 'value': 2},
                                                               {'label': 3, 'value': 3},
                                                               {'label': 4, 'value': 4},
                                                            ]
                                                        ,
                                                     value = 4,
                                                    id="select_channel"
                                                ),
                                                width="4"
                                            )
                                        ]
                                    )
                                ]
                                 
                            )
                        ],
                        justify="center"
                    )
                    
                ]
            )
        )
    ])
])


@callback(
    Output('image_cells_all', 'figure'),
    Input('UMAP_Scatter', 'clickData'))

def update_image_cells_all(clickData):
    cellID = clickData['points'][0]["customdata"][0]
    return create_image_data_all([cellID], project_location)

@callback(
    Output('image_cells_single', 'figure'),
    Input('UMAP_Scatter', 'clickData'),
    Input("select_colormap", "value"),
    Input("select_channel", "value"))

def update_image_cells(clickData, value1, value2):
    
    cellID = clickData['points'][0]["customdata"][0]
    return create_image_data([cellID], project_location, value1, value2)

@callback(
    Output('UMAP_Scatter', 'figure'),
    Input('select_var_color', 'value')
)

def update_graph(value):
    return plot_umap(df, value)


#define layout elements


def _get_layout():
    print(test)

def start_app(host):

    #initialize app
    app = JupyterDash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
    
    #get layout
    app.layout = _get_layout()

    #run server
    app.run_server(host = host)