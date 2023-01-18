import os
import yaml
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

def plot_CSV_collection(
    base_dir="./lightning_logs",
    metrics=None,
    train=True,
    val=True
):   
    """
    function to fancy plot a list of metric 
    reported by multiple pytorch lighting csv 
    logger with the same name, and dir

    Parameters
    ----------
    base_dir
        The name of the directory with 
        subdirectory called version_n  
        each subdirectory contains a file called metrics.csv
    metrics
        The subsample of metric to consider,
        example {nb_loss, kl_loss, crossentropy, recall, ...}
        if None take all metric
    val
        Choice if include val metrics in the plot
    train
        Choice if include train metrics in the plot
    """
    dirs = os.listdir(base_dir)
    palettes = [px.colors.qualitative.Set1, px.colors.qualitative.Pastel1]
    modes = list(l for l, c in  zip(["val", "train"], [val, train]) if c) 
    button_label = []
    fig = go.Figure()

    for i, dir in enumerate(dirs):
      # leggo il file csv
      try:
        df = pd.read_csv(
            base_dir+dir+"/metrics.csv",
            index_col="epoch"
        ).groupby(level=0).mean()
      except: continue
      for mode, palette in zip(modes, palettes):
        # filtro per modo 
        df_mode = df.filter(like=mode, axis=1)
        df_mode.columns = df_mode.columns.str.replace(mode+"_","")
        # se le metriche non sono specificate le prendo tutte
        if metrics is None:
          metrics=df_mode.columns
        for metric in (metrics):
          # seleziono la metrica
          sr_metric = df_mode.loc[:,df_mode.columns.str.contains(metric)].squeeze()
          # salto se la metrica non e presente
          try:
            if sr_metric.shape[0] == 0: continue
          except:
            raise Exception(dir)
          fig.add_trace(go.Scatter(
                            x=sr_metric.index,
                            y=sr_metric.values,
                            visible=True,
                            name=dir+"/"+mode,
                            line=dict(color=palette[i])
                           )
                )
          button_label.append(metric)
    
    # create button
    buttons = []
    button_label = np.array(button_label)
    for metric in set(metrics):
      traces = button_label == metric
      buttons.append(dict(method='update',
                          label=metric,
                          visible=True,
                          args=[{'visible': traces}],
                          )
                  )
    # create the layout 
    layout = go.Layout(
        updatemenus=[
            dict(
                type='buttons',
                direction='right',
                x=0.8,
                y=1.2,
                showactive=True,
                buttons=buttons
            )
        ],
        title=dict(text='metric plot'),
        showlegend=True
    )
    return fig.update_layout(layout)


def df_yaml_collection(
    base_dir="./lightning_logs",
    hparams_label=None,
):
    """
    function to return a dataframe  with the parameter 
    of each model.
    """
    dirs=os.listdir(base_dir)
    filename="hparams.yaml"
    df = pd.DataFrame()
    hparams={}
    cells = []
    fig = go.Figure()
    for sub_dir in dirs:
      with open("/".join([base_dir,sub_dir,filename]), 'r') as file:
          hparams = yaml.safe_load(file)

      df=df.append(
          {label: hparams[label] for label in hparams}, 
          ignore_index=True
          )
    if hparams_label is not None:
        df = df[hparams_label]

    df.index = pd.Index(dirs, name="version")
    return df

def plot_scatter(
    X_pca, 
    variance_ratio=None, 
    color=None, 
    color_discrete_sequence=None,
    symbol=None
):
    if variance_ratio is None:
        variance_ratio = ["",""]
    else:
        variance_ratio = [f"({variance_ratio[0]:.2})", f"({variance_ratio[1]:.2})"]
    
    color_discrete_map = None
    if color_discrete_sequence is not None:
        color_discrete_map = dict(zip(
            color.cat.categories,
            color_discrete_sequence
        ))
    
    fig = px.scatter(
        x=X_pca[:,0],
        y=X_pca[:,1],
        color=color,
        color_discrete_map=color_discrete_map,
        opacity=0.3,
        marginal_x="histogram",
        marginal_y="histogram",
        #color_continuous_midpoint=color.quantile(0.75)
        range_color=[0, color.quantile(0.8)],
        symbol=symbol
    )
    
    fig.update_layout(title = "PCA",
                      xaxis_title = f"PCA1"+variance_ratio[0],
                      yaxis_title = f"PCA2"+variance_ratio[1],
                      width = 2000,
                      height = 700,
                      #legend_y = 1.05,
                      #legend_x = 1.035,
                      coloraxis_colorbar_x = -0.10
                      )

    #config = dict({'scrollZoom': True})
    # add dropdown menus to the figure
    return fig







