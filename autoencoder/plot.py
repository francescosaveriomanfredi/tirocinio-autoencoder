import seaborn as sns
from sklearn.decomposition import PCA
import pandas as pd
import plotly.express as px

def joinplot_pca(df:pd.DataFrame, cluster=None, palette=None, size=10, alpha=0.3):
    pca = PCA(n_components=2)
    pca_df= pd.DataFrame(pca.fit_transform(df), columns=["pca_1", "pca_2"])
    sns.jointplot(data=pca_df, x="pca_1", y="pca_2", hue=cluster, alpha=alpha, palette=palette, height=size)
    return pca_df

def plot_CSVLogger(data, metrics,):
    try:
        history = pd.read_csv(data, index_col="epoch")
    except:
        history = data
    history = history.groupby(level=0).mean()
    fig = px.line(history)
    fig = fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                direction="right",
                x=0.7,
                y=1.2,
                showactive=True,
                buttons=list(
                    [
                    
                        dict(
                            label=metric,
                            method="update",
                            args=[
                                {"visible": list(history.columns.str.endswith(metric))},
                                {"yaxis.title.text": metric},
                            ],
                        )for metric in metrics
                    ]
                ),
            )
        ]
    )
    return fig

def plot_scatter(
    X_pca, 
    variance_ratio=None, 
    color=None, 
    color_discrete_sequence=None,
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
        #range_color=[0,1]
    )

    fig.update_layout(title = "PCA",
                      xaxis_title = f"PCA1"+variance_ratio[0],
                      yaxis_title = f"PCA2"+variance_ratio[1],
                      width = 2000,
                      height = 700,
                      )

    config = dict({'scrollZoom': True})
    # add dropdown menus to the figure
    return fig.show(config=config)
