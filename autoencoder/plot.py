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
