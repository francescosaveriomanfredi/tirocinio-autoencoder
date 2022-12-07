import seaborn as sns
from sklearn.decomposition import PCA
import pandas as pd

def joinplot_pca(df:pd.DataFrame, cluster=None, palette=None, size=10, alpha=0.3):
    pca = PCA(n_components=2)
    pca_df= pd.DataFrame(pca.fit_transform(df), columns=["pca_1", "pca_2"])
    sns.jointplot(data=pca_df, x="pca_1", y="pca_2", hue=cluster, alpha=alpha, palette=palette, height=size)
    return pca_df

def print_CSVlogger(trainer, metric):
    history = pd.read_csv(f"{trainer.logger.log_dir}/metrics.csv", index_col="epoch")
    axe=history["metric"].plot.line()
    return axe
