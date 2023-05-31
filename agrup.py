import pandas as pd
import seaborn as sns
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

if __name__ == "__main__":

    dataset = pd.read_csv('./data/dset.csv')
    #print(dataset.head(10))

    X = dataset.drop('Target', axis=1)
    #Selecionamos 4 grupos
    kmeans = MiniBatchKMeans(n_clusters=4, batch_size=8).fit(X)
    print("Total de centros: " , len(kmeans.cluster_centers_))
    print("="*64)
    print(kmeans.predict(X))

    dataset['agroup'] = kmeans.predict(X)
    print(dataset)
    sns.pairplot(dataset[['Marital','Course','Target','agroup']], hue = 'agroup')
    plt.show()
    #implementacion_k_means