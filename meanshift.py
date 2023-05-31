import pandas as pd
from sklearn.cluster import MeanShift

if __name__ == "__main__":

    dataset = pd.read_csv("./data/sensor.csv")
    #print(dataset.head(5))

    X = dataset.drop('rain', axis=1)

    meanshift = MeanShift().fit(X)
    print(max(meanshift.labels_))
    print("="*64)
    print(meanshift.cluster_centers_)

    dataset['meanshift'] = meanshift.labels_
    print("="*64)
    print(dataset)

    #implementacion_meanshift