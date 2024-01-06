import re

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

num_clusters = 200
occasion_factor = 0.5

if __name__ == '__main__':
    print("hello")
    df = pd.read_csv('amazon.csv', usecols=['product_name', 'about_product', 'discounted_price', 'discount_percentage'])

    names = df['product_name'].to_numpy()
    abouts = df['about_product'].to_numpy()
    for i in range(len(df)):
        abouts[i] = names[i] + ";" + abouts[i]
    prices = df['discounted_price'].to_numpy()
    percentage = df['discount_percentage'].to_numpy()

    vectorizer = TfidfVectorizer(stop_words='english')
    vectorized_documents = vectorizer.fit_transform(abouts)

    kmeans = KMeans(n_clusters=num_clusters, n_init=5,
                    max_iter=500, random_state=42)
    kmeans.fit(vectorized_documents)
    results = kmeans.labels_

    clusters = [[] for i in range(num_clusters)]
    for i in range(len(df)):
        price = prices[i]
        price = re.sub("[^0-9.]", "", price)
        clusters[results[i]].append({"price": float(price), "desc": abouts[i]})

    averages = [0]*num_clusters
    for i in range(len(clusters)):
        cluster = clusters[i]
        for j in range(len(cluster)):
            element = cluster[j]
            averages[i] = averages[i] + element["price"]
        averages[i] = averages[i]/len(clusters[i])

    for i in range(len(clusters)):
        cluster = clusters[i]
        for j in range(len(cluster)):
            element = cluster[j]
            if element["price"] < occasion_factor*averages[i]:
                print(element)

    print(clusters[10])

    print("end")
