import re
import spacy
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.cluster import rand_score, v_measure_score, adjusted_rand_score
import seaborn as sns
from pprint import pprint
from sklearn.cluster import KMeans

occasion_factor = 0.5
num_clusters = 928

df = pd.read_csv('amazon.csv', usecols=['product_name', 'about_product', 'discounted_price', 'discount_percentage', 'category'])

names = df['product_name'].to_numpy()
abouts = df['about_product'].to_numpy()
for i in range(len(df)):
    abouts[i] = names[i] + ";" + abouts[i]
prices = df['discounted_price'].to_numpy()
percentage = df['discount_percentage'].to_numpy()
categories = df['category'].tolist()

vectorizer = TfidfVectorizer(ngram_range=(2, 2), stop_words='english')
vectorized_documents = vectorizer.fit_transform(abouts)


kmeans = KMeans(n_clusters=num_clusters, n_init=5,
                max_iter=500, random_state=42)
kmeans.fit(vectorized_documents)
results = kmeans.labels_

clusters = [[] for i in range(num_clusters)]
for i in range(len(df)):
    price = prices[i]
    price = re.sub("[^0-9.]", "", price)
    clusters[results[i]].append({"price": float(price), "name": names[i]})

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


pprint(clusters[0])

distance_thresholds = []
rand_scores = []
v_measures = []
adjusted_rand_scores = []
linkages = []
metrics = []
for metric in ['euclidean', 'cosine', 'manhattan', 'l1', 'l2']:
    for linkage in ['single', 'complete', 'average']:
        for distance_threshold in np.linspace(0, 1, 11):
            clustering = AgglomerativeClustering(metric=metric, linkage=linkage, distance_threshold=distance_threshold, compute_full_tree=True, n_clusters=None)
            clustering.fit(vectorized_documents.toarray())
            results = clustering.labels_
            metrics.append(metric)
            linkages.append(linkage)
            rand_scores.append(rand_score(results, categories))
            v_measures.append(v_measure_score(results, categories))
            adjusted_rand_scores.append(adjusted_rand_score(results, categories))
            distance_thresholds.append(distance_threshold)


metrics_df = pd.DataFrame({
    'distance_thresholds': distance_thresholds,
    'rand_scores': rand_scores,
    'v_measures': v_measures,
    'adjusted_rand_scores': adjusted_rand_scores,
    'linkages': linkages,
    'metrics': metrics
})

from matplotlib import pyplot as plt
import random
plt.rcParams["figure.figsize"] = (20,10)

fig, ax = plt.subplots()

grouped = metrics_df.groupby(['linkages', 'metrics'])
line_styles = ['dashed', 'solid', 'dotted', 'dashdot']

for style_num, ((linkage, metric), group) in enumerate(grouped):
    sns.lineplot(x='distance_thresholds', y='rand_scores', data=group, label=f'{linkage}-{metric}', ax=ax, linestyle=random.choice(line_styles))

ax.set_title('Rand Scores by Distance Thresholds')
ax.set_xlabel('Distance Thresholds')
ax.set_ylabel('Rand Scores')
ax.set_ylim(0, 1)

plt.legend(title='Linkage-Metric')
plt.show()

fig, ax = plt.subplots()

grouped = metrics_df.groupby(['linkages', 'metrics'])
line_styles = ['dashed', 'solid', 'dotted', 'dashdot']

for style_num, ((linkage, metric), group) in enumerate(grouped):
    sns.lineplot(x='distance_thresholds', y='v_measures', data=group, label=f'{linkage}-{metric}', ax=ax, linestyle=random.choice(line_styles))

ax.set_title('V-measures by Distance Thresholds')
ax.set_xlabel('Distance Thresholds')
ax.set_ylabel('V-measures')
ax.set_ylim(0, 1)

plt.legend(title='Linkage-Metric')
plt.show()

fig, ax = plt.subplots()

grouped = metrics_df.groupby(['linkages', 'metrics'])
line_styles = ['dashed', 'solid', 'dotted', 'dashdot']

for style_num, ((linkage, metric), group) in enumerate(grouped):
    sns.lineplot(x='distance_thresholds', y='adjusted_rand_scores', data=group, label=f'{linkage}-{metric}', ax=ax, linestyle=random.choice(line_styles))

ax.set_title('Adjusted Rand Scores by Distance Thresholds')
ax.set_xlabel('Distance Thresholds')
ax.set_ylabel('Adjusted Rand Scores')
ax.set_ylim(0, 1)

plt.legend(title='Linkage-Metric')
plt.show()

metrics_df

for (linkage, metric), group in grouped:
    print(group)
