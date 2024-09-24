from ML.model import SentenceTransformersOnnxInference
from ML.distance import Meter
import pandas as pd
import sys
import argparse
import numpy as np
import ast
import json


def find_nearest_cluster(tag):
    # инициализация модели
    model = SentenceTransformersOnnxInference("ML/checkpoints/sentence_transformer.onnx", "ML/tokenizer")
    # Чтение CSV
    centroids = pd.read_csv('data/centroids.csv', sep=',')
    # Преобразование строк обратно в numpy массивы
    centroids['centroid_emb'] = centroids['centroid_emb'].apply(lambda x: np.array(json.loads(x)))
    meter = Meter()
    tag_emb = model.get_embedding(tag)
    min_dist = meter.get_distance(tag_emb, centroids['centroid_emb'].values[0])
    nearest_tag_group = centroids['centroid_tag_group'].values[0]
    nearest_centroid = centroids['centroid_emb'].values[0]
    for index, row in centroids.iterrows():
        distance = meter.get_distance(tag_emb, row['centroid_emb'])
        if distance < min_dist:
            min_dist = distance
            nearest_centroid = row['centroid_emb']
            nearest_tag_group = row['centroid_tag_group']
    #расстояние с каким центроидом
    return min_dist, nearest_tag_group


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process words to find embeddings and clusters")
    parser.add_argument('-w', '--word', dest='word', required=True,
                        help='Word to find cluster')
    args = parser.parse_args()
    #возвращаем результат
    print(find_nearest_cluster(args.word))