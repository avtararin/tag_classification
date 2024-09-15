from ML.model import SentenceTransformersOnnxInference
from ML.distance import Meter
import pandas as pd
import sys


def find_nearest_cluster(tag):
    # инициализация модели
    model = SentenceTransformersOnnxInference("ML/checkpoints/sentence_transformer.onnx", "ML/tokenizer")
    centroids = pd.read_csv("data/kmeans_centroids.csv", sep=" , ").to_numpy()
    meter = Meter()
    tag_emb = model.get_embedding(tag)
    min_dist = meter.get_distance(tag_emb, centroids[0])
    nearest_centroid = centroids[0]
    for centroid in centroids:
        distance = meter.get_distance(tag_emb, centroid)
        if distance < min_dist:
            min_dist = distance
            nearest_centroid = centroid
    return nearest_centroid


if __name__ == "__main__":
    print(find_nearest_cluster(sys.argv[1]))
