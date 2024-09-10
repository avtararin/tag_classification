from ML.model import SentenceTransformersOnnxInference
from sentence_transformers import SentenceTransformer
from ML.distance import Meter
import pandas as pd

onnx_model = SentenceTransformersOnnxInference("ML/checkpoints/sentence_transformer.onnx", "ML/tokenizer")
df = pd.read_csv("data/df_embs.csv", sep=',')

def test_embeddings_1():
    onnx_embed = onnx_model.get_embedding(df.loc[:, 'tag'].tolist()[0])
    meter_cosine = Meter()
    distance = meter_cosine.get_distance(df.T[0].to_list()[1:], onnx_embed[0])
    assert distance <= 0, f"Embedding distance is too high: {distance}"
