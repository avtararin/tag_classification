from ML.model import SentenceTransformersOnnxInference
import pandas as pd
a = SentenceTransformersOnnxInference("ML/checkpoints/sentence_transformer.onnx", "ML/tokenizer")


df = pd.read_csv("data/df_embs.csv", sep=',')
print(len(df.T[0].tolist()[1:]))
print(df.loc[:, 'tag'].tolist()[0])
print(df.T[0].tolist()[1:])