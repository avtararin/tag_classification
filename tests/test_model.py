import pytest
import pandas as pd
from ML.model import SentenceTransformersOnnxInference
from ML.distance import Meter

#тесты не проходят, как грамотно импортировать модель и с какой точностью это нужно делать

# Инициализация модели ONNX и загрузка эмбендингов для тестировния
onnx_model = SentenceTransformersOnnxInference("ML/checkpoints/sentence_transformer.onnx", "ML/tokenizer")
df = pd.read_csv("data/df_embs.csv", sep=',')


@pytest.mark.parametrize("index", range(10))  # Параметризация для тестирования нескольких примеров
def test_embedding_similarity(index):
    """
    Тест на сравнение эмбеддингов модели ONNX и сохраненного в CSV.
    :param index: индекс строки в CSV, для которой будет тестироваться эмбеддинг.
    """

    # Получаем тэг и его предобученный эмбеддинг из DataFrame
    tag = df.iloc[index, 0]  # Столбец с тэгами (первый столбец)
    saved_embedding = df.iloc[index, 1:].to_numpy()  # Остальные столбцы — это эмбеддинги

    # Получаем эмбеддинг через модель ONNX
    onnx_embedding = onnx_model.get_embedding(tag)

    # Вычисляем косинусное расстояние между эмбеддингами
    meter = Meter('cosine')
    distance = meter.get_distance(saved_embedding, onnx_embedding[0])

    # Проверка, что расстояние мало (эмбеддинги похожи)
    assert distance < 0.01, f"Embedding distance too high for tag '{tag}': {distance}"
