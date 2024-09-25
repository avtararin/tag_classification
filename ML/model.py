import onnxruntime as ort
import numpy as np
from transformers import AutoTokenizer


class SentenceTransformersOnnxInference:
    def __init__(self, model_path: str, tokenizer_path: str):
        """
        Инициализация модели ONNX и токенизатора.

        :type tokenizer_path: str
        :param model_path: Путь к файлу модели в формате .onnx.
        :param tokenizer_path: Название предобученного токенизатора BERT.
        """
        # Загрузка модели ONNX
        self.session = ort.InferenceSession(model_path)

        # Загрузка токенизатора
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    def get_embedding(self, tag):
        """
        Токенизация тэга и получение его эмбединга
        :param tag: Тэг для которого нужно получить эмбединг
        :return: эмбединг
        """
        tokens = self.tokenizer(tag,  padding=True, truncation=True, return_tensors="np")

        # Убедитесь, что input_ids и attention_mask имеют форму (1, sequence_length)
        input_ids = tokens['input_ids']
        attention_mask = tokens['attention_mask']

        onnx_inputs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }
        onnx_embedding: object = self.session.run(None, onnx_inputs)[0]
        return onnx_embedding
