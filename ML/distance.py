import numpy as np


class Meter:
    def __init__(self, method='cosine'):
        self.method = method

    @staticmethod
    def cosine_distance(embedding_1, embedding_2):
        """Рассчитывает косинусное расстояние между двумя эмбеддингами"""
        cosine_similarity = np.dot(embedding_1, embedding_2) / (
                np.linalg.norm(embedding_1) * np.linalg.norm(embedding_2))
        return 1 - cosine_similarity

    @staticmethod
    def euclidean_distance(embedding_1, embedding_2):
        """Рассчитывает евклидово расстояние между двумя эмбеддингами"""
        return np.linalg.norm(embedding_1 - embedding_2)

    def get_distance(self, embedding_1, embedding_2):
        """Вызывает нужный метод расчета расстояния в зависимости от выбранного метода"""
        if self.method == 'cosine':
            return Meter.cosine_distance(embedding_1, embedding_2)
        elif self.method == 'euclidean':
            return Meter.euclidean_distance(embedding_1, embedding_2)
        else:
            raise ValueError(f"Метод {self.method} не поддерживается")
