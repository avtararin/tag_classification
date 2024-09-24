# Tag classification
Консольное приложение для поиска ближайшего кластера для тэга по его эмбедингу.

A CLI utility that find nearest cluster for tag by its embedding.
## Installation
1. Клонируйте репозиторий
```commandline
git clone https://github.com/avtararin/tag_classification.git
cd tag_classification
```
2. Создайте и активируйте виртуальное окружение
```commandline
# For Unix/macOS
python3 -m venv venv
source venv/bin/activate

# For Windows
python -m venv venv
.\venv\Scripts\activate
```
3. Установить необходимые зависимости
```commandline
pip install -r requirements.txt

```
## Примеры
Использование приложения с длинным флагом и коротким тэгом. Коротким считается тэг, состоящий из одного слова.
```python
python cli.py --word бег
# (array([0.26998966]), 'Спорт')
```
Вывод модели: кортеж, состоящий из двух элементов
1. Расстояние до ближайшего центроида
2. Название кластера, в который попал тэг

Использование приложения с длинным флагом и сложным тэгом, состоящим из нескольких слов.

Чтобы подать на вход приложению предложение, его нужно обернуть в <b>кавычки</b>.
```python
python cli.py -w "люблю готовить торты"
# (array([0.43309414]), 'Кулинария')
```
## Аргументы
```
usage: cli.py [-h] -w WORD

Process words to find embeddings and clusters

options:
  -h, --help            show this help message and exit
  -w WORD, --word WORD  Word to find cluster

```