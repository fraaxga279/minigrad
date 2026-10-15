Реализация небольшого фреймворка для нейросетей на NumPy, чтобы разобраться, как устроены основные части deep learning фреймворков: forward/backward, слои, функции потерь, оптимизаторы и обучение по батчам:

### слои
- Linear
- ReLU
- Sigmoid
- GELU
- Softmax
- LogSoftmax
- BatchNormalization
- Dropout
- Sequential

### функции потерь
- MSELoss
- CrossEntropyLoss

### оптимизаторы

- SGD
- Momentum
- Nesterov
- Adam
- Weight decay

### еще добавлены

- DataLoader
- Trainer
- train / eval режимы
- тесты
- gradient checking

для наглядности работы реализации взял датасет из открытого доступа, на нем обучил модель мультиклассовой классификации в ноутбуке wine.ipynb


### для запуска 
```bash
conda create -n minigrad python=3.11 -y
conda activate minigrad
pip install -e .