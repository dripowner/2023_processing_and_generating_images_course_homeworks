ФИО: Голдобин Илья

Предмет: Обработка и генерация изображений

Задача: Множественная классификация объектов (10 классов)

Классы:

- plane
- car
- bird
- cat
- deer
- dog
- frog
- horse
- ship
- truck

Датасет: CIFAR-10

Архитектура: ResNet-50

Гиперпараметры:
- batch size: 128
- optimizer: SGD
- learning rate: 0.001
- momentum: 0.9
- loss: cross-entropy
- device: cuda
- epochs: 20


Эксперименты:
1. 100 % датасета
    - Лоссы в обучении
        - ![Предобученный feature extractor](./exp1/results/SL_pretrained/validation_loss.png)
        - ![Рандомно инициализированный feature extractor](./exp1/results/SL/validation_loss.png)
    - Метрики в обучении
        - ![Предобученный feature extractor](./exp1/results/SL_pretrained/validation_metrics.png)
        - ![Рандомно инициализированный feature extractor](./exp1/results/SL/validation_metrics.png)
2. 100 % датасета
    - Лоссы в обучении
        - ![Предобученный feature extractor](./exp2/results/SL_pretrained/validation_loss.png)
        - ![Рандомно инициализированный feature extractor](./exp1/results/SL/validation_loss.png)
    - Метрики в обучении
        - ![Предобученный feature extractor](./exp2/results/SL_pretrained/validation_metrics.png)
        - ![Рандомно инициализированный feature extractor](./exp1/results/SL/validation_metrics.png)
3. 100 % датасета
    - Лоссы в обучении
        - ![Предобученный feature extractor](./exp3/results/SL_pretrained/validation_loss.png)
        - ![Рандомно инициализированный feature extractor](./exp1/results/SL/validation_loss.png)
    - Метрики в обучении
        - ![Предобученный feature extractor](./exp3/results/SL_pretrained/validation_metrics.png)
        - ![Рандомно инициализированный feature extractor](./exp1/results/SL/validation_metrics.png)

Вывод:
Во всех экспериментах модель с рандомно инициализированным feature extractor обучается быстрее и достигает больших метрик на валидации