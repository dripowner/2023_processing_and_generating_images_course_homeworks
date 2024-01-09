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
- batch size 128
- learning rate 0.001
- momentum 0.9
- loss cross-entropy
- device cuda
- epochs 15/25

Логгирование(wandb)
- [Метрки по классам](./wandb_report/wandb1.png)
- [Лоссы и метрики](./wandb_report/wandb2.png)

Результаты обучения:
- График лосс функции в процессе обучения
    - 15 эпох
    [лосс](./results/runs/2024-01-09_16_58_34_927901/validation_loss.png)
    - 25 эпох
    [лосс](./results/runs/2024-01-09_17_04_46_222598/validation_loss.png)
- График метрик на валидационной выборке во время обучения
    - 15 эпох
    [метрики](./results/runs/2024-01-09_16_58_34_927901/validation_metrcis.png)
    - 25 эпох
    [метрики](./results/runs/2024-01-09_17_04_46_222598/validation_metrcis.png)
- 5-10 примеров изображений с результатом работы сети
    [ноутбук](./show_images.ipynb)
- Значения метрик
    [ноутбук](./test.ipynb)