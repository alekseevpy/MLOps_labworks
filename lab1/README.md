## Module 1

* Необходимо из создать простейший конвейер для автоматизации работы с моделью машинного обучения. 
* Отдельные этапы конвейера машинного обучения описываются в разных python–скриптах, которые потом соединяются в единую цепочку действий с помощью bash-скрипта.
* Все файлы необходимо разместить в подкаталоге lab1 корневого каталога

Этапы:
1. Создайте python-скрипт (data_creation.py), который создает различные наборы данных, описывающие некий процесс (например, изменение дневной температуры). Таких наборов должно быть несколько, в некоторые данные можно включить аномалии или шумы. 
Часть наборов данных должны быть сохранены в папке “train”, другая часть в папке “test”. Одним из вариантов выполнения этого этапа может быть скачивание набора данных из сети, и разделение выборки на тестовую и обучающую. Учтите, что файл должен быть доступен и методы скачивания либо есть в ubuntu либо устанавливаются через pip в файле pipeline.sh
2. Создайте python-скрипт (data_preprocessing.py), который выполняет предобработку данных, например, с помощью sklearn.preprocessing.StandardScaler. Трансформации выполняются и над тестовой и над обучающей выборкой. 
3. Создайте python-скрипт (model_preparation.py), который создает и обучает модель машинного обучения на построенных данных из папки “train”. Для сохранения модели в файл можно воспользоваться [pickle](https://docs.python.org/3/library/pickle.html) (см. [пример](https://rukovodstvo.net/posts/id_1322/))
4. Создайте python-скрипт (model_testing.py), проверяющий модель машинного обучения на построенных данных из папки “test”.
5. Напишите bash-скрипт (pipeline.sh), последовательно запускающий все python-скрипты. При необходимости усложните скрипт. В результате выполнения скрипта на терминал в стандартный поток вывода печатается одна строка с оценкой метрики на вашей модели, например:

```shell
Model test accuracy is: 0.876
```

Настоятельно рекомендуем вам проверить работоспособность скрипта в окружении отличном от того в котором происходила разработка.