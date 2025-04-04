## Module 4

В практическом задании данного модуля вам необходимо продемонстрировать навыки практического использования утилиты dvc для работы с данными. В результате выполнения этих заданий вы выполните все основные операции с dvc и закрепите полученные теоретические знания практическими действиями.

Этапы задания:

1. Создайте папку lab4 в корне проекта.
2. Установите git и dvc. Настройте папку проекта для работы с git и dvc.
3. Настройте удаленное хранилище файлов, например на Google Disk или S3.
4. Создайте датасет, например, о пассажирах “Титаника” catboost.titanic().
5. Модифицируйте датасет, в котором содержится информация о классе (“Pclass”),  поле (“Sex”) и возрасте (“Age”) пассажира. Сделайте коммит в git и push в dvc.
6. Создайте новую версию датасета, в котором пропущенные (nan) значения в поле “Age” будут заполнены средним значением. Сделайте коммит в git и push в dvc.
7. Создайте новый признак с использованием one-hot-encoding для строкового признака “Пол” (“Sex”). Сделайте коммит в git и push в dvc.
8. Выполните переключение между всеми созданными версиями датасета.

При правильном выполнении задания и вас появится git репозиторий с опубликованной метаинформацией и папка на Google Disk, в которой хранятся различные версии датасетов.
Вам необходимо подготовить отчет в тех функциональностях которые вы настроили. Дополнительно можно настроить DAG, запуск и версионирование экспериментов, например, с использованием Hydra.

В постановке задачи используется датасет из конкурса “Titanic Disaster”, однако вы можете использовать свои наборы данных, в этом случае в п.п.4-8 необходимо использовать информацию и признаки из вашего датасета.
