"""Задачи первой части лабораторной: Pandas и Titanic."""
from __future__ import annotations

import pandas as pd

from grader_contracts.pandas_tasks import TitanicInput, TitanicSummary


def analyze_titanic(data: TitanicInput) -> TitanicSummary:

    """Выполните загрузку и анализ датасета Titanic.

    Нужно: посчитать пропуски, число пассажиров старше 30 лет, средний возраст
    и долю выживших по классам, а также пять наибольших тарифов по убыванию.

    """
    # Шаг 1. Загружаем CSV-файл с помощью pandas по пути, указанному в объекте data.path
    file_path = data.path
    df = pd.read_csv(file_path)

    # Шаг 2. Считаем количество пропусков в каждом столбце таблицы.
    # .isna() возвращает True для пропусков, а .sum() суммирует их количество по столбцам.
    # Превращаем результат в обычный словарь Python.
    missing_values = df.isna().sum().to_dict()

    # Шаг 3. Считаем количество пассажиров, у которых возраст (Age) строго больше 30 лет.
    # Условие df['Age'] > 0 отфильтровывает строки, но лучше просто делать сравнение,
    # так как NaN сравнения с 30 дают False.
    older_than_30_mask = df["Age"] > 30
    older_than_30_count = int(older_than_30_mask.sum())

    # Шаг 4. Вычисляем средний возраст для каждого класса пассажиров (Pclass).
    # Используем groupby по столбцу 'Pclass', берем столбец 'Age' и считаем среднее (.mean()).
    # Пропуски сами игнорируются pandas, в 0 не превращаются.
    mean_age_by_pclass_series = df.groupby("Pclass")["Age"].mean()

    # Превращаем Series в словарь, где ключи — целые числа (классы 1, 2, 3).
    mean_age_by_pclass = {int(k): float(v) for k, v in mean_age_by_pclass_series.items()}

    # Шаг 5. Вычисляем долю выживших (среднее арифметическое по столбцу Survived, т.к. там 0 и 1) для каждого Pclass.
    survival_rate_by_pclass_series = df.groupby("Pclass")["Survived"].mean()

    # Превращаем в словарь с целочисленными ключами.
    survival_rate_by_pclass = {int(k): float(v) for k, v in survival_rate_by_pclass_series.items()}

    # Шаг 6. Находим пять наибольших значений стоимости билета (Fare) в порядке убывания.
    # Сортируем столбец Fare по убыванию, убираем возможные дубликаты (или берем первые 5 элементов)
    # и преобразуем в список чисел.
    sorted_fares = df["Fare"].sort_values(ascending=False)
    top_5_fares = sorted_fares.head(5).tolist()

    # Шаг 7. Возвращаем заполненный объект структуры TitanicSummary (по контракту проверяющей системы).
    return TitanicSummary(
        missing_values=missing_values,
        older_than_30_count=older_than_30_count,
        mean_age_by_pclass=mean_age_by_pclass,
        survival_rate_by_pclass=survival_rate_by_pclass,
        top_5_fares=top_5_fares,)
    raise NotImplementedError
