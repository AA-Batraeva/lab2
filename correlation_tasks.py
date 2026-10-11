"""Задачи второй части лабораторной: корреляционный анализ."""

from __future__ import annotations

import pandas as pd

from grader_contracts.correlation_tasks import BrainCorrelationSummary, BrainDataInput

def analyze_brain_correlations(data: BrainDataInput) -> BrainCorrelationSummary:
    """Проанализируйте brainsize.txt.

    Разделите наблюдения по полу и для каждой группы вычислите корреляции
    признаков FSIQ, VIQ, PIQ, Weight, Height с MRI_Count методом Пирсона.
    В strongest_mri_feature верните название признака с наибольшим модулем
    корреляции с MRI_Count среди объединённых результатов двух групп.
    """
    # Шаг 1. Загружаем таблицу с разделителем tab и корректно обрабатываем 'NA' как пропуски
    file_path = data.path
    dataframe = pd.read_csv(file_path, sep="\t", na_values=["NA"])

    # Шаг 2. Разделяем наблюдения на мужчин и женщин
    grouped_by_gender = dataframe.groupby("Gender")
    female_group = grouped_by_gender.get_group("Female")
    male_group = grouped_by_gender.get_group("Male")

    # Шаг 3. Список признаков для анализа корреляции с MRI_Count
    features_to_check = ["FSIQ", "VIQ", "PIQ", "Weight", "Height"]
    target_column = "MRI_Count"

    # Шаг 4. Считаем коэффициенты корреляции Пирсона для каждой группы
    female_correlations = {}
    for feature in features_to_check:
        corr_value = female_group[feature].corr(female_group[target_column], method="pearson")
        female_correlations[feature] = float(corr_value)

    male_correlations = {}
    for feature in features_to_check:
        corr_value = male_group[feature].corr(male_group[target_column], method="pearson")
        male_correlations[feature] = float(corr_value)

    # Шаг 5. Находим признак с максимальным модулем корреляции среди всех результатов обеих групп
    all_correlations = {**female_correlations, **male_correlations}
    strongest_mri_feature = max(all_correlations, key=lambda feature_name: abs(all_correlations[feature_name]))

    # Шаг 6. Возвращаем итоговый объект контракта
    return BrainCorrelationSummary(female_correlations=female_correlations, male_correlations=male_correlations, strongest_mri_feature=strongest_mri_feature)
