import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Загрузка данных
file_path = 'hakaton_nn_1month.xlsx'
data = pd.read_excel(file_path)

print(data.info)
print(data.isnull().sum())

# Предобработка данных
data['offencedate'] = pd.to_datetime(data['offencedate'])
data['offencetime'] = pd.to_datetime(data['offencetime'], format='%H:%M:%S').dt.time

# Преобразование столбца engine_power в числовой тип, игнорирование ошибок
data['engine_power'] = pd.to_numeric(data['engine_power'], errors='coerce')

# Проверка и удаление строк с NaN в столбце engine_power
data = data.dropna(subset=['engine_power']) 

# Создание папки для сохранения графиков, если ее нет
output_dir = 'output_charts_excluding_top3_avg'
os.makedirs(output_dir, exist_ok=True)

# Текстовый вывод
text_output = []

# 1. Основная статистика по числовым данным через describe()
desc_stats = data.describe()
desc_stats.to_csv(f'{output_dir}/numerical_statistics.csv', encoding='utf-8-sig')
text_output.append("Основная статистика по числовым данным:\n" + desc_stats.to_string())

# 2. Выбор трёх лидирующих регионов по количеству нарушений
top_regions = data['region'].value_counts().nlargest(3).index

# Исключаем данные для трёх лидирующих регионов
other_data = data[~data['region'].isin(top_regions)]

# 3. Среднее значение мощности автомобилей для всех остальных регионов
mean_power_other_regions = other_data['engine_power'].mean().round(2)
text_output.append(f"Средняя мощность автомобилей для всех регионов, кроме трёх лидирующих: {mean_power_other_regions}")

# 4. Количество машин выше и ниже средней мощности по остальным регионам
overall_mean_power = data['engine_power'].mean().round(2)

above_mean_power = other_data[other_data['engine_power'] > overall_mean_power]
below_mean_power = other_data[other_data['engine_power'] <= overall_mean_power]

above_mean_count = above_mean_power.shape[0]
below_mean_count = below_mean_power.shape[0]
text_output.append(f"Количество машин с мощностью выше средней: {above_mean_count}")
text_output.append(f"Количество машин с мощностью ниже средней: {below_mean_count}")

# 5. Количество правонарушений за машинами с высокой и низкой мощностью
offences_above_mean = above_mean_power.shape[0]
offences_below_mean = below_mean_power.shape[0]
text_output.append(f"Количество правонарушений за мощными машинами: {offences_above_mean}")
text_output.append(f"Количество правонарушений за маломощными машинами: {offences_below_mean}")

# 6. Процентное соотношение мощных и маломощных машин к количеству нарушений
# Процентное соотношение мощных и маломощных машин к количеству нарушений
total_cars = above_mean_count + below_mean_count
percent_above_mean = round((above_mean_count / total_cars * 100), 2)
percent_below_mean = round((below_mean_count / total_cars * 100), 2)
text_output.append(f"Процент машин с мощностью выше средней: {percent_above_mean}%")
text_output.append(f"Процент машин с мощностью ниже средней: {percent_below_mean}%")


# Дополнительная статистика: средний возраст водителей
mean_age_other_regions = other_data['age'].mean().round(2)
text_output.append(f"Средний возраст водителей для всех регионов, кроме трёх лидирующих: {mean_age_other_regions}")

# Визуализация

# График 1: Количество машин выше и ниже средней мощности для остальных регионов в совокупности
labels = ['Выше средней мощности', 'Ниже средней мощности']
sizes = [above_mean_count, below_mean_count]
colors = ['#66b3ff', '#ff9999']

plt.figure(figsize=(6, 6))
plt.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
plt.title('Процентное соотношение мощных и маломощных машин для остальных регионов')
plt.tight_layout()
plt.savefig(f'{output_dir}/above_below_mean_pie_other_regions.png')

# График 2: Количество правонарушений за мощными и маломощными машинами
labels = ['Мощные машины', 'Маломощные машины']
sizes = [offences_above_mean, offences_below_mean]

plt.figure(figsize=(6, 6))
plt.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
plt.title('Количество правонарушений за мощными и маломощными машинами')
plt.tight_layout()
plt.savefig(f'{output_dir}/offences_above_below_mean_pie.png')

# График 3: Тепловая карта корреляционной матрицы по числовым данным для остальных регионов
plt.figure(figsize=(10, 8))
corr_matrix_other = other_data.corr()
sns.heatmap(corr_matrix_other, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Корреляционная матрица по числовым данным для остальных регионов')
plt.tight_layout()
plt.savefig(f'{output_dir}/correlation_matrix_heatmap_other_regions.png')

# Сохранение текстового вывода
with open(f'{output_dir}/text_summary.txt', 'w', encoding='utf-8-sig') as f:
    for item in text_output:
        f.write("%s\n\n" % item)
