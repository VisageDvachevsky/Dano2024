import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

file_path = 'hakaton_nn_1month.xlsx'
data = pd.read_excel(file_path)

data['offencedate'] = pd.to_datetime(data['offencedate'])
data['offencetime'] = pd.to_datetime(data['offencetime'], format='%H:%M:%S').dt.time

data['engine_power'] = pd.to_numeric(data['engine_power'], errors='coerce')

# Проверка и удаление строк с NaN в столбце engine_power
data = data.dropna(subset=['engine_power'])

output_dir = 'output_charts_extended'
os.makedirs(output_dir, exist_ok=True)

text_output = []

# 1. Основная статистика по числовым данным
desc_stats = data.describe()
desc_stats.to_csv(f'{output_dir}/numerical_statistics.csv')
text_output.append("Основная статистика по числовым данным:\n" + desc_stats.to_string())

# 2. Выбор трёх лидирующих регионов по количеству нарушений
top_regions = data['region'].value_counts().nlargest(3).index
top_data = data[data['region'].isin(top_regions)]

# Среднее значение мощности автомобилей для трёх лидирующих регионов
mean_power_top = top_data.groupby('region')['engine_power'].mean().round(2)
mean_power_top.to_csv(f'{output_dir}/mean_power_top_regions.csv')
text_output.append(f"Среднее значение мощности для лидирующих регионов:\n{mean_power_top.to_string()}")

# Среднее значение мощности для остальных регионов
other_data = data[~data['region'].isin(top_regions)]
mean_power_others = other_data['engine_power'].mean().round(2)
text_output.append(f"Среднее значение мощности для остальных регионов вместе взятых: {mean_power_others:.2f}")

# 3. Количество машин выше и ниже средней мощности по каждому из трёх лидирующих регионов
overall_mean_power = data['engine_power'].mean().round(2)

above_below_power_top = top_data.groupby('region').apply(lambda x: pd.Series({
    'above_mean': (x['engine_power'] > overall_mean_power).sum(),
    'below_mean': (x['engine_power'] <= overall_mean_power).sum()
}))
above_below_power_top.to_csv(f'{output_dir}/above_below_mean_power_top.csv')
text_output.append(f"Количество машин выше и ниже средней мощности для трёх лидирующих регионов:\n{above_below_power_top.to_string()}")

# 4. Процентное соотношение мощных и маломощных машин к количеству нарушений
percent_above_below = above_below_power_top.apply(lambda x: (x / x.sum()) * 100).round(2)
percent_above_below.to_csv(f'{output_dir}/percent_above_below_power.csv')
text_output.append(f"Процентное соотношение мощных и маломощных машин к количеству нарушений:\n{percent_above_below.to_string()}")

# Дополнительная статистика
mean_age_by_region = top_data.groupby('region')['age'].mean().round(2)
mean_age_by_region.to_csv(f'{output_dir}/mean_age_by_region.csv')
text_output.append(f"Средний возраст водителей по регионам:\n{mean_age_by_region.to_string()}")

# Визуализация

# График 1: Средняя мощность двигателей в лидирующих регионах с точными значениями
plt.figure(figsize=(10, 6))
sns.barplot(x=mean_power_top.index, y=mean_power_top.values)
plt.title('Средняя мощность автомобилей в трёх лидирующих регионах')
plt.xlabel('Регион')
plt.ylabel('Средняя мощность (л.с.)')

for index, value in enumerate(mean_power_top.values):
    plt.text(index, value + 1, f'{value:.2f}', ha='center')

plt.tight_layout()
plt.savefig(f'{output_dir}/mean_power_top_regions.png')

# График 2: Количество машин выше и ниже средней мощности с точными значениями
above_below_power_top.plot(kind='bar', figsize=(10, 6), stacked=True)
plt.title('Количество машин выше и ниже средней мощности для лидирующих регионов')
plt.xlabel('Регион')
plt.ylabel('Количество автомобилей')
plt.legend(title='Мощность автомобиля', labels=['Выше средней', 'Ниже средней'])

# Добавляем значения на график
for index, (above, below) in enumerate(zip(above_below_power_top['above_mean'], above_below_power_top['below_mean'])):
    plt.text(index, above + below / 2, f'{below}', ha='center', color='white')
    plt.text(index, above / 2, f'{above}', ha='center', color='white')

plt.tight_layout()
plt.savefig(f'{output_dir}/above_below_mean_power_top.png')

# График 3: Тепловая карта корреляционной матрицы по числовым данным
plt.figure(figsize=(10, 8))
corr_matrix = data.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Корреляционная матрица по числовым данным')
plt.tight_layout()
plt.savefig(f'{output_dir}/correlation_matrix_heatmap.png')

# График 4: Распределение мощности двигателя (гистограмма)
plt.figure(figsize=(10, 6))
sns.histplot(data['engine_power'], bins=30, kde=True, color='blue')
plt.title('Распределение мощности двигателя')
plt.xlabel('Мощность (л.с.)')
plt.ylabel('Частота')
plt.tight_layout()
plt.savefig(f'{output_dir}/engine_power_distribution.png')

# График 5: Распределение возраста водителей (гистограмма)
plt.figure(figsize=(10, 6))
sns.histplot(data['age'], bins=20, kde=True, color='green')
plt.title('Распределение возраста водителей')
plt.xlabel('Возраст')
plt.ylabel('Частота')
plt.tight_layout()
plt.savefig(f'{output_dir}/age_distribution.png')

with open(f'{output_dir}/text_summary.txt', 'w') as f:
    for item in text_output:
        f.write("%s\n\n" % item)
