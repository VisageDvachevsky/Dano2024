import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Загрузка данных
file_path = 'hakaton_nn_1month.xlsx'
data = pd.read_excel(file_path)

# Предобработка данных
data['offencedate'] = pd.to_datetime(data['offencedate'])
data['offencetime'] = pd.to_datetime(data['offencetime'], format='%H:%M:%S').dt.time

# Преобразование столбца engine_power в числовой тип, игнорирование ошибок
data['engine_power'] = pd.to_numeric(data['engine_power'], errors='coerce')

# Проверка и удаление строк с NaN в столбце engine_power
data = data.dropna(subset=['engine_power'])

# Создание папки для сохранения графиков, если ее нет
output_dir = 'output_charts_two'
os.makedirs(output_dir, exist_ok=True)

# Текстовый вывод
text_output = []

# 1. Основная статистика по числовым данным
desc_stats = data.describe()
desc_stats.to_csv(f'{output_dir}/numerical_statistics.csv')
text_output.append("Основная статистика по числовым данным:\n" + desc_stats.to_string())

# 2. Выбор трёх лидирующих регионов по количеству нарушений
top_regions = data['region'].value_counts().nlargest(3).index
top_data = data[data['region'].isin(top_regions)]

# Среднее значение мощности автомобилей для трёх лидирующих регионов
mean_power_top = top_data.groupby('region')['engine_power'].mean()
mean_power_top.to_csv(f'{output_dir}/mean_power_top_regions.csv')
text_output.append(f"Среднее значение мощности для лидирующих регионов:\n{mean_power_top.to_string()}")

# Среднее значение мощности для остальных регионов
other_data = data[~data['region'].isin(top_regions)]
mean_power_others = other_data['engine_power'].mean()
text_output.append(f"Среднее значение мощности для остальных регионов вместе взятых: {mean_power_others:.2f}")

# 3. Количество машин выше и ниже средней мощности по каждому из трёх лидирующих регионов
overall_mean_power = data['engine_power'].mean()

above_below_power_top = top_data.groupby('region').apply(lambda x: pd.Series({
    'above_mean': (x['engine_power'] > overall_mean_power).sum(),
    'below_mean': (x['engine_power'] <= overall_mean_power).sum()
}))
above_below_power_top.to_csv(f'{output_dir}/above_below_mean_power_top.csv')
text_output.append(f"Количество машин выше и ниже средней мощности для трёх лидирующих регионов:\n{above_below_power_top.to_string()}")

# 4. Процентное соотношение мощных и маломощных машин к количеству нарушений
percent_above_below = above_below_power_top.apply(lambda x: (x / x.sum()) * 100)
percent_above_below.to_csv(f'{output_dir}/percent_above_below_power.csv')
text_output.append(f"Процентное соотношение мощных и маломощных машин к количеству нарушений:\n{percent_above_below.to_string()}")

# Визуализация

# График 1: Средняя мощность двигателей в лидирующих регионах
plt.figure(figsize=(10, 6))
sns.barplot(x=mean_power_top.index, y=mean_power_top.values)
plt.title('Средняя мощность автомобилей в трёх лидирующих регионах')
plt.xlabel('Регион')
plt.ylabel('Средняя мощность (л.с.)')
plt.tight_layout()
plt.savefig(f'{output_dir}/mean_power_top_regions.png')

# График 2: Количество машин выше и ниже средней мощности
above_below_power_top.plot(kind='bar', figsize=(10, 6), stacked=True)
plt.title('Количество машин выше и ниже средней мощности для лидирующих регионов')
plt.xlabel('Регион')
plt.ylabel('Количество автомобилей')
plt.legend(title='Мощность автомобиля', labels=['Выше средней', 'Ниже средней'])
plt.tight_layout()
plt.savefig(f'{output_dir}/above_below_mean_power_top.png')

# Сохранение текстового вывода
with open(f'{output_dir}/text_summary.txt', 'w') as f:
    for item in text_output:
        f.write("%s\n\n" % item)
