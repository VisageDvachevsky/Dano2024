import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Загрузка данных
file_path = 'hakaton_nn_1month.xlsx'
data = pd.read_excel(file_path)

# Предобработка данных
data['offencedate'] = pd.to_datetime(data['offencedate'])
data['offencetime'] = pd.to_datetime(data['offencetime'], format='%H:%M:%S').dt.time

# Создание папки для сохранения графиков, если ее нет
import os
output_dir = 'output_charts_two'
os.makedirs(output_dir, exist_ok=True)

# Текстовый вывод
text_output = []

# 1. Основная статистика по числовым данным
desc_stats = data.describe()
desc_stats.to_csv(f'{output_dir}/numerical_statistics.csv')
text_output.append("Основная статистика по числовым данным:\n" + desc_stats.to_string())

# 2. Пол водителей (круговая диаграмма и текст)
plt.figure(figsize=(8, 8))
gender_counts = data['gender_cd'].value_counts()
plt.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%', colors=['#ff9999','#66b3ff'], startangle=90)
plt.title('Распределение пола водителей')
plt.savefig(f'{output_dir}/gender_distribution.png')
plt.show()
plt.close()
text_output.append(f"Распределение водителей по полу:\n{gender_counts.to_string()}")

# 3. Возрастные группы водителей (столбцовая диаграмма и текст)
age_bins = [18, 25, 35, 45, 55, 65, 75, 85]
age_labels = ['18-24', '25-34', '35-44', '45-54', '55-64', '65-74', '75+']
data['age_group'] = pd.cut(data['age'], bins=age_bins, labels=age_labels)

age_group_counts = data['age_group'].value_counts()
plt.figure(figsize=(10, 6))
sns.countplot(x='age_group', data=data, palette='viridis', hue=None, legend=False)
plt.title('Количество правонарушений по возрастным группам')
plt.xlabel('Возрастная группа')
plt.ylabel('Количество правонарушений')
plt.xticks(rotation=45)
plt.savefig(f'{output_dir}/age_group_distribution.png')
plt.show()
plt.close()
text_output.append(f"Количество правонарушений по возрастным группам:\n{age_group_counts.to_string()}")

# 4. Тип кузова автомобиля (столбцовая диаграмма и текст)
body_type_counts = data['body_type'].value_counts()
plt.figure(figsize=(12, 6))
sns.countplot(x='body_type', data=data, palette='Set2', hue=None, legend=False)
plt.title('Распределение правонарушений по типу кузова автомобиля')
plt.xlabel('Тип кузова')
plt.ylabel('Количество правонарушений')
plt.xticks(rotation=45)
plt.savefig(f'{output_dir}/body_type_distribution.png')
plt.show()
plt.close()
text_output.append(f"Распределение правонарушений по типу кузова автомобиля:\n{body_type_counts.to_string()}")

# 5. День недели (столбцовая диаграмма и текст)
day_of_week_counts = data['day_of_week'].value_counts()
plt.figure(figsize=(10, 6))
sns.countplot(x='day_of_week', data=data, palette='muted', hue=None, legend=False)
plt.title('Распределение правонарушений по дням недели')
plt.xlabel('День недели')
plt.ylabel('Количество правонарушений')
plt.xticks(rotation=45)
plt.savefig(f'{output_dir}/day_of_week_distribution.png')
plt.show()
plt.close()
text_output.append(f"Распределение правонарушений по дням недели:\n{day_of_week_counts.to_string()}")

# 6. Тепловая карта правонарушений по времени суток и дням недели
data['offencetime_hour'] = pd.to_datetime(data['offencetime'], format='%H:%M:%S').dt.hour
heatmap_data = data.pivot_table(index='day_of_week', columns='offencetime_hour', values='party_rk', aggfunc='count')

plt.figure(figsize=(12, 6))
sns.heatmap(heatmap_data, cmap='coolwarm', linewidths=0.1, linecolor='white', annot=True, fmt=".0f")
plt.title('Тепловая карта правонарушений по времени суток и дням недели')
plt.xlabel('Час суток')
plt.ylabel('День недели')
plt.savefig(f'{output_dir}/offence_heatmap.png')
plt.show()
plt.close()
text_output.append("Тепловая карта правонарушений по времени суток и дням недели показывает наибольшую концентрацию правонарушений в определенные часы и дни.")

# 7. Корреляционная матрица для числовых переменных
numeric_columns = ['age', 'engine_power', 'car_price', 'children_cnt', 'person_monthly_income_amt']
corr_matrix = data[numeric_columns].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5, linecolor='white', fmt=".2f")
plt.title('Корреляционная матрица')
plt.savefig(f'{output_dir}/correlation_matrix.png')
plt.show()
plt.close()
text_output.append("Корреляционная матрица для числовых переменных:\n" + corr_matrix.to_string())

# 8. Распределение доходов водителей (гистограмма и текст)
plt.figure(figsize=(12, 6))
sns.histplot(data['person_monthly_income_amt'], bins=20, kde=True, color='blue')
plt.title('Распределение доходов водителей')
plt.xlabel('Месячный доход (условные единицы)')
plt.ylabel('Частота')
plt.savefig(f'{output_dir}/income_distribution.png')
plt.show()
plt.close()
text_output.append(f"Распределение доходов водителей:\nСредний доход: {data['person_monthly_income_amt'].mean():.2f}, медианный доход: {data['person_monthly_income_amt'].median():.2f}")

# 9. Семейный статус водителей (круговая диаграмма и текст)
marital_status_counts = data['marital_status_cd'].value_counts()
plt.figure(figsize=(8, 8))
plt.pie(marital_status_counts, labels=marital_status_counts.index, autopct='%1.1f%%', colors=sns.color_palette('Set3'), startangle=140)
plt.title('Распределение по семейному статусу водителей')
plt.savefig(f'{output_dir}/marital_status_distribution.png')
plt.show()
plt.close()
text_output.append(f"Распределение водителей по семейному статусу:\n{marital_status_counts.to_string()}")

# 10. Распределение правонарушений по марке автомобиля (топ-10 марок и текст)
top10_auto_marks = data['auto_mark'].value_counts().nlargest(10)
plt.figure(figsize=(12, 6))
sns.barplot(x=top10_auto_marks.index, y=top10_auto_marks.values, palette='tab10', hue=None, legend=False)
plt.title('Топ-10 марок автомобилей с наибольшим количеством правонарушений')
plt.xlabel('Марка автомобиля')
plt.ylabel('Количество правонарушений')
plt.xticks(rotation=45)
plt.savefig(f'{output_dir}/top10_auto_marks.png')
plt.show()
plt.close()
text_output.append(f"Топ-10 марок автомобилей с наибольшим количеством правонарушений:\n{top10_auto_marks.to_string()}")

# 11. Количество детей у водителей (столбцовая диаграмма и текст)
children_cnt_counts = data['children_cnt'].value_counts()
plt.figure(figsize=(10, 6))
sns.countplot(x='children_cnt', data=data, palette='Set1', hue=None, legend=False)
plt.title('Количество детей у водителей')
plt.xlabel('Количество детей')
plt.ylabel('Количество правонарушений')
plt.savefig(f'{output_dir}/children_count_distribution.png')
plt.show()
plt.close()
text_output.append(f"Распределение водителей по количеству детей:\n{children_cnt_counts.to_string()}")

# 12. Количество правонарушений по типу правонарушения (столбцовая диаграмма и текст)
offence_types = data['offenceshortstatement'].value_counts()
plt.figure(figsize=(12, 6))
sns.barplot(x=offence_types.index, y=offence_types.values, palette='husl', hue=None, legend=False)
plt.title('Распределение правонарушений по типу')
plt.xlabel('Тип правонарушения')
plt.ylabel('Количество правонарушений')
plt.xticks(rotation=45)
plt.savefig(f'{output_dir}/offence_type_distribution.png')
plt.show()
plt.close()
text_output.append(f"Распределение правонарушений по типу:\n{offence_types.to_string()}")

# Сохранение текстового вывода
with open(f'{output_dir}/text_summary.txt', 'w') as f:
    for item in text_output:
        f.write("%s\n\n" % item)
