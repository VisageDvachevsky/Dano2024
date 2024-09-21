import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats

# Функция для сохранения графиков
def save_plot(fig, filename):
    fig.savefig(filename, bbox_inches='tight')
    plt.close(fig)

# Загрузка данных
df = pd.read_csv('traffic_violations.csv', parse_dates=['offencedate'])

# Преобразование типов данных
df['offencetime'] = pd.to_datetime(df['offencetime'], format='%H:%M:%S').dt.time
df['engine_type'] = df['engine_type'].astype(float)
df['engine_power'] = df['engine_power'].astype(float)
df['auto_year'] = df['auto_year'].astype(int)
df['car_price'] = df['car_price'].astype(float)
df['children_cnt'] = df['children_cnt'].astype(int)
df['person_monthly_income_amt'] = df['person_monthly_income_amt'].astype(float)

# Первичный осмотр данных
print(df.head())
print(df.info())

# Проверка на пропущенные значения
missing_values = df.isnull().sum()
print("Пропущенные значения:\n", missing_values)

# Базовая статистика
print(df.describe())

# Анализ числовых переменных
numeric_columns = ['engine_type', 'engine_power', 'auto_year', 'car_price', 'age', 'children_cnt', 'person_monthly_income_amt']
fig, axs = plt.subplots(len(numeric_columns), 1, figsize=(12, 6*len(numeric_columns)))
for i, col in enumerate(numeric_columns):
    sns.histplot(df[col].dropna(), ax=axs[i], kde=True)
    axs[i].set_title(f'Распределение {col}')
plt.tight_layout()
save_plot(fig, 'numeric_distributions.png')

# Анализ категориальных переменных
categorical_columns = ['region', 'offenceshortstatement', 'body_type', 'auto_mark', 'color', 'gear_type', 'gender_cd', 'marital_status_cd', 'education_level_cd', 'day_of_week', 'public_holiday']
for col in categorical_columns:
    fig, ax = plt.subplots(figsize=(12, 6))
    df[col].value_counts(normalize=True).plot(kind='bar', ax=ax)
    ax.set_title(f'Распределение значений в столбце {col}')
    ax.set_ylabel('Доля')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    save_plot(fig, f'{col}_distribution.png')

# Анализ времени совершения правонарушений
df['hour'] = df['offencetime'].apply(lambda x: x.hour)
df['month'] = df['offencedate'].dt.month

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
sns.countplot(x='hour', data=df, ax=ax1)
ax1.set_title('Распределение правонарушений по часам')
sns.countplot(x='month', data=df, ax=ax2)
ax2.set_title('Распределение правонарушений по месяцам')
plt.tight_layout()
save_plot(fig, 'time_distribution.png')

# Анализ дней недели и праздников
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
sns.countplot(x='day_of_week', data=df, order=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'], ax=ax1)
ax1.set_title('Распределение правонарушений по дням недели')
sns.countplot(x='public_holiday', data=df, ax=ax2)
ax2.set_title('Распределение правонарушений по праздничным/рабочим дням')
plt.tight_layout()
save_plot(fig, 'weekday_holiday_distribution.png')

# Анализ корреляций между числовыми переменными
corr_matrix = df[numeric_columns].corr()
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
plt.title('Корреляционная матрица числовых переменных')
save_plot(plt.gcf(), 'correlation_matrix.png')

# Анализ зависимости количества правонарушений от возраста и пола
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
sns.boxplot(x='gender_cd', y='age', data=df, ax=ax1)
ax1.set_title('Распределение возраста по полу')
sns.violinplot(x='gender_cd', y='age', data=df, ax=ax2)
ax2.set_title('Распределение возраста по полу (violin plot)')
plt.tight_layout()
save_plot(fig, 'age_gender_distribution.png')

# Анализ зависимости вида правонарушения от марки автомобиля
top_10_marks = df['auto_mark'].value_counts().nlargest(10).index
top_5_offences = df['offenceshortstatement'].value_counts().nlargest(5).index
df_top = df[df['auto_mark'].isin(top_10_marks) & df['offenceshortstatement'].isin(top_5_offences)]

plt.figure(figsize=(15, 10))
sns.heatmap(pd.crosstab(df_top['auto_mark'], df_top['offenceshortstatement'], normalize='index'),
            annot=True, fmt='.2f', cmap='YlOrRd')
plt.title('Зависимость вида правонарушения от марки автомобиля (топ-10 марок, топ-5 правонарушений)')
plt.tight_layout()
save_plot(plt.gcf(), 'offence_by_auto_mark.png')

# Анализ зависимости количества правонарушений от уровня дохода
df['income_group'] = pd.qcut(df['person_monthly_income_amt'], q=5, labels=['Очень низкий', 'Низкий', 'Средний', 'Высокий', 'Очень высокий'])
plt.figure(figsize=(10, 6))
sns.countplot(x='income_group', data=df)
plt.title('Распределение правонарушений по уровню дохода')
plt.xticks(rotation=45)
plt.tight_layout()
save_plot(plt.gcf(), 'offences_by_income.png')

# Анализ зависимости количества правонарушений от стоимости автомобиля
df['car_price_group'] = pd.qcut(df['car_price'], q=5, labels=['Очень дешевый', 'Дешевый', 'Средний', 'Дорогой', 'Очень дорогой'])
plt.figure(figsize=(10, 6))
sns.countplot(x='car_price_group', data=df)
plt.title('Распределение правонарушений по стоимости автомобиля')
plt.xticks(rotation=45)
plt.tight_layout()
save_plot(plt.gcf(), 'offences_by_car_price.png')

# Анализ зависимости вида правонарушения от образования
plt.figure(figsize=(12, 8))
sns.heatmap(pd.crosstab(df['education_level_cd'], df['offenceshortstatement'], normalize='index'),
            annot=True, fmt='.2f', cmap='YlOrRd')
plt.title('Зависимость вида правонарушения от уровня образования')
plt.tight_layout()
save_plot(plt.gcf(), 'offence_by_education.png')

# Статистический анализ: t-тест для сравнения возраста нарушителей по полу
male_age = df[df['gender_cd'] == 'M']['age']
female_age = df[df['gender_cd'] == 'F']['age']
t_stat, p_value = stats.ttest_ind(male_age, female_age)
print(f"T-тест для сравнения возраста по полу: t-statistic = {t_stat}, p-value = {p_value}")

# ANOVA для сравнения возраста нарушителей по уровню образования
education_groups = [group for _, group in df.groupby('education_level_cd')['age']]
f_stat, p_value = stats.f_oneway(*education_groups)
print(f"ANOVA для сравнения возраста по уровню образования: F-statistic = {f_stat}, p-value = {p_value}")

# Хи-квадрат тест для проверки зависимости между полом и видом правонарушения
chi2, p_value, dof, expected = stats.chi2_contingency(pd.crosstab(df['gender_cd'], df['offenceshortstatement']))
print(f"Хи-квадрат тест для зависимости между полом и видом правонарушения: chi2 = {chi2}, p-value = {p_value}")

# Анализ зависимости вида правонарушения от семейного положения
plt.figure(figsize=(12, 8))
sns.heatmap(pd.crosstab(df['marital_status_cd'], df['offenceshortstatement'], normalize='index'),
            annot=True, fmt='.2f', cmap='YlOrRd')
plt.title('Зависимость вида правонарушения от семейного положения')
plt.tight_layout()
save_plot(plt.gcf(), 'offence_by_marital_status.png')

# Анализ зависимости количества правонарушений от количества детей
plt.figure(figsize=(10, 6))
sns.countplot(x='children_cnt', data=df)
plt.title('Распределение правонарушений по количеству детей')
plt.tight_layout()
save_plot(plt.gcf(), 'offences_by_children_count.png')

print("Анализ завершен. Все графики сохранены в файлы.")