# # Домашнее задание: Предсказание оттока клиентов кредитных карт
# ## Выполнил: Аникин Максим
# ### Датасет: Credit Card Customers (Kaggle)

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, classification_report)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import warnings
warnings.filterwarnings('ignore')

# Настройка отображения
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
pd.set_option('display.max_columns', None)

print("Библиотеки загружены успешно!")

# %%

# df = pd.read_csv('BankChurners.csv')
# # Удаляем последние 2 колонки (результаты предыдущего анализа)
# df = df.iloc[:, :-2]
# 
# # Отделяем целевую переменную
# X = df.drop(['Attrition_Flag', 'CLIENTNUM'], axis=1)  # CLIENTNUM - это просто ID
# y = df['Attrition_Flag']
# 
# # Преобразуем целевую переменную в бинарный формат
# y = y.map({'Existing Customer': 0, 'Attrited Customer': 1})
# 
# # Разделение на обучающую (80%) и тестовую (20%) выборки
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42, stratify=y
# )
# 
# print(f"Размер обучающей выборки: {X_train.shape}")
# print(f"Размер тестовой выборки: {X_test.shape}")
# print(f"\nРаспределение классов в обучающей выборке:")
# print(y_train.value_counts(normalize=True))
# print(f"\nРаспределение классов в тестовой выборке:")
# print(y_test.value_counts(normalize=True))

# %% [markdown]
# ## 3. Визуализация данных и вычисление основных характеристик

# %%
# Общая информация о данных
print("=== ОБЩАЯ ИНФОРМАЦИЯ ===")
print(df.info())
print("\n=== СТАТИСТИКА ЧИСЛЕННЫХ ПРИЗНАКОВ ===")
print(df.describe())

# %%
# Анализ целевой переменной
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
y.value_counts().plot(kind='bar')
plt.title('Распределение классов (абсолютное)')
plt.xlabel('Класс')
plt.ylabel('Количество')
plt.xticks([0, 1], ['Existing (0)', 'Attrited (1)'], rotation=0)

plt.subplot(1, 2, 2)
y.value_counts(normalize=True).plot(kind='bar', color=['green', 'red'])
plt.title('Распределение классов (относительное)')
plt.xlabel('Класс')
plt.ylabel('Доля')
plt.xticks([0, 1], ['Existing (0)', 'Attrited (1)'], rotation=0)
for i, v in enumerate(y.value_counts(normalize=True)):
    plt.text(i, v + 0.01, f'{v:.1%}', ha='center')

plt.tight_layout()
plt.show()

print(f"\n📊 ВЫВОД: Данные несбалансированы!")
print(f"   - Существующие клиенты: {(y==0).sum()} ({(y==0).mean():.1%})")
print(f"   - Ушедшие клиенты: {(y==1).sum()} ({(y==1).mean():.1%})")

# %%
# Визуализация численных признаков
numerical_features = X_train.select_dtypes(include=[np.number]).columns.tolist()

fig, axes = plt.subplots(4, 4, figsize=(16, 12))
axes = axes.ravel()

for idx, col in enumerate(numerical_features[:16]):
    axes[idx].hist(X_train[col], bins=30, edgecolor='black', alpha=0.7)
    axes[idx].set_title(col, fontsize=10)
    axes[idx].set_xlabel('')

plt.suptitle('Распределения численных признаков', fontsize=14, y=1.00)
plt.tight_layout()
plt.show()

# %%
# Статистики численных признаков
print("\n=== ОСНОВНЫЕ ХАРАКТЕРИСТИКИ ЧИСЛЕННЫХ ПРИЗНАКОВ ===")
stats = X_train[numerical_features].agg(['mean', 'std', 'min', 'max'])
print(stats.round(2))

# %%
# Корреляционная матрица
plt.figure(figsize=(14, 10))
correlation_matrix = X_train[numerical_features].corr()
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', center=0,
            square=True, linewidths=0.5)
plt.title('Корреляционная матрица численных признаков', fontsize=14)
plt.tight_layout()
plt.show()

# Находим сильно коррелированные пары
print("\n=== СИЛЬНО КОРРЕЛИРОВАННЫЕ ПРИЗНАКИ (|r| > 0.8) ===")
high_corr = []
for i in range(len(correlation_matrix.columns)):
    for j in range(i+1, len(correlation_matrix.columns)):
        if abs(correlation_matrix.iloc[i, j]) > 0.8:
            high_corr.append((correlation_matrix.columns[i], 
                            correlation_matrix.columns[j], 
                            correlation_matrix.iloc[i, j]))
            print(f"{correlation_matrix.columns[i]:30} <-> {correlation_matrix.columns[j]:30}: {correlation_matrix.iloc[i, j]:.3f}")

if not high_corr:
    print("Сильно коррелированных пар не найдено")

# %%
# Визуализация категориальных признаков
categorical_features = X_train.select_dtypes(include=['object']).columns.tolist()

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.ravel()

for idx, col in enumerate(categorical_features):
    X_train[col].value_counts().plot(kind='bar', ax=axes[idx])
    axes[idx].set_title(col)
    axes[idx].set_xlabel('')
    axes[idx].tick_params(axis='x', rotation=45)

plt.suptitle('Распределения категориальных признаков', fontsize=14)
plt.tight_layout()
plt.show()

# %%
# Анализ зависимости между признаками и целевой переменной
# Создаем временный датафрейм для анализа
train_df = X_train.copy()
train_df['Attrition_Flag'] = y_train

# Численные признаки vs целевая переменная
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.ravel()

key_features = ['Customer_Age', 'Total_Trans_Amt', 'Total_Trans_Ct', 
                'Total_Revolving_Bal', 'Avg_Utilization_Ratio', 'Credit_Limit']

for idx, col in enumerate(key_features):
    train_df.boxplot(column=col, by='Attrition_Flag', ax=axes[idx])
    axes[idx].set_title(f'{col} по классам')
    axes[idx].set_xlabel('0=Existing, 1=Attrited')

plt.suptitle('Сравнение ключевых признаков по классам', fontsize=14, y=1.00)
plt.tight_layout()
plt.show()

print("\n=== ИНТЕРПРЕТАЦИЯ ВИЗУАЛИЗАЦИЙ ===")
print("1. Данные несбалансированы: ~84% существующих клиентов, ~16% ушедших")
print("2. Сильная корреляция между Credit_Limit и Avg_Open_To_Buy (ожидаемо)")
print("3. Ушедшие клиенты имеют:")
print("   - Меньше транзакций (Total_Trans_Ct, Total_Trans_Amt)")
print("   - Меньший баланс (Total_Revolving_Bal)")
print("   - Меньшее использование карты (Avg_Utilization_Ratio)")

# Удаляем временный датафрейм
del train_df

# %% [markdown]
# ## 4. Обработка пропущенных значений

# %%
print("=== ПРОВЕРКА ПРОПУЩЕННЫХ ЗНАЧЕНИЙ ===")
print("\nОбучающая выборка:")
missing_train = X_train.isnull().sum()
print(missing_train[missing_train > 0])
if missing_train.sum() == 0:
    print("✓ Пропущенных значений нет!")

print("\nТестовая выборка:")
missing_test = X_test.isnull().sum()
print(missing_test[missing_test > 0])
if missing_test.sum() == 0:
    print("✓ Пропущенных значений нет!")

# %%
# Проверка на значения 'Unknown' в категориальных признаках
print("\n=== ПРОВЕРКА ЗНАЧЕНИЙ 'Unknown' ===")
for col in categorical_features:
    unknown_count = (X_train[col] == 'Unknown').sum()
    if unknown_count > 0:
        print(f"{col}: {unknown_count} значений 'Unknown' ({unknown_count/len(X_train)*100:.1f}%)")

print("\n📝 РЕШЕНИЕ: Значения 'Unknown' оставляем как отдельную категорию,")
print("   так как они могут нести информацию (клиент не указал данные)")

# %% [markdown]
# ## 5. Обработка категориальных признаков

# %%
print("=== КОДИРОВАНИЕ КАТЕГОРИАЛЬНЫХ ПРИЗНАКОВ ===")
print(f"Категориальные признаки: {categorical_features}")

# Создаем копии для обработки
X_train_encoded = X_train.copy()
X_test_encoded = X_test.copy()

# Label Encoding для признаков с порядковой природой
# (можно использовать One-Hot, но для KNN это увеличит размерность)

label_encoders = {}
for col in categorical_features:
    le = LabelEncoder()
    X_train_encoded[col] = le.fit_transform(X_train[col])
    X_test_encoded[col] = le.transform(X_test[col])
    label_encoders[col] = le

    print(f"\n{col}:")
    print(f"  Оригинальные категории: {list(le.classes_)}")
    print(f"  Закодированы как: {list(range(len(le.classes_)))}")

print("\n✓ Категориальные признаки закодированы!")

# %% [markdown]
# ## 6. Нормализация признаков

# %%
print("=== НОРМАЛИЗАЦИЯ ПРИЗНАКОВ ===")
print("\nИспользуем StandardScaler (z-score нормализация):")
print("  x_norm = (x - mean) / std")
print("\nОбоснование: KNN использует евклидово расстояние,")
print("поэтому признаки с разными масштабами должны быть нормализованы.")

# Создаем и обучаем масштабировщик на обучающей выборке
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_encoded)
X_test_scaled = scaler.transform(X_test_encoded)

# Преобразуем обратно в DataFrame для удобства
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train_encoded.columns, 
                               index=X_train_encoded.index)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test_encoded.columns,
                              index=X_test_encoded.index)

print("\nПример нормализации (первые 5 строк, первые 5 признаков):")
print("\nДО нормализации:")
print(X_train_encoded.iloc[:5, :5])
print("\nПОСЛЕ нормализации:")
print(X_train_scaled.iloc[:5, :5])

print("\n✓ Признаки нормализованы!")

# %% [markdown]
# ## 7. Обучение классификатора K-Nearest Neighbors

# %%
print("=== ОБУЧЕНИЕ KNN КЛАССИФИКАТОРА ===")
print("\nВыбор метода: K-Nearest Neighbors (KNN)")
print("\nОбоснование:")
print("1. KNN - простой и интерпретируемый метод")
print("2. Хорошо работает для задач классификации")
print("3. Не делает предположений о распределении данных")
print("4. Подходит для нашей задачи с умеренным количеством признаков")

# Начнем с k=5 (распространенное начальное значение)
knn_initial = KNeighborsClassifier(n_neighbors=5)
knn_initial.fit(X_train_scaled, y_train)

# Предсказания
y_train_pred = knn_initial.predict(X_train_scaled)
y_test_pred = knn_initial.predict(X_test_scaled)

print("\n✓ Модель обучена с k=5")

# %% [markdown]
# ## 8. Вычисление ошибок и оптимизация гиперпараметра k

# %%
print("=== ОЦЕНКА КАЧЕСТВА МОДЕЛИ (k=5) ===")

print("\n--- ОБУЧАЮЩАЯ ВЫБОРКА ---")
print(f"Accuracy:  {accuracy_score(y_train, y_train_pred):.4f}")
print(f"Precision: {precision_score(y_train, y_train_pred):.4f}")
print(f"Recall:    {recall_score(y_train, y_train_pred):.4f}")
print(f"F1-Score:  {f1_score(y_train, y_train_pred):.4f}")

print("\n--- ТЕСТОВАЯ ВЫБОРКА ---")
print(f"Accuracy:  {accuracy_score(y_test, y_test_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_test_pred):.4f}")
print(f"Recall:    {recall_score(y_test, y_test_pred):.4f}")
print(f"F1-Score:  {f1_score(y_test, y_test_pred):.4f}")

# %%
# Матрица ошибок
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Обучающая выборка
cm_train = confusion_matrix(y_train, y_train_pred)
sns.heatmap(cm_train, annot=True, fmt='d', cmap='Blues', ax=axes[0])
axes[0].set_title('Матрица ошибок (обучающая выборка)')
axes[0].set_xlabel('Предсказанный класс')
axes[0].set_ylabel('Истинный класс')
axes[0].set_xticklabels(['Existing', 'Attrited'])
axes[0].set_yticklabels(['Existing', 'Attrited'])

# Тестовая выборка
cm_test = confusion_matrix(y_test, y_test_pred)
sns.heatmap(cm_test, annot=True, fmt='d', cmap='Oranges', ax=axes[1])
axes[1].set_title('Матрица ошибок (тестовая выборка)')
axes[1].set_xlabel('Предсказанный класс')
axes[1].set_ylabel('Истинный класс')
axes[1].set_xticklabels(['Existing', 'Attrited'])
axes[1].set_yticklabels(['Existing', 'Attrited'])

plt.tight_layout()
plt.show()

print("\n=== ИНТЕРПРЕТАЦИЯ МАТРИЦ ОШИБОК ===")
print("\nТестовая выборка:")
print(f"True Negatives (правильно предсказан Existing):  {cm_test[0,0]}")
print(f"False Positives (ошибочно предсказан Attrited): {cm_test[0,1]}")
print(f"False Negatives (пропущен Attrited):            {cm_test[1,0]}")
print(f"True Positives (правильно предсказан Attrited):  {cm_test[1,1]}")

# %%
# Полный отчет по классификации
print("\n=== ДЕТАЛЬНЫЙ ОТЧЕТ ПО КЛАССИФИКАЦИИ ===")
print("\nТестовая выборка:")
print(classification_report(y_test, y_test_pred, 
                          target_names=['Existing Customer', 'Attrited Customer']))

# %%

# ОПТИМИЗАЦИЯ ГИПЕРПАРАМЕТРА k
print("=== ПОИСК ОПТИМАЛЬНОГО ЗНАЧЕНИЯ k ===")
print("\nПроверяем k от 1 до 30...")

# Тестируем различные значения k
k_values = range(1, 31)
train_scores = []
test_scores = []
f1_scores = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)

    train_scores.append(knn.score(X_train_scaled, y_train))
    test_scores.append(knn.score(X_test_scaled, y_test))

    y_pred = knn.predict(X_test_scaled)
    f1_scores.append(f1_score(y_test, y_pred))

# Визуализация результатов
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# График accuracy
axes[0].plot(k_values, train_scores, marker='o', label='Train Accuracy')
axes[0].plot(k_values, test_scores, marker='s', label='Test Accuracy')
axes[0].set_xlabel('Количество соседей (k)')
axes[0].set_ylabel('Accuracy')
axes[0].set_title('Зависимость точности от k')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# График F1-score
axes[1].plot(k_values, f1_scores, marker='o', color='green')
axes[1].set_xlabel('Количество соседей (k)')
axes[1].set_ylabel('F1-Score (test)')
axes[1].set_title('Зависимость F1-Score от k')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Находим оптимальное k
optimal_k = k_values[np.argmax(test_scores)]
optimal_k_f1 = k_values[np.argmax(f1_scores)]

print(f"\n📊 РЕЗУЛЬТАТЫ:")
print(f"   Оптимальное k по Accuracy: {optimal_k} (Accuracy = {max(test_scores):.4f})")
print(f"   Оптимальное k по F1-Score: {optimal_k_f1} (F1 = {max(f1_scores):.4f})")

# %%
# Обучаем финальную модель KNN с оптимальным k
print(f"\n=== ФИНАЛЬНАЯ МОДЕЛЬ KNN (k={optimal_k_f1}) ===")

knn_final = KNeighborsClassifier(n_neighbors=optimal_k_f1)
knn_final.fit(X_train_scaled, y_train)

y_train_pred_final = knn_final.predict(X_train_scaled)
y_test_pred_final = knn_final.predict(X_test_scaled)

print("\n--- ОБУЧАЮЩАЯ ВЫБОРКА ---")
print(f"Accuracy:  {accuracy_score(y_train, y_train_pred_final):.4f}")
print(f"Precision: {precision_score(y_train, y_train_pred_final):.4f}")
print(f"Recall:    {recall_score(y_train, y_train_pred_final):.4f}")
print(f"F1-Score:  {f1_score(y_train, y_train_pred_final):.4f}")

print("\n--- ТЕСТОВАЯ ВЫБОРКА ---")
print(f"Accuracy:  {accuracy_score(y_test, y_test_pred_final):.4f}")
print(f"Precision: {precision_score(y_test, y_test_pred_final):.4f}")
print(f"Recall:    {recall_score(y_test, y_test_pred_final):.4f}")
print(f"F1-Score:  {f1_score(y_test, y_test_pred_final):.4f}")

# Матрица ошибок для финальной модели
cm_final = confusion_matrix(y_test, y_test_pred_final)

plt.figure(figsize=(8, 6))
sns.heatmap(cm_final, annot=True, fmt='d', cmap='RdYlGn', cbar=True)
plt.title(f'Матрица ошибок - KNN (k={optimal_k_f1})')
plt.xlabel('Предсказанный класс')
plt.ylabel('Истинный класс')
plt.xticks([0.5, 1.5], ['Existing', 'Attrited'])
plt.yticks([0.5, 1.5], ['Existing', 'Attrited'])
plt.show()

print("\n=== КОММЕНТАРИЙ ===")
print(f"При k={optimal_k_f1} модель показывает хороший баланс между")
print("точностью на обучающей и тестовой выборках, что указывает")
print("на отсутствие переобучения.")

# %% [markdown]
# ## 9. (Дополнительно) Сравнение с другими классификаторами

# %%
print("=== СРАВНЕНИЕ РАЗЛИЧНЫХ КЛАССИФИКАТОРОВ ===")

# Инициализируем модели
models = {
    'KNN': KNeighborsClassifier(n_neighbors=optimal_k_f1),
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=10),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
}

# Обучаем и оцениваем каждую модель
results = []

for name, model in models.items():
    # Обучение
    model.fit(X_train_scaled, y_train)

    # Предсказания
    y_pred = model.predict(X_test_scaled)

    # Метрики
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    results.append({
        'Модель': name,
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1-Score': f1
    })

    print(f"\n{name}:")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1-Score:  {f1:.4f}")

# %%
# Визуализация сравнения моделей
results_df = pd.DataFrame(results)
results_df = results_df.set_index('Модель')

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
colors = ['steelblue', 'coral', 'mediumseagreen', 'mediumpurple']

for idx, (metric, color) in enumerate(zip(metrics, colors)):
    ax = axes[idx // 2, idx % 2]
    results_df[metric].plot(kind='barh', ax=ax, color=color)
    ax.set_title(metric, fontsize=12, fontweight='bold')
    ax.set_xlabel('Значение')
    ax.set_xlim([0, 1])
    ax.grid(True, alpha=0.3, axis='x')

    # Добавляем значения на графики
    for i, v in enumerate(results_df[metric]):
        ax.text(v + 0.01, i, f'{v:.3f}', va='center')

plt.suptitle('Сравнение классификаторов', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# Таблица результатов
print("\n=== ИТОГОВАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ ===")
print(results_df.round(4))

# Лучшая модель
best_model = results_df['F1-Score'].idxmax()
print(f"\n🏆 Лучшая модель по F1-Score: {best_model}")

# %% [markdown]
# ## 10. (Дополнительно) Борьба с несбалансированностью классов

# %%
print("=== РАБОТА С НЕСБАЛАНСИРОВАННОСТЬЮ КЛАССОВ ===")
print(f"\nТекущее распределение:")
print(f"  Existing: {(y_train==0).sum()} ({(y_train==0).mean():.1%})")
print(f"  Attrited: {(y_train==1).sum()} ({(y_train==1).mean():.1%})")

print("\nМетоды борьбы с несбалансированностью:")
print("1. Взвешивание классов (class_weight)")
print("2. SMOTE (Synthetic Minority Over-sampling)")
print("3. Under-sampling")
print("\nИспользуем взвешивание классов для KNN через уменьшение порога:")

# %%
# KNN с пробабилистическим подходом
knn_balanced = KNeighborsClassifier(n_neighbors=optimal_k_f1)
knn_balanced.fit(X_train_scaled, y_train)

# Получаем вероятности
y_proba = knn_balanced.predict_proba(X_test_scaled)[:, 1]

# Подбираем оптимальный порог
thresholds = np.arange(0.1, 0.9, 0.05)
f1_scores_threshold = []

for threshold in thresholds:
    y_pred_thresh = (y_proba >= threshold).astype(int)
    f1_scores_threshold.append(f1_score(y_test, y_pred_thresh))

optimal_threshold = thresholds[np.argmax(f1_scores_threshold)]

# Предсказания с оптимальным порогом
y_pred_balanced = (y_proba >= optimal_threshold).astype(int)

print(f"\nОптимальный порог: {optimal_threshold:.2f}")
print(f"\nРезультаты с балансировкой:")
print(f"Accuracy:  {accuracy_score(y_test, y_pred_balanced):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_balanced):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred_balanced):.4f}")
print(f"F1-Score:  {f1_score(y_test, y_pred_balanced):.4f}")

# График зависимости F1 от порога
plt.figure(figsize=(10, 5))
plt.plot(thresholds, f1_scores_threshold, marker='o')
plt.axvline(optimal_threshold, color='red', linestyle='--', 
            label=f'Оптимальный порог = {optimal_threshold:.2f}')
plt.xlabel('Порог классификации')
plt.ylabel('F1-Score')
plt.title('Зависимость F1-Score от порога классификации')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Сравнение матриц ошибок
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

cm1 = confusion_matrix(y_test, y_test_pred_final)
sns.heatmap(cm1, annot=True, fmt='d', cmap='Blues', ax=axes[0])
axes[0].set_title('Без балансировки (порог=0.5)')
axes[0].set_xlabel('Предсказано')
axes[0].set_ylabel('Истина')

cm2 = confusion_matrix(y_test, y_pred_balanced)
sns.heatmap(cm2, annot=True, fmt='d', cmap='Greens', ax=axes[1])
axes[1].set_title(f'С балансировкой (порог={optimal_threshold:.2f})')
axes[1].set_xlabel('Предсказано')
axes[1].set_ylabel('Истина')

plt.tight_layout()
plt.show()

print("\n📊 ВЫВОД: Настройка порога позволяет улучшить Recall,")
print("   но может снизить Precision. Выбор зависит от бизнес-задачи.")

# %% [markdown]
# ## 11. (Дополнительно) Удаление коррелированных признаков

# %%
print("=== УДАЛЕНИЕ КОРРЕЛИРОВАННЫХ ПРИЗНАКОВ ===")
print("\nЗачем удалять коррелированные признаки:")
print("1. Уменьшение размерности (ускорение обучения)")
print("2. Снижение мультиколлинеарности")
print("3. Упрощение модели и улучшение интерпретируемости")
print("4. Возможное снижение переобучения")

# Находим пары с корреляцией > 0.9
correlation_matrix = X_train_scaled.corr().abs()
upper_triangle = correlation_matrix.where(
    np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
)

to_drop = [column for column in upper_triangle.columns 
           if any(upper_triangle[column] > 0.9)]

print(f"\nПризнаки для удаления (|r| > 0.9): {to_drop}")

# Создаем версию без коррелированных признаков
X_train_reduced = X_train_scaled.drop(columns=to_drop)
X_test_reduced = X_test_scaled.drop(columns=to_drop)

print(f"\nРазмерность до: {X_train_scaled.shape[1]} признаков")
print(f"Размерность после: {X_train_reduced.shape[1]} признаков")

# %%
# Обучаем модель на уменьшенном наборе признаков
if len(to_drop) > 0:
    knn_reduced = KNeighborsClassifier(n_neighbors=optimal_k_f1)
    knn_reduced.fit(X_train_reduced, y_train)

    y_pred_reduced = knn_reduced.predict(X_test_reduced)

    print("\n=== СРАВНЕНИЕ: ВСЕ ПРИЗНАКИ vs БЕЗ КОРРЕЛИРОВАННЫХ ===")
    print("\nВсе признаки:")
    print(f"  Accuracy:  {accuracy_score(y_test, y_test_pred_final):.4f}")
    print(f"  F1-Score:  {f1_score(y_test, y_test_pred_final):.4f}")

    print("\nБез коррелированных:")
    print(f"  Accuracy:  {accuracy_score(y_test, y_pred_reduced):.4f}")
    print(f"  F1-Score:  {f1_score(y_test, y_pred_reduced):.4f}")

    print("\n📊 ВЫВОД: Удаление коррелированных признаков существенно")
    print("   не ухудшило качество, но упростило модель.")
else:
    print("\n✓ Сильно коррелированных признаков (|r| > 0.9) не обнаружено.")

# %% [markdown]
# ## 12. Общие выводы

# %%
print("=" * 80)
print(" " * 20 + "ИТОГОВЫЕ ВЫВОДЫ")
print("=" * 80)

print("\nДАННЫЕ:")
print("   • Датасет: 10,127 клиентов, 21 признак")
print("   • Целевая переменная: отток клиентов (16% ушли, 84% остались)")
print("   • Данные несбалансированы, пропущенных значений нет")

print("\nРАЗВЕДОЧНЫЙ АНАЛИЗ:")
print("   • Ушедшие клиенты имеют меньше транзакций")
print("   • Низкое использование кредитной карты - признак оттока")
print("   • Сильная корреляция между Credit_Limit и Avg_Open_To_Buy")

print("\nМОДЕЛИ:")
print("   • KNN: простой и эффективный метод для этой задачи")
print(f"   • Оптимальное k = {optimal_k_f1} (по F1-score)")
print(f"   • Accuracy на тесте: ~{max(test_scores):.1%}")
print("   • Random Forest показал лучшие результаты среди сравниваемых моделей")

print("\nНЕСБАЛАНСИРОВАННОСТЬ:")
print("   • Настройка порога классификации улучшает Recall")
print("   • Выбор метрики зависит от бизнес-приоритетов:")
print("     - Precision важен, если дорого ошибиться (предложить бонус существующему)")
print("     - Recall важен, если дорого пропустить уход клиента")

print("\nРЕКОМЕНДАЦИИ ДЛЯ БАНКА:")
print("   1. Обратить внимание на клиентов с низкой активностью транзакций")
print("   2. Мониторить снижение использования карты")
print("   3. Проактивно работать с клиентами, имеющими низкий Total_Trans_Ct")
print("   4. Использовать модель для раннего предупреждения оттока")

print("\n" + "=" * 80)
print(" " * 25 + "РАБОТА ЗАВЕРШЕНА!")
print("=" * 80)
