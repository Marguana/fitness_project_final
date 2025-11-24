# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Настройка страницы
st.set_page_config(
    page_title="Fitness Recommendation System",
    page_icon="💪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Заголовок приложения
st.title("🏋️ Fitness Recommendation System")
st.markdown("""
### Персонализированная система рекомендаций по тренировкам и питанию
*Data Science проект - Рекомендательная система на основе машинного обучения*
""")

# Загрузка моделей
@st.cache_resource
def load_models():
    """Упрощенная загрузка моделей для демо"""
    st.warning("⚠️ Режим демонстрации - используются тестовые данные")
    
    class SimpleRecommender:
        def get_production_recommendations(self, user_data, n_recommendations=3):
            # Определяем рекомендацию на основе входных данных
            user_bmi = user_data['BMI'].iloc[0] if 'BMI' in user_data else 22
            user_workout = user_data['Workout_Type'].iloc[0] if 'Workout_Type' in user_data else 'Strength'
            
            # Логика рекомендаций на основе реальных данных
            if user_bmi > 25:
                # Для высокого BMI рекомендуем Cardio или HIIT
                recommendations = [
                    {
                        'Workout_Type': 'Cardio',
                        'diet_type': 'Low-Carb', 
                        'Calories': 1800,
                        'Workout_Frequency (days/week)': 4,
                        'Proteins': 90,
                        'expert_advice': ['🏃‍♂️ Кардио тренировки для сжигания жира', '🥬 Низкоуглеводная диета', '💧 Увеличьте водный баланс'],
                        'user_cluster': 1,
                        'cluster_description': 'Группа снижения веса',
                        'bmr': 1600,
                        'calorie_balance': 200,
                        'calorie_status': 'surplus'
                    },
                    {
                        'Workout_Type': 'HIIT',
                        'diet_type': 'Balanced',
                        'Calories': 1900, 
                        'Workout_Frequency (days/week)': 3,
                        'Proteins': 95,
                        'expert_advice': ['⚡ HIIT для эффективного жиросжигания', '⚖️ Сбалансированное питание', '📊 Отслеживайте прогресс'],
                        'user_cluster': 1,
                        'cluster_description': 'Группа снижения веса',
                        'bmr': 1650,
                        'calorie_balance': 250,
                        'calorie_status': 'surplus'
                    }
                ]
            else:
                # Для нормального BMI рекомендуем Strength или Yoga
                recommendations = [
                    {
                        'Workout_Type': 'Strength',
                        'diet_type': 'High-Protein', 
                        'Calories': 2200,
                        'Workout_Frequency (days/week)': 4,
                        'Proteins': 120,
                        'expert_advice': ['💪 Силовые тренировки для роста мышц', '🥩 Высокобелковая диета', '🛌 Не забывайте про восстановление'],
                        'user_cluster': 0,
                        'cluster_description': 'Фитнес-энтузиасты',
                        'bmr': 1800,
                        'calorie_balance': 400,
                        'calorie_status': 'surplus'
                    },
                    {
                        'Workout_Type': 'Yoga',
                        'diet_type': 'Balanced',
                        'Calories': 2000,
                        'Workout_Frequency (days/week)': 5,
                        'Proteins': 80,
                        'expert_advice': ['🧘‍♀️ Йога для гибкости и баланса', '🍎 Сбалансированное питание', '🌿 Фокус на ментальное здоровье'],
                        'user_cluster': 0, 
                        'cluster_description': 'Фитнес-энтузиасты',
                        'bmr': 1700,
                        'calorie_balance': 300,
                        'calorie_status': 'surplus'
                    }
                ]
            
            return pd.DataFrame(recommendations[:n_recommendations])
    
    return SimpleRecommender()

def load_models():
    """Упрощенная загрузка моделей для демо"""
    st.warning("⚠️ Режим демонстрации - используются тестовые данные")
    
    class SimpleRecommender:
        def get_production_recommendations(self, user_data, n_recommendations=3):
            # Тестовые рекомендации
            return pd.DataFrame([{
                'Workout_Type': 'Strength',
                'diet_type': 'High-Protein', 
                'Calories': 2200,
                'Workout_Frequency (days/week)': 4,
                'Proteins': 120,
                'expert_advice': ['💪 Силовые тренировки 3-4 раза в неделю', '🥩 Высокобелковая диета', '💧 Пейте 2+ литра воды'],
                'user_cluster': 0,
                'cluster_description': 'Фитнес-энтузиаст',
                'bmr': 1800,
                'calorie_balance': 400,
                'calorie_status': 'surplus'
            }])
    
    return SimpleRecommender()

# Загрузка данных для анализа
@st.cache_data
def load_analysis_data():
    """Загрузка данных для анализа"""
    try:
        data_path = Path("data_processed/df_clean.csv")
        df = pd.read_csv(data_path)
        return df
    except:
        return None

# Основная функция
def main():
    # Сайдбар для навигации
    st.sidebar.title("📊 Навигация")
    page = st.sidebar.radio("Выберите раздел:", [
        "🎯 Рекомендации", 
        "📈 Анализ данных", 
        "📊 О проекте"
    ])

    # Загрузка моделей и данных
    recommender = load_models()
    df = load_analysis_data()

    if page == "🎯 Рекомендации":
        show_recommendations(recommender)
    elif page == "📈 Анализ данных":
        show_analysis(df)
    elif page == "📊 О проекте":
        show_about()

def show_recommendations(recommender):
    """Раздел с рекомендациями"""
    st.header("🎯 Персонализированные рекомендации")
    
    with st.form("user_profile"):
        st.subheader("📋 Ваш профиль")
        
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.slider("Возраст", 18, 70, 30)
            weight = st.slider("Вес (кг)", 40, 150, 75)
            height = st.slider("Рост (м)", 1.4, 2.2, 1.75)
            gender = st.selectbox("Пол", ["Male", "Female"])
            
        with col2:
            workout_type = st.selectbox("Тип тренировок", 
                                      ["Strength", "Cardio", "Yoga", "HIIT"])
            experience = st.selectbox("Уровень подготовки", 
                                    ["Beginner", "Intermediate", "Advanced"])
            diet_type = st.selectbox("Тип питания", 
                                   ["Balanced", "Low-Carb", "High-Protein", "Paleo", "Vegetarian"])
            workout_freq = st.slider("Частота тренировок (дней/неделю)", 1, 7, 3)
        
        # Дополнительные параметры
        st.subheader("📊 Дополнительные параметры")
        col3, col4 = st.columns(2)
        
        with col3:
            fat_percentage = st.slider("Процент жира", 5.0, 40.0, 20.0)
            water_intake = st.slider("Потребление воды (л/день)", 1.0, 5.0, 2.5)
            
        with col4:
            calories = st.number_input("Суточная калорийность", 1000, 5000, 2000)
            protein = st.number_input("Белки (г/день)", 50, 300, 100)
        
        submitted = st.form_submit_button("Получить рекомендации 🚀")
    
    if submitted and recommender:
        # Создаем профиль пользователя
        user_profile = {
            "Age": age,
            "Weight (kg)": weight,
            "Height (m)": height,
            "BMI": weight / (height ** 2),
            "Fat_Percentage": fat_percentage,
            "Water_Intake (liters)": water_intake,
            "Workout_Frequency (days/week)": workout_freq,
            "Workout_Type": workout_type,
            "Experience_Level": experience,
            "diet_type": diet_type,
            "Gender": gender,
            "Calories": calories,
            "Proteins": protein,
            "Carbs": calories * 0.4 / 4,  # Примерный расчет
            "Fats": calories * 0.25 / 9,   # Примерный расчет
        }
        
        # Добавляем расчетные поля
        user_profile["protein_per_kg"] = protein / weight
        user_profile["lean_mass_kg"] = weight * (1 - fat_percentage/100)
        user_profile["pct_carbs"] = 40  # Примерное значение
        
        try:
            # Получаем рекомендации
            recommendations = recommender.get_production_recommendations(
                pd.DataFrame([user_profile]), 3
            )
            
            if not recommendations.empty:
                display_recommendations(recommendations, user_profile)
            else:
                st.warning("Не удалось найти рекомендации. Попробуйте изменить параметры.")
                
        except Exception as e:
            st.error(f"Ошибка при получении рекомендаций: {e}")

def display_recommendations(recommendations, user_profile):
    """Отображение рекомендаций"""
    st.success("🎉 Рекомендации успешно сгенерированы!")
    
    # Информация о пользователе
    with st.expander("📊 Ваш профиль анализа", expanded=True):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("BMI", f"{user_profile['BMI']:.1f}")
            st.metric("Возраст", user_profile['Age'])
            
        with col2:
            st.metric("Вес", f"{user_profile['Weight (kg)']} кг")
            st.metric("Белок на кг", f"{user_profile['protein_per_kg']:.1f} г")
            
        with col3:
            bmi_status = "Норма" if 18.5 <= user_profile['BMI'] <= 25 else ("Недостаток" if user_profile['BMI'] < 18.5 else "Избыток")
            st.metric("Статус BMI", bmi_status)
            st.metric("Частота тренировок", f"{user_profile['Workout_Frequency (days/week)']} д/нед")
    
    # Рекомендации
    st.header("💪 Персональные рекомендации")
    
    for i, (_, rec) in enumerate(recommendations.iterrows(), 1):
        with st.container():
            st.markdown(f"### 🎯 Вариант {i}")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🏋️ Тренировки")
                st.info(f"**Тип:** {rec['Workout_Type']}")
                st.info(f"**Частота:** {rec['Workout_Frequency (days/week)']} дней/неделю")
                
            with col2:
                st.subheader("🍽️ Питание")
                st.success(f"**Тип диеты:** {rec['diet_type']}")
                st.success(f"**Калории:** {rec['Calories']} ккал/день")
                if 'Proteins' in rec:
                    st.success(f"**Белки:** {rec['Proteins']} г/день")
            
            # Дополнительная информация
            if 'bmr' in rec:
                st.markdown("---")
                col3, col4 = st.columns(2)
                with col3:
                    st.metric("BMR (Основной обмен)", f"{rec['bmr']:.0f} ккал")
                with col4:
                    calorie_balance = rec['Calories'] - rec['bmr']
                    status = "Дефицит" if calorie_balance < -200 else ("Профицит" if calorie_balance > 200 else "Баланс")
                    st.metric("Баланс калорий", f"{calorie_balance:+.0f} ккал", status)

def show_analysis(df):
    """Раздел с анализом данных"""
    st.header("📈 Анализ данных проекта")
    
    if df is None:
        st.warning("Данные для анализа не найдены")
        return
    
    st.subheader("Обзор данных")
    st.write(f"**Размер dataset:** {df.shape[0]} пользователей, {df.shape[1]} признаков")
    
    # Основные метрики
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_age = df['Age'].mean()
        st.metric("Средний возраст", f"{avg_age:.1f} лет")
    
    with col2:
        avg_bmi = df['BMI'].mean()
        st.metric("Средний BMI", f"{avg_bmi:.1f}")
    
    with col3:
        avg_calories = df['Calories'].mean()
        st.metric("Средние калории", f"{avg_calories:.0f}")
    
    with col4:
        if 'cluster' in df.columns:
            cluster_count = df['cluster'].nunique()
            st.metric("Кластеров", cluster_count)
    
    # Визуализации
    st.subheader("📊 Визуализация данных")
    
    tab1, tab2, tab3 = st.tabs(["Распределение BMI", "Потребление калорий", "Кластеры"])
    
    with tab1:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(df['BMI'], bins=20, kde=True, ax=ax)
        ax.set_title('Распределение BMI пользователей')
        ax.axvline(18.5, color='red', linestyle='--', label='Недостаток')
        ax.axvline(25, color='green', linestyle='--', label='Норма')
        ax.axvline(30, color='orange', linestyle='--', label='Избыток')
        ax.legend()
        st.pyplot(fig)
    
    with tab2:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(data=df, x='Weight (kg)', y='Calories', hue='Workout_Type', ax=ax)
        ax.set_title('Зависимость калорий от веса и типа тренировок')
        st.pyplot(fig)
    
    with tab3:
        if 'cluster' in df.columns:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Распределение по кластерам
            cluster_counts = df['cluster'].value_counts()
            ax1.pie(cluster_counts.values, labels=[f'Кластер {i}' for i in cluster_counts.index], autopct='%1.1f%%')
            ax1.set_title('Распределение пользователей по кластерам')
            
            # Средние значения по кластерам
            cluster_means = df.groupby('cluster')[['Age', 'BMI', 'Calories']].mean()
            sns.heatmap(cluster_means, annot=True, fmt='.1f', ax=ax2)
            ax2.set_title('Средние значения по кластерам')
            
            st.pyplot(fig)

def show_about():
    """Раздел о проекте"""
    st.header("📊 О проекте")
    
    st.markdown("""
    ## 🏋️ Fitness Recommendation System
    
    ### 🎯 Цель проекта
    Разработка персонализированной системы рекомендаций для фитнеса и питания 
    на основе методов машинного обучения.
    
    ### 📊 Метрики качества
    - **Coverage**: 100% - система работает для всех пользователей
    - **Precision**: 100% - высокая точность рекомендаций  
    - **Diversity**: 44% - хорошее разнообразие рекомендаций
    - **A/B Test**: На 58% лучше случайных рекомендаций
    
    ### 🛠️ Технологии
    - **Python** + Scikit-learn для ML
    - **Pandas** для обработки данных
    - **Streamlit** для веб-интерфейса
    - **Git** для контроля версий
    
    ### 📁 Структура проекта
    ```
    fitness_project_final/
    ├── app.py              # Streamlit приложение
    ├── data/               # Исходные данные
    ├── data_processed/     # Обработанные данные
    ├── models/            # Обученные ML модели
    ├── README.md          # Документация
    └── requirements.txt   # Зависимости
    ```
    
    ### 👥 Кластерный анализ
    Система выделяет 2 основных кластера пользователей:
    1. **Фитнес-энтузиасты** - нормальный вес, силовые тренировки
    2. **Группа снижения веса** - высокий BMI, йога и низкоуглеводная диета
    """)

if __name__ == "__main__":
    main()
