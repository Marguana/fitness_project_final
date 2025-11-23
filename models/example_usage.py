
# ПРИМЕР ИСПОЛЬЗОВАНИЯ ФИНАЛЬНОЙ СИСТЕМЫ:

import pandas as pd
import joblib

# Загрузка модели
recommender = joblib.load("models/production_fitness_recommender.pkl")

# Создание профиля пользователя
user_profile = {
    "Age": 35,
    "Weight (kg)": 95, 
    "Height (m)": 1.75,
    "BMI": 31.0,
    "Workout_Type": "Yoga",
    "Workout_Frequency (days/week)": 2,
    "diet_type": "Balanced",
    "Experience_Level": "Beginner",
    "Gender": "Female"
}

# Получение рекомендаций
recommendations = recommender.get_production_recommendations(
    pd.DataFrame([user_profile]), 
    n_recommendations=3
)

print("Ваши персонализированные рекомендации:")
for i, (_, rec) in enumerate(recommendations.iterrows(), 1):
    print(f"{i}. {rec['Workout_Type']} + {rec['diet_type']}")
    print(f"   Калории: {rec['Calories']} | Частота: {rec['Workout_Frequency (days/week)']} дней/неделю")
