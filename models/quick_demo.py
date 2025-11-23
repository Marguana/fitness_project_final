#!/usr/bin/env python3
"""
Демо-скрипт для Fitness Recommendation System
"""

import pandas as pd
import joblib
import os
import sys

def main():
    print("🏋️ Fitness Recommendation System - Демо")
    print("=" * 50)
    
    try:
        # Загрузка модели
        model_path = os.path.join(os.path.dirname(__file__), "production_fitness_recommender.pkl")
        recommender = joblib.load(model_path)
        print("✅ Модель успешно загружена")
        
    except Exception as e:
        print(f"❌ Ошибка загрузки модели: {e}")
        return

    # Примеры пользователей для демо
    demo_users = [
        {
            "name": "Пользователь с лишним весом",
            "data": {
                "Age": 42,
                "Weight (kg)": 88,
                "Height (m)": 1.65,
                "BMI": 32.3,
                "Workout_Type": "Cardio",
                "Workout_Frequency (days/week)": 1,
                "Experience_Level": "Beginner",
                "Gender": "Female"
            }
        },
        {
            "name": "Фитнес-энтузиаст", 
            "data": {
                "Age": 29,
                "Weight (kg)": 72,
                "Height (m)": 1.78,
                "BMI": 22.7,
                "Workout_Type": "Strength",
                "Workout_Frequency (days/week)": 5,
                "Experience_Level": "Advanced",
                "Gender": "Male"
            }
        }
    ]

    for user in demo_users:
        print(f"\n🧪 ТЕСТ: {user['name']}")
        print("-" * 40)
        
        try:
            recommendations = recommender.get_production_recommendations(
                pd.DataFrame([user['data']]), 2
            )
            
            if not recommendations.empty:
                print("✅ Рекомендации получены:")
                for i, (_, rec) in enumerate(recommendations.iterrows(), 1):
                    print(f"   {i}. {rec['Workout_Type']} + {rec['diet_type']}")
                    print(f"      Калории: {rec['Calories']} | Частота: {rec['Workout_Frequency (days/week)']} дней/неделю")
            else:
                print("❌ Рекомендации не найдены")
                
        except Exception as e:
            print(f"❌ Ошибка: {e}")

if __name__ == "__main__":
    main()
