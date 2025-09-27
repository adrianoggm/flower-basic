#!/usr/bin/env python3
"""Análisis directo de por qué SWELL da 99.6% accuracy sospechoso."""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def analyze_swell_real_data():
    """Analizar los datos reales de SWELL para encontrar el problema."""
    
    print("🔍 ANÁLISIS DIRECTO DE DATOS SWELL REALES")
    print("=" * 50)
    
    # Cargar el archivo computer interaction que sabemos que existe
    computer_file = "data/SWELL/3 - Feature dataset/per sensor/A - Computer interaction features (Ulog - All Features per minute)-Sheet_1.csv"
    
    try:
        df = pd.read_csv(computer_file)
        print(f"✓ Cargado: {computer_file}")
        print(f"  Shape: {df.shape}")
        print(f"  Columns: {list(df.columns)}")
        print()
        
        # Examinar las primeras filas
        print("📊 Primeras 3 filas:")
        print(df.head(3))
        print()
        
        # Analizar condiciones
        print("🏷️ Análisis de condiciones:")
        if 'Condition' in df.columns:
            conditions = df['Condition'].value_counts()
            print(f"  Distribución: {dict(conditions)}")
            
            # Verificar patrones sospechosos
            print(f"  Condiciones únicas: {df['Condition'].unique()}")
            
            # Análisis de participantes
            if 'PP' in df.columns:
                participants = df['PP'].value_counts()
                print(f"  Participantes: {len(participants)} únicos")
                print(f"  Muestras por participante: min={participants.min()}, max={participants.max()}, mean={participants.mean():.1f}")
        
        # Análizar features numéricas
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        print(f"\n🔢 Features numéricas: {len(numeric_cols)}")
        
        if len(numeric_cols) > 0:
            # Verificar varianza
            variances = df[numeric_cols].var()
            zero_var = (variances == 0).sum()
            print(f"  Features con varianza 0: {zero_var}/{len(numeric_cols)}")
            
            # Verificar si todas las features son constantes (señal de problema)
            all_zeros = (df[numeric_cols] == 0).all().sum()
            print(f"  Features que son siempre 0: {all_zeros}/{len(numeric_cols)}")
            
            if all_zeros > len(numeric_cols) * 0.8:
                print("  🚨 PROBLEMA DETECTADO: >80% features son constantes cero")
                print("     Esto explica por qué Random Forest da 99.6%")
                print("     El modelo está 'memorizando' patrones artificiales")
        
        # Simular el problema que está causando 99.6%
        print("\n🧪 SIMULACIÓN DEL PROBLEMA:")
        
        # Crear labels binarias desde condiciones
        if 'Condition' in df.columns:
            # Mapeo como en el script original
            stress_conditions = ['T', 'I', 'R']  # Time pressure, Interruptions, Combined
            labels = df['Condition'].isin(stress_conditions).astype(int)
            
            print(f"  Labels creadas: {np.bincount(labels)}")
            
            # Solo usar features numéricas no constantes
            valid_features = df[numeric_cols].loc[:, df[numeric_cols].var() > 0]
            print(f"  Features válidas: {valid_features.shape[1]}")
            
            if len(valid_features.columns) > 0:
                # División por participante (como debería ser)
                participants = df['PP'].unique()
                train_pp = participants[:len(participants)//2]
                
                train_mask = df['PP'].isin(train_pp)
                
                X_train = valid_features[train_mask].values
                X_test = valid_features[~train_mask].values  
                y_train = labels[train_mask].values
                y_test = labels[~train_mask].values
                
                print(f"  Train: {X_train.shape}, Test: {X_test.shape}")
                
                # Entrenar modelo simple
                rf = RandomForestClassifier(n_estimators=10, random_state=42)
                rf.fit(X_train, y_train)
                
                y_pred = rf.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                
                print(f"  🎯 Accuracy con datos reales: {accuracy:.3f}")
                
                if accuracy > 0.9:
                    print("  🚨 CONFIRMADO: Los datos reales también dan alta accuracy")
                    print("     Posibles causas:")
                    print("     1. Task design muy predecible")
                    print("     2. Features demasiado discriminativas")
                    print("     3. Participantes muy consistentes")
                else:
                    print("  ✓ Accuracy normal con datos reales")
                    print("     El problema está en el procesamiento del script")
    
    except Exception as e:
        print(f"❌ Error: {e}")
        
        # Análisis alternativo si no se puede cargar
        print("\n🔄 ANÁLISIS ALTERNATIVO - Teoría del problema:")
        print("El 99.6% en SWELL probablemente se debe a:")
        print()
        print("1. 📊 DATOS SINTÉTICOS MAL GENERADOS:")
        print("   - Los 'sujetos' P01-P500 son artificiales")
        print("   - Creados con patrones demasiado regulares")
        print("   - No hay variabilidad real entre sujetos")
        print()
        print("2. 🔗 DATA LEAKAGE EN EL MERGE:")
        print("   - Merge de 4 modalidades crea dependencias artificiales")  
        print("   - Features correlacionadas perfectamente con conditions")
        print("   - Random sampling puede preservar patrones")
        print()
        print("3. 🎭 OVERFITTING EXTREMO:")
        print("   - Random Forest memoriza patrones de merge")
        print("   - División por 'sujetos' sintéticos no es realista") 
        print("   - 500 'sujetos' de 100 muestras cada uno = artificial")
        print()
        print("💡 SOLUCIÓN: Usar datos reales SWELL sin merge complejo")

if __name__ == "__main__":
    analyze_swell_real_data()
