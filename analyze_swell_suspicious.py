#!/usr/bin/env python3
"""
SWELL Dataset Analysis - Investigación del 99.6% accuracy sospechoso
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

def analyze_swell_suspicion():
    """Analizar por qué SWELL da 99.6% accuracy - puede ser data leakage."""
    
    print("🔍 INVESTIGACIÓN: ¿Por qué SWELL tiene 99.6% accuracy?")
    print("=" * 60)
    
    # Buscar archivos SWELL reales
    swell_files = []
    data_path = Path("data")
    
    # Buscar en subdirectorios
    for pattern in ["*SWELL*.csv", "*Computer*.csv", "*Facial*.csv", "*Posture*.csv", "*Physiology*.csv"]:
        swell_files.extend(list(data_path.rglob(pattern)))
    
    print(f"📁 Archivos SWELL encontrados: {len(swell_files)}")
    for f in swell_files[:5]:
        print(f"  - {f.name}")
    
    if not swell_files:
        print("❌ No se encontraron archivos SWELL - creando datos sintéticos para análisis")
        return create_synthetic_analysis()
    
    # Analizar el primer archivo disponible
    try:
        first_file = swell_files[0]
        print(f"\n📊 Analizando: {first_file.name}")
        
        df = pd.read_csv(first_file)
        print(f"  Shape: {df.shape}")
        print(f"  Columns: {list(df.columns[:10])}...")
        
        # Buscar columnas de condición/label
        condition_cols = [col for col in df.columns if 'condition' in col.lower() or 'label' in col.lower() or 'stress' in col.lower()]
        print(f"  Condition columns: {condition_cols}")
        
        if condition_cols:
            condition_col = condition_cols[0]
            conditions = df[condition_col].value_counts()
            print(f"  Condition distribution:\n{conditions}")
            
            # Verificar si hay patrones sospechosos
            if len(conditions) == 2 and conditions.min() / conditions.max() > 0.9:
                print("  ⚠️  SOSPECHOSO: Balance perfecto entre clases")
            
            if len(df) > 1000 and df[condition_col].nunique() <= 4:
                print("  ⚠️  SOSPECHOSO: Pocas condiciones para tantos datos")
        
        # Verificar duplicados
        duplicates = df.duplicated().sum()
        print(f"  Duplicados: {duplicates} ({duplicates/len(df)*100:.1f}%)")
        
        if duplicates > len(df) * 0.1:
            print("  🚨 PROBLEMA: >10% duplicados - posible data leakage")
        
        # Análisis de variabilidad
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            zero_var_cols = (df[numeric_cols].var() == 0).sum()
            print(f"  Columnas varianza cero: {zero_var_cols}/{len(numeric_cols)}")
            
            if zero_var_cols > len(numeric_cols) * 0.5:
                print("  🚨 PROBLEMA: >50% columnas constantes")
        
    except Exception as e:
        print(f"  ❌ Error leyendo archivo: {e}")
        return create_synthetic_analysis()

def create_synthetic_analysis():
    """Crear análisis sintético para entender el patrón sospechoso."""
    
    print("\n🧪 ANÁLISIS SINTÉTICO: Recreando condiciones SWELL")
    print("-" * 50)
    
    # Recrear el escenario que puede causar 99.6%
    np.random.seed(42)
    n_samples = 10000
    
    # Escenario 1: Data Leakage por merge incorrecto
    print("\n📊 Escenario 1: Data leakage por merge")
    
    # Simular datos como en SWELL script
    participant_ids = [f"P{(i//100) + 1:02d}" for i in range(n_samples)]
    conditions = np.random.choice(['N', 'T', 'I', 'R'], n_samples)
    
    # Crear features que dependen directamente de la condición (data leakage)
    features = np.random.randn(n_samples, 20)
    
    # AQUÍ ESTÁ EL PROBLEMA: features correlacionadas perfectamente con labels
    labels = (pd.Series(conditions).isin(['T', 'I', 'R'])).astype(int)
    
    # Simular el bug: hacer que algunas features sean casi deterministas
    features[:, 0] = labels + np.random.normal(0, 0.01, n_samples)  # Casi determinista
    features[:, 1] = labels * 2 + np.random.normal(0, 0.05, n_samples)  # Muy correlacionada
    
    print(f"  Labels distribution: {np.bincount(labels)}")
    print(f"  Feature 0 correlation with labels: {np.corrcoef(features[:, 0], labels)[0,1]:.3f}")
    print(f"  Feature 1 correlation with labels: {np.corrcoef(features[:, 1], labels)[0,1]:.3f}")
    
    # División por sujetos (como debería ser)
    unique_participants = list(set(participant_ids))
    train_participants = unique_participants[:len(unique_participants)//2]
    
    train_mask = pd.Series(participant_ids).isin(train_participants)
    
    X_train = features[train_mask]
    X_test = features[~train_mask]
    y_train = labels[train_mask]
    y_test = labels[~train_mask]
    
    print(f"  Train: {len(X_train)}, Test: {len(X_test)}")
    
    # Entrenar Random Forest
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_train, y_train)
    
    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"  🎯 Accuracy: {accuracy:.3f}")
    
    if accuracy > 0.95:
        print("  🚨 CONFIRMADO: Data leakage reproduce el problema")
        
        # Análizar importancia de features
        importances = rf.feature_importances_
        top_features = np.argsort(importances)[::-1][:3]
        
        print(f"  Top 3 features importantes:")
        for i, feat_idx in enumerate(top_features):
            print(f"    Feature {feat_idx}: importance={importances[feat_idx]:.3f}")
    
    print("\n💡 DIAGNÓSTICO:")
    print("  El 99.6% accuracy en SWELL probablemente se debe a:")
    print("  1. 🔗 Data leakage en el merge de modalidades")
    print("  2. 🔄 Features sintéticas correlacionadas con labels")
    print("  3. 📊 Distribución artificial de 'sujetos'")
    print("  4. 🎭 Random Forest overfitting a patrones artificiales")
    
    print("\n🔧 SOLUCIONES:")
    print("  ✓ Usar datos reales de SWELL sin merge complejo")
    print("  ✓ Validar que features no contengan información de labels")
    print("  ✓ Cross-validation más estricta")
    print("  ✓ Probar con modelos más simples primero")

if __name__ == "__main__":
    analyze_swell_suspicion()
