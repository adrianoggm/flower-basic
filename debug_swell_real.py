#!/usr/bin/env python3
"""AnÃ¡lisis directo de por quÃ© SWELL da 99.6% accuracy sospechoso."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def analyze_swell_real_data():
    """Analizar los datos reales de SWELL para encontrar el problema."""

    print("ð ANÃLISIS DIRECTO DE DATOS SWELL REALES")
    print("=" * 50)

    # Cargar el archivo computer interaction que sabemos que existe
    computer_file = "data/SWELL/3 - Feature dataset/per sensor/A - Computer interaction features (Ulog - All Features per minute)-Sheet_1.csv"

    try:
        df = pd.read_csv(computer_file)
        print(f"â Cargado: {computer_file}")
        print(f"  Shape: {df.shape}")
        print(f"  Columns: {list(df.columns)}")
        print()

        # Examinar las primeras filas
        print("ð Primeras 3 filas:")
        print(df.head(3))
        print()

        # Analizar condiciones
        print("ð·ï¸ AnÃ¡lisis de condiciones:")
        if "Condition" in df.columns:
            conditions = df["Condition"].value_counts()
            print(f"  DistribuciÃ³n: {dict(conditions)}")

            # Verificar patrones sospechosos
            print(f"  Condiciones Ãºnicas: {df['Condition'].unique()}")

            # AnÃ¡lisis de participantes
            if "PP" in df.columns:
                participants = df["PP"].value_counts()
                print(f"  Participantes: {len(participants)} Ãºnicos")
                print(
                    f"  Muestras por participante: min={participants.min()}, max={participants.max()}, mean={participants.mean():.1f}"
                )

        # AnÃ¡lizar features numÃ©ricas
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        print(f"\nð¢ Features numÃ©ricas: {len(numeric_cols)}")

        if len(numeric_cols) > 0:
            # Verificar varianza
            variances = df[numeric_cols].var()
            zero_var = (variances == 0).sum()
            print(f"  Features con varianza 0: {zero_var}/{len(numeric_cols)}")

            # Verificar si todas las features son constantes (seÃ+/-al de problema)
            all_zeros = (df[numeric_cols] == 0).all().sum()
            print(f"  Features que son siempre 0: {all_zeros}/{len(numeric_cols)}")

            if all_zeros > len(numeric_cols) * 0.8:
                print("  ð¨ PROBLEMA DETECTADO: >80% features son constantes cero")
                print("     Esto explica por quÃ© Random Forest da 99.6%")
                print("     El modelo estÃ¡ 'memorizando' patrones artificiales")

        # Simular el problema que estÃ¡ causando 99.6%
        print("\nð§ª SIMULACIÃN DEL PROBLEMA:")

        # Crear labels binarias desde condiciones
        if "Condition" in df.columns:
            # Mapeo como en el script original
            stress_conditions = [
                "T",
                "I",
                "R",
            ]  # Time pressure, Interruptions, Combined
            labels = df["Condition"].isin(stress_conditions).astype(int)

            print(f"  Labels creadas: {np.bincount(labels)}")

            # Solo usar features numÃ©ricas no constantes
            valid_features = df[numeric_cols].loc[:, df[numeric_cols].var() > 0]
            print(f"  Features vÃ¡lidas: {valid_features.shape[1]}")

            if len(valid_features.columns) > 0:
                # DivisiÃ³n por participante (como deberÃ­a ser)
                participants = df["PP"].unique()
                train_pp = participants[: len(participants) // 2]

                train_mask = df["PP"].isin(train_pp)

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

                print(f"  ð¯ Accuracy con datos reales: {accuracy:.3f}")

                if accuracy > 0.9:
                    print(
                        "  ð¨ CONFIRMADO: Los datos reales tambiÃ©n dan alta accuracy"
                    )
                    print("     Posibles causas:")
                    print("     1. Task design muy predecible")
                    print("     2. Features demasiado discriminativas")
                    print("     3. Participantes muy consistentes")
                else:
                    print("  â Accuracy normal con datos reales")
                    print("     El problema estÃ¡ en el procesamiento del script")

    except Exception as e:
        print(f"â Error: {e}")

        # AnÃ¡lisis alternativo si no se puede cargar
        print("\nð ANÃLISIS ALTERNATIVO - TeorÃ­a del problema:")
        print("El 99.6% en SWELL probablemente se debe a:")
        print()
        print("1. ð DATOS SINTÃTICOS MAL GENERADOS:")
        print("   - Los 'sujetos' P01-P500 son artificiales")
        print("   - Creados con patrones demasiado regulares")
        print("   - No hay variabilidad real entre sujetos")
        print()
        print("2. ð DATA LEAKAGE EN EL MERGE:")
        print("   - Merge de 4 modalidades crea dependencias artificiales")
        print("   - Features correlacionadas perfectamente con conditions")
        print("   - Random sampling puede preservar patrones")
        print()
        print("3. ð­ OVERFITTING EXTREMO:")
        print("   - Random Forest memoriza patrones de merge")
        print("   - DivisiÃ³n por 'sujetos' sintÃ©ticos no es realista")
        print("   - 500 'sujetos' de 100 muestras cada uno = artificial")
        print()
        print("ð¡ SOLUCIÃN: Usar datos reales SWELL sin merge complejo")


if __name__ == "__main__":
    analyze_swell_real_data()
