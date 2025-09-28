# 🚀 Multi-Dataset Federated Learning Demo – Executive Summary

**Fecha**: 27 Septiembre 2025  
**Demo Ejecutado**: Federación WESAD + SWELL (sujeto a particiones por participante)

---

## 📊 Resultados Baseline Actualizados (splits por sujeto)

### 🩺 WESAD – Stress fisiológico controlado
- **Modalidad**: Señales fisiológicas (BVP, EDA, ACC, TEMP)
- **Participantes**: 15 sujetos reales (S2-S17)
- **División**: 10 sujetos para entrenamiento (572 ventanas) / 5 sujetos para test (287 ventanas)
- **Características**: 30 estadísticas por ventana (60 s, 50% solape)
- **Modelos**:
  - Logistic Regression → **93.0%** accuracy, **0.921** macro-F1
  - Random Forest → **96.2%** accuracy, **0.959** macro-F1
- 5-fold CV (por sujeto): LR 86.5% +/- 7.9% acc / 85.4% +/- 8.7% macro-F1; RF 76.8% +/- 8.3% acc / 73.8% +/- 8.1% macro-F1.
- **Insight**: El protocolo TSST genera diferencias fisiológicas claras incluso con menos sujetos en train (manteniendo la separación por participante).

### 💻 SWELL – Stress en escenarios de oficina
- **Modalidad**: Computer interaction (mouse/teclado/app switching)
- **Participantes**: 25 (20 sujetos para train / 5 para test tras split por participante)
- **Características**: 17 métricas limpias + canónicas (incluye SCL↔EDA cuando está presente)
- **Modelos**:
  - Logistic Regression → **95.3%** accuracy, **0.948** macro-F1
  - Random Forest → **99.2%** accuracy, **0.991** macro-F1
- 5-fold CV (por sujeto): LR 95.1% +/- 0.9% acc / 94.6% +/- 0.9% macro-F1; RF 98.9% +/- 0.6% acc / 98.7% +/- 0.8% macro-F1.
- **Insight**: Las estadísticas de interacción, combinadas con imputaciones y reducción de varianza, distinguen con claridad sesiones con presión/interruptions vs baseline.

---

## 🤝 Escenario Federado / Multimodal

- **Loaders**: `load_real_multimodal_dataset()` aplica el mapa canónico (EDA) y mantiene bloques específicos por modalidad.
- **Particiones por sujeto**: 24 sujetos en train (2 453 ventanas), 6 sujetos en validación (619), 10 sujetos en test (926) – sin fuga de datos.
- **Resultados (test)**:
  - Logistic Regression → **90.8%** accuracy, **0.906** macro-F1
  - Random Forest → **97.5%** accuracy, **0.975** macro-F1
- 5-fold CV (por sujeto): LR 93.1% +/- 2.6% acc / 92.8% +/- 2.9% macro-F1; RF 94.5% +/- 3.3% acc / 94.1% +/- 3.7% macro-F1.
- **Métricas de validación** (monitorización previa al retrain): LR 88.2% / RF 94.5% accuracy.
- **Indicador de dataset**: Se mantiene la columna `dataset_is_wesad` para diferenciar procedencia durante el entrenamiento global.

---

## 🔑 Claves Técnicas
- **Alineación canónica**: EDA/SCL comparten las mismas columnas; se evita inflar dimensionalidad con prefijos duplicados.
- **Imputación controlada**: Se rellenan NaN con medias por característica y se eliminan features de varianza cero en SWELL.
- **Splits por sujeto**: Todos los scripts (`run_multi_dataset_demo.py`, `evaluate_multimodal_baseline.py`, `demo_multidataset_fl.py`) usan `_split_by_subject`, preservando privacidad y evitando leakage.
- **Metadatos ricos**: Los loaders devuelven `n_subjects`, `n_samples`, `feature_names` y escaladores para reproducibilidad (véanse `multi_dataset_demo_report.json` y `multimodal_baseline_results.json`).

---

## ✅ Próximos Pasos Recomendados
1. **Explorar modalidad fisiológica completa de SWELL** para incorporar HR/HRV y reforzar la porción común con WESAD.
2. **Evaluar modelos multimodales jerárquicos** (ramas separadas por dataset + fusión tardía) para aprovechar los indicadores específicos.
3. **Añadir métricas por origen** en la evaluación combinada (p. ej., rendimiento del modelo global sobre ventanas SWELL vs WESAD por separado).
4. **Documentar pipeline reproducible** (Makefile/README) con los nuevos parámetros (`test_size=5/15` para WESAD) y scripts actualizados.

---

ℹ️ Para valores exactos, consulte:
- `multi_dataset_demo_report.json`
- `WESAD_BASELINE_RESULTS.md`
- `SWELL_BASELINE_RESULTS.md`
- `multimodal_baseline_results.json`
- `subject_cv_results/subject_cv_summary.csv`
- `subject_cv_results/subject_cv_summary.json`
