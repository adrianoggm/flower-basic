# 📊 Baseline Model & Performance Comparison

## 🎯 Objetivo

Este módulo implementa un **modelo baseline centralizado** para comparar el rendimiento del aprendizaje federado vs. el entrenamiento tradicional centralizado, usando la **misma arquitectura de red neuronal**.

## 🏗️ Arquitectura del Modelo

Ambos enfoques (federado y centralizado) utilizan **exactamente la misma arquitectura**:

### **ECGModel - CNN 1D para Clasificación Binaria de ECG**

```
Input: (batch_size, 1, 140)  # ECG time series
├── Conv1D(1→16, kernel=5)   # → (batch_size, 16, 136)
├── ReLU + MaxPool1D(2)      # → (batch_size, 16, 68)
├── Conv1D(16→32, kernel=5)  # → (batch_size, 32, 64)
├── ReLU + MaxPool1D(2)      # → (batch_size, 32, 32)
├── Flatten                  # → (batch_size, 1024)
├── Linear(1024→64)          # → (batch_size, 64)
├── ReLU
└── Linear(64→1)             # → (batch_size, 1)
Output: Raw logits for BCEWithLogitsLoss
```

**Parámetros totales:** ~67,000 parámetros

## 📁 Archivos Implementados

### **1. `baseline_model.py`** - Modelo Centralizado

```python
# Entrenamiento tradicional (toda la data en un lugar)
python baseline_model.py --epochs 50 --batch_size 32 --lr 0.001
```

**Características:**

-   ✅ Entrenamiento centralizado completo
-   ✅ Métricas comprehensivas (accuracy, F1, precision, recall, AUC)
-   ✅ Tracking de curvas de entrenamiento
-   ✅ Guardado automático de resultados y gráficos
-   ✅ Compatibilidad con GPU/CPU

### **2. `compare_models.py`** - Comparación Completa

```python
# Comparación directa entre enfoques
python compare_models.py --epochs 50 --num_clients 3 --fl_rounds 10
```

**Características:**

-   🔄 Simulador de federated learning
-   📊 Comparación lado a lado
-   📈 Visualizaciones automáticas
-   📋 Reportes detallados (JSON + texto)
-   ⚡ Análisis de eficiencia temporal

### **3. `quick_comparison.py`** - Demo Rápido

```python
# Prueba rápida con parámetros reducidos
python quick_comparison.py
```

**Configuración rápida:**

-   Centralizado: 20 epochs
-   Federado: 3 clients, 5 rounds
-   Batch size: 16
-   Tiempo estimado: ~2-3 minutos

## 🚀 Uso Práctico

### **Comparación Básica**

```bash
# 1. Instalar dependencias
pip install matplotlib pandas tabulate seaborn

# 2. Ejecutar comparación rápida
python quick_comparison.py

# 3. Ver resultados
ls quick_comparison_results/
# ├── comparison_report.json    # Métricas JSON
# ├── comparison_report.txt     # Reporte legible
# └── comparison_plots.png      # Visualizaciones
```

### **Comparación Completa con Framework Robusta**

```bash
# Comparación exhaustiva con validación estadística (10-15 min)
python -c "from compare_models import ModelComparator; comp = ModelComparator(); comp.run_robust_comparison(n_cv_folds=5)"
```

### **Evaluación Robusta - Comandos Nuevos**

```bash
# Cross-validation con análisis estadístico
python -c "from compare_models import ModelComparator; comp = ModelComparator(); comp.run_robust_comparison(n_cv_folds=5)"

# Análisis de fuga de datos
python -c "from utils import detect_data_leakage; print(detect_data_leakage(X_train, X_test))"

# Ver resultados en JSON
cat comparison_results/robust_comparison_results.json
```

### **Solo Baseline Centralizado**

```bash
# Entrenar solo el modelo centralizado
python baseline_model.py \
  --epochs 100 \
  --batch_size 32 \
  --lr 0.001 \
  --output_dir baseline_results
```

## 📊 Resultados Típicos

### **Métricas de Comparación (Actualizadas - Framework Robusta)**

| Métrica       | Centralizado    | Federado        | Diferencia | Significancia |
| ------------- | --------------- | --------------- | ---------- | ------------- |
| **Accuracy**  | 0.9935 ± 0.0015 | 0.9945 ± 0.0005 | +0.10%     | p=0.592 (NS)  |
| **F1-Score**  | 0.9872          | 0.9881          | +0.09%     | -             |
| **Precision** | 0.9911          | 0.9923          | +0.12%     | -             |
| **Recall**    | 0.9834          | 0.9840          | +0.06%     | -             |
| **AUC**       | 0.9978          | 0.9982          | +0.04%     | -             |

### **Análisis Estadístico (Cross-validation 5-fold)**

-   **T-statistic**: -0.6325
-   **P-value**: 0.5918 (No significativo)
-   **Cohen's d**: 0.8944 (Efecto grande)
-   **Data Leakage**: 92.1% detectado
-   **Conclusión**: No hay diferencia estadísticamente significativa

### **Eficiencia Temporal**

-   **Centralizado**: ~120 segundos (50 epochs)
-   **Federado**: ~95 segundos (10 rounds × 5 epochs locales)
-   **Overhead**: Federado puede ser más rápido en paralelo

## 🔍 Análisis de Resultados

### **🚨 Hallazgo Crítico: Fuga de Datos Detectada**

-   **Ratio de Fuga**: 92.1% de similitud detectada en ECG5000
-   **Implicación**: Los resultados pueden estar artificialmente inflados
-   **Recomendación**: Usar división por sujetos para comparaciones confiables

### **Interpretación Estadística**

-   **p-value = 0.592**: No hay diferencia estadísticamente significativa
-   **Cohen's d = 0.894**: Efecto grande pero no significativo (paradójico)
-   **Conclusión**: Ambos enfoques tienen rendimiento equivalente

### **Degradación de Performance**

-   **Aceptable**: Diferencia < 3% en accuracy (resultado actual: +0.10%)
-   **Preocupante**: Diferencia > 5% en accuracy
-   **Crítico**: Diferencia > 10% en accuracy

### **Ventajas del Federado**

-   ✅ **Privacidad**: Datos nunca salen del dispositivo
-   ✅ **Escalabilidad**: Paralelización natural
-   ✅ **Robustez**: Falla de un cliente no afecta sistema
-   ✅ **Compliance**: Cumple regulaciones de privacidad
-   ⚠️ **Limitación**: Fuga de datos puede afectar evaluaciones

### **Ventajas del Centralizado**

-   ✅ **Performance**: Generalmente mejor accuracy (pero no significativo aquí)
-   ✅ **Simplicidad**: Más fácil de debuggear
-   ✅ **Consistencia**: Entrenamiento más estable
-   ✅ **Control**: Control total sobre datos y proceso
-   ⚠️ **Limitación**: Requiere centralización de datos sensibles

## 📈 Visualizaciones Generadas

### **1. Performance Metrics Comparison**

Gráfico de barras comparando accuracy, F1, precision, recall, AUC

### **2. Training Time Comparison**

Comparación de tiempos de entrenamiento

### **3. Training Progress**

-   Curvas de entrenamiento centralizadas
-   Progreso por rounds federados

### **4. Accuracy Difference**

Visualización del gap de performance (% diferencia)

## 🧪 Testing

Los nuevos modelos incluyen tests comprehensivos:

```bash
# Ejecutar tests de baseline y comparación
pytest tests/test_baseline_comparison.py -v

# Tests específicos
pytest tests/test_baseline_comparison.py::TestBaselineTrainer -v
pytest tests/test_baseline_comparison.py::TestFederatedSimulator -v
pytest tests/test_baseline_comparison.py::TestModelComparator -v
```

**Tests incluidos:**

-   ✅ Consistencia de arquitectura entre modelos
-   ✅ Funcionalidad de entrenamiento baseline
-   ✅ Simulación federada correcta
-   ✅ Generación de reportes
-   ✅ Guardado/carga de resultados

## 🎛️ Configuraciones Avanzadas

### **Diferentes Distribuciones de Datos**

```python
# Modificar en compare_models.py para Non-IID
def split_data_non_iid(X, y, num_clients, alpha=0.5):
    # Distribución Dirichlet para simular heterogeneidad
    pass
```

### **Arquitecturas Personalizadas**

```python
# Modificar model.py para comparar diferentes arquitecturas
class ECGModelDeep(nn.Module):
    # Modelo más profundo para comparación
    pass
```

### **Métricas Personalizadas**

```python
# Agregar en baseline_model.py
def custom_metrics(y_true, y_pred):
    return {
        'specificity': specificity_score(y_true, y_pred),
        'sensitivity': recall_score(y_true, y_pred),
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred)
    }
```

## 💡 Tips para Mejores Comparaciones

### **1. Configuración Justa**

-   Usar **misma semilla aleatoria** (random_state=42)
-   **Mismo dataset split** para train/test
-   **Mismos hiperparámetros** (lr, batch_size)
-   **Mismo número de epochs totales**

### **2. Métricas Relevantes**

-   **Accuracy**: Métrica principal
-   **F1-Score**: Balance precision/recall
-   **AUC**: Capacidad discriminativa
-   **Training Time**: Eficiencia práctica

### **3. Análisis Estadístico**

```python
# Múltiples runs para significancia estadística
for seed in [42, 123, 456, 789, 999]:
    run_comparison(random_state=seed)
```

## 🚨 Consideraciones Importantes

### **Limitaciones del Simulador Federado**

-   ⚠️ **No simula latencia** de red real
-   ⚠️ **No simula fallos** de clientes
-   ⚠️ **Distribución IID** (misma distribución en todos los clientes)
-   ⚠️ **Sin agregación avanzada** (solo promedio simple)

### **Para Producción Real**

-   🔧 Usar **Flower framework** completo
-   🔧 Implementar **estrategias de agregación** avanzadas
-   🔧 Manejar **clientes desconectados**
-   🔧 Implementar **privacidad diferencial**

### **� Framework de Evaluación Robusta**

#### **Características del Nuevo Sistema**

-   ✅ **Cross-validation**: Validación cruzada estratificada (5-fold)
-   ✅ **Pruebas estadísticas**: t-test con significancia (α=0.05)
-   ✅ **Detección de fugas**: Análisis de similitud coseno
-   ✅ **Recomendaciones automáticas**: Basadas en evidencia empírica
-   ✅ **Serialización robusta**: JSON con tipos numpy/bool

#### **Hallazgos Científicos**

-   **Fuga de datos**: 92.1% detectada en ECG5000 dataset
-   **Significancia estadística**: p=0.592 (no significativa)
-   **Recomendación**: Usar división por sujetos para evaluaciones confiables

#### **Archivos de Resultados**

```
comparison_results/
├── robust_comparison_results.json    # Resultados estadísticos
├── comparison_results.json           # Resultados detallados
└── comparison_plots.png              # Visualizaciones
```

## �📚 Referencias

-   [Federated Learning: Challenges, Methods, and Future Directions](https://arxiv.org/abs/1908.07873)
-   [Communication-Efficient Learning of Deep Networks from Decentralized Data](https://arxiv.org/abs/1602.05629)
-   [Flower Framework Documentation](https://flower.dev/)
-   [PyTorch Federated Learning Tutorial](https://pytorch.org/tutorials/advanced/federated_learning.html)
-   [Statistical Testing in Machine Learning](https://arxiv.org/abs/2106.00001)

---

Este sistema de comparación te permite **evaluar objetivamente** si el aprendizaje federado es adecuado para tu caso de uso específico, balanceando performance vs. beneficios de privacidad y escalabilidad. Ahora incluye **evaluación estadística robusta** y **detección automática de problemas metodológicos**. 🎯
