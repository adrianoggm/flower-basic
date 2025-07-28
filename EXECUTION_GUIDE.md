# Guía de Ejecución - Federated Fog Demo

## ¡Configuración Completada! 

Tu entorno está completamente configurado y listo para ejecutar la demo de aprendizaje federado con fog computing.

## Pasos para Ejecutar la Demo

### Opción 1: Usando Scripts .bat (Windows - Recomendado)

Ejecuta los siguientes archivos .bat en **orden secuencial**, cada uno en una ventana de terminal separada:

```
1️⃣ run_server.bat        - Servidor Flower central
2️⃣ run_fog_broker.bat    - Broker fog que agrega updates locales  
3️⃣ run_fog_client.bat    - Cliente Flower en el nodo fog
4️⃣ run_client.bat        - Cliente local (ejecutar múltiples veces)
5️⃣ run_debug.bat         - Monitor MQTT (opcional)
```

**Orden de ejecución:**
1. Haz doble clic en `run_server.bat` 
2. Espera hasta ver "Flower server running"
3. Haz doble clic en `run_fog_broker.bat`
4. Espera hasta ver "MQTT Broker connected"
5. Haz doble clic en `run_fog_client.bat`
6. Espera hasta ver "Connected to Flower server"
7. Haz doble clic en `run_client.bat` (puedes ejecutar múltiples instancias)
8. (Opcional) Haz doble clic en `run_debug.bat` para monitorear tráfico MQTT

### Opción 2: Línea de Comandos Manual

Si prefieres usar la línea de comandos:

**Terminal 1 - Servidor Central:**
```bash
.venv\Scripts\python.exe server.py
```

**Terminal 2 - Fog Broker:**
```bash
.venv\Scripts\python.exe broker_fog.py
```

**Terminal 3 - Fog Client:**
```bash
.venv\Scripts\python.exe fog_flower_client.py
```

**Terminal 4+ - Clientes Locales:**
```bash
.venv\Scripts\python.exe client.py
```

**Terminal Opcional - Debug:**
```bash
.venv\Scripts\python.exe debug.py
```

## Qué Esperar Durante la Ejecución

### 1. Servidor Central (`server.py`)
```
[INFO] Flower server running on port 8080
[MQTT] Connected to test.mosquitto.org
[SERVER] Ready to aggregate...
```

### 2. Fog Broker (`broker_fog.py`)
```
[MQTT] Connected to test.mosquitto.org (rc=0)
[BROKER] Waiting for client updates...
[BROKER] Region 'region_0' has 3/3 updates, computing partial...
```

### 3. Fog Client (`fog_flower_client.py`)
```
[FOG CLIENT] Connected to Flower server at localhost:8080
[FOG CLIENT] Received partial for region=region_0
[FOG CLIENT] Sending partial aggregate to central server...
```

### 4. Clientes Locales (`client.py`)
```
[CLIENT] Training locally on ECG5000 data...
[CLIENT] Publishing update to fog broker...
[CLIENT] Waiting for global model...
[CLIENT] Received new global model, starting next round...
```

### 5. Debug Monitor (`debug.py`)
```
[DEBUG] fl/updates → {"region": "region_0", "client_id": "client_1", ...}
[DEBUG] fl/partial → {"region": "region_0", "partial_weights": {...}}
[DEBUG] fl/global_model → {"global_weights": {...}, "round": 1}
```

## Flujo de Entrenamiento

1. **Ronda 1**: Los clientes entrenan localmente con datos ECG5000
2. **Agregación Fog**: El broker fog agrega K=3 updates por región
3. **Agregación Central**: El servidor Flower agrega todas las regiones
4. **Distribución**: El modelo global se distribuye via MQTT
5. **Ronda 2+**: Se repite el proceso con el nuevo modelo global

## Métricas y Resultados

El sistema mostrará:
- **Precisión local** de cada cliente
- **Pérdida de entrenamiento** por ronda
- **Tiempo de agregación** en fog y central
- **Número de clientes** participantes por ronda

Típicamente verás convergencia en **3-5 rondas** con mejora progresiva de la precisión.

## Troubleshooting

### ❌ Error de conexión al servidor Flower
- Verificar que `server.py` esté ejecutándose
- Comprobar que el puerto 8080 no esté ocupado: `netstat -ano | findstr :8080`

### ❌ Error de conexión MQTT
- Verificar conectividad a internet
- Probar con broker local: cambiar `test.mosquitto.org` por `localhost`

### ❌ Clientes no participan
- Verificar que K≤número de clientes activos
- Esperar a que se acumulen suficientes updates (K=3 por defecto)

### ❌ Modelo no converge
- Aumentar número de épocas locales en `client.py`
- Verificar que hay suficientes clientes (mínimo 2)
- Revisar distribución de datos entre clientes

## Personalización

### Cambiar Número de Agregación Fog (K)
En `broker_fog.py`, línea 20:
```python
K = 3  # Cambiar a número deseado
```

### Cambiar Broker MQTT
En todos los archivos, cambiar:
```python
MQTT_BROKER = "test.mosquitto.org"  # Cambiar a tu broker
```

### Modificar Modelo
Editar `model.py` para cambiar arquitectura del CNN:
```python
class ECGModel(nn.Module):
    # Modificar capas aquí
```

## Archivos de Log

Los logs se muestran en las consolas de cada componente. Para logs persistentes, modifica los scripts para redirigir salida:

```bash
.venv\Scripts\python.exe server.py > server.log 2>&1
```

## Parar la Demo

1. Presiona `Ctrl+C` en cada terminal
2. O cierra las ventanas de los scripts .bat
3. Los archivos .bat tienen `pause` al final para mantener la ventana abierta

---

¡La demo está lista para ejecutarse! Consulta el README.md para más detalles sobre la arquitectura.
