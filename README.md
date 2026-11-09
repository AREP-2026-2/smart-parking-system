# Asignación dinámica de parqueaderos con predicción híbrida

Prototipo del proyecto de Arquitectura Empresarial (AREP, Escuela Colombiana de Ingeniería Julio Garavito).
Implementa el alcance de la Sección 7 del paper: el motor predictivo híbrido (Módulo 4), el motor de
asignación reducido (Módulo 5) y una versión reducida de los módulos 2 y 3.

> **Los datos actuales son simulados.** `prediction-service/parking_ml/simulator.py` genera series con supuestos
> plausibles de horario y capacidad. Las métricas describen a ese simulador: **no validan las hipótesis sobre la
> ECI**. Para validarlas hay que entrenar con los datos de campo (ver [Usar datos reales](#usar-datos-reales)).

## Ejecutar el demo

Requisito: Docker Desktop encendido.

```bash
docker compose up --build
```

Abrir <http://127.0.0.1:8088>. Los modelos entrenados y los datos ya están en el repositorio, así que no hace
falta GPU para el demo. La primera construcción descarga PyTorch y puede tardar varios minutos.

Guion sugerido para la presentación:

1. Cargar los escenarios **A, B, C y D** y mostrar qué espacio se asigna, el desglose de costo y qué rama respondió.
2. En **Pronóstico por horizonte**, comparar lo previsto con lo observado y ver los pesos de atención.
3. En **Evaluación del prototipo**, revisar las hipótesis H1 y H2 y ejecutar la simulación de H3.
4. En **Operación**, simular la caída de una rama e inyectar deriva: la rama afectada pasa a la línea base.
5. Ejecutar `docker compose stop prediction` y asignar otro espacio: el sistema sigue respondiendo con la
   línea base local. Se recupera con `docker compose start prediction`.

## Arquitectura

```text
Navegador ──> nginx (dashboard estático)
                ├── /api ──> allocation-service (Spring Boot)     Módulos 3 y 5
                │              ├─ DigitalTwin / OccupancyHistory   estado e histórico
                │              ├─ CostFunction / AllocationService  Ecuación (2)
                │              ├─ AllocationExperiment              Hipótesis 3
                │              └─ ForecastGateway ──HTTP──┐         línea base si no hay respuesta
                └── /ml ───> prediction-service (FastAPI) <┘        Módulo 4
                               ├─ router: h ≤ τ → LSTM, si no → Transformer
                               ├─ BranchHealth: caídas y error móvil
                               └─ artifacts/: modelos versionados y registro
```

| Decisión (Sección 4.4) | Dónde está |
|---|---|
| Inferencia desacoplada tras una API única | `POST /v1/forecasts`; Spring solo conoce el puerto `PredictionClient` |
| Modelos versionados | `artifacts/registry.json`: versión, hash de datos y MAE por rama y horizonte |
| Degradación controlada | `BranchHealth` y `ForecastGateway`: rama caída, error sobre el límite o servicio caído |
| Presupuesto de latencia | Inferencia sobre series agregadas; cada respuesta informa su latencia |
| Observabilidad | `ObservabilityService` compara lo previsto con lo observado y lo reporta al enrutador |
| Alcance del modelo | Un modelo global por rama y horizonte, con la zona como entrada |

### Decisiones de implementación que conviene conocer

- **Umbral τ.** El servicio arranca con τ = 1 h, la política inicial de la Tabla 2 (variable `ROUTER_TAU`). El valor
  calibrado con validación se guarda en `artifacts/registry.json` y es el que usa el reporte de H2. El dashboard
  permite cambiar el τ activo en vivo.
- **Horizontes intermedios.** Hay modelos para 1, 3, 5, 8, 12 y 24 h. Otros valores se interpolan linealmente
  entre los dos horizontes vecinos; por debajo de 1 h se interpola desde la ocupación observada.
- **Saturación a corto plazo.** Con h ≤ 1 h, S es el máximo entre lo previsto y la ocupación actual.
- **Accesibilidad.** Un espacio accesible nunca se asigna a quien no lo necesita. Quien lo necesita recibe uno si queda alguno.
- **Pesos.** Son de juicio experto (`application.yml`) y no se ajustaron con datos, como indica la Sección 7.1.
- **Espacios individuales.** Los datos solo traen conteos por zona; qué espacio concreto está ocupado se fija de
  forma reproducible. El plano y sus distancias son ilustrativos.

## Protocolo de evaluación

- Partición cronológica 70/15/15 por instante objetivo; ventana de 24 h, como en Vakrinou et al.
- Cada rama y horizonte se entrena con 3 semillas; se publica la de menor MAE en validación.
- τ se elige solo con validación entre 0 (todo Transformer) y 24 (todo LSTM). Prueba se usa únicamente para reportar.
- MAPE, MAE, MSE y R² por horizonte. El MAPE omite ocupaciones reales menores al 5%.
- Importancia por permutación por variable en `artifacts/report.json`. SHAP no está implementado.
- **H3:** en las horas pico del periodo de prueba se comparan cuatro políticas con los mismos vehículos: vecino más
  cercano, costo sin saturación, costo con ocupación actual y costo con predicción. Una asignación es *adecuada* si
  respeta tamaño y accesibilidad y la zona no supera el 90% al llegar. El tiempo incluye recorrido y una cola por
  zona con servicio de 15 s por vehículo; ambos valores son supuestos del simulador.

## Usar datos reales

El CSV debe tener `timestamp, zone, occupied, capacity` y, de forma opcional, `inflow, outflow, class_change,
no_class_day, rain`. Acepta resolución de 15 minutos o una hora y rechaza huecos, duplicados y ocupaciones imposibles.

```bash
cd prediction-service
py -3.11 -m venv .venv
.venv/Scripts/pip install torch --index-url https://download.pytorch.org/whl/cu128
.venv/Scripts/pip install -r requirements-dev.txt
.venv/Scripts/python -m parking_ml.train --data ../data/mis_datos.csv --out artifacts --dataset-kind field
```

Antes de entrenar, borre `data/zones.json` si no corresponde a sus zonas: el entrenamiento solo crea ese archivo si
no existe. El plano admite cualquier número de zonas, y con `field` desaparece el aviso de datos simulados.

## Regenerar y probar

```bash
cd prediction-service
.venv/Scripts/python -m parking_ml.simulator --out ../data --weeks 18 --seed 7
.venv/Scripts/python -m parking_ml.train --data ../data/occupancy_15min.csv --out artifacts
.venv/Scripts/python -m pytest

cd ../allocation-service
mvn test
```

## Fuera de alcance

Clasificación vehicular, navegación interna con recálculo, aplicación móvil y módulo de administración completo
(Sección 7.1). Tampoco hay autenticación: los endpoints administrativos solo se publican en `127.0.0.1`, por lo
que este despliegue no debe exponerse a una red.
