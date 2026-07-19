# Kairos — Análisis de viabilidad sobre datos públicos (WESAD)

Primera medición directa de la **incertidumbre técnica central de Kairos** —la
especificidad— usando un dataset público, **antes** de recolectar datos propios.

## La pregunta

Kairos debe distinguir una activación fisiológica **genuina** (estrés sostenido) de una
activación **benigna** (café, discusión, emoción positiva), usando solo señales de
muñeca: PPG/HRV, EDA, temperatura e IMU. La duda no es si el estrés altera la fisiología
—claro que sí— sino si esa alteración es **separable** con una **tasa de falsas alarmas**
lo bastante baja para que el sistema se siga usando.

Este repo mide esa separabilidad corriendo el pipeline de extracción de características de
Kairos sobre **WESAD** (Schmidt et al., 2018), que trae exactamente las mismas modalidades.

## Por qué WESAD y por qué la muñeca

WESAD incluye dos dispositivos; usamos **solo el de muñeca (Empatica E4)** porque su
form factor coincide con Kairos:

| Señal E4 (muñeca) | Frecuencia | Uso en Kairos |
|---|---|---|
| BVP | 64 Hz | HRV (análogo al PPG de Kairos) |
| EDA | 4 Hz  | nivel tónico (SCL) + respuestas fásicas (SCR) |
| TEMP | 4 Hz | temperatura periférica |
| ACC | 32 Hz | movimiento → **gating por IMU** (descarta el confusor de ejercicio) |

Etiquetas WESAD: comparamos **baseline** (reposo) vs **stress**.

## Las dos comparaciones que importan

1. **Solo fisiología** vs **fisiología + IMU** — ¿el gating por movimiento mejora la separación?
2. **Normalización global** vs **baseline personal por sujeto** — la hipótesis de Kairos es
   que la "normalidad" fisiológica es idiosincrática, así que normalizar contra el reposo
   de **cada usuario** debería separar mejor que una norma poblacional. La normalización
   personal usa **solo las ventanas de reposo** de cada sujeto como referencia (no toca las
   de estrés), replicando la línea base personal del producto.

Se evalúa con **Leave-One-Subject-Out (LOSO)**: se entrena con N−1 sujetos y se prueba en
el que quedó fuera, de modo que las métricas reflejan generalización a personas nuevas.

## Métrica principal

No reportamos "accuracy". Reportamos, a una **sensibilidad fija (80%)**, la
**especificidad** = 1 − tasa de falsas alarmas. Traduce directo la pregunta de Kairos:
*si detectamos el 80% del estrés real, ¿qué fracción del reposo marcamos por error?*

## Cómo correrlo

```bash
# 1. Descargar WESAD (~2.1 GB) y descomprimir a esta estructura:
#    WESAD/S2/S2.pkl, WESAD/S3/S3.pkl, ...
#    https://ubicomp.eti.uni-siegen.de/home/datasets/icmi18/

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Correr
python kairos_wesad_separability.py --data-dir ./WESAD --out-dir ./out
```

## Salidas (`out/`)

- `figures/fig_distributions.png` — distribuciones baseline vs stress por feature, crudo y
  normalizado por baseline personal, con tamaño de efecto (Cohen's d).
- `figures/fig_roc.png` — curvas ROC de las 4 condiciones (LOSO).
- `figures/fig_specificity.png` — especificidad @ 80% sensibilidad por condición.
- `results/feature_table.csv` — features por ventana (para inspección/reproducibilidad).
- `results/metrics.json` — todas las métricas.

## Alcance y honestidad

- Este repo **no trae resultados incluidos**: los números salen de correrlo sobre WESAD
  real. Es un laboratorio para **medir** la incertidumbre, no una afirmación de desempeño.
- WESAD es estrés **inducido en laboratorio** (TSST) en adultos; **no** es el escenario
  final de Kairos (uso libre, otra población). Sirve para poner un piso realista y validar
  el pipeline, no para prometer el desempeño en producción.
- Estos son datos públicos ajenos; Kairos construirá su propio dataset etiquetado con
  contexto verificado (paso posterior, con la aprobación ética correspondiente).

## Referencia

Schmidt, P., Reiss, A., Duerichen, R., Marberger, C., & Van Laerhoven, K. (2018).
*Introducing WESAD, a Multimodal Dataset for Wearable Stress and Affect Detection.* ICMI.
