# Kairos

Wearable de estimación fisiológica e intervención local.

**→ [agesfranciscoteran.github.io/kairos-tech](https://agesfranciscoteran.github.io/kairos-tech/)**

Kairos estima estados de activación fisiológica a partir de señales de muñeca —PPG/HRV,
actividad electrodérmica, temperatura y movimiento— y ejecuta intervenciones locales de
regulación. Cuando la activación no cede, escala a una persona de confianza que el usuario
configuró previamente.

**Kairos no detecta emociones.** No infiere estados clínicos ni etiqueta lo que alguien siente.
Opera sobre correlatos fisiológicos de activación, con clasificación probabilística y umbrales
conservadores por diseño.

Este repositorio reúne la documentación técnica y la evidencia verificable del proyecto,
etiquetada por lo que realmente es.

## Contenido

| Carpeta | Qué contiene | Estado |
|---|---|---|
| [`docs/`](docs/) | Documento de arquitectura técnica v1.1: capas del sistema, umbrales de intervención, frontera de privacidad y riesgos conocidos | Especificación |
| [`wesad-feasibility/`](wesad-feasibility/) | Análisis de separabilidad sobre el dataset público WESAD | Listo para ejecutar |
| [`imu-demo/`](imu-demo/) | Prototipo de gating por movimiento con los sensores del teléfono | Funcional |
| [`render/`](render/) | Render 3D interactivo del wearable, con vista de despiece | Diseño |
| [`mockups/`](mockups/) | Flujos de interfaz: monitoreo, detección, intervención y escalamiento | Diseño |

## El problema técnico central

La activación autonómica **no es específica**. Una caída de HRV con aumento de EDA aparece en una
crisis de ansiedad, pero también al discutir con alguien, tomar café o pasar frío. A eso se suma
un problema de tasas base: los eventos reales son raros y el monitoreo es continuo, así que
incluso una especificidad alta produce más falsas alarmas que detecciones correctas.

Los dos errores son costosos en direcciones opuestas: una falsa alarma quema la confianza del
usuario y del contacto de emergencia; una omisión falla en lo único para lo que el sistema existe.

De ahí las dos decisiones que estructuran la arquitectura:

- **Gating por movimiento.** Si el IMU detecta actividad física, la fisiología no se interpreta
  como estrés. Descarta el confusor más frecuente antes de cualquier inferencia.
  Implementado en [`imu-demo/`](imu-demo/).
- **Línea base personal.** La normalidad fisiológica es idiosincrática, así que cada usuario se
  compara contra su propio reposo y no contra una norma poblacional.

## Análisis de viabilidad

[`wesad-feasibility/`](wesad-feasibility/) mide cuánta separabilidad se alcanza entre activación
por estrés y reposo usando **WESAD** (Schmidt et al., 2018), un dataset público con las mismas
modalidades de muñeca que usa Kairos. Es una primera medición de la incertidumbre central del
proyecto sobre datos ajenos, antes de recolectar los propios.

Compara dos ejes —fisiología sola frente a fisiología con gating por IMU, y normalización
poblacional frente a línea base personal— con validación **Leave-One-Subject-Out**, de modo que
las métricas reflejan generalización a personas nuevas.

No reporta exactitud. Reporta, a una sensibilidad fija, la **especificidad**: si detectamos el 80%
de los eventos reales, qué fracción del reposo marcamos por error.

```bash
cd wesad-feasibility
pip install -r requirements.txt

# Descargar WESAD y descomprimir a WESAD/S2/S2.pkl, WESAD/S3/S3.pkl, ...
# https://ubicomp.eti.uni-siegen.de/home/datasets/icmi18/

python kairos_wesad_separability.py --data-dir ./WESAD --out-dir ./out
```

Genera curvas ROC, distribuciones por característica con tamaño de efecto, y las métricas en
`out/results/metrics.json`.

## Estado del proyecto

Kairos está en **etapa de diseño**. Para que no quede implícito:

- No hay producto desplegado.
- No existe un dataset propio; la validación disponible proviene de datos públicos de terceros.
- La aprobación ética no se ha gestionado. Es requisito previo a cualquier recolección con
  participantes.
- Los umbrales de intervención documentados son valores de diseño, no calibrados empíricamente.
- La plataforma de sensado actual transmite señal cruda al teléfono, donde ocurre todo el
  procesamiento. La frontera de privacidad vive en el teléfono, no en el wearable. La sección 8
  del documento de arquitectura detalla la ruta de hardware para cerrar esa brecha.

La apuesta del proyecto es construir el primer dataset fisiológico **etiquetado con contexto
verificado**: cada evento asociado a lo que la persona efectivamente estaba haciendo. Ese
emparejamiento señal-contexto no existe en los datasets públicos y es lo que hace utilizable
el dato.

## Referencia

Schmidt, P., Reiss, A., Duerichen, R., Marberger, C., & Van Laerhoven, K. (2018).
*Introducing WESAD, a Multimodal Dataset for Wearable Stress and Affect Detection.* ICMI.
