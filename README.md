# kairos-tech

Evidencia técnica de **Kairos** — wearable de estimación fisiológica e intervención local.

Kairos está en etapa de diseño. Este repositorio reúne la evidencia verificable del proyecto,
etiquetada por lo que realmente es: lo medido, lo especificado y lo que todavía no existe.

**Portada del proyecto:** abre `index.html` o la URL de GitHub Pages del repositorio.

## Contenido

| Carpeta | Qué es | Estado |
|---|---|---|
| `docs/` | Documento de arquitectura técnica v1.1 | Especificación |
| `wesad-feasibility/` | Análisis de separabilidad sobre el dataset público WESAD | Listo para ejecutar |
| `imu-demo/` | Prototipo de gating por movimiento con sensores reales del teléfono | Funciona |
| `render/` | Render 3D interactivo del wearable | Diseño |
| `mockups/` | Flujos de interfaz: monitoreo, detección, intervención, escalamiento | Diseño |

## Publicar en GitHub Pages

1. Sube el repositorio a GitHub como **público**.
2. Ve a **Settings → Pages**.
3. En *Source*, elige **Deploy from a branch**; rama `main`, carpeta `/ (root)`. Guarda.
4. Espera un par de minutos. El sitio queda en:

```
https://TU-USUARIO.github.io/kairos-tech/
```

El prototipo de gating necesita HTTPS para acceder a los sensores del teléfono; GitHub Pages
lo provee automáticamente. Ábrelo desde el celular, no desde la computadora.

## Antes de publicar

Falta un archivo: copia tu render 3D a `render/index.html` (ver `render/LEEME.txt`).
El nombre debe ser exactamente `index.html`, sin espacios ni paréntesis.

## Ejecutar el análisis de viabilidad

```bash
cd wesad-feasibility
pip install -r requirements.txt
# Descarga WESAD y descomprime a WESAD/S2/S2.pkl, WESAD/S3/S3.pkl, ...
python kairos_wesad_separability.py --data-dir ./WESAD --out-dir ./out
```
