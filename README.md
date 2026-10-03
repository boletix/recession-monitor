# recession-monitor

Panel de riesgo de recesión en EE. UU.: https://boletix.github.io/recession-monitor/

Un score compuesto de 0 a 100 con 13 indicadores de mercados, crédito, empleo y consumo, más los umbrales de capitulación de Hartnett/Carpatos. Se actualiza de lunes a viernes tras el cierre de Nueva York con GitHub Actions.

## Estructura

| Archivo | Qué es |
|---|---|
| `scripts/update_data.py` | Descarga los datos, puntúa y escribe `docs/dashboard.json` y `docs/history.json` |
| `scripts/backfill_history.py` | Reconstruye el histórico desde el git de la v3 (se ejecutó una vez) |
| `docs/index.html` | El panel. Solo lee JSON: no hay que tocarlo para actualizar datos |
| `docs/manual.json` | Datos que no se pueden descargar (Bull & Bear de BofA). Se editan a mano |
| `scripts/legacy/` | Scripts de la v1–v2 que generaban PNG; ya no se ejecutan |

## Reglas del score

- Un dato que falta **no puntúa**: el score se renormaliza sobre los pesos disponibles y publica su cobertura.
- Si la cobertura baja del 60% o el JSON no es estricto, el workflow falla y la web conserva el último dato bueno.
- Con `FRED_API_KEY` (secreto del repo) el crédito HY usa el diferencial ICE BofA real; sin ella, un proxy LQD−HYG.

## Ejecutar en local

```bash
pip install -r requirements.txt
FRED_API_KEY=... python scripts/update_data.py
python -m http.server --directory docs
```
