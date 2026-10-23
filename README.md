# recession-monitor

Panel de riesgo de recesión con indicadores adelantados: https://boletix.github.io/recession-monitor/

v5 (oct-2026): tres scores de 0 a 100. EE. UU. tiene 22 indicadores en cuatro pilares: crédito y tipos, economía adelantada, confirmación y mercados. La zona euro y el bloque global tienen 6 cada uno. Incluye las probabilidades de los modelos publicados (curva de la Fed de NY, prima de riesgo de los bonos de la Fed, Michaillat-Saez, Chauvet-Piger) y un backtest mensual desde 1988 con las recesiones del NBER. La pestaña «Método y estudios» explica de qué estudio sale cada indicador. El histórico de la v4 está en `docs/history_v4.json`. Se actualiza de lunes a viernes tras el cierre de Nueva York con GitHub Actions.

## Estructura

| Archivo | Qué es |
|---|---|
| `scripts/update_data.py` | Descarga los datos, puntúa y escribe `docs/dashboard.json` y `docs/history.json` |
| `scripts/backfill_history.py` | Reconstruye el histórico desde el git de la v3 (se ejecutó una vez) |
| `docs/index.html` | El panel. Solo lee JSON: no hay que tocarlo para actualizar datos |
| `docs/backtest.json` | Score mensual de EE. UU. desde 1988, pilares, recesiones, avisos y falsas alarmas |
| `docs/manual.json` | Datos que no se pueden descargar (almacén de gas de la UE, Bull & Bear de BofA). Se editan a mano |
| `scripts/legacy/` | Scripts de la v1–v2 que generaban PNG; ya no se ejecutan |

## Reglas del score

- Un dato que falta **no puntúa**: el score se renormaliza sobre los pesos disponibles y publica su cobertura.
- Si la cobertura baja del 60% o el JSON no es estricto, el workflow falla y la web conserva el último dato bueno.
- Con `FRED_API_KEY` (secreto del repo) FRED se lee por la API; sin ella, por `fredgraph.csv`. Eurostat, BCE y la Fed no piden clave.

## Ejecutar en local

```bash
pip install -r requirements.txt
FRED_API_KEY=... python scripts/update_data.py
python -m http.server --directory docs
```
