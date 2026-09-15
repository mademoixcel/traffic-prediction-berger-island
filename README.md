# Traffic Congestion Prediction App

This project predicts the congestion ratio for the Berger–Lagos Island route using date, time and weather condition.

## Files
- `traffic_data.csv` — the supplied dataset
- `train_model.py` — cleans the data, engineers features, trains and evaluates the model, then saves it
- `app.py` — Streamlit prediction interface
- `requirements.txt` — Python packages
- `traffic_model.joblib` — created after training
- `model_metadata.json` — created after training

## Run locally

Open a terminal in this folder and run:

```bash
python -m venv .venv
```

Windows:
```bash
.venv\Scripts\activate
```

macOS/Linux:
```bash
source .venv/bin/activate
```

Install packages:
```bash
pip install -r requirements.txt
```

Train the model:
```bash
python train_model.py
```

Start the app:
```bash
streamlit run app.py
```

## Important modelling choice
The app predicts `congestion_ratio` from information that can realistically be known when making a prediction: route, weather, date and time. `travel_time_s`, `traffic_delay_s`, and `traffic_length_m` are intentionally excluded from the model inputs because using current traffic measurements to predict a congestion ratio derived from current traffic would create data leakage.

## Congestion levels used by the interface
- Low: ratio < 1.15
- Moderate: 1.15 to < 1.40
- High: 1.40 to < 1.80
- Severe: >= 1.80

These labels are presentation thresholds. The machine-learning target itself is the numeric `congestion_ratio`.
