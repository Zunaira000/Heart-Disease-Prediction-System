# ❤️ Heart Disease Prediction System

Streamlit web app that predicts heart disease from 4 inputs (no label encoding needed).

| Inputs | Meaning |
|---|---|
| `ca` | Major vessels coloured by fluoroscopy (0-3) |
| `oldpeak` | ST depression induced by exercise |
| `thalach` | Maximum heart rate achieved |
| `exang` | Exercise-induced angina (0 = No, 1 = Yes) |

**Output:** heart disease present (1) / absent (0). Model: Logistic Regression, ~83% test accuracy.

## Run locally
```bash
python -m venv venv
venv\Scripts\activate        # Windows  (macOS/Linux: source venv/bin/activate)
pip install -r requirements.txt
python train_model.py        # optional: re-creates heart_model.pkl
streamlit run app.py
```

Data: UCI Heart Disease (Cleveland). Educational use only - not medical advice.
