# TP — Automatisation d’un entraînement avec MLflow (Local + CI)

## Prérequis
- Windows + PowerShell, VS Code
- Python 3.11+, Git, Docker **facultatif** (non requis ici)
- Port **5000** libre

## Installation & Serveur MLflow (local)
```powershell
# À la racine du projet
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt

# Démarrer le serveur MLflow (garde ce terminal ouvert)
mlflow server --backend-store-uri sqlite:///mlflow.db --artifacts-destination ./artifacts --host 127.0.0.1 --port 5000
```

Dans **un nouveau terminal** PowerShell :
```powershell
.\.venv\Scripts\Activate.ps1
$env:MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
```

## Lancer un entraînement
```powershell
python .\src	train.py --experiment-name "Iris-Local" --registered-model-name "IrisClassifier" `
  --C 1.0 --max-iter 200 --seed 42 --min-accuracy 0.8
```

## Voir l’UI & la Registry
- Ouvre le navigateur : http://127.0.0.1:5000
- Tu verras les **runs**, **artefacts**, **metrics**, et le **Registered Model** `IrisClassifier` avec ses **versions**.

## CI/CD (GitHub Actions)
- Pousse ce dossier dans un repo GitHub.
- Le workflow `.github/workflows/train.yml` :
  - démarre un serveur MLflow **local au runner** (SQLite + artifacts),
  - exécute `pytest`,
  - entraîne et enregistre la version suivante de `IrisClassifier`,
  - imprime les top runs.

Déclencheurs : `push` sur `main` ou **Run workflow** manuellement.

## Commandes utiles
```powershell
# Tests unitaires (serveur MLflow lancé)
pytest -q

# Autres runs pour comparer
python .\src	rain.py --experiment-name "Iris-Local" --registered-model-name "IrisClassifier" --C 0.5 --max-iter 300 --seed 123 --min-accuracy 0.8
python .\src	rain.py --experiment-name "Iris-Local" --registered-model-name "IrisClassifier" --C 2.0 --max-iter 150 --seed 7 --min-accuracy 0.8
```

## Nettoyage
```powershell
deactivate
Remove-Item -Recurse -Force .\.venv
Remove-Item -Recurse -Force .rtifacts, .\mlruns
Remove-Item -Force .\mlflow.db
```

---

**Bonnes pratiques appliquées**
- Versions **pinnées** dans `requirements.txt` (reproductibilité).
- **Seed** fixé, **signature** & **input_example** loggués.
- **Seuil de qualité** – entraîne échoue si accuracy < 0.8 (utile en CI).
- **Registry** MLflow utilisée pour versionner le modèle `IrisClassifier`.
