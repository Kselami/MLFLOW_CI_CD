# TP — Automatisation d’un entraînement avec MLflow

## Objectifs pédagogiques
- **Versioning & suivi** : tracer paramètres, métriques, artefacts et **versions de modèles** avec MLflow.
- **Expérimentations** : structurer des **expériences** (noms, paramètres, seeds), comparer les runs.
- **CI/CD d’entraînement** : pipeline GitHub Actions qui **entraîne** et **registre** un modèle.
- **Bonnes pratiques** : reproductibilité (seed, requirements), signature, seuils de qualité, promotion de versions.

---

## ÉNONCÉ (sans réponses)
1) **Environnement & serveur MLflow**
   - Crée un environnement Python 3.11, installe les dépendances.
   - Lance un **serveur MLflow** local avec backend **SQLite** et artefacts sur le disque.
   - Configure `MLFLOW_TRACKING_URI` pour pointer sur ce serveur.

2) **Entraînement avec suivi MLflow**
   - Implémente `src/train.py` :
     - charge **Iris** (scikit-learn), split train/test avec un **seed**.
     - entraîne une **LogisticRegression** paramétrable (`C`, `max_iter`, `seed`).
     - loggue **paramètres**, **métriques** (accuracy, precision, recall), **artefacts** (matrice de confusion).
     - enregistre le modèle avec **signature** et **input_example**.
     - **registre** le modèle dans la **Model Registry** MLflow sous un nom donné.

3) **Expérimentations**
   - Exécute plusieurs runs en changeant `C` et `max_iter` sous un **même nom d’expérience**.
   - Identifie le **meilleur run** (accuracy max).

4) **CI/CD d’entraînement**
   - Ajoute un workflow `.github/workflows/train.yml` qui :
     - installe Python et dépendances,
     - démarre un **serveur MLflow** local,
     - exécute les **tests** puis un **entraînement**,
     - affiche les **meilleures métriques** dans les logs du job.

5) **Bonnes pratiques**
   - Implémente un **seuil de qualité** (ex. accuracy ≥ 0.8) qui **échoue** le job si non atteint.
   - Assure la **reproductibilité** : requirements pinés, seed, signature d’entrée, pas de “latest”.
   - Prépare une **promotion manuelle** (Staging/Production) via la Model Registry.

---

## CORRIGÉ (extraits & commandes)

### 1) Serveur MLflow (local, SQLite + artifacts)
```powershell
# À la racine du projet
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt

# Lancer le serveur MLflow (laisse ce terminal ouvert)
mlflow server --backend-store-uri sqlite:///mlflow.db --artifacts-destination ./artifacts --host 127.0.0.1 --port 5000
```

Dans un **nouveau** terminal PowerShell, active l’environnement et :
```powershell
$env:MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
```

### 2) Lancer un entraînement
```powershell
# Exemple : expérience "Iris-Local", modèle "IrisClassifier"
python .\src	rain.py --experiment-name "Iris-Local" --registered-model-name "IrisClassifier" `
  --C 1.0 --max-iter 200 --seed 42 --min-accuracy 0.8
```

### 3) Explorer dans l’UI
Ouvre http://127.0.0.1:5000 et compare les **runs** (tri par accuracy).

### 4) CI/CD (GitHub Actions)
Le workflow démarre un serveur MLflow **local au runner**, puis lance **tests** et **entraînement**.
Déclenche-le par un push sur `main` ou via “Run workflow”.

### 5) Bonnes pratiques (incluses)
- **Seed** fixé, **requirements** pinés, **signature** & **input_example**, **seuils** de qualité.
- Model Registry alimentée (versions incrémentées). Promotion de version à faire via l’UI.

---

**Nettoyage local**
```powershell
deactivate
Remove-Item -Recurse -Force .\.venv  # optionnel
Remove-Item -Recurse -Force .\mlruns, .rtifacts  # si tu as lancé sans serveur
Remove-Item -Force .\mlflow.db
```
