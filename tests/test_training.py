import os

# Les tests supposent que le serveur MLflow local tourne (http://127.0.0.1:5000)
os.environ.setdefault("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")

def test_dummy_threshold():
    assert 1 + 1 == 2
