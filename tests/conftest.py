"""
Fixtures compartilhadas.

O URI de tracking do MLflow é estado global do processo: um teste que aponta o MLflow para
um store temporário contaminaria todos os testes seguintes (por exemplo, `policy_store`
carregaria a política sintética em vez de cair no arquivo local). A fixture abaixo restaura
o valor original ao fim de cada teste, tornando a ordem de execução irrelevante.
"""

import pytest


@pytest.fixture(autouse=True)
def restore_mlflow_tracking_uri():
    import mlflow

    original = mlflow.get_tracking_uri()
    yield
    mlflow.set_tracking_uri(original)
