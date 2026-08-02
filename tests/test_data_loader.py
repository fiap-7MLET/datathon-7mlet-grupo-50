"""
Testes de `datathon.data_loader.BankMarketingPreprocessor` — o pipeline de limpeza do
dataset Bank Marketing.

Roda contra um CSV sintético minúsculo em vez do dataset real (41k linhas, não versionado
no repositório): o comportamento testado é a lógica de limpeza, não os números do Bank
Marketing em si. `monkeypatch.chdir` isola cada teste no seu próprio `tmp_path`, já que a
classe deriva os caminhos de `os.getcwd()`.
"""

import os

import pandas as pd
import pytest

from datathon.data_loader import BankMarketingPreprocessor

RAW_COLUMNS = [
    "age", "job", "marital", "education", "default", "housing", "loan",
    "contact", "month", "day_of_week", "duration", "campaign", "pdays",
    "previous", "poutcome", "emp.var.rate", "cons.price.idx", "cons.conf.idx",
    "euribor3m", "nr.employed", "y",
]


def _raw_row(**overrides):
    row = {
        "age": 35, "job": "admin.", "marital": "married", "education": "university.degree",
        "default": "no", "housing": "yes", "loan": "no", "contact": "cellular",
        "month": "may", "day_of_week": "mon", "duration": 120, "campaign": 1,
        "pdays": 999, "previous": 0, "poutcome": "nonexistent", "emp.var.rate": 1.1,
        "cons.price.idx": 93.2, "cons.conf.idx": -36.4, "euribor3m": 4.857,
        "nr.employed": 5191.0, "y": "no",
    }
    row.update(overrides)
    return row


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    """Um `cwd` isolado com `data/kaggle/bank-additional-full.csv` já populado."""
    monkeypatch.chdir(tmp_path)
    source_dir = tmp_path / "data" / "kaggle"
    source_dir.mkdir(parents=True)

    rows = [
        _raw_row(),
        _raw_row(pdays=999, previous=0),  # duplicata exata da primeira linha
        _raw_row(job="student", age=90, pdays=3, previous=2, campaign=50, poutcome="success"),
        _raw_row(job="retired", age=None, education=None, month="oct"),
    ]
    df = pd.DataFrame(rows, columns=RAW_COLUMNS)
    df.to_csv(source_dir / "bank-additional-full.csv", sep=";", index=False)
    return tmp_path


def test_paths_are_derived_from_the_current_working_directory(workspace):
    preprocessor = BankMarketingPreprocessor()

    assert preprocessor.source_path == os.path.join(
        str(workspace), "data", "kaggle", "bank-additional-full.csv"
    )
    assert preprocessor.output_path == os.path.join(
        str(workspace), "data", "processed", "bank_marketing_processed.parquet"
    )


def test_load_data_raises_a_clear_error_when_the_source_csv_is_missing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    preprocessor = BankMarketingPreprocessor()

    with pytest.raises(FileNotFoundError, match="Arquivo não encontrado"):
        preprocessor.load_data()


def test_load_data_reads_the_semicolon_separated_csv(workspace):
    preprocessor = BankMarketingPreprocessor().load_data()

    assert preprocessor.df_raw.shape[0] == 4
    assert "duration" in preprocessor.df_raw.columns


def test_compute_md5_is_deterministic_for_the_same_file_contents(workspace):
    preprocessor = BankMarketingPreprocessor()

    first = preprocessor._compute_md5(preprocessor.source_path)
    second = preprocessor._compute_md5(preprocessor.source_path)

    assert first == second
    assert len(first) == 32  # tamanho de um hex digest MD5


def test_clean_data_drops_duplicates_and_leakage_columns(workspace):
    preprocessor = BankMarketingPreprocessor().load_data().clean_data()
    clean = preprocessor.df_clean

    assert len(clean) == 3, "a linha duplicada deve ser removida"
    for leaky_col in ("duration", "pdays", "emp.var.rate", "nr.employed"):
        assert leaky_col not in clean.columns


def test_clean_data_derives_contact_flags_from_pdays_and_previous(workspace):
    clean = BankMarketingPreprocessor().load_data().clean_data().df_clean

    never_contacted = clean[clean["previous"] == 0]
    contacted = clean[clean["previous"] == 2]
    assert (never_contacted["contacted_before"] == 0).all()
    assert (contacted["contacted_before"] == 1).all()  # pdays=3 != 999
    assert (contacted["had_previous_contact"] == 1).all()


def test_clean_data_fills_missing_numeric_and_categorical_values(workspace):
    clean = BankMarketingPreprocessor().load_data().clean_data().df_clean

    assert clean["age"].isnull().sum() == 0
    assert clean["job"].isnull().sum() == 0
    assert "unknown" in clean["education"].values


def test_clean_data_discretizes_age_into_buckets(workspace):
    clean = BankMarketingPreprocessor().load_data().clean_data().df_clean

    assert set(clean["age_group"].cat.categories) == {"<30", "30-40", "40-50", "50-60", "60+"}
    assert clean.loc[clean["age"] == 90, "age_group"].iloc[0] == "60+"


def test_run_writes_a_parquet_file_and_returns_the_cleaned_dataframe(workspace):
    result = BankMarketingPreprocessor().run()

    output_path = workspace / "data" / "processed" / "bank_marketing_processed.parquet"
    assert output_path.exists()
    assert pd.read_parquet(output_path).shape == result.shape
