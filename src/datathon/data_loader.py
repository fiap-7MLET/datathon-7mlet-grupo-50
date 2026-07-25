import pandas as pd
import numpy as np
import logging
import hashlib
import warnings
import os

warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class BankMarketingPreprocessor:
    """
    Pipeline de pré-processamento do dataset Bank Marketing.
 
    Responsabilidades:
        - Carregar o CSV bruto
        - Logar metadados do arquivo de origem (hash MD5 e tamanho)
        - Aplicar limpeza e feature engineering
        - Salvar o resultado em Parquet
    """
    SOURCE_FILENAME = "bank-additional-full.csv"
    OUTPUT_FILENAME = "bank_marketing_processed.parquet"
 
    def __init__(self):
        self.working_dir   = os.getcwd()
        self.source_path   = os.path.join(self.working_dir, "data", "kaggle", self.SOURCE_FILENAME)
        self.output_path   = os.path.join(self.working_dir, "data", "processed", self.OUTPUT_FILENAME)
        self.df_raw        = None
        self.df_processed  = None

    def _compute_md5(self, file_path: str) -> str:
        """Calcula o hash MD5 do arquivo para garantir reprodutibilidade."""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
 
    def _log_source_metadata(self) -> None:
        """Loga versão (tamanho em bytes) e hash MD5 do arquivo de origem."""
        size_bytes = os.path.getsize(self.source_path)
        md5        = self._compute_md5(self.source_path)
        logger.info("Metadados do arquivo de origem:")
        logger.info(f"Arquivo : {self.source_path}")
        logger.info(f"Tamanho : {size_bytes:,} bytes")
        logger.info(f"MD5     : {md5} \n")

    def load_data(self) -> "BankMarketingPreprocessor":
        if not os.path.exists(self.source_path):
            raise FileNotFoundError(f"Arquivo não encontrado: {self.source_path}")
    
        self._log_source_metadata()
        self.df_raw = pd.read_csv(self.source_path, sep=";")
        logger.info(f"Dados carregados. Shape: {self.df_raw.shape}")
        return self

    def clean_data(self) -> "BankMarketingPreprocessor":
        logger.info("Iniciando limpeza de dados...")
        df = self.df_raw.copy()

        # Dropando duplicados
        logger.info(f"Duplicados antes de limpeza: {df.duplicated().sum()}")
        df_clean = df.drop_duplicates()
        duplicates_clean = df_clean.duplicated().sum()
        pct_dup_clean = df_clean.duplicated().sum() / len(df_clean) * 100

        logger.info(f"Duplicados depois de limpeza: {duplicates_clean} ({pct_dup_clean:.2f}%)")

        # Correção de pdays e previous como flag
        df_clean['contacted_before'] = (df_clean['pdays'] != 999).astype(int)
        df_clean['had_previous_contact'] = (df_clean['previous'] > 0).astype(int)
        logger.info(f"Flags criadas. Novo shape do DataFrame: {df_clean.shape}")

        # Dropando colunas de during e pdays
        cols_to_drop = [
            "duration", # leakage
            "pdays", #c convertida em flag
            "emp.var.rate", # multicolinearidade
            "nr.employed" # multicolinearidade
            ]
        
        df_clean = df_clean.drop(columns=cols_to_drop)
        logger.info(f"Colunas {cols_to_drop} removidas. Novo shape do DataFrame: {df_clean.shape}")

        # Tratamento de outliers na coluna campaign
        df_clean['campaign'] = df_clean['campaign'].clip(upper=df_clean['campaign'].quantile(0.99))
        logger.info(f"Outliers na coluna 'campaign' tratados. Valor máximo: {df_clean['campaign'].max()}.")

        # Discretização da coluna age
        df_clean['age_group'] = pd.cut(df_clean['age'], 
                            bins=[0, 30, 40, 50, 60, 100],
                            labels=['<30','30-40','40-50','50-60','60+'])
        logger.info(f"Coluna 'age' discretizada. Novo shape do DataFrame: {df_clean.shape}")

        # Tratamento de NaN
        logger.info(f'Número de nulos antes do tratamento: \n{df.isnull().sum()}')

        cat_cols = df_clean.select_dtypes(include="object").columns 
        num_cols = df_clean.select_dtypes(include=np.number).columns

        df_clean[num_cols] = df_clean[num_cols].fillna(df_clean[num_cols].median())
        df_clean[cat_cols] = df_clean[cat_cols].fillna("unknown")

        logger.info(f"NaN tratados. \nNúmero de nulos: \n{df_clean.isnull().sum()}")

        self.df_clean = df_clean
        logger.info("Limpeza de dados concluída.")
        logger.info(f"Shape final do DataFrame: {self.df_clean.shape}")
        return self
    
    def save_data(self) -> "BankMarketingPreprocessor":
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        self.df_clean.to_parquet(self.output_path, index=False)
        logger.info(f"Arquivo salvo em: {self.output_path}")
        return self

    def run(self) -> pd.DataFrame:
        """Executa o pipeline completo e retorna o DataFrame processado."""
        return (
            self
            .load_data()
            .clean_data()
            .save_data()
            .df_clean
        )
 
if __name__ == "__main__":
    preprocessor = BankMarketingPreprocessor()
    df = preprocessor.run()
    logger.info(f"Pipeline concluído. Shape final: {df.shape}")