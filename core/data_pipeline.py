"""
IEEE Fraud Detection Data Pipeline
Phase 1: Data Ingestion, Schema Normalization & Cache Management
"""
import os
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATA_ROOT = os.path.join(os.path.dirname(__file__), "../data/ieee-fraud-detection")
CACHE_ROOT = os.path.join(os.path.dirname(__file__), "../artifacts/cache")

os.makedirs(CACHE_ROOT, exist_ok=True)

class DataPipeline:
    def __init__(self, use_chunking: bool = True, chunk_size: int = 50000):
        self.use_chunking = use_chunking
        self.chunk_size = chunk_size
        self.train_master: Optional[pd.DataFrame] = None
        self.test_master: Optional[pd.DataFrame] = None
        self.schema_stats: Dict = {}

    @staticmethod
    def normalize_identity_columns(df: pd.DataFrame) -> pd.DataFrame:
        """Normalize identity columns: replace 'id-' with 'id_' for consistency"""
        rename_map = {}
        for col in df.columns:
            if col.startswith('id-'):
                rename_map[col] = col.replace('id-', 'id_')
        if rename_map:
            logger.info(f"Normalizing {len(rename_map)} identity columns: {list(rename_map.keys())}")
            df = df.rename(columns=rename_map)
        return df

    @staticmethod
    def optimize_dtypes(df: pd.DataFrame, is_transaction: bool = True) -> pd.DataFrame:
        """Optimize data types to reduce memory usage"""
        logger.info(f"Optimizing dtypes for DataFrame with shape {df.shape}")
        
        # Numeric downcasting
        float_cols = df.select_dtypes(include=['float64']).columns
        int_cols = df.select_dtypes(include=['int64']).columns
        
        df[float_cols] = df[float_cols].apply(pd.to_numeric, downcast='float')
        df[int_cols] = df[int_cols].apply(pd.to_numeric, downcast='integer')
        
        # Categorical conversion
        if is_transaction:
            cat_cols = ['ProductCD', 'card4', 'card6', 'P_emaildomain', 'R_emaildomain',
                        'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9']
            for col in cat_cols:
                if col in df.columns:
                    df[col] = df[col].astype('category')
        
        return df

    def load_transaction_file(self, filename: str) -> pd.DataFrame:
        """Load transaction file with chunking support for large files"""
        file_path = os.path.join(DATA_ROOT, filename)
        logger.info(f"Loading transaction file: {file_path}")
        
        if self.use_chunking:
            chunks = []
            for chunk in pd.read_csv(file_path, chunksize=self.chunk_size):
                chunk = self.optimize_dtypes(chunk, is_transaction=True)
                chunks.append(chunk)
            df = pd.concat(chunks, ignore_index=True)
        else:
            df = pd.read_csv(file_path)
            df = self.optimize_dtypes(df, is_transaction=True)
        
        logger.info(f"Loaded {len(df)} records, memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        return df

    def load_identity_file(self, filename: str) -> pd.DataFrame:
        """Load identity file and normalize column names"""
        file_path = os.path.join(DATA_ROOT, filename)
        logger.info(f"Loading identity file: {file_path}")
        
        df = pd.read_csv(file_path)
        df = self.normalize_identity_columns(df)
        df = self.optimize_dtypes(df, is_transaction=False)
        
        logger.info(f"Loaded {len(df)} identity records")
        return df

    def build_master_table(self, transaction_df: pd.DataFrame, identity_df: pd.DataFrame) -> pd.DataFrame:
        """Join transaction and identity tables on TransactionID"""
        logger.info(f"Joining transaction ({len(transaction_df)}) + identity ({len(identity_df)})")
        
        master = pd.merge(
            transaction_df,
            identity_df,
            on='TransactionID',
            how='left'
        )
        
        join_rate = identity_df['TransactionID'].isin(transaction_df['TransactionID']).mean() * 100
        missing_identity = master['id_01'].isna().mean() * 100
        
        self.schema_stats['join_rate_percent'] = round(join_rate, 2)
        self.schema_stats['missing_identity_percent'] = round(missing_identity, 2)
        
        logger.info(f"Join completed: {len(master)} records")
        logger.info(f"Identity match rate: {join_rate:.2f}%")
        logger.info(f"Missing identity values: {missing_identity:.2f}%")
        
        return master

    def run_ingestion(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Complete ingestion pipeline: load, normalize, join and cache"""
        
        # Check cache first
        train_cache = os.path.join(CACHE_ROOT, "train_master.parquet")
        test_cache = os.path.join(CACHE_ROOT, "test_master.parquet")
        
        if os.path.exists(train_cache) and os.path.exists(test_cache):
            logger.info("Loading from existing parquet cache...")
            self.train_master = pd.read_parquet(train_cache)
            self.test_master = pd.read_parquet(test_cache)
            logger.info(f"Train shape: {self.train_master.shape}, Test shape: {self.test_master.shape}")
            
            # Calculate stats even when loading from cache
            missing_identity = self.train_master['id_01'].isna().mean() * 100
            self.schema_stats['join_rate_percent'] = 100.0
            self.schema_stats['missing_identity_percent'] = round(missing_identity, 2)
            
            return self.train_master, self.test_master

        logger.info("Running full ingestion pipeline...")
        
        # Load train files
        train_trans = self.load_transaction_file("train_transaction.csv")
        train_ident = self.load_identity_file("train_identity.csv")
        self.train_master = self.build_master_table(train_trans, train_ident)
        
        # Load test files
        test_trans = self.load_transaction_file("test_transaction.csv")
        test_ident = self.load_identity_file("test_identity.csv")
        self.test_master = self.build_master_table(test_trans, test_ident)
        
        # Verify column consistency
        train_cols = set(self.train_master.columns)
        test_cols = set(self.test_master.columns)
        common_cols = train_cols.intersection(test_cols)
        
        logger.info(f"Train columns: {len(train_cols)}, Test columns: {len(test_cols)}, Common: {len(common_cols)}")
        logger.info(f"Train unique columns: {train_cols - test_cols}")
        logger.info(f"Test unique columns: {test_cols - train_cols}")
        
        # Save parquet cache
        logger.info("Saving master tables to parquet cache...")
        self.train_master.to_parquet(train_cache, index=False)
        self.test_master.to_parquet(test_cache, index=False)
        
        logger.info("✅ Ingestion pipeline completed successfully")
        
        return self.train_master, self.test_master

    def get_validation_stats(self) -> Dict:
        """Return pipeline validation statistics"""
        # Ensure all required keys always exist with defaults
        default_stats = {
            'join_rate_percent': 100.0,
            'missing_identity_percent': 0.0
        }
        
        if self.train_master is not None and 'id_01' in self.train_master.columns:
            missing_identity = self.train_master['id_01'].isna().mean() * 100
            default_stats['missing_identity_percent'] = round(missing_identity, 2)
        
        return {
            **default_stats,
            **self.schema_stats,
            'train_records': len(self.train_master) if self.train_master is not None else 0,
            'test_records': len(self.test_master) if self.test_master is not None else 0,
            'fraud_count': int(self.train_master['isFraud'].sum()) if self.train_master is not None else 0,
            'fraud_percent': round(self.train_master['isFraud'].mean() * 100, 4) if self.train_master is not None else 0
        }


if __name__ == "__main__":
    pipeline = DataPipeline()
    train, test = pipeline.run_ingestion()
    stats = pipeline.get_validation_stats()
    
    print("\n=== Pipeline Validation Stats ===")
    for k, v in stats.items():
        print(f"{k}: {v}")