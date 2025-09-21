# paths
MOVIELENS_1M_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
DATA_DIR = "data"
TENSORBOARD_DIR = "lightning_logs"

# data
ITEMS_PARQUET = "data/ml-1m/items.parquet"
USERS_PARQUET = "data/ml-1m/users.parquet"
EVENTS_PARQUET = "data/ml-1m/events.parquet"

# model
PRETRAINED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
METRIC = {"name": "val/retrieval_normalized_dcg", "mode": "max"}
TOP_K = 20

# serving
EMBEDDER_PATH = "embedder"
ENCODER_PATH = "encoder"
ITEMS_TABLE_NAME = "items"
LANCE_DB_PATH = "lance_db"
MODEL_NAME = "xfmr_rec"
TRANSFORMER_PATH = "transformer"
USERS_TABLE_NAME = "users"
