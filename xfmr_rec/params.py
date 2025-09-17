# paths
MOVIELENS_1M_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
DATA_DIR = "data"
TENSORBOARD_DIR = "lightning_logs"
MLFLOW_DIR = "mlruns"

# data
ITEMS_PARQUET = "data/ml-1m/items.parquet"
USERS_PARQUET = "data/ml-1m/users.parquet"
EVENTS_PARQUET = "data/ml-1m/events.parquet"
TARGET_COL = "event_value"
ITEM_IDX_COL = "item_rn"
ITEM_ID_COL = "item_id"
ITEM_TEXT_COL = "item_text"
USER_IDX_COL = "user_rn"
USER_ID_COL = "user_id"
USER_TEXT_COL = "user_text"

# model
BATCH_SIZE = 2**5
PADDING_IDX = 0
PRETRAINED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
METRIC = {"name": "val/RetrievalNormalizedDCG", "mode": "max"}
TOP_K = 20

# serving
EMBEDDER_PATH = "embedder"
ENCODER_PATH = "encoder"
ITEMS_TABLE_NAME = "items"
LANCE_DB_PATH = "lance_db"
MODEL_NAME = "xfmr_rec"
PROCESSORS_JSON = "processors.json"
SEQ_MODEL_NAME = "xfmr_seq_rec"
SEQ_EMBEDDED_MODEL_NAME = "xfmr_seq_embedded_rec"
TRANSFORMER_PATH = "transformer"
USERS_TABLE_NAME = "users"
