from pathlib import Path

# ============================================================
# Experiment root
# ============================================================
ROOT = Path("x3_lds_exp")
DATA_DIR = ROOT / "data"
MASK_DIR = ROOT / "subset_masks"
MODEL_DIR = ROOT / "models"
QUERY_DIR = ROOT / "queries"
ATTR_DIR = ROOT / "attribution"
LDS_DIR = ROOT / "lds"
LOG_DIR = ROOT / "logs"

# ============================================================
# Base data
# ============================================================
DATA_SEED = 67
N_TRAIN = 5000
BASE_CSV = DATA_DIR / f"{DATA_SEED}_{N_TRAIN}.csv"

# ============================================================
# Training
# ============================================================
FAMILIES = ("prompted", "unprompted")

# Two full/base models.
TRAIN_SEED = 67
BASE_MODEL_SEED = TRAIN_SEED

# LDS: 3 independent mask banks x 64 masks.
LDS_SEEDS = (0, 1, 2)
SUBSETS_PER_SEED = 64
SUBSET_FRACTION = 0.25
SUBSET_SIZE = int(N_TRAIN * SUBSET_FRACTION)  # 1250

# A "subset slot" is one mask. By default each slot trains BOTH family
# variants on the exact same 1250 points. Thus:
#   3 x 64 = 192 subset masks / jobs
#   192 x 2 = 384 final subset checkpoints
# Set this to ("prompted",) if you truly want only 192 trained models total.
SUBSET_TRAIN_FAMILIES = ("prompted", "unprompted")

# Aligned training settings.
BATCH_SIZE = 256
EPOCHS = 200
PEAK_LR = 1e-4
WARMUP_RATIO = 0.10
WEIGHT_DECAY = 1e-4
ADAM_B1 = 0.9
ADAM_B2 = 0.999
ADAM_EPS = 1e-8
GRAD_CLIP = 1.0
EMA_DECAY = 0.999

T = 1000
BASE_CH = 64
TIME_DIM = 128

# Full/base models need trajectory checkpoints for Traj-TracIn.
BASE_SAVE_EVERY_EPOCHS = 4   # 50 checkpoints across 200 epochs.
# LDS subset models only need their final checkpoint.
SUBSET_SAVE_EVERY_EPOCHS = 200

CUDA_IDS = (0, 1, 2, 3)

# ============================================================
# Query bank: 7 initial noises -> 16 queries
# ============================================================
INITIAL_SEEDS = (0, 1, 2, 3, 4, 5, 6)
PROMPTED_INITIAL_SEEDS = (0, 1, 2)
UNPROMPTED_INITIAL_SEEDS = (3, 4, 5, 6)
RANDOM_PROMPTS_PER_PROMPTED_INITIAL = 4  # 3*4 + 4 = 16
QUERY_PROMPT_SEED = 20260818

DDIM_STEPS = 1000
TRAJ_SNAPSHOTS = 50

# Query-gradient diagnostic cache. Exact Traj scoring still uses the
# full gradients in memory; this 4096-D cache is saved for reuse/analysis.
QUERY_GRAD_CACHE_DIM = 4096

# ============================================================
# Attribution
# ============================================================
TRACIN_PARAM_SOURCE = "raw"
TRACIN_TRAIN_MC = 8
TRACIN_SCORE_BATCH_SIZE = 512
TRACIN_FAST_JVP = True
TRACIN_USE_LR_WEIGHTS = True

DAS_PARAM_SOURCE = "ema"
DAS_PROJ_DIM = 4096
DAS_FEATURE_BATCH_SIZE = 64
DAS_FAST_VMAP = True
DAS_TIMESTEPS = (0,)
DAS_NUM_MC = 1
DAS_NORMALIZE_PROJECTED_GRADS = True
DAS_USE_SM_DENOMINATOR = False
DAS_LAMBDAS = (0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0)

# ============================================================
# LDS response metrics
# ============================================================
# Both are collected so you can evaluate either target without retraining.
LDS_METRICS = ("simple_loss", "traj_ref")
LDS_EVAL_TIMESTEPS = (0, 111, 222, 333, 444, 555, 666, 777, 888, 999)
LDS_EVAL_MC = 4
