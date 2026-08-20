from exp_config import *
print("ROOT =", ROOT)
print("DATA_SEED =", DATA_SEED)
print("TRAIN_SEED =", TRAIN_SEED)
print("N_TRAIN =", N_TRAIN)
print("SUBSET_FRACTION =", SUBSET_FRACTION)
print("SUBSET_SIZE =", SUBSET_SIZE)
print("LDS mask count =", len(LDS_SEEDS) * SUBSETS_PER_SEED)
print("families =", FAMILIES)
print("DAS_PROJ_DIM =", DAS_PROJ_DIM)
print("TRACIN_SCORE_BATCH_SIZE =", TRACIN_SCORE_BATCH_SIZE)
print("DAS_FEATURE_BATCH_SIZE =", DAS_FEATURE_BATCH_SIZE)

assert DATA_SEED == 67
assert TRAIN_SEED == 67
assert N_TRAIN == 50000
assert SUBSET_SIZE == 12500
assert len(LDS_SEEDS) * SUBSETS_PER_SEED == 192
print("[OK] 50k configuration verified")

print("TRACIN_PARAM_SOURCES =", TRACIN_PARAM_SOURCES)
print("DAS lambdas =", DAS_LAMBDAS)
assert TRACIN_PARAM_SOURCES == ("raw", "ema")
assert max(DAS_LAMBDAS) == 100000.0
