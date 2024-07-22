import os

# Base directory for the project
# BASE_DIR = '/nfs/workspace/coarse-to-fine-dataset'
BASE_DIR = '/nfs'

# Paths to external data directories
DATA_DIR = os.path.join(BASE_DIR, 'datasets')

# Logs directory
LOGS_DIR = os.path.join(BASE_DIR, 'logs')

# Charades STA
CHARADES_STA = "charades_sta"
CHARADES_DIR = os.path.join(DATA_DIR, 'Charades_v1')
CHARADES_VIDEOS_DIR = os.path.join(CHARADES_DIR, 'videos')
CHARADES_ANNOTATIONS_DIR = os.path.join(CHARADES_DIR, 'annotations')
CHARADES_ANNOTATIONS_TRAIN = os.path.join(CHARADES_ANNOTATIONS_DIR, 'charades_sta_train.txt')
CHARADES_ANNOTATIONS_TEST = os.path.join(CHARADES_ANNOTATIONS_DIR, 'charades_sta_test.txt')

# Save model directory
SAVE_MODEL_DIR = os.path.join(BASE_DIR, 'saved_models')
COARSE_GRAINED_MODELS_DIR = os.path.join(SAVE_MODEL_DIR, 'coarse_grained')
FINE_GRAINED_MODELS_DIR = os.path.join(SAVE_MODEL_DIR, 'fine_grained')
