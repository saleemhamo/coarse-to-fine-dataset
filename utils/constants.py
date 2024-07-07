import os

# Paths to external data directories
MOUNTED_CLAIM_DIRECTORY = os.getenv('MOUNTED_CLAIM_DIRECTORY', '/Users/saleemhamo/Desktop/MSc Project/project')
DATA_DIR = os.path.join(MOUNTED_CLAIM_DIRECTORY, 'datasets')

# Charades STA
CHARADES_STA = "charades_sta"
CHARADES_DIR = os.path.join(DATA_DIR, 'Charades_v1')
CHARADES_VIDEOS_DIR = os.path.join(CHARADES_DIR, 'videos')
CHARADES_ANNOTATIONS_DIR = os.path.join(CHARADES_DIR, 'annotations')
CHARADES_ANNOTATIONS_TRAIN = os.path.join(CHARADES_ANNOTATIONS_DIR, 'charades_sta_train.txt')
CHARADES_ANNOTATIONS_TEST = os.path.join(CHARADES_ANNOTATIONS_DIR, 'charades_sta_test.txt')
