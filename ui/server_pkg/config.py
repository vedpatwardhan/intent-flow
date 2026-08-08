import os
import sys

# Global state variables
LOGS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "logs", "training", "latent-flow")
)

active_camera = "world_center"
encoder_processing_enabled = True
combostoc_noise = {"torso": 0.0, "arm": 0.0, "hand": 0.0, "vision": 0.0}
attack_active = False

colab_url = None
for i, arg in enumerate(sys.argv):
    if arg == "--colab-url" and i + 1 < len(sys.argv):
        colab_url = sys.argv[i + 1]
colab_url = colab_url or os.environ.get("COLAB_URL")

cached_checkpoints = []
click_x = None
click_y = None
click_type = None
text_prompt = "right hand to the red cube"
text_modifier = None
ui_annotations = {}

colab_is_processing = False
is_training_active = False
needs_colab_processing = True
last_colab_query_time = 0.0

from collections import deque

frame_history = deque(maxlen=5)
frame_all_views = {}

cached_dino_attn = None
cached_clip_sim = None
cached_sam_mask = None
cached_motion_field = None
cached_task_isolated_features = None
