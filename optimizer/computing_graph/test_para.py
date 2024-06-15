import sys

from DNN_model_tf.vgg_tf import VGG16_tf
from slurm_util import get_slurm_available_nodes, run_srun_command, SLURM_RUN_CONF

sys.path.append("../../")

import warnings

warnings.filterwarnings("ignore")


nodes = get_slurm_available_nodes()
output_comp = run_srun_command(nodes, SLURM_RUN_CONF.COMPUTING_COST, 'VGG16_tf')
