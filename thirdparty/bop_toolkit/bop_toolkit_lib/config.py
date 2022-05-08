# Author: Tomas Hodan (hodantom@cmp.felk.cvut.cz)
# Center for Machine Perception, Czech Technical University in Prague

"""Configuration of the BOP Toolkit."""

import os


######## Basic ########

# Folder with the BOP datasets.
if 'BOP_PATH' in os.environ:
  datasets_path = os.environ['BOP_PATH']
else:
  # (nmerrill67) NOTE: that this assumes being run in suo_slam root
  datasets_path = r'data/bop_datasets/'

# Folder with pose results to be evaluated.
results_path = r''

# Folder for the calculated pose errors and performance scores.
eval_path = r'./results'

######## Extended ########

# Folder for outputs (e.g. visualizations).
output_path = r'./viz'

# For offscreen C++ rendering: Path to the build folder of bop_renderer (github.com/thodan/bop_renderer).
bop_renderer_path = r'/path/to/bop_renderer/build'

# Executable of the MeshLab server.
meshlab_server_path = r'/path/to/meshlabserver.exe'
