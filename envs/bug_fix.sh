# lcuda找不到
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$CONDA_PREFIX/x86_64-conda-linux-gnu/lib:$LD_LIBRARY_PATH
export LIBRARY_PATH=$CONDA_PREFIX/lib:$CONDA_PREFIX/x86_64-conda-linux-gnu/lib:$LIBRARY_PATH
# mpi4py
conda install mpi4py