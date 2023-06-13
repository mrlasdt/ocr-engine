export CUDA_VISIBLE_DEVICES=1 
# export PATH=/usr/local/cuda-11.6/bin${PATH:+:${PATH}}
# export LD_LIBRARY_PATH=/usr/local/cuda-11.6/lib64\ {LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
# export CUDA_HOME=/usr/local/cuda-11.6
# export PATH=/usr/local/cuda-11.6/bin:$PATH
# export CPATH=/usr/local/cuda-11.6/include:$CPATH
# export LIBRARY_PATH=/usr/local/cuda-11.6/lib64:$LIBRARY_PATH
# export LD_LIBRARY_PATH=/usr/local/cuda-11.6/lib64:/usr/local/cuda-11.6/extras/CUPTI/lib64:$LD_LIBRARY_PATH
python test/test_deskew_dir.py