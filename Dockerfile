FROM pytorch/pytorch:1.4-cuda10.1-cudnn7-devel

RUN conda install -c conda-forge \
        tensorboardx \
        toolz \
        matplotlib \
        hdf5storage \
        numpy \
        scikit-learn \
        pyyaml \
        attrdict \
        tensorboard

# nvidia-docker hooks
LABEL com.nvidia.volumes.needed="nvidia_driver"
ENV PATH /usr/local/nvidia/bin:${PATH}
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64:${LD_LIBRARY_PATH}
