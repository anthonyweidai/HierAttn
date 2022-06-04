# Installation

The code was tested with [Anaconda](https://www.anaconda.com/download) Python 3.8, CUDA 11.3, and [PyTorch]((http://pytorch.org/)) 1.10.1.
After installing Anaconda:

0. [Optional but highly recommended] create a new conda environment. 

    ~~~
    conda create -n hier python=3.8
    ~~~
    And activate the environment.
    
    ~~~
    conda activate hier
    ~~~

1. Install PyTorch:

    ~~~
    conda install pytorch=1.10.1 torchvision=0.11.2 cudatoolkit=11.3 -c pytorch
    ~~~

3. Clone this repo:

    ~~~
    git clone https://github.com/anthonyweidai/HierAttn.git
    ~~~

4. Install the requirements

    ~~~
    cd $HierAttn_ROOT
    pip install -r requirements.txt
    ~~~
    

