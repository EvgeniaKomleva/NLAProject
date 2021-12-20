# NLAProject

## Randomized CP Decomposition
***************************
The CP decomposition is an equation-free, data-driven tensor decomposition that is capable of providing 
accurate reconstructions of multi-mode data arising in signal processing, computer vision, neuroscience and elsewhere. 

The ctensor package includes the following tensor decomposition routines:
* CP using the alternating least squares method (ALS)
* CP using the block coordinate descent method (BCD)
* Both methods can be used either using a deterministic or a fast randomized algorithm.
* The package is build ontop of the [scikit-tensor package](https://github.com/mnick/scikit-tensor)

The following figure illustrates the performance of the randomized accelerated CP decomposition routines.
![toy](https://raw.githubusercontent.com/Benli11/data/master/img/tensor_speed.png)


Installation
************

`git clone https://github.com/EvgeniaKomleva/NLAProject.git`

`python demo.py `

