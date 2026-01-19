# Instruction
## 1.here, we have three possible entrance:
* src/fem/topologyOptimization.py using 3D 1x1x1 RVE model
* src/fem/topologyOptimization33.py using 3D 3x3x3 RVE model
* src/fem/twoDTopologyOptimization33.py using 2D 3x3 RVE model

## 2.RVE models are defined at data:
* data/EVG for 1x1x1 RVE model
* data/Kernels for 3x3x3 and 3x3 RVE model, parameters can be changed in SIgmaInterpreter_**.py

## 3.msh is used to define design space, and can be generated in 1.

## 4.stp are used for 1x1x1 RVE models, for constructing stl from RVE model
## 5. src/dynamicGenerator/stlConstructor.py is used to obtain stl
### while this process is not integated into the main procedure, you can run it manully.
### parameters should be defined correctly.

## 6. CPU is used explicitly.
### GPU is supported in theropy, if you have enough GPU memory, maybe you can give it a try.

## 7. conda, conda-forge, and pip are employed:
* to create conda environment with requirements
* and the name of environment can be changed in **environment.yaml**
```shell
conda env create -f environment.yml
```
* to update conda environment
```shell
conda install -f environment.yml
```
* To export conda environment
```shell
conda env export --from-history > environment.yml
```

<!-- * to install pip requirements
```shell
pip install -r requirements.txt
``` -->

* to export pip requirements
```shell
pipreqs . --force
```

* to install jax-cuda12
```shell
pip install -U "jax[cuda12]"
```

<!-- * to install julia environment
```shell
julia Pkg.instantiate(julia_env)
``` -->
## 8. Use AI tool for understanding.
#### For me, English Code Comments are not intuition. So some comments are in Chinese. And some codes are vibe coding, but human supervised. 
#### Using AI tool to help you understand the algorithm will be a good idea.
#### When I started this project, I was not good at JAX coding, and when I can code well, the project is to heavy to recode.
#### Maybe implementing your own dWFC instead of using my code will be a good choice.