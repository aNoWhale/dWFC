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

* to install pip requirements
```shell
pip install -r requirements.txt
```

* to export pip requirements
```shell
pipreqs . --force
```

* to install jax-cuda12
```shell
pip install -U "jax[cuda12]"
```

* to install julia environment
```shell
julia Pkg.instantiate(julia_env)
```