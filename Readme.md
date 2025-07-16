* to create conda environment with requirements
* and the name of environment can be changed in **requirements.yml**
```shell
conda env create -f requirements.yml
```
* to update conda environment
```shell
conda env install -f requirements.yml
```

* to install pip requirements
```shell
pip install -r requirements.txt
```
* 
* to export pip requirements
```shell
pipreqs .
```