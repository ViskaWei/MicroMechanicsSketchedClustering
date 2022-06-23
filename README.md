# MicroMechanicsUMAPClustering
If using this code for research or industrial purposes, please cite:

A UMAP-based Clustering Method for Multi-scale Damage Analysis of Laminates. Applied Mathematical Modelling





# Before Running the Code

The 'data_Kmatrix.csv' in 'data' folder contains each row contains information of one element. 

If you want to run your own data, put your own .csv file in 'data' folder. Please set the first row (i.e. 'Number', 'k11'...,'vol') in your .csv exactly same as in 'data_Kmatrix.csv', otherwise the code won't run.

'Number'column represent the element number. 'k11~k66' represent 36 elements in K matrix defined in Eq.8 in the papaer. The last column 'vol' stands for the volume (area) of this element. 

Code will be public once the paper is published.


1. For those who are familiar with basic computer science knowledge, follow the Instruction section below, download the code and run it in Linux. 

2. For those who are not familiar with Linux system, the code can be run directly on google colab: https://colab.research.google.com/drive/1rKWBU73XQN4fKByDSpMiiwynw9U7IGEb?usp=sharing




# Instruction
### Install and setup virtual environment
```sudo apt-get install python3-venv ```

```mkdir -p ~/venv```

```cd ~/venv```

```python3 -m venv mmuc```

```source ~/venv/mmuc/bin/activate```

### Upgrade setuptools 
```pip3 install --upgrade setuptools```

### Install MMUC
```python3 setup.py install```

### Run test
```mmuc-test ```

### Clean 
```python3 setup.py clean --all ```
