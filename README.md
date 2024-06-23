# \[Group EconTechVets\] Overview
**Group members:**
Rasmus Darling Wegener, hgs252
Frederik Nellerod Nordentoft, hnw220

This repository contains:  
1. Inaugural project incl. a README, a notebook file and a py-file. We model an exchange economy.
2. Data project incl. a README, a notebook file, a py-file and plots/graphs from the project. We fetch data from **Danish Statistics, table IDRAKT01 and FOLK1A** on Danish football players.
3. Model project incl. a README, a notebook file and a py-file. We model an IS-LM model.
3. Exam project incl. a README, a notebook file and 3 py-files (1 for each Problem in the exam).

# Inaugural project
The purpose of this project is to analyze an exchange economy with two consumers, A and B, and two goods, x1 and x2. 
The **results** of the project can be seen from running [inauguralproject.ipynb](inauguralproject.ipynb), which depends on [inauguralproject.py](inauguralproject.py).

# Data analysis project
Our project is titled **Football in Denmark: Where are we playing?** and is about locating in which regions the danes are members of football clubs. The **results** of the project can be seen from running [dataproject.ipynb](dataproject.ipynb), which depends on [dataproject.py](dataproject.py).

We apply the **following datasets**:
1. DstApi('IDRAKT01') (*Danish Statistics, table IDRAKT01*) 
1. DstApi('FOLK1A') (*Danish Statistics, table FOLK1A*)

# Model analysis project
Our project is titled **Modelproject, IS-LM** and considers an IS-LM model describing the interaction between the goods and money market on the basis of the IS- and LM-curve. The **results** of the project can be seen from running [modelproject.ipynb](modelproject.ipynb), which depends on [modelproject.py](modelproject.py).

# Dependencies
Apart from a standard Anaconda Python 3 installation, the project requires no packages not used during the course lectures nor exercises. However, we have here outlined a few used that may need installations:
``pip install pandas``
``pip install matplotlib.pyplot``
``pip install matplotlib.ticker``
``pip install numpy``
``pip install IPython``
``pip install git+https://github.com/alemartinello/dstapi``
``pip install geopandas``
``pip install ipywidgets``