# Data analysis project

Our project is titled **Football in Denmark: Where are we playing?** and is about locating in which regions the danes are members of football clubs. Furthermore, we choose to look at the distribution of football players across genders. We believe to find that the share of women footballers is increasing. Furthermore, the project takes into account the differences in population sizes across regions, as it is a factor that can influence the number of football players. Therefore we merge the data of football players with population data.

The **results** of the project can be seen from running [dataproject.ipynb](dataproject.ipynb). We will refer to the dataproject.py file to see further calculations and functions used in the project.

We apply the **following datasets**:

1. DstApi('IDRAKT01') (*Danish Statistics, table IDRAKT01*) 
1. DstApi('FOLK1A') (*Danish Statistics, table FOLK1A*)

**Dependencies:** Apart from a standard Anaconda Python 3 installation, the project requires the following installations:

``pip install pandas``
``pip install matplotlib.pyplot``
``pip install matplotlib.ticker``
``pip install numpy``
``pip install IPython``
``pip install git+https://github.com/alemartinello/dstapi``
``pip install geopandas``
``pip install ipywidgets``
