# Overview

The script 

# Setup

1. Create a virtual environment and activate it

```sh
python3 -m venv env
source env/bin/activate
```

Then install the requirements by

```sh
pip install -r requirements.txt
```

1. In main.py modify MODEL (if necessary), PATH_TO_DATA and NAME. 

![](img/Screenshot%202022-03-14%20at%2018.12.23.png)

- The model can be personal or selected from Stardist presets, depending on which segments the nuclear channel best. [Stardist GitHub](https://github.com/stardist/stardist)
- The path to data has to contain all channels of an image in the same folder (nuclear stain channel - ch00, YFP channel or other channel indicative of target presence - ch01, and measurement channel - ch02).

1. Run the script

```sh
python3 main.py
```

# Exporting results

