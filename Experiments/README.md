# Basic Tools For Pedestrian Trajectory Models

This repository contains basic tools such as data downloading and data loader for pedestrian trajectory models.

### Requirements

* python version 3.6.2
* install ```requirements.txt```

### Data
For pedestrian trajectory prediction, we work on the following datasets. 

| Datasets| 
|:-------| 
|ETH | 
|HOTEL | 
|UNIV|
|ZARA1| 
|ZARA2|
|STANFORD DRONE DATASET (SDD)|
|SDD SYNTHETIC HYANG 4|

Download the datasets with the download file from the TUM webshare server. 
* LINUX: ```download_dataset.sh```
* MAC: ```download_dataset_mac.sh```

* ```data/experiments``` contains information such as format and homography matrices of different datasets.
 ### Dataloader
 
 * ```data/trajectories.py``` prepares trajectories as single instances and does not consider different agents per scene.
 * ```data/trajectories_scene.py``` prepares entire scenes and considers multiple agents per scene (needed for models with social interactions).
 
 ### Models
 * ```model/modules.py``` contains Linear and LSTM baseline models.
