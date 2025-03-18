# OpenTraj

## Abstract

We introduce existing datasets for Human Trajectory Prediction (HTP) task, and also provide tools to load, visualize and analyze datasets. 

So far multiple datasets are supported.

## Datasets

<!--begin(table_main)-->
| Sample | Name | &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Description&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; | Ref | 
|----|----|----|----|
| ![](datasets/ETH/seq_eth/reference.png) | [ETH](datasets/ETH) | 2 top view scenes containing walking pedestrians <code>#Traj:[Peds=750]</code> <code>Coord=world-2D</code> <code>FPS=2.5</code> | [website](http://www.vision.ee.ethz.ch/en/datasets/) [paper](https://ethz.ch/content/dam/ethz/special-interest/baug/igp/photogrammetry-remote-sensing-dam/documents/pdf/pellegrini09iccv.pdf) | 
| ![](datasets/UCY/zara01/reference.png) | [UCY](datasets/UCY) | 3 scenes (Zara/Arxiepiskopi/University). Zara and University close to top view. Arxiepiskopi more inclined. <code>#Traj:[Peds=786]</code> <code>Coord=world-2D</code> <code>FPS=2.5</code> | [website](https://graphics.cs.ucy.ac.cy/research/downloads/crowd-data) [paper](https://onlinelibrary.wiley.com/doi/full/10.1111/j.1467-8659.2007.01089.x) | 
| ![](datasets/PETS-2009/reference.jpg) | [PETS 2009](datasets/PETS-2009) | different crowd activities <code>#Traj:[?]</code> <code>Coord=image-2D</code> <code>FPS=7</code> | [website](http://www.cvg.reading.ac.uk/PETS2009/data.html) [paper](https://projet.liris.cnrs.fr/imagine/pub/proceedings/AVSS-2010/data/4264a143.pdf) | 
| ![](datasets/SDD/coupa/video3/reference.jpg) | [SDD](datasets/SDD) | 8 top view scenes recorded by drone contains various types of agents <code>#Traj:[Bikes=4210 Peds=5232 Skates=292 Carts=174 Cars=316 Buss=76 Total=10,300]</code> <code>Coord=image-2D</code> <code>FPS=30</code> | [website](http://cvgl.stanford.edu/projects/uav_data) [paper](http://svl.stanford.edu/assets/papers/ECCV16social.pdf) [dropbox](https://www.dropbox.com/s/v9jvt4ln7t42m6m/StanfordDroneDataset.zip) | 
| ![](datasets/GC/reference.jpg) | [GC](datasets/GC) | Grand Central Train Station Dataset: 1 scene of 33:20 minutes of crowd trajectories <code>#Traj:[Peds=12,684]</code> <code>Coord=image-2D</code> <code>FPS=25</code> | [dropbox](https://www.dropbox.com/s/7y90xsxq0l0yv8d/cvpr2015_pedestrianWalkingPathDataset.rar) [paper](http://openaccess.thecvf.com/content_cvpr_2015/html/Yi_Understanding_Pedestrian_Behaviors_2015_CVPR_paper.html) | 
| ![](datasets/HERMES/reference.png) | [HERMES](datasets/HERMES) | Controlled Experiments of Pedestrian Dynamics (Unidirectional and bidirectional flows) <code>#Traj:[?]</code> <code>Coord=world-2D</code> <code>FPS=16</code> | [website](https://www.fz-juelich.de/ias/ias-7/EN/AboutUs/Projects/Hermes/_node.html) [data](https://www.fz-juelich.de/ias/ias-7/EN/Research/Pedestrian_Dynamics-Empiricism/_node.html) | 
| ![](datasets/Waymo/reference.jpg) | [Waymo](datasets/Waymo) | High-resolution sensor data collected by Waymo self-driving cars <code>#Traj:[?]</code> <code>Coord=2D and 3D</code> <code>FPS=?</code> | [website](https://waymo.com/open/) [github](https://github.com/waymo-research/waymo-open-dataset) | 
| ![](datasets/KITTI/reference.jpg) | [KITTI](datasets/KITTI) | 6 hours of traffic scenarios. various sensors <code>#Traj:[?]</code> <code>Coord=image-3D + Calib</code> <code>FPS=10</code> |  [website](http://www.cvlibs.net/datasets/kitti/) | 
| ![](datasets/InD/reference.png) | [inD](datasets/InD) | Naturalistic Trajectories of Vehicles and Vulnerable Road Users Recorded at German Intersections <code>#Traj:[Total=11,500]</code> <code>Coord=world-2D</code> <code>FPS=25</code> | [website](https://www.ind-dataset.com/) [paper](https://arxiv.org/pdf/1911.07602.pdf) | 
| ![](datasets/L-CAS/reference.png) | [L-CAS](datasets/L-CAS) | Multisensor People Dataset Collected by a Pioneer 3-AT robot <code>#Traj:[?]</code> <code>Coord=0</code> <code>FPS=0</code> | [website](https://lcas.lincoln.ac.uk/wp/research/data-sets-software/l-cas-multisensor-people-dataset/) | 
| ![](datasets/Edinburgh/reference.jpg) | [Edinburgh](datasets/Edinburgh) | People walking through the Informatics Forum (University of Edinburgh) <code>#Traj:[ped=+92,000]</code> <code>FPS=0</code> | [website](http://homepages.inf.ed.ac.uk/rbf/FORUMTRACKING/) | 
| ![](datasets/Town-Center/reference.jpg) | [Town Center](datasets/Town-Center) | CCTV video of pedestrians in a busy downtown area in Oxford <code>#Traj:[peds=2,200]</code> <code>Coord=0</code> <code>FPS=0</code> | [website](https://megapixels.cc/datasets/oxford_town_centre/) | 
| ![](datasets/Wild-Track/reference.jpg) | [Wild Track](datasets/Wild-Track) | surveillance video dataset of students recorded outside the ETH university main building in Zurich. <code>#Traj:[peds=1,200]</code> | [website](https://megapixels.cc/wildtrack/) | 
| ![](datasets/ATC/reference.png) | [ATC](datasets/ATC) | 92 days of pedestrian trajectories in a shopping center in Osaka, Japan <code>#Traj:[?]</code> <code>Coord=world-2D + Range data</code> | [website](https://irc.atr.jp/crest2010_HRI/ATC_dataset) | 
| ![](datasets/VIRAT/reference.png) | [VIRAT](datasets/VIRAT) | Natural scenes showing people performing normal actions <code>#Traj:[?]</code> <code>Coord=0</code> <code>FPS=0</code> | [website](http://viratdata.org/) | 
| ![](datasets/Forking-Paths-Garden/reference.png) | [Forking Paths Garden](datasets/Forking-Paths-Garden) | **Multi-modal** _Synthetic_ dataset, created in [CARLA](https://carla.org) (3D simulator) based on real world trajectory data, extrapolated by human annotators <code>#Traj:[?]</code> | [website](https://next.cs.cmu.edu/multiverse/index.html) [github](https://github.com/JunweiLiang/Multiverse) [paper](https://arxiv.org/abs/1912.06445) | 
| ![](datasets/DUT/reference.png) | [DUT](datasets/DUT) | Natural Vehicle-Crowd Interactions in crowded university campus <code>#Traj:[Peds=1,739 vehicles=123 Total=1,862]</code> <code>Coord=world-2D</code> <code>FPS=23.98</code> | [github](https://github.com/dongfang-steven-yang/vci-dataset-dut) [paper](https://arxiv.org/pdf/1902.00487.pdf) | 
| ![](datasets/CITR/reference.png) | [CITR](datasets/CITR) | Fundamental Vehicle-Crowd Interaction scenarios in controlled experiments <code>#Traj:[Peds=340]</code> <code>Coord=world-2D</code> <code>FPS=29.97</code> | [github](https://github.com/dongfang-steven-yang/vci-dataset-dut) [paper](https://arxiv.org/pdf/1902.00487.pdf) | 
| ![](datasets/NuScenes/reference.png) | [nuScenes](datasets/NuScenes) | Large-scale Autonomous Driving dataset <code>#Traj:[peds=222,164 vehicles=662,856]</code> <code>Coord=World + 3D Range Data</code> <code>FPS=2</code> | [website](www.nuscences.org) | 
| ![](datasets/VRU/reference.png) | [VRU](datasets/VRU) | consists of pedestrian and cyclist trajectories, recorded at an urban intersection using cameras and LiDARs <code>#Traj:[peds=1068 Bikes=464]</code> <code>Coord=World (Meter)</code> <code>FPS=25</code> | [website](https://www.th-ab.de/ueber-uns/organisation/labor/kooperative-automatisierte-verkehrssysteme/trajectory-dataset) | 
| ![](datasets/City-Scapes/reference.png) | [City Scapes](datasets/City-Scapes) | 25,000 annotated images (Semantic/ Instance-wise/ Dense pixel annotations) <code>#Traj:[?]</code> | [website](https://www.cityscapes-dataset.com/dataset-overview/) | 
| ![](datasets/Argoverse/reference.jpg) | [Argoverse](datasets/Argoverse) | 320 hours of Self-driving dataset <code>#Traj:[objects=11,052]</code> <code>Coord=3D</code> <code>FPS=10</code> | [website](https://www.argoverse.org) | 
| ![](datasets/Ko-PER/reference.png) | [Ko-PER](datasets/Ko-PER) | Trajectories of People and vehicles at Urban Intersections (Laserscanner + Video) <code>#Traj:[peds=350]</code> <code>Coord=world-2D</code> | [paper](https://www.uni-ulm.de/fileadmin/website_uni_ulm/iui.inst.110/Bilder/Forschung/Datensaetze/20141010_DatasetDocumentation.pdf) | 
| ![](datasets/TRAF/reference.png) | [TRAF](datasets/TRAF) | small dataset of dense and heterogeneous traffic videos in India (22 footages) <code>#Traj:[Cars=33 Bikes=20 Peds=11]</code> <code>Coord=image-2D</code> <code>FPS=10</code> | [website](https://gamma.umd.edu/researchdirections/autonomousdriving/trafdataset/) [gDrive](https://drive.google.com/drive/folders/1zKaeboslkqoLdTJbRMyQ0Y9JL3007LRr) [paper](https://arxiv.org/pdf/1812.04767.pdf) | 
| ![](datasets/ETH-Person/reference.png) | [ETH-Person](datasets/ETH-Person) | Multi-Person Data Collected from Mobile Platforms | [website](https://data.vision.ee.ethz.ch/cvl/aess/) | 

<!--end(table_main)-->

<!--
#### Other Trajectory Datasets
- [NGSim](https://catalog.data.gov/dataset/next-generation-simulation-ngsim-vehicle-trajectories)
- [Daimler](http://www.gavrila.net/Datasets/Daimler_Pedestrian_Benchmark_D/daimler_pedestrian_benchmark_d.html)
- [Cyclist](No Link)
- [highD](No Link)
-->

## Benchmarks
- [Trajnet](http://trajnet.stanford.edu/): Trajectory Forecasting Challenge
- [Trajnet++](https://www.aicrowd.com/challenges/trajnet-a-trajectory-forecasting-challenge): Trajectory Forecasting Challenge
- [MOT-Challenge](https://motchallenge.net): Multiple Object Tracking Benchmark
- [JackRabbot](https://jrdb.stanford.edu/): Detection And Tracking Dataset and Benchmark
