## ImageSURF

This repository corresponds to my MSc thesis 
[Reconstructing Surfaces with Appearance using
Neural Implicit 3D Representations](https://drive.google.com/file/d/1RWD3citDTZecb4INWwU3IhnqYikTQWd-/view?usp=sharing) implementation.

Specifically we proposed `ImageSURF` as a both surface and volumetric rendering
pipeline, for object specific `novel view synthesis` and `3D reconstruction`, as an alternative
to [pixelnerf](https://alexyu.net/pixelnerf/).

## Pipeline
![](./media/pipeline.png)

## Installation
In orded to install the proper dependencies after cloning the repository, 
run the following commands to create the `nerf` environment.

```
conda env create -f environment.yaml
conda activate imagesurf
```

## Dataset
For the object specifc scenes we employed [ShapeNet](https://shapenet.org/) instances
similarly preprocessed to the [srn](https://github.com/vsitzmann/scene-representation-networks).
Despite of the fact that out method is appliclable to all shapenet instances, our evaluation contains data only from the `srn_cars` instance due to resources limitation.
In case you are interested in evaluating our results, you can just download [srn_cars](https://drive.google.com/file/d/1BIVGjPK86L4G5i5zm63oYGeNE_k3Z-zp/view?usp=sharing).
For the remaining shapenet instances, we provide the following process to convert the shapenet data
to the desired compressed form, after downloading the desired shapenet category from here
[ShapeNet](https://shapenet.org/).
```
python preprocessing.py --shapenet_path <shapenet_path> --data_path <data_path>

```
SRN dataset like our preprocessed one is organized to three subsets train, val and test
Each one of them contains images that correspons to different viewpoints and their camera parameters. All viewpoints are onganized in a 3D sphere of radius 1.3. In terms of number of views being provided, following table wiil give you a good insight in terms of instances and correponding views.
|subset|instances|views|
|:-------:|:-------:|:-------:|
|train|2458|50|
|val|352|251|
|test|703|251|

After downloading either our preprocessed srn_cars dataset or any other 
shapenet category (and turn it into the desired form though prerpocessing.py), 
please attach the data path into the dataloading section of the corresponding yaml file(**e.g.** SRN_cars.yaml)
## Train
Our codebase is built on [UniSURF](https://github.com/autonomousvision/unisurf) structure.
Following UniSURF implementation we propose a new framework that employs image features 
in order to render novel views and accurate 3D shapes of objects that belong to the 
same semantic class. 
To train an imageSURF from scratch run the following

```
python train.py SRN_cars.yaml
```
In case you are into other shapenet categories you can easily convert SRN_cars.yaml as wou wish.

## Render
To evaluate the trained model qualitatively run 

```
python reconstruction.py SRN_cars.yaml
```
## Extract Mesh
For 3D reconstruction you can extract colorless meshes through
that utilizes [marching cubes](https://github.com/pmneila/PyMCubes)
implemented by [pmneila](https://github.com/pmneila)
```
python extracet_mesh.py SRN_cars.yaml
```

## Qualitative Results
### Appearance Renderings
![](./media/GT_pixelnerf_imagesurf.png)

![](./media/gif_rgb.gif)
### Depth Renderings
![](./media/pixelnerf_imagesurf.png)

![](./media/gif_depth.gif)
### Surface Renderings
![](./media/GT_pixelnerf_imagesurf.png)

![](./media/gifs_surf.gif)

## Quantitative Results
![](./media/metrics.png)
