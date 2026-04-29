# Shapes2Labels
>_Pre-release_. Work in progress.

`Shapes2Labels` is a Napari based annotator tool to enable ground truth, multiclass annotation of large images using spatialdata's Zarr protocol. For a fuller, more comprehensive package please refer to [`napari-spatialdata-repo`](https://github.com/scverse/napari-spatialdata).

Full github repository can be found [here](https://github.com/mjbcowan/reimagined_guacamole/tree/main).

![Demo](assets/shapes2label_annotation.gif)

*Shapes2Labels annotation workflow: drawing shapes and generating pixel classification masks. Image credit: [skimage.data (skin)](https://scikit-image.org/docs/stable/api/skimage.data.html).*

## Scope and Design Intent
`Shapes2Labels` has been designed as a lightweight method for:

* Multiscale visualisation of bioimaging formats, specifically multiplexed images, RGB images, and paganin-filtered synchrotron images
* Annotation of shapes of multiple class types across whole image in single layer which can be visualised dynamically
* Generation of semantic pixel classification masks from annotated shapes
* Refinement of both of the above

Many of [`napari-spatialdata-repo`](https://github.com/scverse/napari-spatialdata)'s are more extensive and cover broader use cases, however, this repo was designed for this sole use case to increase cycle time of mask-inference generation in custom segmentation model training.

## CREDITS AND REFERENCES
Developed by MJBC, this is a pre-release work in progress. **_This package is not officially affilitated with SpatialData, nor Napari_**.


Please cite the following if using this package:

<div class="grid" markdown>
<div markdown>

### SpatialData
SpatialData: an open and universal framework for processing spatial omics data ([paper](https://www.nature.com/articles/s41592-024-02212-x), [repository](https://github.com/scverse/spatialdata), [documentation](https://spatialdata.scverse.org/en/stable/))
```bibtex
@article{marconato2025spatialdata,
  title={SpatialData: an open and universal data framework for spatial omics},
  author={Marconato, Luca and Palla, Giovanni and Yamauchi, Kevin A and Virshup, Isaac and Heidari, Elyas and Treis, Tim and Vierdag, Wouter-Michiel and Toth, Marcella and Stockhaus, Sonja and Shrestha, Rahul B and others},
  journal={Nature methods},
  volume={22},
  number={1},
  pages={58--62},
  year={2025},
  publisher={Nature Publishing Group US New York}
}
```

</div>
<div markdown>

### Napari
Multi-dimensional image viewer for python ([repository](https://github.com/napari/napari), [documentation](https://napari.org/stable/))
```bibtex
@Manual{napari2019,
    title={napari: a multi-dimensional image viewer for Python},
    author={Sofroniew, N., Lambert, T., Bokota, G., Nunez-Iglesias, J., Sobolewski, P., Sweet, A., Gaifas, L., Evans, K., Burt, A., Doncila Pop, D., Yamauchi, K., Weber Mendonça, M., Buckley, G., Vierdag, W.-M., Royer, L., Can Solak, A., Harrington, K. I. S., Ahlers, J., Althviz Moré, D., … Zhao, R},
    year={2019},
    url={https://github.com/napari/napari},
    doi={https://doi.org/10.5281/zenodo.14427406}
}
```

</div>
</div>
