# SDDR-Species
Species Distribution Modeling of Disease Vector Species using Semi-Structured Deep Distributional Regression

**Disclaimer**: The analysis is not entirely reproducible as it relies on some confidential data that could not be made public.


## Folder structure
Overview of project files and folders:

- Preliminary notes:
    + The initial steps of this project requiere access to the **`mastergrids`** folder at Maleria Atlas Project and will thus not be fully reproducible

- **`mastergrids`**:
An **`R`** package that facilitates the import of environmental
raster data from the **`mastergrids`** folder at BDI MAP and some
utility functions to transform rasters to data frames and vice versa
(the data import won't work outside of the BDI/without connection to mastergrids). Also contains two functions `grid_to_df` and `df_to_grid` which
convert RasterLayer/RasterBrick object to a data frame and vice versa.


