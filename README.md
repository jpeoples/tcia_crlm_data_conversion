This repository demonstrates how to prepare the Colorectal-Liver-Metastastases data set (from TCIA) into a set of nifti files ready for machine learning use.

# The data set
	
Simpson, A. L., Peoples, J., Creasy, J. M., Fichtinger, G., Gangai, N., Lasso, A., Keshava Murthy, K. N., Shia, J., D’Angelica, M. I., & Do, R. K. G. (2023). Preoperative CT and Survival Data for Patients Undergoing Resection of Colorectal Liver Metastases (Colorectal-Liver-Metastases) (Version 2) [Data set]. The Cancer Imaging Archive. https://doi.org/10.7937/QXK2-QG03

# Downloading the Data
Download the [NBIA Data Retriever](https://wiki.cancerimagingarchive.net/display/NBIA/Downloading+TCIA+Images#DownloadingTCIAImages-DownloadingtheNBIADataRetriever) and the data manifest from the [dataset page](https://doi.org/10.7937/QXK2-QG03).

Load the manifest into the data retriever and commence the download. Once the data is downloaded, take note of the directory to which it was downloaded. At the top-level of the directory you will find a `metadata.csv` file. This directory will be the root directory you will provide to the conversion script.

# Installing Dependencies

The dependencies for this package are `SimpleITK`, `numpy`, `pandas`, `pydicom<3.0`, `pydicom-seg`, and optionally `joblib` for parallelism.

Because of a quirk of `pydicom-seg` which is not yet fixed, `pydicom` should be from major version 2, so it must be specified.

```
pip install SimpleITK numpy pandas "pydicom<3.0" pydicom-seg joblib
```

# Converting the data

TL;DR

```
python tcia_crlm.py --input_root <ROOT> --output_root <OUTPUT_DIRECTORY>
```

will convert all images and segmentations into nifti format, and dump everything into a single directory, with files named as

```
<OUTPUT_DIRECTORY>/<SUBJECT>_ct.nii.gz
<OUTPUT_DIRECTORY>/<SUBJECT>_seg_<LABEL>.nii.gz
```

where LABEL refers to the segmentation labels, `{Liver, Liver_Remnant, Portal, Hepatic, Tumor_1, Tumor_2, ...}`

All segmentatations are output as binary masks, and all images for a given patient are perfectly aligned (same size, same spacing, same origin, same direction, etc, so every voxel, array-wise, is corresponding between all images for a given patient).

# Questions

If you have any questions please reach out to me: `jacob.peoples@queensu.ca`