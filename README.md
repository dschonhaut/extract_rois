# extract_rois

A general-purpose tool for extracting regional values from neuroimaging data. This command-line utility simplifies the extraction of mean values from regions-of-interest (ROIs) in NIfTI format neuroimaging files, supporting both binary masks and parcellation-based approaches.

## Features

- Extract mean values from binary ROI masks
- Extract values from parcellation files (e.g., FreeSurfer's aparc+aseg)
- Built-in FreeSurfer ROI definitions
- Support for multiple input images and ROIs
- Flexible output formats (long or wide)
- Command-line interface for easy integration into processing pipelines

## Installation

```bash
pip install extract_rois
```

## Usage

The tool operates in two modes:

### Binary Mask Mode (-m)
Use this mode when working with binary masks that define your ROIs. Each mask should contain 0s and 1s, where 1s indicate the ROI to extract.

### Parcellation Mode (-a)
Use this mode with labeled images where different regions are defined by integer values, such as FreeSurfer's aparc+aseg. A reference CSV file (-f) defines the mapping between region names and their corresponding integer labels.

### Input Requirements

The program requires the -i flag and either -m or -a. All other flags are optional. Input files must meet these requirements:
- All images, masks, and parcellations must be NIfTI files storing 3D numeric arrays
- ROI masks and parcellations must have the same shape and be in the same space as their target images

### Output Formats

The program prints an output table of ROI means and volumes to standard output, with an option to save to a CSV file using the -o flag. Two output formats are available:

Long format (default):
- One row for each target image x ROI pair
- Columns: image path, ROI path, ROI name, mean value within ROI, and voxel count

Wide format:
- One row for each target image
- Two columns for each ROI (mean and voxel count)

### Command Line Interface

```
usage: extract_rois -i IMAGE [IMAGE ...] (-m [MASK ...] | -a [PARCELLATION ...])
                    [-r [ROI ...]] [-f ROI_FILE] [-l] [-o OUTPUT]
                    [-s {long,wide,l,w}] [-q]

Required Arguments:
  -i, --image         One or more target NIfTI images
  -m, --mask          One or more binary mask files
  -a, --parcellation  One or more parcellation files

Optional Arguments:
  -r, --roi           Names of ROIs to extract (parcellation mode only)
  -f, --roi_file      Custom ROI definition file (parcellation mode only)
  -l, --list_rois     List all available ROIs (parcellation mode only)
  -o, --output        Output CSV file path
  -s, --shape         Output format (long/wide)
  -q, --quiet         Suppress stdout output
```

### Examples

```bash
# 1. Binary mask mode, simple case:
extract_rois -i fbb_suvr-wcbl.nii -m mask-amyloid-cortical-summary.nii
# Description: Extract mean PET SUVR within the amyloid-PET cortical summary ROI
# (combination of FreeSurfer regions used by ADNI to calculate Centiloids)

# 2. Binary mask mode, multiple masks and target images:
extract_rois -i wr*_suvr-wcbl.nii -m mni_rois/*.nii
# Description: Wildcard expansion is used to extract multiple ROIs from multiple PET images
# at once. You can also type the path to each file individually, with a space between files
# (e.g. extract_rois -i 'pet1.nii' 'pet2.nii' -m 'mask1.nii' 'mask2.nii' 'mask3.nii').
# The output table will be n_images x n_masks rows long

# 3. Binary mask mode, output saved to CSV:
extract_rois -i wr*_suvr-wcbl.nii -m mni_rois/*.nii -o roi_means.csv -s wide
# Description: The output table is saved to roi_means.csv in wide format

# 4. Parcellation mode, simple case:
extract_rois -i ftp_suvr-infcblgm.nii -a aparc+aseg.nii
# Description: Extract mean PET SUVR from all FreeSurfer ROIs in the default reference file

# 5. Parcellation mode, select ROIs:
extract_rois -i ftp_suvr-infcblgm.nii -a aparc+aseg.nii -r meta_temporal
# Description: Extract only the tau-PET meta-temporal ROI

# 6. Parcellation mode, multiple target images all in the same space:
extract_rois -i ftp_suvr-infcblgm.nii ftp_suvr-eroded-subcortwm.nii -a aparc+aseg.nii
# Description: Extract all FreeSurfer ROIs for SUVR images that were referenced against
# inferior cerebellar gray matter and eroded subcortical white matter, respectively

# 7. Parcellation mode, multiple target images and parcellations:
extract_rois \\
         -i $(tail -n +2 EXTRACTION_PATHS.csv | cut -d',' -f1) \\
         -a $(tail -n +2 EXTRACTION_PATHS.csv | cut -d',' -f2) \\
         -r Amygdala Hippocampus ctx_entorhinal ctx_parahippocampal \\
         -o roi_means.csv
# Description: Here the user has saved a CSV file in advance that contains paths to target
# images (first column) and their corresponding parcellations (second column). extract_rois
# reads these paths and proceeds to extract values from selected MTL regions, for each
# target image

# 8. Parcellation mode, custom ROIs:
     extract_rois -i fdg.nii -a brainstem-rois.nii \\
       -f brainstem_rois.csv \\
       -o brainstem_roi_means.csv -q
# Description: Extract custom ROIs from a user-defined reference file. The output table
# is saved, while command line output is suppressed with the -q flag
```

## Author

Daniel Schonhaut

## Version

1.0.0

## License

GNU General Public License v3.0