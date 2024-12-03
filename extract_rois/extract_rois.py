#!/usr/bin/env python

"""
General purpose ROI extraction tool.
"""

import argparse
import importlib.resources
import os.path as op
import sys
import time
from collections import OrderedDict as od
from inspect import isroutine

import nibabel as nib
import numpy as np
import pandas as pd


class Timer(object):
    """A simple timer."""

    def __init__(self, msg="Time elapsed: "):
        """Start the global timer."""
        self.reset()
        self.msg = msg

    def __str__(self):
        """Print how long the global timer has been running."""
        self.check()
        hours, remainder = divmod(self.elapsed, 3600)
        minutes, seconds = divmod(remainder, 60)
        if hours:
            return self.msg + f"{hours:d}h, {minutes:d}m, {seconds:.2f}s"
        if minutes:
            return self.msg + f"{minutes:d}m, {seconds:.2f}s"
        else:
            return self.msg + f"{self.elapsed:.2f}s"

    def check(self):
        """Report the global runtime."""
        self.elapsed = time.time() - self.start

    def reset(self):
        """Reset the global timer."""
        self.start = time.time()


class TextFormatter(argparse.RawTextHelpFormatter):
    """Custom formatter for argparse help text."""

    # use defined argument order to display usage
    def _format_usage(self, usage, actions, groups, prefix):
        if prefix is None:
            prefix = "usage: "

        # if usage is specified, use that
        if usage:
            usage = usage % dict(prog=self._prog)

        # if no optionals or positionals are available, usage is just prog
        elif usage is None and not actions:
            usage = "%(prog)s" % dict(prog=self._prog)
        elif usage is None:
            prog = "%(prog)s" % dict(prog=self._prog)
            # build full usage string
            action_usage = self._format_actions_usage(actions, groups)  # NEW
            usage = " ".join([s for s in [prog, action_usage] if s])
            # omit the long line wrapping code
        # prefix with 'usage:'
        return "%s%s\n\n" % (prefix, usage)


def _fmt_long_str(x, maxlen=50):
    """Truncate long strings of semicolon-separated values."""
    if len(x) <= maxlen:
        return x
    elif len(x) > maxlen:
        stop = x[maxlen + 4 :].find(";")
        if stop == -1:
            return x
        else:
            # find the last ';'
            start_last = x.rfind(";") + 1
            return x[: stop + maxlen] + "..." + x[start_last:]


def load_rois(roi_file):
    """Load dictionary of ROI names to lists of 1+ int labels."""
    rois = pd.read_csv(roi_file)
    rois = od(zip(rois.iloc[:, 0], rois.iloc[:, 1]))
    try:
        rois = {
            k: list(np.unique([int(x) for x in v.split(";")])) for k, v in rois.items()
        }
    except AttributeError:
        pass
    rois = pd.Series(rois)
    return rois


def roi_desc(dat, rois, subrois=None, aggf=np.mean, conv_nan=0):
    """Apply `aggf` over `dat` values within each ROI mask.

    Parameters
    ----------
    dat :
        Filepath string, nifti image, or array-like object.
    rois : str, list[str], or dict-like {str: obj}
        Map each ROI name to its filepath string(s), nifti image, or
        array.
    subrois : dict of {str: int or list}
        Map each sub-ROI within the main ROI mask to a value or list of
        mask values that comprise it. The classic example is of an
        aparc+aseg file containing multiple regions with different
        labels. Note: subrois cannot be passed if len(rois) > 1.
    aggf : function, list of functions, or dict of functions
        Function or functions to apply over `dat` values within each
        ROI.
    conv_nan : bool, number, or NoneType object
        Convert NaNs in `dat` to `conv_nan`. No conversion is applied if
        `conv_nan` is np.nan, None, or False.

    Returns
    -------
    output : DataFrame
        `aggf` output for each agg function, for each ROI. Index is the
        ROI names, columns are the function names. The last column is
        ROI volume (number of voxels in the mask).
    """
    if (not isinstance(rois, str)) and (len(rois) > 1) and (subrois is not None):
        raise ValueError("Cannot define multiple rois and subrois")

    # Load the data array.
    dat = load_nii(dat, flatten=True, conv_nan=conv_nan)

    # Format the ROIs to be dict-like.
    if isinstance(rois, str):
        rois = [rois]

    if isinstance(rois, (list, tuple)):
        rois_dict = od([])
        for roi in rois:
            roi_name = op.basename(roi)
            i = roi_name.find("mask-")
            if i != -1:
                roi_name = roi_name[i + 5 :]
            roi_name = ".".join(roi_name.split(".")[:-1])
            rois_dict[roi_name] = roi
        rois = rois_dict
    elif isinstance(rois, dict):
        pass
    else:
        raise ValueError("rois must be str, list, tuple, or dict")

    # Format the aggregation functions to be dict-like.
    if isroutine(aggf):
        aggf = od({aggf.__name__: aggf})
    elif not isinstance(aggf, dict):
        aggf = od({func.__name__: func for func in aggf})

    # Prepare the output DataFrame.
    if subrois is not None:
        output_idx = list(subrois.keys())
    else:
        output_idx = list(rois.keys())
    output_cols = list(aggf.keys()) + ["voxels"]
    output = pd.DataFrame(index=output_idx, columns=output_cols)
    output = output.rename_axis("roi")

    # Loop over the ROIs and sub-ROIs.
    for roi, roi_mask in rois.items():
        if subrois is not None:
            mask = load_nii(roi_mask, flatten=True, binarize=False)
            assert dat.shape == mask.shape
            for subroi, subroi_vals in subrois.items():
                mask_idx = np.where(np.isin(mask, subroi_vals))
                for func_name, func in aggf.items():
                    output.at[subroi, func_name] = func(dat[mask_idx])
                output.at[subroi, "voxels"] = mask_idx[0].size
        else:
            mask = load_nii(roi_mask, flatten=True, binarize=True)
            assert dat.shape == mask.shape
            mask_idx = np.where(mask)
            for func_name, func in aggf.items():
                output.at[roi, func_name] = func(dat[mask_idx])
            output.at[roi, "voxels"] = mask_idx[0].size

    return output


def load_nii(
    infile,
    dtype=np.float32,
    squeeze=True,
    flatten=False,
    conv_nan=0,
    binarize=False,
    int_rounding="nearest",
):
    """Load a NIfTI file and return the NIfTI image and data array.

    Returns (img, dat), with dat being an instance of img.dataobj loaded
    from disk. You can modify or delete dat and get a new version from
    disk: ```dat = np.asanyarray(img.dataobj)```

    Parameters
    ----------
    infile : str
        The nifti file to load.
    dtype : data-type
        Determines the data type of the data array returned.
    flatten : bool
        If true, `dat` is returned as a flattened copy of the
        `img`.dataobj array. Otherwise `dat`.shape == `img`.shape.
    conv_nan : bool, number, or NoneType object
        Convert NaNs to `conv_nan`. No conversion is applied if
        `conv_nan` is np.nan, None, or False.
    binarize : bool
        If true, `dat` values > 0 become 1 and all other values are 0.
        `dat` type is recast to np.uint8.
    int_rounding : str
        Determines how the data array is recast if `binarize` is false
        and `dtype` is an integer.
        `nearest` : round to the nearest integer
        `floor` : round down
        `ceil` : round up

    Returns
    -------
    img : Nifti1Image
    dat : ndarray or ndarray subclass
    """

    # Get the right file extension.
    def _format_array(
        dat,
        dtype=np.float32,
        squeeze=True,
        flatten=False,
        conv_nan=0,
        binarize=False,
        int_rounding="nearest",
    ):
        """Format an array.

        Formatting options:
        - Flattening
        - NaN handling
        - Data type conversion

        Parameters
        ----------
        dtype : data-type
            Determines the data type returned.
        flatten : bool
            Return `dat` as a flattened copy of the input array.
        conv_nan : bool, number, or NoneType object
            Convert NaNs to `conv_nan`. No conversion is applied if
            `conv_nan` is np.nan, None, or False.
        binarize : bool
            If true, `dat` values > 0 become 1 and all other values are 0.
            `dat` type is recast to np.uint8.
        int_rounding : str
            Determines how the data array is recast if `binarize` is false
            and `dtype` is an integer.
            `nearest` : round to the nearest integer
            `floor` : round down
            `ceil` : round up

        Returns
        -------
        dat : ndarray or ndarray subclass
        """
        # Flatten the array.
        if flatten:
            dat = dat.ravel()

        # Squeeze the array.
        elif squeeze:
            dat = np.squeeze(dat)

        # Convert NaNs.
        if not np.any((conv_nan is None, conv_nan is False, conv_nan is np.nan)):
            dat[np.invert(np.isfinite(dat))] = conv_nan

        # Recast the data type.
        if binarize or (dtype is bool):
            idx = dat > 0
            dat[idx] = 1
            dat[~idx] = 0
            if dtype is bool:
                dat = dat.astype(bool)
            else:
                dat = dat.astype(np.uint8)
        elif "int" in str(dtype):
            if int_rounding == "nearest":
                dat = np.rint(dat)
            elif int_rounding == "floor":
                dat = np.floor(dat)
            elif int_rounding == "ceil":
                dat = np.ceil(dat)
            else:
                raise ValueError("int_rounding='{}' not valid".format(int_rounding))
            dat = dat.astype(dtype)
        else:
            dat = dat.astype(dtype)

        return dat

    infile = find_gzip(infile)

    # Load a NIfTI file and get its data array.
    img = nib.load(infile)
    dat = np.asanyarray(img.dataobj)

    # Format the data array.
    dat = _format_array(
        dat,
        dtype=dtype,
        squeeze=squeeze,
        flatten=flatten,
        conv_nan=conv_nan,
        binarize=binarize,
        int_rounding=int_rounding,
    )

    return dat


def find_gzip(infile, raise_error=False, return_infile=False):
    """Find the existing file, gzipped or gunzipped.

    Return the infile if it exists, otherwise return the gzip-toggled
    version of the infile if it exists, otherwise return None or raise
    a FileNotFoundError.

    Parameters
    ----------
    infile : str
        The input file string.
    raise_error : bool
        If true, a FileNotFoundError is raised if the outfile does not
        exist.
    return_infile : bool
        If true, the infile is returned if the outfile does not exist.
        Otherwise None is returned if the outfile does not exist. This
        argument is ignored if raise_error is true.
    """
    if op.isfile(infile):
        outfile = infile
        return outfile
    elif op.isfile(toggle_gzip(infile)):
        outfile = toggle_gzip(infile)
        return outfile
    else:
        if raise_error:
            raise FileNotFoundError(
                "File not found: {}[.gz]".format(infile.replace(".gz", ""))
            )
        elif return_infile:
            return infile
        else:
            return None


def toggle_gzip(infile):
    """Return the gzip-toggled filepath.

    Parameters
    ----------
    infile : str
        The input file string.

    Returns
    -------
    outfile : str
        The output file string, which is the input file string minus
        the ".gz" extension if it exists in infile, or the input file
        string plus the ".gz" extension if it does not exist in infile.
    """
    if infile.endswith(".gz"):
        outfile = infile[:-3]
    else:
        outfile = infile + ".gz"
    return outfile


def get_default_roi_file():
    """Get the path to the default ROI file included with the package."""
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        # Running as compiled executable
        return op.join(sys._MEIPASS, "extract_rois", "freesurfer_rois.csv")
    else:
        # Running in normal Python environment
        try:
            # Try the new Python 3.9+ way first
            with (
                importlib.resources.files("extract_rois")
                .joinpath("freesurfer_rois.csv")
                .open("r") as f
            ):
                return str(f.name)
        except AttributeError:
            # Fall back to the older way for Python <3.9
            import pkg_resources

            return pkg_resources.resource_filename(
                "extract_rois", "freesurfer_rois.csv"
            )


def _parse_args():
    """Parse and return command line arguments."""
    parser = argparse.ArgumentParser(
        description="""
General purpose ROI extraction tool.

extraction modes:
  1. BINARY MASK (-m): Extract means from 1+ target images within 1+ ROI masks
  2. PARCELLATION (-a): Extract means from 1+ target images using 1+ parcellation files.
     Parcellation files (e.g. FreeSurfer's aparc+aseg) have integer values that
     correspond to specific regions. Region-to-value mappings are defined within a
     reference CSV file. See description under the -f flag for more information

input requirements:
  *  -i and either -m or -a (but not both) must be passed. All other flags are optional
  *  All images, masks, and parcellations must be NIfTI files with 3D numeric arrays
  *  ROI masks and parcellations must have the same shape and be in the same space as
     their target image(s)

output:
  *  An output table of ROI means and volumes is printed to the standard output
  *  The output table can also be saved to a CSV file with the -o flag
  *  Two output formats are available: long (default) and wide:
     > Long format has one row for each target image x ROI pair, and five columns
       corresponding to the target image path, ROI path, ROI name, mean of the target
       image within the ROI, and number of voxels in the ROI
     > Wide format has one row for each target image, and two columns for each ROI,
       corresponding to the ROI mean and voxel count

examples:
  1. Binary mask mode, simple case:
     $ extract_rois -i fbb_suvr-wcbl.nii -m mask-amyloid-cortical-summary.nii
     ---
     Extract mean PET SUVR within the amyloid-PET cortical summary ROI (combination of
     FreeSurfer regions used by ADNI to calculate Centiloids)

  2. Binary mask mode, multiple masks and target images:
     $ extract_rois -i wr*_suvr-wcbl.nii -m mni_rois/*.nii
     ---
     Wildcard expansion is used to extract multiple ROIs from multiple PET images at once.
     You can also type the path to each file individually, with a space between files
     (e.g. extract_rois -i 'pet1.nii' 'pet2.nii' -m 'mask1.nii' 'mask2.nii' 'mask3.nii').
     The output table will be n_images x n_masks rows long

  3. Binary mask mode, output saved to CSV:
     $ extract_rois -i wr*_suvr-wcbl.nii -m mni_rois/*.nii -o roi_means.csv -s wide
     ---
     The output table is saved to roi_means.csv in wide format

  4. Parcellation mode, simple case:
     $ extract_rois -i ftp_suvr-infcblgm.nii -a aparc+aseg.nii
     ---
     Extract mean PET SUVR from all FreeSurfer ROIs in the default reference file

  5. Parcellation mode, select ROIs:
     $ extract_rois -i ftp_suvr-infcblgm.nii -a aparc+aseg.nii -r meta_temporal
     ---
     Extract only the tau-PET meta-temporal ROI

  6. Parcellation mode, multiple target images all in the same space:
     $ extract_rois -i ftp_suvr-infcblgm.nii ftp_suvr-eroded-subcortwm.nii -a aparc+aseg.nii
     ---
     Extract all FreeSurfer ROIs for SUVR images that were referenced against inferior
     cerebellar gray matter and eroded subcortical white matter, respectively

  7. Parcellation mode, multiple target images and parcellations:
     $ extract_rois \\
         -i $(tail -n +2 EXTRACTION_PATHS.csv | cut -d',' -f1) \\
         -a $(tail -n +2 EXTRACTION_PATHS.csv | cut -d',' -f2) \\
         -r Amygdala Hippocampus ctx_entorhinal ctx_parahippocampal \\
         -o roi_means.csv
     ---
     Here the user has saved a CSV file in advance that contains paths to target images
     (first column) and their corresponding parcellations (second column). extract_rois
     reads these paths and proceeds to extract values from selected MTL regions, for each
     target image

  8. Parcellation mode, custom ROIs:
     $ extract_rois -i fdg.nii -a brainstem-rois.nii \\
         -f brainstem_rois.csv \\
         -o brainstem_roi_means.csv -q
     ---
     Extract custom ROIs from a user-defined reference file. The output table is saved,
     while command line output is suppressed with the -q flag
""",
        epilog="""
author: Daniel Schonhaut
version: 1.0.0
""",
        formatter_class=TextFormatter,
        exit_on_error=False,
    )
    parser.add_argument(
        "-i",
        "--image",
        type=str,
        nargs="+",
        help=(
            "Paths to one or more NIfTI target images from which regional means are\n"
            "extracted. Multiple images must be separated by spaces. Wildcard\n"
            "expansion can also be used."
        ),
    )
    parser.add_argument(
        "-m",
        "--mask",
        type=str,
        nargs="*",
        help="Paths to one or more NIfTI binary masks to apply to each target image",
    )
    parser.add_argument(
        "-a",
        "--parcellation",
        type=str,
        nargs="*",
        help=(
            "Paths to one or more parcellation files to apply to the target images.\n"
            "*  If there is only one target image, there must be one parcellation file\n"
            "*  If there are multiple target images and only one parcellation file,\n"
            "   then the parcellation file is broadcast over each target image\n"
            "*  If there are multiple target images and multiple parcellation files,\n"
            "   then length(images) must equal length(parcellations), and\n"
            "   images[n] is paired with parcellations[n] for n = 1..length(images)\n"
        ),
    )
    parser.add_argument(
        "-r",
        "--roi",
        type=str,
        nargs="*",
        help=(
            "Names of one or more ROIs to extract.\n"
            "*  ROIs must be defined in the reference CSV file (see -f)\n"
            "*  A list of all defined ROIs can be printed with -l\n"
            "*  If -r is not specified, then all ROIs are extracted by default"
        ),
    )
    parser.add_argument(
        "-f",
        "--roi_file",
        type=str,
        help=(
            "Path to a 2-column reference CSV file of defined ROIs.\n"
            "*  Column 1 must contain ROI names (each row = 1 ROI)\n"
            "*  Column 2 must contain corresponding integer labels. If a region is\n"
            "   defined by multiple integer labels, they must be semicolon-separated\n"
            "*  If -f is not specified, the default FreeSurfer ROI file is used,\n"
            "   and parcellation files are assumed to be aparc+aseg\n"
            "*  This flag is only relevant in parcellation mode (see -a)"
        ),
    )
    parser.add_argument(
        "-l",
        "--list_rois",
        action="store_true",
        help="List all ROI names and labels from the reference CSV file (see -f)",
    )

    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help=(
            "Name of the output CSV file that you want to save. Will overwrite any\n"
            "existing files with the same name. If -o is not specified, results are\n"
            "printed to the standard output but are not saved to disk"
        ),
    )
    parser.add_argument(
        "-s",
        "--shape",
        type=str,
        default="long",
        choices=["long", "wide", "l", "w"],
        help="Shape of the output table. Default: %(default)s",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Don't print the output table to the standard output",
    )
    # Print help if no arguments are given.
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)
    else:
        args = parser.parse_args()
        return args


def main():
    """Main function to run the ROI extraction."""
    timer = Timer()

    # Get command line arguments.
    args = _parse_args()

    # If roi_file wasn't specified, use the default
    if args.roi_file is None:
        args.roi_file = get_default_roi_file()

    # Print ROI names and labels in a nicely-formatted table.
    if args.list_rois:
        all_rois = pd.read_csv(args.roi_file)
        all_rois["n_labels"] = all_rois.iloc[:, 1].apply(lambda x: len(x.split(";")))
        all_rois.iloc[:, 1] = all_rois.iloc[:, 1].apply(_fmt_long_str)
        print(all_rois.to_markdown(index=False, tablefmt="rst"))
        print(args.roi_file, end="\n" * 2)
        sys.exit(0)

    # Load the ROI dictionary
    all_rois = load_rois(args.roi_file)

    # Check that -m or -a is specified
    if args.mask is None and args.parcellation is None:
        print(
            "ERROR: -m or -a must be specified. See help documentation for details.\n"
        )
        sys.exit(1)

    if args.mask is not None and args.parcellation is not None:
        print(
            "ERROR: -m and -a cannot both be specified. See help documentation for details.\n"
        )
        sys.exit(1)

    # Check that the number of images and parcellations match
    if args.parcellation:
        if (len(args.parcellation) > 1) and (len(args.image) != len(args.parcellation)):
            print(
                "ERROR: Number of images and parcellation files must match,\n"
                + "or there must be only 1 parcellation file specified.\n"
                + "Found {} images and {} parcellation files".format(
                    len(args.image), len(args.parcellation)
                )
            )
            sys.exit(1)

    # Extract ROI values from masks
    output = []
    if args.mask:
        for img in args.image:
            _output = roi_desc(dat=img, rois=args.mask)
            _output = _output.reset_index()
            _output.insert(0, "image_file", img)
            _output.insert(1, "roi_file", args.mask)
            output.append(_output)

    # Extract ROI values from parcellations
    if args.parcellation:
        # Get ROIs
        keep_rois = {}
        if args.roi is None:
            keep_rois = all_rois
        else:
            for roi in args.roi:
                if roi not in all_rois.keys():
                    print(f"WARNING: {roi} missing from {args.roi_file}")
                else:
                    keep_rois[roi] = all_rois[roi]
        # Broadcast inputs if needed
        if (len(args.image) > 1) and (len(args.parcellation) == 1):
            args.parcellation = args.parcellation * len(args.image)
        # Extract ROI values
        for img, parc in zip(args.image, args.parcellation):
            _output = roi_desc(dat=img, rois=parc, subrois=keep_rois)
            _output = _output.reset_index()
            _output.insert(0, "image_file", img)
            _output.insert(1, "roi_file", parc)
            output.append(_output)

    output = pd.concat(output).reset_index(drop=True)
    output = output.rename(columns={"voxels": "voxel_count"})

    # Pivot the output dataframe
    if args.shape in ["wide", "w"]:
        if args.mask:
            output = output.pivot(
                index="image_file",
                columns="roi",
                values=["mean", "voxel_count"],
            )
        else:
            output = output.pivot(
                index=["image_file", "roi_file"],
                columns="roi",
                values=["mean", "voxel_count"],
            )
        output.columns = ["_".join(col[::-1]).strip() for col in output.columns.values]
        output = output.reset_index()

    # Save output.
    if args.output:
        output.to_csv(args.output, index=False)
        print(f"\nSaved output to {op.abspath(args.output)}")

    # Print output.
    if not args.quiet:
        output["image_file"] = output["image_file"].apply(op.basename)
        if "roi_file" in output.columns:
            output["roi_file"] = output["roi_file"].apply(op.basename)
        for col in output.columns:
            if "mean" in col:
                output[col] = output[col].astype(float)
            elif "voxel_count" in col:
                output[col] = output[col].astype(int)
        output.columns = output.columns.str.replace("_", "\n")
        print(
            output.to_markdown(
                index=False,
                tablefmt="rst",
                floatfmt=".4f",
                intfmt=",",
            )
        )

    print(timer)
    sys.exit(0)


if __name__ == "__main__":
    main()
