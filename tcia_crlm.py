import argparse
import glob
import os, os.path

import pandas
import pydicom
import pydicom_seg
import SimpleITK as sitk
import numpy

try:
    from joblib import Parallel, delayed
except ImportError:
    HAS_JOB_LIB=False
else:
    HAS_JOB_LIB=True

# NOTE: This is not a very robust way to load DICOM images, but it works for this curated data set.
def load_ct(d, expected_number_of_files=None):
    reader = sitk.ImageSeriesReader()
    files = reader.GetGDCMSeriesFileNames(d)

    if expected_number_of_files:
        assert len(files) == expected_number_of_files

    reader.SetFileNames(files)
    img = reader.Execute()

    return img

def load_seg(d, expected_series_id=None):
    names = os.listdir(d)
    assert len(names) == 1
    dpath = os.path.join(d, names[0])

    reader = pydicom_seg.SegmentReader()
    result = reader.read(pydicom.dcmread(dpath))
    if expected_series_id:
        assert result.dataset.SeriesInstanceUID == expected_series_id

    results = {}
    for segint in result.available_segments:
        seg = result.segment_image(segint)
        metadata = result.segment_infos[segint]
        # No spaces in returned labels!
        label = metadata.SegmentLabel.replace(" ", "_")
        results[label] = seg

    return results

def im_information_agrees(ima, imb):
    attribs = (
        lambda x: x.GetSpacing(),
        lambda x: x.GetDirection(),
        lambda x: x.GetOrigin(),
        lambda x: x.GetSize(),
    )

    for a in attribs:
        if not numpy.allclose(a(ima), a(imb)):
            return False

    return True

def expand_subsized_segmentation(ct, m):
    assert numpy.allclose(m.GetDirection(), ct.GetDirection(), atol=1e-3)
    assert numpy.allclose(m.GetSpacing(), ct.GetSpacing(), atol=1e-3)
    size = ct.GetSize()

    if ct.GetSize() == m.GetSize(): 
        # The images already are in same coord system and same number of slices, so we are fine
        assert numpy.allclose(ct.GetOrigin(), m.GetOrigin())
        return m
    
    # Mask volume is cropped, within CT volume, so we need to create our own image with the same size
    arr = numpy.zeros(size[::-1], dtype=numpy.uint8)
    mor = m.GetOrigin()
    mask_origin_index = ct.TransformPhysicalPointToIndex(mor)
    mask_bound = m.TransformIndexToPhysicalPoint(m.GetSize())
    mask_bound_index = ct.TransformPhysicalPointToIndex(mask_bound)
    
    # Copy the loaded mask into the CT.
    arr[mask_origin_index[2]:mask_bound_index[2], mask_origin_index[1]:mask_bound_index[1], mask_origin_index[0]:mask_bound_index[0]] = sitk.GetArrayViewFromImage(m)

    # Convert array to sitk image with correct information to match the CT.
    mout = sitk.GetImageFromArray(arr)
    mout.CopyInformation(ct)
    return mout

def ensure_segments_match_image(img, seg):
    out_seg = {}
    for key, msk in seg.items():
        out_seg[key] = expand_subsized_segmentation(img, msk)

    return out_seg

def metadata_rows(subject, ct, seg, root):
    rows = [dict(Subject=subject, Modality="CT", Label="ct", Path=os.path.relpath(ct, root))]

    for lab, spath in seg.items():
        rows.append(dict(
            Subject=subject, Modality="SEG", Label=lab, Path=os.path.relpath(spath, root)
        ))

    return pandas.DataFrame.from_records(rows)

def do_conversion(row_iter, args):
    subject, tab = row_iter

    assert tab.shape[0] == 2
    assert set(tab.Modality.unique()) == set(["SEG", "CT"])
    ct_row = tab.loc[tab.Modality=="CT"].iloc[0]
    seg_row = tab.loc[tab.Modality=="SEG"].iloc[0]

    ctdir = os.path.join(args.input_root, ct_row['File Location'])
    segdir = os.path.join(args.input_root, seg_row['File Location'])

    ct = load_ct(ctdir, expected_number_of_files=int(ct_row['Number of Images']))
    seg = load_seg(segdir, expected_series_id=seg_row['Series UID'])

    seg = ensure_segments_match_image(ct, seg)

    out_path = lambda label: os.path.join(args.output_root, f"{subject}_{label}.nii.gz")

    ct_path = out_path("ct")
    sitk.WriteImage(ct, ct_path)
    seg_paths = {}
    for lab, msk in seg.items():
        seg_path = out_path(lab)
        seg_paths[lab] = seg_path
        sitk.WriteImage(msk, seg_path)

    rows = metadata_rows(subject, ct_path, seg_paths, args.output_root)
    return rows
    

def convert(in_path, out_path, args):
    in_meta = os.path.join(in_path, "metadata.csv")
    out_meta = os.path.join(out_path, "metadata.csv")

    os.makedirs(out_path, exist_ok=True)

    table = pandas.read_csv(in_meta)

    grouped = table.groupby("Subject ID")
    if HAS_JOB_LIB:
        results = Parallel(n_jobs=args.jobs, verbose=10)(delayed(do_conversion)(row_iter, args) for row_iter in grouped)
    else:
        print("No joblib, so any parallelism is disabled")
        results = []
        for sub, rows in grouped:
            print("Processing", sub)
            results.append(do_conversion((sub, rows), args))

    results = pandas.concat(results, axis=0)
    results.to_csv(out_meta, index=False)

def main():
    if HAS_JOB_LIB:
        default_n_jobs=-1
    else:
        default_n_jobs=1

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_root", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--jobs", type=int, default=default_n_jobs)
    args = parser.parse_args()

    convert(args.input_root, args.output_root, args)
    
if __name__=="__main__":
    main()
