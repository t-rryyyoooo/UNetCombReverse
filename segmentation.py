import SimpleITK as sitk
import numpy as np
import argparse
from functions import getSizeFromString, createParentPath
from featureMapCreater import FeatureMapCreater
from extractor import extractor as extor
from tqdm import tqdm
import torch
import cloudpickle


def ParseArgs():
    parser = argparse.ArgumentParser()

    parser.add_argument("image_path", help="$HOME/Desktop/data/kits19/case_00000/imaging.nii.gz")
    parser.add_argument("modelweightfile_fmc", help=".pkl")
    parser.add_argument("modelweightfile", help="Trained model weights file (*.pkl).")
    parser.add_argument("save_path", help="Segmented label file.(.mha)")
    parser_add_argument("--image_patch_size", default="132-132-116")
    parser_add_argument("--label_patch_size", default="44-44-28")
    parser_add_argument("--image_patch_size_fmc", default="48-48-32")
    parser_add_argument("--label_patch_size_fmc", default="44-44-28")
    parser.add_argument("--image_patch_width_fmc", default=8, type=int)
    parser.add_argument("--label_patch_width_fmc", default=8, type=int)
    parser.add_argument("--plane_size", default="512-512")
    parser.add_argument("-g", "--gpuid", help="0 1", nargs="*", default=0, type=int)

    args = parser.parse_args()
    return args

def segment(image_array, image_array_final, model, device="cpu"):
    while image_array.ndim < 5:
        image_array = image_array[np.newaxis, ...]

    assert image_array_final.ndim == 4

    image_array = torch.from_numpy(image_array).to(device, dtype=torch.float)
    image_array_final = torch.from_numpy(image_array_final).to(device, dtype=torch.float)

    segmented_array = model(image_array, image_array_final).to("cpu").detach().numpy().astype(np.float)
    segmented_array = np.squeeze(segmented_array)

    return segmented_array


def main(args):
    """ Get the patch size etc from string."""
    image_patch_size = getSizeFromString(args.image_patch_size)
    label_patch_size = getSizeFromString(args.label_patch_size)
    image_patch_size_fmc = getSizeFromString(args.image_patch_size_fmc)
    label_patch_size_fmc = getSizeFromString(args.label_patch_size_fmc)
    plane_size = getSizeFromString(args.plane_size, digit=2)

    """ Slice module. """
    image = sitk.ReadImage(args.image_path)

    dummy = sitk.Image(image.GetSize(), sitk.sitkUInt8)
    dummy.SetOrigin(image.GetOrigin())
    dummy.SetSpacing(image.GetSpacing())
    dummy.SetDirection(image.GetDirection())

    """ Get model for FeatureMapCreater. """
    with open(args.modelweightfile_fmc, 'rb') as f:
        model_fmc = cloudpickle.load(f)

    model_fmc.eval()


    """ Get feature maps. """
    fmc = FeatureMapCreater(
            image = image,
            model = model_fmc,
            image_patch_size = image_patch_size_fmc,
            label_patch_size = label_patch_size_fmc,
            image_patch_width = args.image_patch_width,
            label_patch_width = args.label_patch_width,
            plane_size = plane_size
            )

    fmc.execute()
    feature_map_list = fmc.output()
    del fmc

    """ Get patches. """
    extractor = extor(
            image = image,
            label = dummy,
            image_patch_size = image_patch_size,
            label_patch_size = label_patch_size
            )

    extractor.execute()
    image_array_list = extractor.output()[0]

    assert len(image_array_list) == len(feature_map_list)

    """ Load model for segmentation. """
    with open(args.modelweightfile, 'rb') as f:
        model = cloudpickle.load(f)

    model.eval()

    """ Confirm if GPU is available. """
    use_cuda = torch.cuda.is_available() and True
    device = torch.device("cuda" if use_cuda else "cpu")

    """ Segmentation module. """
    segmented_array_list = []
    with tqdm(total=len(image_array_list), desc="Segmenting images...", ncols=60):
        for image_array, feature_map in zip(image_array_list, feature_map_list):
            segmented_array = segment(image_array, feature_map, model, device=device)
            segmented_array = np.argmax(segmented_array, axis=0).astype(np.uint8)

        segmented_array_list.append(segmented_array)

    """ Restore module. """
    segmented = tpc.restore(segmented_array_list)

    createParentPath(args.save_path)
    print("Saving image to {}".format(args.save_path))
    sitk.WriteImage(segmented, args.save_path, True)


if __name__ == '__main__':
    args = ParseArgs()
    main(args)
    
