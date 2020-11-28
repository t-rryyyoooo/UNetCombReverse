import argparse
import SimpleITK as sitk
import cloudpickle
from featureMapCreater import FeatureMapCreater
from functions import getSizeFromString, sendToLineNotify

def parseArgs():
    parser = argparse.ArgumentParser()

    parser.add_argument("image_path")
    parser.add_argument("model_path")
    parser.add_argument("save_path")
    parser.add_argument("--mask_path")
    parser.add_argument("--image_patch_size", default="48-48-32")
    parser.add_argument("--label_patch_size", default="44-44-28")
    parser.add_argument("--image_patch_width", default=8, type=int)
    parser.add_argument("--label_patch_width", default=2, type=int)
    parser.add_argument("--plane_size", default="512-512")

    args = parser.parse_args()

    return args

def main(args):
    image = sitk.ReadImage(args.image_path)
    with open(args.model_path, "rb") as f:
        model = cloudpickle.load(f)

    model.eval()
    if args.mask_path is not None:
        mask = sitk.ReadImage(args.mask_path)

    else:
        mask = None

    image_patch_size = getSizeFromString(args.image_patch_size)
    label_patch_size = getSizeFromString(args.label_patch_size)
    plane_size = getSizeFromString(args.plane_size, digit=2)

    fmc = FeatureMapCreater(
            image = image,
            model = model,
            mask = mask,
            image_patch_size = image_patch_size,
            label_patch_size = label_patch_size,
            image_patch_width = args.image_patch_width,
            label_patch_width = args.label_patch_width,
            plane_size = plane_size
            )

    fmc.execute()
    fmc.save(args.save_path)

    message = args.save_path + " DONE."
    sendToLineNotify(message)

if __name__ == "__main__":
    args = parseArgs()
    main(args)

