import SimpleITK as sitk
import numpy as np
import torch
from thinPatchCreater import ThinPatchCreater
from extractor import extractor as extor
from functions import croppingForNumpy
from pathlib import Path
from tqdm import tqdm

class FeatureMapCreater():
    def __init__(self, image, model, mask=None, image_patch_size=[48, 48, 32], label_patch_size=[44, 44, 28], image_patch_width=8, label_patch_width=2, plane_size=[512, 512]):
        self.image = image
        self.image_array = sitk.GetArrayFromImage(image)
        self.model = model
        self.mask = mask
        self.image_patch_size = image_patch_size
        self.label_patch_size = label_patch_size
        self.image_patch_width = image_patch_width
        self.label_patch_width = label_patch_width
        self.plane_size = plane_size

        """ Check GPU. """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model  = model.to(self.device)

    """
    1. Get thin patch for UNetThin.
    2. Get feature map from model.
    3. Restore feature map to the original size.
    4. Extract feature map to desired size.

    """

    def preprocess(self, image_array):
        while image_array.ndim < 5:
            image_array = image_array[np.newaxis, ...]

        image_array = torch.from_numpy(image_array).to(self.device, dtype=torch.float)
        
        return image_array

    def predict(self, image_array):
        print("Segmenting image...", end="")
        image_array = self.preprocess(image_array)

        segmented_array = self.model.forwardWithoutSegmentation(image_array)
        segmented_array = segmented_array.to("cpu").detach().numpy().astype(np.float)
        segmented_array = np.squeeze(segmented_array)
        print("Done")

        return segmented_array


    def execute(self):
        """ Make dummy image for ThinPatchCreater. """
        dummy = sitk.Image(self.image.GetSize(), sitk.sitkUInt8)
        dummy.SetOrigin(self.image.GetOrigin())
        dummy.SetSpacing(self.image.GetSpacing())
        dummy.SetDirection(self.image.GetDirection())

        """ Get thin patch. """
        tpc = ThinPatchCreater(
                image = self.image,
                label = dummy,
                image_patch_width = self.image_patch_width,
                label_patch_width = self.label_patch_width,
                plane_size = self.plane_size
                )

        tpc.execute()
        image_patch_array_list, _ = tpc.output(kind="Array")

        diff = self.image_patch_width - self.label_patch_width
        lower_crop_width = diff // 2
        upper_crop_width = (diff + 1) // 2

        """ Get channel. """
        segmented_array = self.predict(image_patch_array_list[0])
        self.ch = segmented_array.shape[0]

        """ Get feature map. """
        lower_crop_size = [0, lower_crop_width, 0, 0]
        upper_crop_size = [0, upper_crop_width, 0, 0]
        segmented_array_list = [[] for _ in range(self.ch)]
        image_patch_array_list.reverse()
        for _ in range(len(image_patch_array_list)):
            segmented_array = self.predict(image_patch_array_list.pop())
            segmented_array = croppingForNumpy(segmented_array, lower_crop_size, upper_crop_size)

            for i in range(self.ch):
                segmented_array_list[i].append(segmented_array[i, ...])

        """ Restore feature map. """
        for i in range(self.ch):
            feature_map_list = tpc.restore(segmented_array_list.pop())

            """ Extract image. """
            e = extor(
                    image = feature_map_list,
                    label = dummy,
                    mask = self.mask,
                    image_patch_size = self.image_patch_size,
                    label_patch_size = self.label_patch_size
                    )
            e.execute()

            feature_map_list = e.output()[0]

            """ Separete data per patch. """
            if i == 0:
                output_len = len(feature_map_list)
                self.output_array_list = [[] for _ in range(output_len)]

            feature_map_list.reverse()
            for j in range(output_len):
                self.output_array_list[j].append(feature_map_list.pop())


        """ Concat separated data per channel. """
        print("Stacking images...")
        for i in range(output_len):
            self.output_array_list[i] = np.stack(self.output_array_list[i])
        print("Done")

    def output(self):
        return self.output_array_list

    def save(self, save_path):
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        with tqdm(total=len(self.output_array_list), desc="Saving feature maps...", ncols=60) as pbar:
            for i, output_array in enumerate(self.output_array_list):
                save_feature_path = save_path / "feature_{:04}.npy".format(i)
                np.save(str(save_feature_path), output_array)

                pbar.update(1)


