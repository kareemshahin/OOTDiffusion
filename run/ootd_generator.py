from pathlib import Path
from PIL import Image
import sys
from utils_ootd import get_mask_location

PROJECT_ROOT = Path(__file__).absolute().parents[1].absolute()
sys.path.insert(0, str(PROJECT_ROOT))

from preprocess.openpose.run_openpose import OpenPose
from preprocess.humanparsing.run_parsing import Parsing
from ootd.inference_ootd_hd import OOTDiffusionHD
from ootd.inference_ootd_dc import OOTDiffusionDC
from datetime import datetime


class OOTDGenerator:
    def __init__(self, gpu_id=0, model_path="", cloth_path="", model_type="hd", category=0, scale=2.0, step=20, sample=4, seed=-1):
        self.gpu_id = gpu_id
        self.model_path = model_path
        self.cloth_path = cloth_path
        self.model_type = model_type
        self.category = category
        self.scale = scale
        self.step = step
        self.sample = sample
        self.seed = seed

        # Initialize models
        self.openpose_model = OpenPose(self.gpu_id)
        self.parsing_model = Parsing(self.gpu_id)

        self.category_dict = ['upperbody', 'lowerbody', 'dress']
        self.category_dict_utils = ['upper_body', 'lower_body', 'dresses']

        self._validate_model_type()
        self._initialize_model()

    def _validate_model_type(self):
        if self.model_type not in ["hd", "dc"]:
            raise ValueError("model_type must be 'hd' or 'dc'!")
        if self.model_type == 'hd' and self.category != 0:
            raise ValueError("model_type 'hd' requires category == 0 (upperbody)!")

    def _initialize_model(self):
        if self.model_type == "hd":
            self.model = OOTDiffusionHD(self.gpu_id)
        elif self.model_type == "dc":
            self.model = OOTDiffusionDC(self.gpu_id)

    def _generate_filename(self, idx=0):
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        return f"{timestamp}_{idx}.png"

    def generate_images(self):
        cloth_img = Image.open(self.cloth_path).resize((768, 1024))
        model_img = Image.open(self.model_path).resize((768, 1024))

        keypoints = self.openpose_model(model_img.resize((384, 512)))
        model_parse, _ = self.parsing_model(model_img.resize((384, 512)))

        mask, mask_gray = get_mask_location(
            self.model_type,
            self.category_dict_utils[self.category],
            model_parse,
            keypoints
        )
        mask = mask.resize((768, 1024), Image.NEAREST)
        mask_gray = mask_gray.resize((768, 1024), Image.NEAREST)

        masked_vton_img = Image.composite(mask_gray, model_img, mask)
        masked_vton_img.save('./images_output/mask.jpg')

        images = self.model(
            model_type=self.model_type,
            category=self.category_dict[self.category],
            image_garm=cloth_img,
            image_vton=masked_vton_img,
            mask=mask,
            image_ori=model_img,
            num_samples=self.sample,
            num_steps=self.step,
            image_scale=self.scale,
            seed=self.seed,
        )

        self._save_images(images)

    def _save_images(self, images):
        image_idx = 0
        image_names = []

        for image in images:
            img_filename = f"/app/run/outputs/{self._generate_filename(image_idx)}"
            image.save(img_filename)
            image_names.append(img_filename)
            image_idx += 1

        return image_names

