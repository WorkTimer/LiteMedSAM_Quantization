
from datetime import datetime
import random
import shutil
import uuid
import json
import logging
import mimetypes
import os
import subprocess
from typing import Dict, List, Optional
from zipfile import ZipFile
import tempfile
import httpx
import numpy as np
import SimpleITK
from fastapi import APIRouter, FastAPI, Response, HTTPException, UploadFile, Form, File, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from jsonschema import ValidationError
import traceback
import pydicom
import toml
from dicomweb_client import DICOMwebClient
from pydantic import BaseModel
from pydicom import dcmread
from pydicom.dataset import Dataset
# import pyplastimatch as pypla
from requests_toolbelt import MultipartEncoder
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from segment_anything.modeling import MaskDecoder, PromptEncoder, TwoWayTransformer
from simpleitk_reader_writer import SimpleITKIO
from tiny_vit_sam import TinyViT
import aiofiles
# from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO
# from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
logger = logging.getLogger(__name__)

device = torch.device("cuda:1")
lite_medsam_checkpoint_path = 'lite_medsam.pth'

with open('config.toml', 'r') as config_file:
    config = toml.load(config_file)
    app_settings = config.get('app_settings', {})

api_app = FastAPI(docs_url="/docs", openapi_url="/openapi.json")
origins = [
    "http://localhost:3000",
    "http://localhost:8800",
]
api_app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
router = APIRouter()

@router.get("/")
async def root():
    return {"message": "Hello LiteMedSAM API"}

def get_mime_type(file):
    m_type = mimetypes.guess_type(file, strict=False)
    logger.debug(f"Guessed Mime Type for Image: {m_type}")

    if m_type is None or m_type[0] is None:
        m_type = "application/octet-stream"
    else:
        m_type = m_type[0]
    logger.debug(f"Final Mime Type: {m_type}")
    return m_type

@router.get("/info/")
async def get_info():
    config_data = {}
    async with aiofiles.open('models.json', 'r') as json_file:
        data = await json_file.read()
        config_data = json.loads(data)["model_info"]

    for model_id in config_data["models"]:
        dataset_path = f'{config_data["models"][model_id]["path"]}/dataset.json'
        del config_data["models"][model_id]["path"]
        async with aiofiles.open(dataset_path, 'r') as dataset_file:
            dataset_data = await dataset_file.read()
            dataset_json = json.loads(dataset_data)
            config_data["models"][model_id]["labels"] = dataset_json["labels"]

    return config_data


def extract_entire_plane(image_arrays, box):
    x_coords = [int(round(p[0])) for p in box]
    y_coords = [int(round(p[1])) for p in box]
    z_coords = [int(round(p[2])) for p in box]
    
    if len(set(z_coords)) == 1: 
        z_value = z_coords[0]
        plane = image_arrays[z_value, :, :]
        plane_coords = [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]
        fixed_axis = "z"
        fixed_value = z_value
    elif len(set(x_coords)) == 1:
        x_value = x_coords[0]
        plane = image_arrays[:, :, x_value]
        plane_coords = [min(y_coords), min(z_coords), max(y_coords), max(z_coords)]
        fixed_axis = "x"
        fixed_value = x_value
    elif len(set(y_coords)) == 1:
        y_value = y_coords[0]
        plane = image_arrays[:, y_value, :]
        plane_coords = [min(x_coords), min(z_coords), max(x_coords), max(z_coords)]
        fixed_axis = "y"
        fixed_value = y_value
    
    min_val = np.min(plane)
    max_val = np.max(plane)
    plane_scaled = (plane - min_val) / (max_val - min_val) * 255
    plane_scaled = plane_scaled.astype(np.uint8)

    plane_rgb = np.stack((plane_scaled,) * 3, axis=-1)

    assert np.max(plane_rgb) < 256, f'Input data should be in range [0, 255], but got {np.unique(plane_rgb)}'
    
    return plane_rgb, plane_coords, fixed_axis, fixed_value

def resize_longest_side(image, target_length=256):
    """
    Resize image to target_length while keeping the aspect ratio
    Expects a numpy array with shape HxWxC in uint8 format.
    """
    oldh, oldw = image.shape[0], image.shape[1]
    scale = target_length * 1.0 / max(oldh, oldw)
    newh, neww = oldh * scale, oldw * scale
    neww, newh = int(neww + 0.5), int(newh + 0.5)
    target_size = (neww, newh)

    return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)

def pad_image(image, target_size=256):
    """
    Pad image to target_size
    Expects a numpy array with shape HxWxC in uint8 format.
    """
    # Pad
    h, w = image.shape[0], image.shape[1]
    padh = target_size - h
    padw = target_size - w
    if len(image.shape) == 3: ## Pad image
        image_padded = np.pad(image, ((0, padh), (0, padw), (0, 0)))
    else: ## Pad gt mask
        image_padded = np.pad(image, ((0, padh), (0, padw)))

    return image_padded

def resize_box_to_256(box, original_size):
    """
    the input bounding box is obtained from the original image
    here, we rescale it to the coordinates of the resized image

    Parameters
    ----------
    box : numpy.ndarray
        bounding box coordinates in the original image
    original_size : tuple
        the original size of the image

    Returns
    -------
    numpy.ndarray
        bounding box coordinates in the resized image
    """
    new_box = np.zeros_like(box)
    ratio = 256 / max(original_size)
    for i in range(len(box)):
        new_box[i] = int(box[i] * ratio)

    return new_box

class MedSAM_Lite(nn.Module):
    def __init__(
            self, 
            image_encoder, 
            mask_decoder,
            prompt_encoder
        ):
        super().__init__()
        self.image_encoder = image_encoder
        self.mask_decoder = mask_decoder
        self.prompt_encoder = prompt_encoder

    def forward(self, image, box_np):
        image_embedding = self.image_encoder(image) # (B, 256, 64, 64)
        # do not compute gradients for prompt encoder
        with torch.no_grad():
            box_torch = torch.as_tensor(box_np, dtype=torch.float32, device=image.device)
            if len(box_torch.shape) == 2:
                box_torch = box_torch[:, None, :] # (B, 1, 4)

        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=None,
            boxes=box_np,
            masks=None,
        )
        low_res_masks, iou_predictions = self.mask_decoder(
            image_embeddings=image_embedding, # (B, 256, 64, 64)
            image_pe=self.prompt_encoder.get_dense_pe(), # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings, # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings, # (B, 256, 64, 64)
            multimask_output=False,
          ) # (B, 1, 256, 256)

        return low_res_masks

    @torch.no_grad()
    def postprocess_masks(self, masks, new_size, original_size):
        """
        Do cropping and resizing

        Parameters
        ----------
        masks : torch.Tensor
            masks predicted by the model
        new_size : tuple
            the shape of the image after resizing to the longest side of 256
        original_size : tuple
            the original shape of the image

        Returns
        -------
        torch.Tensor
            the upsampled mask to the original size
        """
        # Crop
        masks = masks[..., :new_size[0], :new_size[1]]
        # Resize
        masks = F.interpolate(
            masks,
            size=(original_size[0], original_size[1]),
            mode="bilinear",
            align_corners=False,
        )

        return masks

@torch.no_grad()
def medsam_inference(medsam_model, img_embed, box_256, new_size, original_size):
    """
    Perform inference using the LiteMedSAM model.

    Args:
        medsam_model (MedSAMModel): The MedSAM model.
        img_embed (torch.Tensor): The image embeddings.
        box_256 (numpy.ndarray): The bounding box coordinates.
        new_size (tuple): The new size of the image.
        original_size (tuple): The original size of the image.
    Returns:
        tuple: A tuple containing the segmented image and the intersection over union (IoU) score.
    """
    box_torch = torch.as_tensor(box_256[None, None, ...], dtype=torch.float, device=img_embed.device)
    
    sparse_embeddings, dense_embeddings = medsam_model.prompt_encoder(
        points = None,
        boxes = box_torch,
        masks = None,
    )
    low_res_logits, iou = medsam_model.mask_decoder(
        image_embeddings=img_embed, # (B, 256, 64, 64)
        image_pe=medsam_model.prompt_encoder.get_dense_pe(), # (1, 256, 64, 64)
        sparse_prompt_embeddings=sparse_embeddings, # (B, 2, 256)
        dense_prompt_embeddings=dense_embeddings, # (B, 256, 64, 64)
        multimask_output=False
    )

    low_res_pred = medsam_model.postprocess_masks(low_res_logits, new_size, original_size)
    low_res_pred = torch.sigmoid(low_res_pred)  
    low_res_pred = low_res_pred.squeeze().cpu().numpy()  
    medsam_seg = (low_res_pred > 0.5).astype(np.uint8)

    return medsam_seg, iou

medsam_lite_image_encoder = TinyViT(
    img_size=256,
    in_chans=3,
    embed_dims=[
        64, ## (64, 256, 256)
        128, ## (128, 128, 128)
        160, ## (160, 64, 64)
        320 ## (320, 64, 64) 
    ],
    depths=[2, 2, 6, 2],
    num_heads=[2, 4, 5, 10],
    window_sizes=[7, 7, 14, 7],
    mlp_ratio=4.,
    drop_rate=0.,
    drop_path_rate=0.0,
    use_checkpoint=False,
    mbconv_expand_ratio=4.0,
    local_conv_size=3,
    layer_lr_decay=0.8
)

medsam_lite_prompt_encoder = PromptEncoder(
    embed_dim=256,
    image_embedding_size=(64, 64),
    input_image_size=(256, 256),
    mask_in_chans=16
)

medsam_lite_mask_decoder = MaskDecoder(
    num_multimask_outputs=3,
        transformer=TwoWayTransformer(
            depth=2,
            embedding_dim=256,
            mlp_dim=2048,
            num_heads=8,
        ),
        transformer_dim=256,
        iou_head_depth=3,
        iou_head_hidden_dim=256,
)

medsam_lite_model = MedSAM_Lite(
    image_encoder = medsam_lite_image_encoder,
    mask_decoder = medsam_lite_mask_decoder,
    prompt_encoder = medsam_lite_prompt_encoder
)

lite_medsam_checkpoint = torch.load(lite_medsam_checkpoint_path, map_location='cpu')
medsam_lite_model.load_state_dict(lite_medsam_checkpoint)

if device.type == "cpu":
# https://pytorch.org/tutorials/recipes/quantization.html#post-training-dynamic-quantization
    medsam_lite_model = torch.quantization.quantize_dynamic(
        medsam_lite_model, {torch.nn.Linear}, dtype=torch.qint8
    )

medsam_lite_model.to(device)
medsam_lite_model.eval()

class InferenceRequest(BaseModel):
    boxes: Optional[List[List[List[float]]]] = None

@router.post("/infer/{model_id}")
async def infer(model_id: str, 
                image: str = '1.2.826.0.1.3680043.10.1398.347439963527678441841885180737383925',
                params: str = Form("{}"),
                ):
    nifti_dir = app_settings.get('nifti_dir', 'temp/nifti/')
    os.makedirs(nifti_dir, exist_ok=True)
    image_nii_gz = os.path.join(f"{nifti_dir}", f"{image}.nii.gz")
    if not os.path.exists(image_nii_gz):
        client = DICOMwebClient(url=os.environ.get("DICOMWEBURL", "http://192.168.0.16:8042/dicom-web"))
        meta = Dataset.from_json(
            [
                series
                for series in client.search_for_series(search_filters={"SeriesInstanceUID": image})
                if series["0020000E"]["Value"] == [image]
            ][0]
        )
        study_id = str(meta["StudyInstanceUID"].value)
        with tempfile.TemporaryDirectory() as tmpdirname:
            series_dir = os.path.join(tmpdirname, 'dicom', image)
            # series_dir = f"temp/dicom/{image}"
            os.makedirs(series_dir, exist_ok=True)
            instances = client.retrieve_series(study_id, image)
            image_arrays = []
            image_properties = {}
            for instance in instances:
                # image_arrays.append(instance.pixel_array)
                # if not image_properties:
                #     pixel_spacing = instance.PixelSpacing if "PixelSpacing" in instance else None
                #     image_properties = {
                #         "spacing": pixel_spacing,
                #     }

                # 'sitk_stuff':{'spacing': (0.6425780057907104, 0.6425780057907104, 5.0), 'origin': (328.35736083984375, 328.35736083984375, 0.0), 'direction': (-1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0)}
                # 'spacing':[5.0, 0.6425780057907104, 0.6425780057907104]
                # shape:(1, 34, 512, 512)

                instance_id = str(instance["SOPInstanceUID"].value)
                file_name = os.path.join(series_dir, f"{instance_id}.dcm")
                instance.save_as(file_name)

            if os.path.isdir(series_dir) and len(os.listdir(series_dir)) > 1:
                reader = SimpleITK.ImageSeriesReader()
                dicom_names = reader.GetGDCMSeriesFileNames(series_dir)
                reader.SetFileNames(dicom_names)
                image_ITK = reader.Execute()

                logger.info(f"Image size: {image_ITK.GetSize()}")
                # nifti_dir = "temp/nifti/"
                # os.makedirs(nifti_dir, exist_ok=True)
                # image_nii_gz = os.path.join(f"{nifti_dir}", f"{image}.nii.gz")
                SimpleITK.WriteImage(image_ITK, f"{image_nii_gz}")
                # image_nii_gz = '/home/ubuntu/tmpofkh274_.nii.gz'

    image_arrays, image_properties = SimpleITKIO().read_images([image_nii_gz])
    image_arrays = image_arrays[0]
    # sitk_image = SimpleITK.ReadImage(image_nii_gz)
    # image_arrays = SimpleITK.GetArrayFromImage(sitk_image)
    seg = np.zeros_like(image_arrays, dtype=np.uint8)
    params = json.loads(params) if params else {}
    boxes = params['boxes']
    for idx, box in enumerate(boxes, start=1):
        plane_rgb, box, fixed_axis, fixed_value = extract_entire_plane(image_arrays, box)
        H, W, _ = plane_rgb.shape
        img_256 = resize_longest_side(plane_rgb, 256)
        newh, neww = img_256.shape[:2]
        img_256_norm = (img_256 - img_256.min()) / np.clip(
            img_256.max() - img_256.min(), a_min=1e-8, a_max=None
        )
        img_256_padded = pad_image(img_256_norm, 256)
        img_256_tensor = torch.tensor(img_256_padded).float().permute(2, 0, 1).unsqueeze(0).to(device)
        with torch.no_grad():
            image_embedding = medsam_lite_model.image_encoder(img_256_tensor)
        box256 = resize_box_to_256(box, original_size=(H, W))
        box256 = box256[None, ...] # (1, 4)
        sam_mask, iou_pred = medsam_inference(medsam_lite_model, image_embedding, box256,  (newh, neww), (H, W))
        print(f'box: {box}, predicted iou: {np.round(iou_pred.item(), 4)}')
        if fixed_axis == "z":
            seg[fixed_value, :, :][sam_mask > 0] = idx
        elif fixed_axis == "y":
            seg[:, fixed_value, :][sam_mask > 0] = idx
        elif fixed_axis == "x":
            seg[:, :, fixed_value][sam_mask > 0] = idx

    res_json = {"labels": {"background": 0, **{f"label_{i}": int(value) for i, value in enumerate(np.unique(seg[seg != 0]), start=1)}}}
    # with tempfile.TemporaryDirectory() as tmpdirname:
    #     seg_dir = os.path.join(tmpdirname, 'seg')
    seg_2d_dir = app_settings.get('seg_2d_dir', 'temp/seg_2d/')
    os.makedirs(seg_2d_dir, exist_ok=True)
    seg_nii_gz = os.path.join(f"{seg_2d_dir}", f"2d_{image}_seg.nii.gz")
    # seg_nii_gz = os.path.join(f"{seg_dir}", f"{image}_seg.nrrd")
    seg = seg.astype("uint16")
    seg_sitk = SimpleITK.GetImageFromArray(seg) # seg is numpy array
    seg_sitk.SetSpacing(image_properties['sitk_stuff']['spacing'])
    seg_sitk.SetOrigin(image_properties['sitk_stuff']['origin'])
    seg_sitk.SetDirection(image_properties['sitk_stuff']['direction'])
    SimpleITK.WriteImage(seg_sitk, seg_nii_gz)

    m_type = get_mime_type(seg_nii_gz)
    res_fields = dict()
    res_fields["params"] = ('prams.json', json.dumps(res_json), "application/json")
    # seg_nii_gz = 'CT_demo__monai.seg.nii.gz'
    if seg_nii_gz and os.path.exists(seg_nii_gz):
        # seg_nii_gz ='tmpk3_tf3p8.nrrd'
        res_fields["image"] = (os.path.basename(seg_nii_gz), open(seg_nii_gz, "rb"), m_type)
    else:
        logger.info(f"Return only Result Json as Result Image is not available: {seg_nii_gz}")
        return res_json

    return_message = MultipartEncoder(fields=res_fields)
    return Response(content=return_message.to_string(), media_type=return_message.content_type)
  
@router.post("/upload",)
async def upload(
    file: UploadFile = File(...),
    modality: Optional[str] = "CT"
):
    try:
        if not file:
            raise HTTPException(status_code=400, detail="file not provided")
        if not file.filename.endswith(".nii.gz"):
            raise HTTPException(status_code=400, detail="File must be a '.nii.gz' file")

        with tempfile.TemporaryDirectory() as tmpdirname:
            current_time = datetime.now().strftime("%Y%m%d%H%M%S")
            unique_id = uuid.uuid4().hex
            filename = f"{current_time}_{unique_id}_{file.filename}"
            upload_dir = os.path.join(tmpdirname, 'upload')
            os.makedirs(upload_dir, exist_ok=True)
            save_path = f'{upload_dir}/{filename}'
            with open(save_path, 'wb') as out_file:
                content = await file.read()
                out_file.write(content)
            dicom_output_dir = os.path.join(tmpdirname, 'dicom_output')
            os.makedirs(dicom_output_dir, exist_ok=True)
            # plastimatch convert --patient-id patient1 --input Task09_Spleen/imagesTs/spleen_1.nii.gz --modality CT --output-dicom dicom_output
            str_cmd = f'plastimatch convert --patient-name {file.filename} --patient-id {file.filename} --input {save_path} --modality {modality} --output-dicom {dicom_output_dir}'
            p = subprocess.Popen(
                        str_cmd.split(),
                        stdout      = subprocess.PIPE,
                        stderr      = subprocess.PIPE,
            )
            str_stdout, str_stderr = p.communicate()
            print(str_stdout.decode())
            if p.returncode != 0:
                error_msg = str_stderr.decode().strip() or "Unknown error during plastimatch conversion."
                logging.error("Plastimatch error: " + error_msg)
                raise RuntimeError("Plastimatch conversion failed: " + error_msg)
            # convert_args_ct = { "patient-id" : 'patient1', "input": save_path, "modality": modality, "output-dicom": dicom_output_dir }
            # pypla.convert(verbose = True, **convert_args_ct)
            zip_filename = os.path.join(tmpdirname, f'{current_time}_{unique_id}_output.zip')
            return_content = {
                'StudyInstanceUID': None,
                'SeriesInstanceUID': None
            }
            file_count = sum(1 for item in os.listdir(dicom_output_dir) if os.path.isfile(os.path.join(dicom_output_dir, item)) and item.endswith('.dcm'))
            file_count = (file_count + 2) * 2
            base_uid = str(pydicom.uid.generate_uid("1.2.826.0.1.3680043.10.1398."))
            length_file_count = len(str(file_count))
            max_random_value = (10 ** length_file_count - 1) - file_count
            random_number = random.randint(0, max_random_value)
            study_instance_uid = base_uid[:-(length_file_count)] + f"{random_number:0{length_file_count}d}"
            frame_of_reference_uid = base_uid[:-(length_file_count)] + f"{random_number + 2:0{length_file_count}d}"
            series_instance_uid = base_uid[:-(length_file_count)] + f"{random_number + 4:0{length_file_count}d}"
            return_content['StudyInstanceUID'] = study_instance_uid
            return_content['SeriesInstanceUID'] = series_instance_uid
            with ZipFile(zip_filename, 'w') as zipf:
                sorted_files = sorted([item for item in os.listdir(dicom_output_dir) if item.endswith('.dcm')], key=lambda x: x.lower())
                for index, file in enumerate(sorted_files):
                    file_path = os.path.join(dicom_output_dir, file)
                    dicom_file = pydicom.dcmread(file_path)
                    dicom_file.StudyInstanceUID = study_instance_uid
                    dicom_file.SeriesInstanceUID = series_instance_uid
                    dicom_file.FrameOfReferenceUID = frame_of_reference_uid
                    dicom_file.file_meta.MediaStorageSOPInstanceUID = base_uid[:-(length_file_count)] + f"{random_number + 6 + 2 * index:0{length_file_count}d}"
                    dicom_file.SOPInstanceUID = base_uid[:-(length_file_count)] + f"{random_number + 6 + 2 * index:0{length_file_count}d}"
                    dicom_file.save_as(file_path)
                    zipf.write(file_path, os.path.relpath(file_path, dicom_output_dir))
            print("base_uid\t\t\t", base_uid)
            print("series_instance_uid\t\t", series_instance_uid)
            # http://10.1.0.2:8042/instances
            async with httpx.AsyncClient() as client:
                with open(zip_filename, 'rb') as f:
                    data = f.read()
                    headers = {
                        'Expect': '',
                        'Accept': 'application/json',
                        'Content-Type': 'application/x-www-form-urlencoded'
                    }
                    response = await client.post('http://10.1.0.2:8042/instances', data=data, headers=headers)
                    if response.status_code != 200:
                        raise HTTPException(status_code=response.status_code, detail="Failed to upload ZIP file")
                    response_data = response.json()

                    # response = await client.get(f'http://10.1.0.2:8042/instances/{response_data[0]["ID"]}/study')
                    # if response.status_code != 200:
                    #     raise HTTPException(status_code=response.status_code, detail="Failed to find study")
                    # response_data = response.json()
                    # new_filename = f'{response_data["MainDicomTags"]["StudyInstanceUID"]}.nii.gz'
                    print(len(response_data))
                    if not all(item['Status'] in ['Success', 'AlreadyStored'] for item in response_data):
                    # if not all(item['Status'] == 'Success' for item in response_data):
                        raise HTTPException(status_code=response.status_code, detail="Failed to upload file")
                    new_filename = f'{return_content["SeriesInstanceUID"]}.nii.gz'
                    nifti_dir = "temp/nifti/"
                    os.makedirs(nifti_dir, exist_ok=True)
                    new_save_path = os.path.join(nifti_dir, new_filename)                    
                    # os.rename(save_path, new_save_path)
                    shutil.move(save_path, new_save_path)
        return JSONResponse(content=return_content, status_code=200)
    except Exception as e:
        logging.error("Error:", e)
        traceback.print_exc()
        return JSONResponse(content={"detail": str(e)}, status_code=500)
    
api_app.include_router(router)