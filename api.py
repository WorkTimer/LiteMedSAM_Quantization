
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
import torch
import aiofiles
from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
logger = logging.getLogger(__name__)

with open('config.toml', 'r') as config_file:
    config = toml.load(config_file)

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

def get_predictor(model_path):
    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring = False,
        perform_everything_on_device = True,
        device = torch.device('cuda', 0),
        verbose = False,
        verbose_preprocessing = False,
        allow_tqdm = True
    )
    predictor.initialize_from_trained_model_folder(
        model_path,
        use_folds="all",
        checkpoint_name='checkpoint_final.pth',
    )
    return predictor
# model_path = "models/abdomenCT/UMambaBot"
# predictor = get_predictor(model_path)


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

@router.post("/infer/{model_id}")
async def infer(model_id: str, image: str = "1.2.826.0.1.3680043.8.274.1.1.330040320.1512.1710375221.933173"):
    nifti_dir = "temp/nifti/"
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
            series_dir = os.path.join(tmpdirname, 'dicom', {image})
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

    # image_nii_gz = f'./temp/nifti/{image}.nii.gz'
    image_arrays, image_properties = SimpleITKIO().read_images([image_nii_gz])

    async with aiofiles.open('models.json', 'r') as json_file:
        data = await json_file.read()
        config_data = json.loads(data)["model_info"]
        model_path = config_data["models"][model_id]["path"]
            
    predictor = get_predictor(model_path)
    seg = predictor.predict_single_npy_array(
        input_image = image_arrays, # np.stack(image_arrays),
        image_properties = image_properties,
        segmentation_previous_stage = None,
        output_file_truncated = None,
        save_or_return_probabilities = False
    )

    res_json = {"labels": {"background": 0, **{f"label_{i}": int(value) for i, value in enumerate(np.unique(seg[seg != 0]), start=1)}}}
    # with tempfile.TemporaryDirectory() as tmpdirname:
    #     seg_dir = os.path.join(tmpdirname, 'seg')
    seg_dir = f"temp/seg"
    os.makedirs(seg_dir, exist_ok=True)
    seg_nii_gz = os.path.join(f"{seg_dir}", f"{image}_seg.nii.gz")
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