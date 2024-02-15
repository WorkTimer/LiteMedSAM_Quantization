# LLiteMedSAM Quantization

LiteMedSAM Quantization is an optimized version based on the original MedSAM library. The original repository can be found here: [MedSAM GitHub Repository](https://github.com/bowang-lab/MedSAM/). The quantized version of LiteMedSAM has been deployed as a WEB application, accessible at: LiteMedSAM WEB Application: https://medsam.senma.xyz/. This application allows users to upload two-dimensional medical imaging pictures (in PNG, JPG, JPEG formats) and process them using the quantized version of LiteMedSAM for image segmentation masking.


## Installation Guide
#### Cloning and Installing Dependencies
1. Clone the repository of the quantized version of LiteMedSAM:
```bash
git clone https://github.com/WorkTimer/LiteMedSAM_Quantization/
cd LiteMedSAM_Quantization
```
2. Install necessary libraries:
```bash
sudo apt-get install libgl1-mesa-glx libegl1-mesa libxrandr2 libxrandr2 libxss1 libxcursor1 libxcomposite1 libxi6 libxtst6
```
3. Install conda, refer to the link: [Conda Installation Guide](https://conda.io/projects/conda/en/latest/user-guide/install/index.html).

#### Creating a Virtual Environment
1. Create a conda virtual environment named `medsam`:
```bash
conda create -n medsam python=3.10 -y
conda activate medsam
```
2. Install Pytorch and related dependencies:
```bash
conda install pytorch torchvision -c pytorch
pip install streamlit pandas opencv-python numpy matplotlib pillow pyarrow
pip install -e .
```

#### Installing Pytorch 2.0
1. Enter the MedSAM folder:
```bash
cd MedSAM
```
2. Run the installation command:
```bash
pip install -e .
```

#### Downloading Necessary Files
1. Download the LiteMedSAM checkpoint file `lite_medsam.pth` and place it in the `work_dir/LiteMedSAM` directory. Download link: [Google Drive](https://drive.google.com/drive/folders/1t3Rs9QbfGSEv2fIFlk8vi7jc0SclD1cq?usp=sharing).
2. Download the demo data and place it in the `test_demo/` directory. Download link: [Google Drive](https://drive.google.com/drive/folders/1t3Rs9QbfGSEv2fIFlk8vi7jc0SclD1cq?usp=sharing).

### Model Testing

#### Running Test Commands
1. Test using the original model:
```bash
python "CVPR24_LiteMedSAM_infer.py" -i test_demo/imgs/ -o test_demo/segs
```
2. Test using the quantized model for accelerated performance:
```bash
python "CVPR24_LiteMedSAM_infer_accelerating.py" -i test_demo/imgs/ -o test_demo/segs
```

### WEB Application Operation

#### Starting and Accessing
1. Run the following command in the terminal to start the WEB application:
```bash
streamlit run /home/scchat/MedSAM/app_streamlit.py --server.port=8501
```
2. Access the application in a browser:
http://<Server_IP>:8501

---

Please replace `<Server_IP>` with your actual server IP address. This document provides a basic guide for the installation, configuration, and usage of the quantized version of LiteMedSAM.


