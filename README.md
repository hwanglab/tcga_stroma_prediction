# tcga_stroma_prediction

Steps to create masks from WSIs

1. Clone repo.
- git clone https://github.com/hwanglab/tcga_stroma_prediction.git

2. Move to "tcga_stroma_prediction" directory

3. Run python
- python generate_mask_from_WSIs.py with arguments

4. Confirm the masks in 'your output directory'

** Note that you may need to install some python packages.

# check python version and location
python --version

# create virtual environment
virtualenv -p /usr/bin/python3.6 env

# activate
source env/bin/activate

pip install tensorflow-gpu==2.5

pip install large-image[all] --find-links https://girder.github.io/large_image_wheels

pip install distributed==2021.3.0 scikit-image==0.17.2 scikit-learn==0.24.2 scipy==1.5.4

pip install scipy==1.5.4


