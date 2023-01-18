# tcga_stroma_prediction

Steps to create masks from WSIs

1. Clone repo.
- git clone https://github.com/hwanglab/tcga_stroma_prediction.git

2. Move to "tcga_stroma_prediction" directory

3. Run python
- python normalization.py 'input directory' 'output file name' 'tile size'
** Note that this will create a .csv file with mu and std statistics to be used as an input.
- python generate_mask_from_WSIs.py 'input directory' 'output directory' 'model path' 'norm stats'
** Use the .csv file for 'norm stats'

4. Confirm the masks in 'your output directory'

** Note that you may need to install some python packages.

# check python version and location
python --version

# create virtual environment
virtualenv -p /usr/bin/python3.8 env

# activate
source env/bin/activate

# package versions
numpy==1.19.5
tensorflow-gpu==2.6.2
scipy==1.4.1
scikit-image==0.17.2
scikit-learn==0.24.2
pandas==1.1.5
Pillow==8.4.0

