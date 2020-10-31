import os
from PIL import Image
import glob

def convertjpg(jpgfile, outdir, width=64, height=64):
    img = Image.open(jpgfile)
    new_img = img.resize((width, height), Image.ANTIALIAS)
    new_img.save(os.path.join(outdir, os.path.basename(jpgfile)))

for jpgfile in glob.glob("D:/MyProjects/PythonProjects/FYP/final_year_project/dataset/lfw_cropped/faces/*.jpg"):
    convertjpg(jpgfile, "D:/MyProjects/PythonProjects/FYP/final_year_project/dataset/lfw_cropped/faces_resize/")

