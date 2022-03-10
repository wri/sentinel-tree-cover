import os
import urllib.request
import zipfile

if __name__ == "__main__":

	model_path = "../../models/"
	if not os.path.exists(os.path.realpath(model_path)):
	        os.makedirs(os.path.realpath(model_path))

	if not os.path.exists(os.path.realpath(model_path + "master/")):
		print("Downloading model file from cloud storage")
		urllib.request.urlretrieve("https://storage.googleapis.com/rm-models/master.zip", "master.zip")
		with zipfile.ZipFile("master.zip", 'r') as zip_ref:
		    zip_ref.extractall("../models/")
