import os

if __name__ == "__main__":

	model_path = "../../models/"
	if not os.path.exists(os.path.realpath(model_path)):
	        os.makedirs(os.path.realpath(model_path))

	if not os.path.exists(os.path.realpath(model_path + "master/")):
		print("Downloading model file from cloud storage")
		#download the file
