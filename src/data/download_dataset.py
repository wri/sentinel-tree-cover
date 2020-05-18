import os

if __name__ == "__main__":

	data_path = "../../data/"
	if not os.path.exists(os.path.realpath(data_path)):
	        os.makedirs(os.path.realpath(data_path))

	if not os.path.exists(os.path.realpath(data_path + "test/")):
		print("Downloading test data files from cloud storage")
		# download test S1
		# download test CSV
		# download test S2
