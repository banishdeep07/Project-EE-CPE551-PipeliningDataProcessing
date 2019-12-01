import preProcessing
import modelTraining
import os


if __name__ == '__main__':
	print("\n\n#################Processing Files#################")
	if any(file.endswith(".pickle") for file in os.listdir("../outputs")): #i.e., pickle file has already been created, no need to repeat this
		pickleCreationRequired=False
	else:
		pickleCreationRequired = True

	if pickleCreationRequired:
		preProcessing.process()


	print("\n\n#################Training Models#################")
	modelTraining.trainModelsAndCreateSubmission()

	print("\n\n#################################################")
	print("####################Completed####################")
	print("###############Please Check Output###############")
	print("#################################################")