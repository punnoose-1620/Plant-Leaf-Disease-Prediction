#libraries need:
	-> tensorflow [to load the model and prediction]
	-> opencv-python [image loading and processing]
	-> numpy [array manipulation]
	-> fastapi [for creating the rest-api]




#there are two rest-api endpoints from which we can get the prediction and the related information

-> option-1: by uploading the image itself [recommended]
this endpoints do follow things:
		# this api endpoint receive an image file for disease prediction
		# It's convert it into numpy array, do processing and predict the class label.
		# return the predicted diseases[target class] along with the follwing information:
		#     -> overview about the dieases predicted
		#     -> control measures that can be taken 
		#     -> further links to read about more 

-> option-2: by providing the image path
this endpoints do follow things:
		# do the same thing as the above function except it receive path of the image
		# and then it will load the file and do the rest of the things
