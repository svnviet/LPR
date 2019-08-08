# RLP


Find plates:
 	Detect plate in scene:
		Preprocess(Img original scene)
		Find posssible chars in scene( img thresh scene)
		Find list of matching chars in scene (List of possible chars in scene)
		→ List of lists of matching chars in scene
		
		Extract Plate
	
		→ List of possible plates
Find chars with each possible plates
	Detect chars in plates:
		Preprocess ( Possible plate )
		Find possible chars in plate ( img thresh)
		findListOfListsOfMatchingChars(listOfListsOfMatchingCharsInPlate)
		removeInnerOverlappingChars(listOfListsOfMatchingCharsInPlate)
		→ listOfListsOfMatchingCharsInPlate
		Within each possible plate, suppose the longest list of potential matching chars is the 			actual list of chars
		If count chars out of range 8 to 10  return next possible plate
		→ License Plate
		recognizeCharsInPlate() - using KNN
→ Licplate
