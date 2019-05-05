import cv2
import os
import NumberPlate
import DetectChars
import DetectPlates
import time


def main():
    blnKNNTrainingSuccessful = NumberPlate.loadKNNDataAndTrainKNN()  # attempt KNN training

    if blnKNNTrainingSuccessful == False:  # if KNN training was not successful
        print("\nerror: KNN traning was not successful\n")  # show error message
    imgOriginalScene = cv2.imread('LicPlateImages/24.jpg')

    if imgOriginalScene is None:  # if image was not read successfully
        print("\nerror: image not read from file \n\n")  # print error message to std out
        os.system("pause")  # pause so user can see error message
         # and exit program
            # end if
        # ret, imgOriginalScene = cap.read()
    listOfPossiblePlates = DetectPlates.detectPlatesInScene(imgOriginalScene)  # detect plates

    listOfPossiblePlates = DetectChars.detectCharsInPlates(listOfPossiblePlates)  # detect chars in scene

    cv2.imshow("imgOriginalScene", imgOriginalScene)  # show scene image

    if len(listOfPossiblePlates) == 0:  # if no plates were found
            print("\nno license plates were detected\n")  # inform user no plates were found
    else:  # else
                # if we get in here list of possible plates has at leat one plate
                # sort the list of possible plates in DESCENDING order (most number of chars to least number of chars)
        listOfPossiblePlates.sort(key=lambda possiblePlate: len(possiblePlate.strChars), reverse=True)

                # suppose the plate with the most recognized chars (the first plate in sorted by string length descending order) is the actual plate
        licPlate = listOfPossiblePlates[0]
        NumberPlate.detectCharsInPlates(licPlate)
        print(
                'license plate read from image =' + licPlate.strChars + ' ')  # write license plate text to std out

        licPlate.imgPlate = cv2.resize(licPlate.imgPlate, (0, 0), fx=1, fy=1)
    cv2.imshow('licPlate',licPlate.imgPlate)
    cv2.waitKey(0)
if __name__ == "__main__":
    main()
