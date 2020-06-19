# Author: Jan Niklas Kolf, 2020
from face_image_quality import InsightFace, SERFIQ, get_embedding_quality
import cv2

if __name__ == "__main__":
    # Sample code of calculating the embedding and it's score
    
    # Create the InsightFace model
    insightface_model = InsightFace()
   
    # Create the SER-FIQ Model
    ser_fiq = SERFIQ()
        
    # Load the test image
    test_img = cv2.imread("./data/test_img.jpeg")
    
    # Calculate the embedding and it's quality score
    # T=100 (default) is a good choice
    # Apply preprocessing if image is not aligned (default)
    embedding, score = get_embedding_quality(test_img,
                                               insightface_model,
                                               ser_fiq
                                               )
   
    print("SER-FIQ quality score of image 1 is", score)
    
    # Load the test image
    test_img2 = cv2.imread("./data/test_img2.jpeg")
    
    # Calculate the embedding and it's quality score
    # T=100 is a good choice
    # Apply preprocessing if image is not aligned
    embedding2, score2 = get_embedding_quality(test_img2, 
                                               insightface_model, 
                                               ser_fiq
                                               )
   
    print("SER-FIQ quality score of image 2 is", score2)
    
# Please note that SER-FIQ on ArcFace produces quality estimates in a very narrow, and thus unconvinient, range.
# You might rescale these values to a more convinient range, such as [0,1].
