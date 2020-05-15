# Author: Jan Niklas Kolf, 2020
# Demo-implementation of SER-FIQ on ArcFace (InsightFace)

import sys
from os import path as os_path

import numpy as np
import mxnet as mx
import cv2

from tqdm import tqdm

from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import euclidean_distances

import keras.backend as K
from keras.models import Model as KerasModel
from keras.layers import Dense, Lambda, Input, Dropout
from keras.layers.normalization import BatchNormalization


class InsightFace:
    
    def __init__(self,
                 insightface_path:str = "./insightface/",
                 gpu:int=0, # Which gpu should be used -> gpu id
                 det:int=0, # Mtcnn option, 1= Use R+O, 0=Detect from beginning
                 flip:int=0 # Whether do lr flip aug
                 ):
        """
        Reimplementing Insightface's FaceModel class.
        Now the dropout output and the network output are returned after a forward pass.

        Parameters
        ----------
        insightface_path : str, optional
            The path to the insightface repository. The default is "./insightface/".
        gpu : int, optional
            The GPU to be used by Mxnet. The default is 0.
        det : int, optional
            Mtcnn option, 1= Use R+0, 0= Detect from beginning. The default is 0.
        flip:int=0 # Whether do lr flip aug.

        Returns
        -------
        None.

        """

        sym, arg_params, aux_params = mx.model.load_checkpoint(
                                        f"{insightface_path}/models/model", 
                                        0
                                        )
    
        all_layers = sym.get_internals()
        sym_dropout = all_layers['dropout0_output']
        sym_fc1 = all_layers["fc1_output"]
        
        sym_grouped = mx.symbol.Group([sym_dropout, sym_fc1])
    
        self.model = mx.mod.Module(symbol=sym_grouped, context=mx.gpu(gpu), label_names = None)
        self.model.bind(data_shapes=[("data", (1,3,112,112))])
        self.model.set_params(arg_params, aux_params)
        
        self.det_minsize = 50
        self.det_threshold = [0.6,0.7,0.8]
        self.det = det

        mtcnn_path = f"{insightface_path}/deploy/mtcnn-model"
        
        sys.path.append(os_path.realpath(os_path.join(insightface_path, "deploy")))
        sys.path.append(os_path.realpath(f"{insightface_path}src/common/"))
        from mtcnn_detector import MtcnnDetector
        from face_preprocess import preprocess
        
        self.preprocess = preprocess
        
        thrs = self.det_threshold if det==0 else [0.0,0.0,0.2]
        
        self.detector = MtcnnDetector(model_folder=mtcnn_path, 
                                      ctx=mx.gpu(0), 
                                      num_worker=1, 
                                      accurate_landmark = True, 
                                      threshold=thrs
                                      )
        

    def get_input(self, face_img):
        """
        Applies preprocessing to the given face image.

        Parameters
        ----------
        face_img : Numpy ndarray
            The face image.

        Returns
        -------
        numpy ndarray of the face image.

        """
        detected = self.detector.detect_face(face_img, det_type=self.det)
        
        if detected is None:
            return None
        
        bbox, points = detected
        
        if bbox.shape[0] == 0:
            return None
        
        points = points[0, :].reshape((2,5)).T
        
        nimg = self.preprocess(face_img, bbox, points, image_size="112,112")
        nimg = cv2.cvtColor(nimg, cv2.COLOR_BGR2RGB)
        
        return np.transpose(nimg, (2,0,1))
      
        
    def get_feature(self, aligned_img):
        """
        Runs the given aligned image on the Mxnet Insightface NN.
        Returns the embedding and the dropout0 layer output.

        Parameters
        ----------
        aligned_img : numpy ndarray
            The aligned image returned by get_input
            (or own alignment method).

        Returns
        -------
        embedding : numpy ndarray, (512,)
            The arcface embedding of the image.
        dropout : numpy ndarray (1, 512, 7, 7)
            The output of the dropout0 layer as numpy array.

        """
        input_blob = np.expand_dims(aligned_img, axis=0)
        data = mx.nd.array(input_blob)
        db = mx.io.DataBatch(data=(data,))
        self.model.forward(db, is_train=False)
        dropout, embedding = self.model.get_outputs()
        
        embedding = normalize(embedding.asnumpy()).flatten()
        
        return embedding, dropout.asnumpy()
        
    
class SERFIQ:
    
    def __init__(self, data_path: str="./data/"):
        """
        Implementing the same-model type model of

        SER-FIQ: Unsupervised Estimation of Face Image Quality 
        Based on Stochastic Embedding Robustness
        
        Philipp Terhörst, Jan Niklas Kolf, Naser Damer, 
        Florian Kirchbuchner, Arjan Kuijper
        
        Accepted at CVPR 2020
        
        Preprint available at https://arxiv.org/abs/2003.09373
        
        Parameters
        ----------
        data_path : str, optional
            Path to the data folder where
            layer weights/bias are located. The default is "./data/".

        
        """
        weights = np.transpose(np.load(
                                f"{data_path}/pre_fc1_weights.npy"
                                       ))
        
        bias = np.load(f"{data_path}/pre_fc1_bias.npy")
        
        def euclid_normalize(x):
            return K.l2_normalize(x, axis=1)

        inputs = Input(shape=(25088,))
        x = inputs
        x = Dropout(0.5)(x, training=True)
        x = Dense(512, name="dense", activation="linear")(x)
        x = BatchNormalization()(x)
        x = Lambda(euclid_normalize)(x)
        output = x
    
        self.model = KerasModel(inputs, outputs=output)
    
        self.model.get_layer("dense").set_weights([weights, bias])
        
    
    def __call__(self, X):
        return self.predict(X)
    
    def predict(self, X):
        return self.model.predict(X)
    

def get_embedding_quality(img_input, 
                        insightface_model : InsightFace, 
                        ser_fiq : SERFIQ, 
                        T:int =100, 
                        use_preprocessing: bool =True,
                        disable_tqdm: bool = False):
    """
    Calculates the SER-FIQ Quality Score for a given img using
    given insightface model and ser-fiq model.
    
    Parameters
    ----------
    img_input : numpy array shape (x,y,3)
        The image to be processed.
    insightface_model : InsightFace
        Instance of InsightFace class
    ser_fiq : SERFIQ
        Instance of SERFIQ class
    T: int, default is 100
        The amount of forward passes the SER-FIQ model should do
    use_preprocessing: bool, default is True
        True: Preprocessing of insightface model is applied (recommended)
        False: No preprocessing is used, needs an already aligned image
    disable_tqdm: bool, default is False
        If True, no tqdm progress bar is displayed

    Returns
    -------
    Arcface/Insightface embedding: numpy array, shape (512,)
    Robustness score : float


    """
    # Apply preprocessing if image is not already aligned
    if use_preprocessing:
        img_input = insightface_model.get_input(img_input)
    
    if img_input is None:
        # No face etc. could be found, no score could be calculated
        return -1.0, -1.0
     
    # Array + prediction with insightface
    dropout_emb = np.empty((1, 25088), dtype=float)
    
    embedding, dropout = insightface_model.get_feature(img_input)
    
    dropout_emb[0] = dropout.flatten()
    
    del dropout

    # Apply T forward passes using keras
    X = np.empty((T, 512), dtype=float)
    for forward_pass in tqdm(range(T),
                             desc="Forward pass",
                             unit="pass",
                             disable=disable_tqdm):
        
        X[forward_pass] = ser_fiq.predict(dropout_emb)
           
    norm = normalize(X, axis=1)
    
    # Only get the upper triangle of the distance matrix
    eucl_dist = euclidean_distances(norm, norm)[np.triu_indices(T, k=1)]
   
    # Calculate score as given in the paper
    return embedding, 2*(1/(1+np.exp(np.mean(eucl_dist)))) 
    

def get_arcface_embedding(img_input, use_preprocessing: bool = True):
    """
    Calculate the Arcface/Insightface Embedding of the given image.
    Applies preprocessing if set so.

    Parameters
    ----------
    img_input : numpy ndarray
        Face image.
    use_preprocessing : bool, optional
        If True, preprocessing with Mtcnn is applied. The default is True.

    Returns
    -------
    numpy ndarray (512,)
        The arcface embedding.

    """
    # Apply preprocessing if image is not already aligned
    if use_preprocessing:
        img_input = insightface_model.get_input(img_input)
    
    if img_input is None:
        # No face etc. could be found, no score could be calculated
        return -1.0
    
    embedding, dropout = insightface_model.get_feature(img_input)
    
    return embedding

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
