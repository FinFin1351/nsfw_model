#! python

import argparse
import json
import requests
from os import listdir
from os.path import isfile, join, exists, isdir, abspath
from PIL import Image
from io import BytesIO

import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow_hub as hub


IMAGE_DIM = 224   # required/default image dimensionality

def load_img_from_url(url, target_size=None):
    try:
        # Download the image from the URL
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for 4xx and 5xx errors

        # Open the downloaded image using PIL
        img = Image.open(BytesIO(response.content))

        # Resize the image if target_size is provided
        if target_size is not None:
            img = img.resize(target_size)

        # Convert the PIL image to a NumPy array
        img_array = keras.processing.image.img_to_array(img)

        return img_array

    except Exception as e:
        # Handle other unexpected errors
        print(f"An error occurred while loading the image from URL: {url}")
        print(e)

    return None  # Return None if there's an error

def load_images(image_urls, image_size, verbose=True):
    '''
    Function for loading images into numpy arrays for passing to model.predict
    inputs:
        image_paths: list of image paths to load
        image_size: size into which images should be resized
        verbose: show all of the image path and sizes loaded
    
    outputs:
        loaded_images: loaded images on which keras model can run predictions
        loaded_image_indexes: paths of images which the function is able to process
    
    '''
    loaded_images = []
    loaded_image_paths = []

    if type(image_urls) is list:
        pass
        # parent = abspath(image_paths)
        # image_paths = [join(parent, f) for f in listdir(image_paths) if isfile(join(parent, f))]
    else:
        image_datas = [image_urls]

    responses = [requests.get(url) for url in image_urls]
    image_datas = [response.content for response in responses if response.status_code == 200]
        
    for img_data in image_datas:
        try:
            if verbose:
                print(img_data, "size:", image_size)
            if type(img_data) != str:
                # image = keras.preprocessing.image.load_img(img_data, target_size=image_size)
                image = keras.preprocessing.image.img_to_array(img_data)
            else:
                image = load_img_from_url(img_data)
            if image is not None:
                image /= 255
                loaded_images.append(image)
                loaded_image_paths.append(img_data)
        except Exception as ex:
            print("Image Load Failure: ", img_data, ex)
    
    return np.asarray(loaded_images), loaded_image_paths

def load_model(model_path):
    if model_path is None or not exists(model_path):
    	raise ValueError("saved_model_path must be the valid directory of a saved model to load.")
    
    model = tf.keras.models.load_model(model_path, custom_objects={'KerasLayer': hub.KerasLayer},compile=False)
    return model


def classify(model, input_paths, image_dim=IMAGE_DIM, predict_args={}):
    """
    Classify given a model, input paths (could be single string), and image dimensionality.
    
    Optionally, pass predict_args that will be passed to tf.keras.Model.predict().
    """
    images, image_paths = load_images(input_paths, (image_dim, image_dim))
    probs = classify_nd(model, images, predict_args)
    return dict(zip(image_paths, probs))


def classify_nd(model, nd_images, predict_args={}):
    """
    Classify given a model, image array (numpy)
    
    Optionally, pass predict_args that will be passed to tf.keras.Model.predict().
    """
    model_preds = model.predict(nd_images, **predict_args)
    # preds = np.argsort(model_preds, axis = 1).tolist()
    
    categories = ['drawings', 'hentai', 'neutral', 'porn', 'sexy']

    probs = []
    for i, single_preds in enumerate(model_preds):
        single_probs = {}
        for j, pred in enumerate(single_preds):
            single_probs[categories[j]] = float(pred)
        probs.append(single_probs)
    return probs


def main(args=None):
    parser = argparse.ArgumentParser(
        description="""A script to perform NFSW classification of images""",
        epilog="""
        Launch with default model and a test image
            python nsfw_detector/predict.py --saved_model_path mobilenet_v2_140_224 --image_source test.jpg
    """, formatter_class=argparse.RawTextHelpFormatter)
    
    submain = parser.add_argument_group('main execution and evaluation functionality')
    submain.add_argument('--image_source', dest='image_source', type=str, required=True, 
                            help='A directory of images or a single image to classify')
    submain.add_argument('--saved_model_path', dest='saved_model_path', type=str, required=True, 
                            help='The model to load')
    submain.add_argument('--image_dim', dest='image_dim', type=int, default=IMAGE_DIM,
                            help="The square dimension of the model's input shape")
    if args is not None:
        config = vars(parser.parse_args(args))
    else:
        config = vars(parser.parse_args())

    if config['image_source'] is None or not exists(config['image_source']):
    	raise ValueError("image_source must be a valid directory with images or a single image to classify.")
    
    model = load_model(config['saved_model_path'])    
    image_preds = classify(model, config['image_source'], config['image_dim'])
    print(json.dumps(image_preds, indent=2), '\n')


if __name__ == "__main__":
	main()
