# Image Caption Generator

### What is Image to Caption Generator?
Image caption generator is a process of recognizing the context of an image and annotating it with relevant captions using deep learning and computer vision. It includes labeling an image with English keywords with the help of datasets provided during model training. CNN is responsible for image feature extraction. These extracted features will be fed to the LSTM model, which generates the image caption.
### What is CNN?
CNN is a subfield of Deep learning and specialized deep neural networks used to recognize and classify images. It processes the data represented as 2D matrix-like images. CNN can deal with scaled, translated, and rotated imagery. It analyzes the visual imagery by scanning them from left to right and top to bottom and extracting relevant features. Finally, it combines all the parts for image classification.
### What is LSTM?
Being a type of RNN (recurrent neural network), LSTM (Long short-term memory) is capable of working with sequence prediction problems. It is mostly used for the next word prediction purposes, as in Google search our system is showing the next word based on the previous text. Throughout the processing of inputs, LSTM is used to carry out the relevant information and to discard non-relevant information.
* To build an image caption generator model we have to merge CNN with LSTM. We can drive that:
* Image Caption Generator Model (CNN-RNN model) = CNN + LSTM
* CNN – To extract features from the image. A pre-trained model called Xception is used for this.
* LSTM – To generate a description from the extracted information of the image.
### Dataset for Image Caption Generator
The Flickr_8K dataset represents the model training of image caption generators. The dataset is downloaded directly from the below links. The downloading process takes some time due to the dataset’s large size(1GB). In the image below, you can check all the files in the Flickr_8k_text folder. The most important file is Flickr 8k.token, which stores all the image names with captions. 8091 images are stored inside the Flicker8k_Dataset folder and the text files with captions of images are stored in the Flickr_8k_text folder.


### Main Model Architecture:
 
![image](https://github.com/pradeepnamile/Image_caption_generator/assets/162186259/d4348251-e21c-4f65-9a0a-865b2ab8bb1a)

This final model is a combination of CNN and RNN models. To train this model we have to give two inputs two the models. (1) Images (2) Corresponding Captions. For each LSTM layer, we input one word for each LSTM layer, and each LSTM layer predicts the next word, and that how the LSTM model optimizes itself by learning from captions. For Image features, we are getting All image features array from the VGG16 pre-trained model and saved in a file so that we can use this file or features directly to correlate captions and image features with each other. Finally the image features and LSTM last layer we input this both outputs combination into decoder model in which we are adding both image features and captions so that model learns to generate captions from images and for a final layer we generate output or captions which length is the maximum length of dataset captions.
* The last layer has a size of the length of the vocab. For this model, we are using ‘categorical cross-entropy ’ because in the last layer we have to predict each word probability and then we are only using high probability words. We are using Adam optimizer for optimization of the network or update the weights of the network.

### What is import os in Python?
What does import OS do in python? The OS module in python provides functions for interacting with the operating system. OS, comes under Python's standard utility modules. This module provides a portable way of using operating system dependent functionality.

### What is import pickle in Python?
“Pickling” is the process whereby a Python object hierarchy is converted into a byte stream, and “unpickling” is the inverse operation, whereby a byte stream (from a binary file or bytes-like object) is converted back into an object hierarchy.

### What is NumPy library in Python?
 NumPy is a Python library used for working with arrays. It also has functions for working in domain of linear algebra, fourier transform, and matrices. NumPy was created in 2005 by Travis Oliphant. It is an open source project and you can use it freely.

### What is tqdm () in Python?
Tqdm is a popular Python library that provides a simple and convenient way to add progress bars to loops and iterable objects. It gets its name from the Arabic name taqaddum, which means 'progress. 

### Why we import TensorFlow in Python?
TensorFlow is a Python library for fast numerical computing created and released by Google. It is a foundation library that can be used to create Deep Learning models directly or by using wrapper libraries that simplify the process built on top of TensorFlow.
### What is TensorFlow and keras in Python?
Pytorch Vs Tensorflow Vs Keras: The Differences You Should Know
TensorFlow is an open-sourced end-to-end platform, a library for multiple machine learning tasks, while Keras is a high-level neural network library that runs on top of TensorFlow. Both provide high-level APIs used for easily building and training models, but Keras is more user-friendly because it's built-in Python
### What is the use of tokenizer?
Tokenizers transform the text into a list of words, which can be cleaned using a text-cleaning function. Afterward, I used the Keras Tokenizer method to transform the text into an array for analysis and to prepare the tokens for the deep learning model.
### What is VGG16 used for?
VGG-16 is a convolutional neural network that is 16 layers deep. You can load a pretrained version of the network trained on more than a million images from the ImageNet database [1]. The pretrained network can classify images into 1000 object categories, such as keyboard, mouse, pencil, and many animals.
VGG stands for Visual Geometry Group (a group of researchers at Oxford who developed this architecture).

### VGG16 Architecture
VGG16, as its name suggests, is a 16-layer deep neural network. VGG16 is thus a relatively extensive network with a total of 138 million parameters—it’s huge even by today’s standards. However, the simplicity of the VGGNet16 architecture is its main attraction. 
The VGGNet architecture incorporates the most important convolution neural network features. 
 
![image](https://github.com/pradeepnamile/Image_caption_generator/assets/162186259/1b7962e2-5393-4c03-b024-80c181d98733)

 
A VGG network consists of small convolution filters. VGG16 has three fully connected layers and 13 convolutional layers.
Here is a quick outline of the VGG architecture:
Input—VGGNet receives a 224×224 image input. In the ImageNet competition, the model’s creators kept the image input size constant by cropping a 224×224 section from the center of each image.
* Convolutional layers—the convolutional filters of VGG use the smallest possible receptive field of 3×3. VGG also uses a 1×1 convolution filter as the input’s linear transformation. 
* ReLu activation—next is the Rectified Linear Unit Activation Function (ReLU) component, AlexNet’s major innovation for reducing training time. ReLU is a linear function that provides a matching output for positive inputs and outputs zero for negative inputs. VGG has a set convolution stride of 1 pixel to preserve the spatial resolution after convolution (the stride value reflects how many pixels the filter “moves” to cover the entire space of the image).
* Hidden layers—all the VGG network’s hidden layers use ReLU instead of Local Response Normalization like AlexNet. The latter increases training time and memory consumption with little improvement to overall accuracy.
* Pooling layers–A pooling layer follows several convolutional layers—this helps reduce the dimensionality and the number of parameters of the feature maps created by each convolution step. Pooling is crucial given the rapid growth of the number of available filters from 64 to 128, 256, and eventually 512 in the final layers.
* Fully connected layers—VGGNet includes three fully connected layers. The first two layers each have 4096 channels, and the third layer has 1000 channels, one for every class.
### Why do we use VGG16?
VGG16 is a convolutional neural network model that's used for image recognition. It's unique in that it has only 16 layers that have weights, as opposed to relying on a large number of hyper-parameters. It's considered one of the best vision model architectures.

### extract the image features
Dictionary 'features' is created and will be loaded with the extracted features of image data
load_img(img_path, target_size=(224, 224)) - custom dimension to resize the image when loaded to the array
image.reshape((1, image.shape[0], image.shape[1], image.shape[2])) - reshaping the image data to preprocess in a RGB type image.
model.predict(image, verbose=0) - extraction of features from the image
img_name.split('.')[0] - split of the image name from the extension to load only the image name.
Now we split and append the captions data with the image
•	Dictionary 'mapping' is created with key as image_id and values as the corresponding caption text
•	Same image may have multiple captions, if image_id not in mapping: mapping[image_id] = [] creates a list for appending captions to the corresponding image
Caption Text Preprocessing Steps
•	Convert sentences into lowercase
•	Remove special characters and numbers present in the text
•	Remove extra spaces
•	Remove single characters
•	Add a starting and an ending tag to the sentences to indicate the beginning and the ending of a sentence
### Tokenization and Encoded Representation¶
•	The words in a sentence are separated/tokenized and encoded in a one hot representation
•	These encodings are then passed to the embeddings layer to generate word embeddings
![image](https://github.com/pradeepnamile/Image_caption_generator/assets/162186259/c9952592-7dc0-4bbb-b2ec-f8e5fcc36ab3)
### Data Generator for Training
This section introduces a data generator used for training your image captioning model. Data generators are essential for efficiently handling large datasets during model training. Here's an explanation of the data generator:
#### Generator Function
•	The data_generator function is defined to generate training data in batches.
•	It takes the following inputs:
	descriptions: A dictionary containing image descriptions.
	photos: A dictionary containing image features extracted from a pretrained CNN.
	wordtoix: A dictionary mapping words to integer indices.
	max_length: The maximum length of a caption sequence.
	num_photos_per_batch: The number of photos to include in each batch.
#### Batch Generation
•	The function enters an infinite loop, allowing you to continuously generate batches of training data.
•	For each image description in the dataset, it retrieves the corresponding image feature.
•	It then encodes the caption sequence and splits it into multiple input-output pairs.
•	For each pair, it pads the input sequence to match the maximum sequence length and encodes the output sequence as one-hot vectors.
•	The image feature, input sequence, and output sequence are stored in separate lists (X1, X2, and y).
•	Once the specified number of photos per batch (num_photos_per_batch) is reached, the function yields the batch as training data and resets the lists.
•	This process continues in an infinite loop, allowing you to train your model efficiently.
Data generators are useful for avoiding memory limitations when working with large datasets during training. They enable you to train your model batch by batch, improving efficiency and resource management.
### Model Creation

•	shape=(4096,) - output length of the features from the VGG model
•	Dense - single dimension linear layer array
•	Dropout() - used to add regularization to the data, avoiding over fitting & dropping out a fraction of the data from the layers
•	model.compile() - compilation of the model
•	loss=’sparse_categorical_crossentropy’ - loss function for category outputs
•	optimizer=’adam’ - automatically adjust the learning rate for the model over the no. of epochs
•	Model plot shows the concatenation of the inputs and outputs into a single layer
•	Feature extraction of image was already done using VGG, no CNN model was needed in this step.
### Model Training
•	The fit method is called on the model to start the training process.
•	It takes the data generator (generator) as input for training data.
•	The training runs for the specified number of epochs, with each epoch comprising multiple steps.
•	The steps_per_epoch parameter is set to the calculated number of training steps per epoch.
•	The verbose parameter controls the verbosity of training output.
•	steps = len(train) // batch_size - back propagation and fetch the next data
•	Loss decreases gradually over the iterations
•	Increase the no. of epochs for better results
•	Assign the no. of epochs and batch size accordingly for quicker results
### Caption Generation Utility Functions
•	Utility functions to generate the captions of input images at the inference time.
•	Here the image embeddings are passed along with the first word, followed by which the text embedding of each new word is passed to generate the next word
•	Captiongenerator appending all the words for an image
•	The caption starts with 'startseq' and the model continues to predict the caption until the 'endseq' appeared
### Conclusion :
Finally, conclude this project we understand VGG16 model Architecture, Long short term memory Network, how to combine this both model, bleu score, How the LSTM network generates captions, How the VGG16 model we can use for our project, and how to generate captions from images using deep learning.


