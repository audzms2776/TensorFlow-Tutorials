import matplotlib.pyplot as plt
import os

# Functions and classes for loading and using the Inception model.
import inception

inception.data_dir = '/tmp/inception/'
inception.maybe_download()
model = inception.Inception()


def classify(image_path):
    # Display the image.
    plt.imshow(plt.imread(image_path))

    # Use the Inception model to classify the image.
    pred = model.classify(image_path=image_path)

    # Print the scores and names for the top-10 predictions.
    model.print_scores(pred=pred, k=10, only_first_name=True)
    plt.show()


image = os.path.join(inception.data_dir, 'cropped_panda.jpg')
classify(image)
