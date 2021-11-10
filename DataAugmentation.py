import os
import shutil
import tensorflow as tf
import numpy as np


def startAugmentation():
    targetnames = ['akiec', 'bcc', 'bkl', 'mel', 'nv']

    # Augmenting images and storing them in temporary directories
    for img_class in targetnames:

        # creating temporary directories
        # creating a base directory
        aug_dir = 'aug_dir'
        os.mkdir(aug_dir)
        # creating a subdirectory inside the base directory for images of the same class
        img_dir = os.path.join(aug_dir, 'img_dir')
        os.mkdir(img_dir)

        img_list = os.listdir('HAM10000/train_dirr/' + img_class)

        # Copy images from the class train dir to the img_dir
        for file_name in img_list:
            # path of source image in training directory
            source = os.path.join('HAM10000/train_dirr/' + img_class, file_name)

            # creating a target directory to send images
            target = os.path.join(img_dir, file_name)

            # copying the image from the source to target file
            shutil.copyfile(source, target)

        # Temporary augumented dataset directory.
        source_path = aug_dir

        # Augmented images will be saved to training directory
        save_path = 'HAM10000/train_dirr/' + img_class

        # Creating Image Data Generator to augment images
        datagen = tf.keras.preprocessing.image.ImageDataGenerator(

            rotation_range=180,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True,
            vertical_flip=True,
            fill_mode='nearest'

        )

        batch_size = 50

        aug_datagen = datagen.flow_from_directory(source_path, save_to_dir=save_path, save_format='jpg',
                                                  target_size=(224, 224), batch_size=batch_size)

        # Generate the augmented images
        aug_images = 8000

        num_files = len(os.listdir(img_dir))
        num_batches = int(np.ceil((aug_images - num_files) / batch_size))

        # creating 8000 augmented images per class
        for i in range(0, num_batches):
            images, labels = next(aug_datagen)

        # delete temporary directory
        shutil.rmtree('aug_dir')
