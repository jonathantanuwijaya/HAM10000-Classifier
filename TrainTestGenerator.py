import shutil
import os


def generateTrainTestDir(train_dir, test_dir, data_pd, train_list, test_list):
    targetnames = ['akiec', 'bcc', 'bkl', 'mel', 'nv']

    for i in targetnames:
        directory1 = train_dir + '/' + i
        directory2 = test_dir + '/' + i
        os.mkdir(directory1)
        os.mkdir(directory2)

    for image in train_list:
        file_name = image + '.jpg'
        label = data_pd.loc[image, 'dx']

        # path of source image
        source = os.path.join('HAM10000', file_name)

        # copying the image from the source to target file
        target = os.path.join(train_dir, label, file_name)

        shutil.copyfile(source, target)

    for image in test_list:
        file_name = image + '.jpg'
        label = data_pd.loc[image, 'dx']

        # path of source image
        source = os.path.join('HAM10000', file_name)

        # copying the image from the source to target file
        target = os.path.join(test_dir, label, file_name)

        shutil.copyfile(source, target)
