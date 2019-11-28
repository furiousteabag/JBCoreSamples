"""Cutting photos and saving them to folders.

Methods to cut all files in folder and put
them into new folders.

  Typical usage example:

  prepare_photos(df, DATA_PATH + "prepared data/rock/cut photos/", 0.2)
  replace_text_in_file_names(DATA_PATH + "prepared data/rock/cut photos/",
                             dict_replace)
"""


import os
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
from importlib import reload
import data_oversampling
import dataframe_processing

reload(data_oversampling)
reload(dataframe_processing)


DATA_PATH = "../data/huge data/"


def display_image_list(image_list: list):
    """Displays image list one under another."""

    y_len = len(image_list)   # Number of plots on y axe.
    cur_column = 0            # Columns iterator.

    fig, ax = plt.subplots(y_len, figsize=(2, 10))

    for i in range(len(y_len)):
        im = image_list[i]
        ax[i].imshow(im)
        ax[i].axis('off')
        cur_column += 1

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()

    return True


def cut_image_into_list(path: str, patter_height_m=0.2, photo_height_m=1.1):
    """Cuts image into list of images.

    Args:
        patter_height_m (float): Height of 1 photo in output.
        photo_height_m (float): Height of given photo.

    Returns:
        list<Image>: List of photos with patter_height_m heigh.
    """

    image = Image.open(path)
    w, h = image.size
    number_of_images = int(photo_height_m // patter_height_m)

    pixels_per_image = (patter_height_m * h) // photo_height_m

    image_list = []
    for i in range(number_of_images):
        image = Image.open(path)
        image = image.crop((0, i * pixels_per_image,
                            w, (i + 1) * pixels_per_image))
        image_list.append(image)

    return image_list


def save_image_array(path, image_list, unload, number, top, UV,
                     rock, carbonate, ruin, saturation, patter_height_m):
    """Saves images in given folders with given names."""

    for i in range(len(image_list)):
        image_list[i].save(
            path + UV + "/{}@{}@{}@{}@{}@{}@{}@{}@{}.jpg".format(
                unload,
                number,
                format(
                    (top + (i * patter_height_m)), '.1f'),
                format(
                    (top + ((i + 1) * patter_height_m)), '.1f'),
                UV,
                rock,
                carbonate,
                ruin,
                saturation))
    return True


def prepare_photos(dataframe: pd.DataFrame, path_to_save, patter_height_m):
    """Cutting all photos and saving them.

    Args:
        path_to_save (string): Path that leads to folder
            with 2 folders: "ДС" and "УФ".
        photo_height_m (float): Height of given photo.

    Returns:
        list<Image>: List of photos with patter_height_m heigh.
    """

    for i in range(len(dataframe)):
        params = list(dataframe.iloc[i, :])
        image_list = cut_image_into_list(path=(DATA_PATH + "raw data/" +
                                               params[0] + "/data/" +
                                               str(params[1]) + ".jpeg"),
                                         patter_height_m=patter_height_m,
                                         photo_height_m=params[11])
        save_image_array(path=path_to_save,
                         image_list=image_list,
                         unload=params[0],
                         number=params[1],
                         top=params[9],
                         UV=params[12],
                         rock=params[15],
                         carbonate=params[16],
                         ruin=params[17],
                         saturation=params[18],
                         patter_height_m=0.2)

    return True


def replace_text_in_file_names(pictures_path, dict_replace):
    """
    Args:
        pictures_path (string): Path to folder, containing 2 subfolders.
        dict_replace (dictionary): What to change in names.
    """

    # Filling list with photo names.
    folders = os.listdir(pictures_path)
    photo_list = []
    for folder in folders:
        photo_list.extend(os.listdir(pictures_path + folder))
    photo_list.sort()

    # Filling list paths.
    paths = []
    for picture_name in photo_list:
        UV_folder = "УФ" if "УФ" in picture_name else "ДС"
        paths.append((pictures_path + UV_folder + "/" + picture_name))

    # Filling target photos names.
    target_paths = []
    for i in range(len(paths)):
        item = paths[i]
        for word, initial in dict_replace.items():
            item = item.replace(word, initial)
        target_paths.append(item)

    # Renaming files.
    for i in range(len(paths)):
        os.renames(paths[i], target_paths[i])

    return True
