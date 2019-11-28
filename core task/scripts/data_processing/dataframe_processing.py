"""Preparing photos to training.

Methods to create dataframe from file name's and put them into
train/validation/test folders.

  Typical usage example:

      pictures_df = dp.create_df_from_picture_names(DATA_PATH +
                                "prepared data/rock/cut photos/")

      pictures_path = DATA_PATH + "prepared data/rock/cut photos/"
      target_path = DATA_PATH + "prepared data/rock/training data/"
      category_column = "Rock"

      prepare_train_folders(
                      pictures_df=pictures_df,
                      pictures_path=pictures_path,
                      target_path=target_path,
                      category_column=category_column)
"""

import shutil
import os
import pandas as pd


def create_df_from_picture_names(df, pictures_path):
    """
    Args:
        pictures_path (string): Path to folder that
            contains 2 folders: UV and non-UV.
    """

    # Getting names of UV and non-UV folder.
    folders = os.listdir(pictures_path)

    # Filling list with photo names.
    photo_list = []
    for folder in folders:
        photo_list.extend(os.listdir(pictures_path + folder))
    photo_list.sort()

    # Getting the columns for our dataframe.
    columns = [df.columns[0], df.columns[1], df.columns[9],
               df.columns[10], df.columns[12], df.columns[15],
               df.columns[16], df.columns[17], df.columns[18]]
    pictures_df = pd.DataFrame(columns=columns)

    # Iterating throw all photo's names and adding their params to dataframe.
    for i in range(len(photo_list)):
        photo_params = photo_list[i].split("@")
        photo_params[-1] = photo_params[-1][:-4]
        pictures_df.loc[i] = photo_params

    return pictures_df


def prepare_train_folders(pictures_df, pictures_path, target_path,
                          category_column, proportions=[0.7, 0.2, 0.1]):
    """Copying files to train/val/test folders with given proportions.
    Args:
        pictures_path (string): Path to folder that
            contains UV and non-UV folder.
        target_path (string): Path to folder that
            contains folders UV, non-UV, UV+non-UV.
        category (string): Column name.
        proportions (list<float>): Proportion of train/validation/test.
    """

    # List of folders that will be created (each folder --- category).
    categories = list(pictures_df[category_column].unique())

    # Creating folders for each category.

    # Folder containing UV and non-UV folders.
    uv_nonuv_folders = os.listdir(target_path)
    # List with test_val_train folders.
    test_val_train_folders = ["test", "validation", "train"]

    # Making a list of paths where to make dirs.
    paths_where_to_make_category_dirs = []
    for uv_folder in uv_nonuv_folders:
        for train_folder in test_val_train_folders:
            paths_where_to_make_category_dirs.append(target_path+uv_folder+"/"+train_folder+"/")

    # Making dirs.
    for where_to_make_dirs in paths_where_to_make_category_dirs:
        for category_name in categories:
            os.makedirs(where_to_make_dirs+category_name, exist_ok=True)

    # Shuffling df.
    pictures_df = pictures_df.sample(frac=1).reset_index(drop=True)

    # List of datarframes.
    # Each dataframe contains elements only from 1 category.
    list_of_df_category = []
    for category in categories:
        list_of_df_category.append(pictures_df[(pictures_df[category_column] == category)])

    # Iterating throw all dataframes and putting images into correct folders.
    for i in range(len(list_of_df_category)):

        # Number of samples in category.
        length = len(list_of_df_category[i])

        # Dataframes for 1 type.
        train_df = list_of_df_category[i].iloc[:int(length*proportions[0]), :]
        validation_df = list_of_df_category[i].iloc[int(
            length*proportions[0]):int(length*(proportions[0] + proportions[1])), :]
        test_df = list_of_df_category[i].iloc[int(length*(proportions[0] + proportions[1])):, :]

        print("'{}' category processing started. \n Train samples: {} \n Validation samples: {} \n Test samples: {}".format(
            categories[i], len(train_df), len(validation_df), len(test_df)))

        # Picture names from dataframes.
        train_pictures_names = []
        validation_pictures_names = []
        test_pictures_names = []
        for a in range(len(train_df)):
            params = list(train_df.iloc[a, :])
            picture_name = '@'.join(map(str, params)) + ".jpg"
            train_pictures_names.append(picture_name)

        for b in range(len(validation_df)):
            params = list(validation_df.iloc[b, :])
            picture_name = '@'.join(map(str, params)) + ".jpg"
            validation_pictures_names.append(picture_name)

        for c in range(len(test_df)):
            params = list(test_df.iloc[c, :])
            picture_name = '@'.join(map(str, params)) + ".jpg"
            test_pictures_names.append(picture_name)

        # Picture paths.
        train_pictures_paths = []
        validation_pictures_paths = []
        test_pictures_paths = []

        for picture_name in train_pictures_names:
            UV_folder = "УФ" if "УФ" in picture_name else "ДС"
            train_pictures_paths.append(pictures_path + UV_folder + "/" + picture_name)

        for picture_name in validation_pictures_names:
            UV_folder = "УФ" if "УФ" in picture_name else "ДС"
            validation_pictures_paths.append(pictures_path + UV_folder + "/" + picture_name)

        for picture_name in test_pictures_names:
            UV_folder = "УФ" if "УФ" in picture_name else "ДС"
            test_pictures_paths.append(pictures_path + UV_folder + "/" + picture_name)

        # Copying pictures.
        for picture_path in train_pictures_paths:
            UV_folder = "UV" if "УФ" in picture_path else "non-UV"
            copy_dir = target_path + UV_folder + "/" + "train/" + categories[i] + "/"
            shutil.copy(picture_path, copy_dir)

        for picture_path in validation_pictures_paths:
            UV_folder = "UV" if "УФ" in picture_path else "non-UV"
            copy_dir = target_path + UV_folder + "/" + "validation/" + categories[i] + "/"
            shutil.copy(picture_path, copy_dir)

        for picture_path in test_pictures_paths:
            UV_folder = "UV" if "УФ" in picture_path else "non-UV"
            copy_dir = target_path + UV_folder + "/" + "test/" + categories[i] + "/"
            shutil.copy(picture_path, copy_dir)

        print("Done copying all photos from category '{}'".format(categories[i]))

    for dirpath, dirnames, filenames in os.walk(target_path):
        if (len(filenames) != 0):
            print("Files in", dirpath, len(filenames))

    return True
