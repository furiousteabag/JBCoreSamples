"""Oversampling photos.

Methods to count all files in folders and
oversample folders with smaller number of photos.

  Typical usage example:

  path = "../data/huge data/prepared data/rock/training data/non-UV/"
  oversample_train_val_test(path)
"""

import os
import shutil


def folder_oversample(path: str, num: int) -> bool:
    """
    Args:
        num: Final number of files in folder.
    """

    filenames = os.listdir(path)
    number_of_files = len(filenames)

    if (number_of_files >= num):
        print("There's more files in folder than your number!")
        return False
    else:
        while (True):
            for filename in filenames:
                src = os.path.join(path, filename)

                # Cuts .jpg or .png.
                dst = os.path.join(path,
                                   filename[:-4]
                                   + "_" + str(number_of_files) + ".jpg")

                shutil.copy(src, dst)
                number_of_files += 1

                if (number_of_files == num):
                    break
            if (number_of_files == num):
                break

        print("Directory {} successfully oversampled to {}.".format(path, num))
        return True


def find_folder_with_biggest_number_of_samples(path: str) -> int:
    """Finding folder with biggest number of elements in it.
    Args:
        path: Path to several folders.
    Returns:
        max_num: Biggest number of elements in directories.
    """

    directories = [f.path for f in os.scandir(path) if f.is_dir()]
    max_num = 0

    for directory in directories:
        number_of_files = len(os.listdir(directory))
        if number_of_files > max_num:
            max_num = number_of_files

    return max_num


def oversample_all_folders(path: str) -> bool:
    """Oversampling all subfolders with biggest number of elements in it.
    Args:
        path: Path to several folders that are need
            to be equalized with number of elements.
    """

    max_num = find_folder_with_biggest_number_of_samples(path)
    directories = [f.path for f in os.scandir(path) if f.is_dir()]

    for directory in directories:
        folder_oversample(directory, max_num)

    return True


def oversample_train_val_test(path: str) -> bool:
    """Oversampling all subfolders in train/val/test folders.
    Args:
        path: Path to folder, containing
            training/validation/test folders.
    """

    directories = [f.path for f in os.scandir(path) if f.is_dir()]

    for directory in directories:
        oversample_all_folders(directory)
        print("Finished oversampling train/val/test folders ({}).".format(path))

    return True
