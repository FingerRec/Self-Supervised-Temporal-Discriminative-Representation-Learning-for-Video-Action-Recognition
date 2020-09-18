import os
import shutil

file_path = "C:/下载"
dest_path = "C:/下载/all"
count = 0
for dir in os.listdir(file_path):
    for subdir in os.listdir(os.path.join(file_path, dir)):
        for subsubdir in os.listdir(os.path.join(file_path, dir, subdir)):
            for subsubsubdir in os.listdir(os.path.join(file_path, dir, subdir, subsubdir)):
                count += 1
                for image in os.listdir(os.path.join(file_path, dir, subdir, subsubdir, subsubsubdir)):
                    shutil.copy(os.path.join(file_path, dir, subdir, subsubdir, subsubsubdir, image), dest_path + "/{}_{}".format(count, image))