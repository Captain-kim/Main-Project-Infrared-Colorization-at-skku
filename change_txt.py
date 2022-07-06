import os

path_dir = "/media/kimhyeongyu/50AEDF33AEDF0FF8/VSIAD/datasets/Our_NIR_RGB/tar"

file_list = os.listdir(path_dir)
for i in range(len(file_list)):
    file_oldname = os.path.join(path_dir + '/' + file_list[i])
    file_newname = os.path.join(path_dir + '/' + file_list[i] + 'g')
    os.rename(file_oldname, file_newname)

