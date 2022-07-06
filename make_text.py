import os

path_dir = "/media/kimhyeongyu/50AEDF33AEDF0FF8/VSIAD/datasets/Our_NIR_RGB/src"

file_list = os.listdir(path_dir)

f = open("all.txt", "w")
for i in range(len(file_list)):
    f.write(file_list[i] + "\n")

f.close()