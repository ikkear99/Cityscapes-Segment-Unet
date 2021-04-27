import os

path = "E:\Dien_AI\Segment_Citydata\data"
x_train_path = os.path.join(path,"leftImg8bit/train")
print(x_train_path)
images = []
j = 0
for i,(dirpath, dirnames, filenames) in enumerate(os.walk(x_train_path)):
    if dirpath is not x_train_path:
        i = 0
        for f in filenames:
            i = i + 1
            print(f"{type(f)}")
            names = os.path.join(dirpath, f"{f}")
            images.append(names)

        j = j + i


for names in images:
    print(names)
print("end program")