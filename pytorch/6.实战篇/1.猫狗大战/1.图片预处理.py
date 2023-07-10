import os
import shutil
print(os.getcwd() )

# ### create dog directory and cat directory
train_root = 'kaggle_dog_vs_cat/data/train'
dog_folder = os.path.join(train_root, 'dog')
cat_folder = os.path.join(train_root, 'cat')
os.mkdir(dog_folder)
os.mkdir(cat_folder)

val_root = 'kaggle_dog_vs_cat/data/val'
dog_folder = os.path.join(val_root, 'dog')
cat_folder = os.path.join(val_root, 'cat')
os.mkdir(dog_folder)
os.mkdir(cat_folder)

# ### move the dog pictures and cat pictures into dog directory and cat directory respectively
data_file = os.listdir('kaggle_dog_vs_cat/zip')
dog_file = list(filter(lambda x: x[:3]=='dog', data_file))
cat_file = list(filter(lambda x: x[:3]=='cat', data_file))

root = 'kaggle_dog_vs_cat/'
for i in range(len(dog_file)):
    pic_path = root + 'zip/' + dog_file[i]
    if i < len(dog_file)*0.9:
        obj_path = train_root + '/dog/' + dog_file[i]
    else:
        obj_path = val_root + '/dog/' + dog_file[i]
    shutil.move(pic_path, obj_path)

for i in range(len(cat_file)):
    pic_path = root + 'zip/' + cat_file[i]
    if i < len(dog_file)*0.9:
        obj_path = train_root + '/cat/' + cat_file[i]
    else:
        obj_path = val_root + '/cat/' + cat_file[i]
    shutil.move(pic_path, obj_path)
