# import numpy as np
# import cv2
# import csv

# def get_x_y(filename):
#     img_files = []
#     mask_files = []

#     filenames = []
#     with open(filename, newline='') as f:
#         reader = csv.reader(f)
#         filenames = list(reader)
#     for i in range(len(filenames)):
#         file = filenames[i][0]
#         img_files.append('./data/' + file[2:])
#         mask_name = file.replace('/img/', '/masks/')
#         mask_files.append('./data/' + mask_name[2:])
#     return np.asarray(img_files), np.asarray(mask_files)

# X, y = get_x_y('data/test.csv')


# ct = 0
# for i, file in enumerate(y):
#     if i % 100 == 0:
#         print(i, '/', len(X))
#     old_file = file[:16] + '_big' + file[16:]
#     img = cv2.imread(old_file)
#     img = cv2.resize(img, (960, 640), interpolation=cv2.INTER_NEAREST)

#     cv2.imwrite(file, img)
