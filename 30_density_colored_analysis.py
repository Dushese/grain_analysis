import os
import pickle

import cv2 as cv
from sklearn.cluster import KMeans
import time

from analysis_functional import *


def name_files_30_density_colored():
    '''
    Подает на вход путь к файлу
    :return:
    '''
    folder_name = "./grains_density_30_colored/img-"
    # folder_name = "C:/Users/Dushese/files_for_jupiter/grains_density_30_colored/img-"
    circle = [num for num in range(10, 90, 10)] + [num for num in range(89, 114)]
    for folder_1 in circle:
        for img_ind_1 in [4, 5]:
            if img_ind_1 == 4:
                indexes = [x for x in range(2, 10)]
            if img_ind_1 == 5:
                indexes = [x for x in range(0, 9)]
            for img_ind_2 in indexes:
                # Clusterisation
                yield folder_name + str(folder_1) + '-' + '49-2550-0.09-0.03/z' + \
                    str(img_ind_1) + '.' + str(img_ind_2) + '.png'


def change_color(color_flag):
    return 255 if color_flag == 1 else 0


def find_markers(clstrs: int, img_name, clusters_matrix, shape):

    result_markers = np.zeros((shape, shape))
    result_image = np.zeros((shape, shape))
    kernel_erode = np.ones((5, 5), np.uint8)
    # kernel_dilate = np.ones((3, 3), np.uint8)
    flag = 0
    for x in range(clstrs):
      amount_markered_in_pix, marker_matrix, cluster_image = show_cluster(clusters_matrix, shape, shape, x)
      if amount_markered_in_pix / (shape * shape) > 0.5:  # если кол-во точек,
                                                           # принадлежащих одному кластеру больше 60% от
                                                           # всей картинки, то это фон, он нам не нужен
        # print('Номер маркера и его доля', x, amount_markered_in_pix / (shape * shape))
        if x == 0:
            continue
        else:
            print('Смена сида инициализации')
            return None, None
      marker_matrix = cv.erode(marker_matrix, kernel_erode, iterations=2)
      result_markers += marker_matrix

      cluster_image = cv.erode(cluster_image, kernel_erode, iterations=2)
      result_image += cluster_image



    # Подготовка к watershed

    img = cv.imread(img_name, cv.IMREAD_GRAYSCALE)
    ret, thresh = cv.threshold(img, 69, 255, cv.THRESH_BINARY_INV)

    # opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel, iterations = 1)
    kernel = np.ones((3, 3), np.uint8)
    sure_bg = cv.dilate(thresh, kernel, iterations=30)

    sure_fg = np.uint8(result_image)
    unknown = cv.subtract(sure_bg, sure_fg)
    # watershed
    img_watershed = cv.imread(img_name)
    img_result = np.copy(img_watershed)
    img_coord = np.copy(img_watershed)
    result_markers = np.array(result_markers, dtype=np.int32)
    result_markers += 1
    result_markers[unknown > 50] = 0
    markers = cv.watershed(img_watershed, result_markers)
    img_watershed[markers == -1] = [255, 0, 0]

    # поиск координат
    # img_coord[::] = [0, 0, 0]
    # print(markers)
    # for i in range(np.shape(thresh)[0] - 1):
    #     for j in range(1, np.shape(thresh)[0] - 1):
    #         # print(color_flag, thresh[i, j], thresh[i, j] == 255)
    #         if markers[i, j] == -1:
    #             if markers[i, j + 1] != -1:
    #                 markers[i, j + 1] = -1
    #             elif markers[i + 1, j] != -1:
    #                 markers[i + 1, j] = -1
    #             markers[i, j + 1] = [255, 0, 0]
    # for i in range(np.shape(thresh)[0] - 1):
    #     for j in range(1, np.shape(thresh)[0] - 1):
    #         if markers[i, j] > 1:
    #             img_coord[i, j] = [255, 0, 0]

    # img_coord[markers == -1] = [255, 0, 0]
    gray = cv.cvtColor(img_coord, cv.COLOR_BGR2GRAY)

    ret, thresh = cv.threshold(gray, 0, 255, cv.THRESH_OTSU)
    # plt.imshow(thresh)
    # plt.show()
    # th = np.copy(thresh)
    # for i in range(np.shape(thresh)[0]):
    #     color_flag = 1
    #     # print('----------------')
    #     for j in range(1, np.shape(thresh)[0] - 1):
    #         # print(color_flag, thresh[i, j], thresh[i, j] == 255)
    #         if thresh[i, j] == 255 and thresh[i, j + 1] != 255:
    #             # print('CHANGE')
    #             color_flag *= -1
    #         else:
    #             thresh[i, j] = change_color(color_flag)


    contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_TC89_L1)

    img_tst = cv.imread(img_name)
    img_tst = cv.drawContours(img_tst, contours, -1, (0, 0, 255), 1)

    # plt.imshow(img_tst)
    # plt.show()

    img_tst = cv.imread(img_name)
    i = 0
    for cnt in contours:
        # if hierarchy[0, i, 2] == -1:
        #     test = np.copy(img_tst)
        #     approx = cv.approxPolyDP(cnt, 0.0001 * cv.arcLength(cnt, True), True)
        #     print(i, hierarchy[0, i])
        #     # draws boundary of contours.
        #     cv.drawContours(test, [approx], 0, (255, 255, 0), 1)
        #     plt.imshow(test)
        #     plt.show()
        i += 1
    # plt.imshow(img_tst)
    # plt.show()
    new_cntrs = [contours[ind] for h, ind in zip(hierarchy[0], range(len(hierarchy[0]))) if h[2] == -1]

    approxed_arr = approximation(new_cntrs)

    img_result = drow_vertices(approxed_arr, img_result)

    # plt.imshow(img_result)
    # plt.show()
    return approxed_arr, img_result #approxed_arr, img # Тут не аппроксимирую, чтобы потом поварьировать коэф аппроксимации


t = time.time()
density_sum = 0
average_grain_area_sum = 0
all_grains_amount = 0
all_angle_amount = 0
amount_images = 0

all_areas = []
all_angles = []
all_perimeters = []
all_grains = {}

CLSTRS = 4

angle_dist = {}


# цикл по изображениям

#
area_dist = {}
prev = 0
intervals = map(int, np.linspace(0, 120000, 10))
for point in intervals:
    area_dist[f'{prev}_{point}'] = []
    prev = point

for img_name in name_files_30_density_colored():
    # C:/Users/an.v.potapov/PycharmProjects/grain_analysis/grains_density_30_colored/img-40-49-2550-0.09-0.03/z5.3.png
    # img_name = "C:/Users/an.v.potapov/PycharmProjects/grain_analysis/grains_density_30_colored/img-40-49-2550-0.09-0.03/z5.3.png"
    print(img_name)
    if not os.path.isfile(img_name):
        print('Такого файла нет!')
        continue
    image_base = cv.imread(img_name)
    amount_images += 1  # подсчитать кол-во снимков
    random_state = 4
    while True:
        img = image_base.reshape((image_base.shape[0] * image_base.shape[1], 3))
        common_params = {
            "init": 'k-means++',
            "n_init": "auto",
            "random_state": random_state
        }
        y_pred = KMeans(n_clusters=CLSTRS, **common_params).fit_predict(img)#cv.medianBlur(image_base,5))

        clusters_matrix = y_pred.reshape(image_base.shape[0], image_base.shape[1])

        approxed_arr, segmented_image = find_markers(CLSTRS, img_name, clusters_matrix, image_base.shape[0])
        if approxed_arr is not None:
            break
        random_state += 1

    if len(approxed_arr) < 20 or len(approxed_arr) > 1000: # аномальное кол-во зерен
        amount_images -= 1
        print(f'кол-во найденных зерен {len(approxed_arr)} => не учитывается в статистике')
        continue
    print(f'Кол-во найденных зерен: {len(approxed_arr)}')

    all_grains[amount_images] = [[el[0] for el in arr] for arr in approxed_arr]
    angles, approxed_arr, density, average_grain_area, areas, perimeters, area_dist_on_img = count_characteristics(np.shape(image_base)[0], np.shape(image_base)[1], approxed_arr, 1, image_base)

    for key in area_dist.keys():
        area_dist[key].append(area_dist_on_img[key])
    # image_segmented_statistic = drow_vertices(approxed_arr, image_base)
    # f = plt.figure(figsize=(9, 18))
    # seg1 = f.add_subplot(121)
    # seg2 = f.add_subplot(122)
    # seg1.imshow(image_base)
    # seg2.imshow(image_segmented_statistic)
    # plt.show()


    # if amount_images % 20 == 1:
    #     cv.imwrite(f'30_density_colored_images/{amount_images}_segmented_statistic.png', image_segmented_statistic)
    #     cv.imwrite(f'30_density_colored_images/{amount_images}_segmented.png', segmented_image)
    #     cv.imwrite(f'30_density_colored_images/{amount_images}.png', image_base)


# print(f'Общие данные \n\tСредняя плотность зерен = {density_sum / amount_images}\n'
#       f'\tСредний размер зерна = {average_grain_area_sum / amount_images}')
# print(f'\n Epsilon = {eps}')
print(f'Кол-во обработанных изображений: {amount_images}')
print(f'Общее время выполнения: {time.time() - t}')

# np.save(f'30_density_colored_distributions/all_areas.npy', np.array(all_areas))
# np.save(f'30_density_colored_distributions/all_angles.npy', np.array(all_angles))
# np.save(f'30_density_colored_distributions/all_perimeters.npy', np.array(all_perimeters))

# with open('30_density_colored_distributions/all_grains.pkl', 'wb') as f:
#     pickle.dump(all_grains, f)

with open('30_density_colored_distributions/area_dist.pkl', 'wb') as f:
    pickle.dump(area_dist, f)
