import cv2 as cv
from sklearn.cluster import KMeans
import time
from analysis_functional import *


def name_files_50_density():
    '''
    Подает на вход путь к файлу
    :return:
    '''
    folder_name = "C:/Users/Dushese/files_for_jupiter/3400/img-"
    for folder_1 in range(10, 80, 10):
        for folder_2 in range(51, 55):
            for img_ind_1 in [4, 5]:
                if img_ind_1 == 4:
                    indexes = [x for x in range(2, 10)]
                else:
                    indexes = [x for x in range(0, 9)]
                for img_ind_2 in indexes:
                    yield folder_name + str(folder_1) + '-' + str(folder_2) + '-3400-0.08-0.03/z' + str(
                        img_ind_1) + '.' + str(img_ind_2) + '.png'


t = time.time()

density_sum = 0
average_grain_area_sum = 0
all_grains_amount = 0
all_angle_amount = 0
amount_images = 0
angle_dist = {}

area_dist = {}

clstrs = 110

# цикл по изображениям
for img_name in name_files_50_density():
    amount_images += 1 # подсчитать кол-во снимков
    print(img_name)
    image_base = cv.imread(img_name)
    img = image_base.reshape((image_base.shape[0] * image_base.shape[1], 3))
    common_params = {
        "init": 'k-means++',
        "n_init": "auto",
        "random_state": 4
    }
    y_pred = KMeans(n_clusters=clstrs, **common_params).fit_predict(img)#cv.medianBlur(image_base,5))

    clusters_matrix = y_pred.reshape(image_base.shape[0], image_base.shape[1])


    result_markers = np.zeros((image_base.shape[0], image_base.shape[1]))
    result_image = np.zeros((image_base.shape[0], image_base.shape[1]))
    kernel_erode = np.ones((5, 5), np.uint8)
    kernel_dilate = np.ones((3, 3), np.uint8)
    flag = 0
    for x in range(clstrs):
      amount_markered_in_pix, marker_matrix, cluster_image = show_cluster(clusters_matrix, image_base.shape[0], image_base.shape[1], x)
      if amount_markered_in_pix > image_base.shape[0] * image_base.shape[1] / 4: # если кол-во точек, принадлежащих одному кластеру равно четверти всей картинке и больше, то это фон, он нам не нужен
        continue
      marker_matrix = cv.erode(marker_matrix, kernel_erode, iterations=2)
      result_markers += marker_matrix

      cluster_image = cv.erode(cluster_image, kernel_erode, iterations=2)
      result_image += cluster_image
    # cv.imshow('cvcv', result_image)
    # cv.waitKey()
    # cv.destroyAllWindows()

    # Подготовка к watershed

    img = cv.imread(img_name, cv.IMREAD_GRAYSCALE)
    ret, thresh = cv.threshold(img, 69, 255, cv.THRESH_BINARY_INV)
    # opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel, iterations = 1)
    kernel = np.ones((3, 3), np.uint8)
    sure_bg = cv.dilate(thresh, kernel, iterations=3)
    sure_fg = np.uint8(result_image)
    unknown = cv.subtract(sure_bg, sure_fg)

    # watershed
    img_real = cv.imread(img_name)
    result_markers = np.array(result_markers, dtype=np.int32)
    result_markers += 1
    result_markers[unknown > 50] = 0
    markers = cv.watershed(img_real, result_markers)

    img_real[markers == -1] = [255, 0, 0]

    # approximation

    img = cv.imread(img_name)
    img[::] = [0, 0, 0]
    img[markers == -1] = [255, 0, 0]
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ret, thresh = cv.threshold(gray, 0, 255, cv.THRESH_OTSU)
    # fig, ax = plt.subplots(figsize=(12, 12))
    # plt.imshow(img_test)
    # plt.show()
    img[:] = 0
    contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    new_cntrs = [contours[ind] for h, ind in zip(hierarchy[0], range(len(hierarchy[0]))) if h[2] == -1]
    if len(new_cntrs) < 5:
        amount_images -= 1
        print(f'кол-во найденных зерен {len(new_cntrs)} => не учитывается в статистике')
        continue
    # cv.drawContours(img, contours, -1, (255, 255, 0), 1, cv.LINE_AA)
    cv.drawContours(img, new_cntrs, -1, (255, 0, 0), 1, cv.LINE_AA)

    starting_color = [0, 0, 100]
    img_new = cv.imread(img_name)

    img = img_new.copy()
    eps = 0.01
    approxed_arr = []
    for cnt_arr in new_cntrs:
        epsilon = eps * cv.arcLength(cnt_arr, True)
        approx = cv.approxPolyDP(cnt_arr, epsilon, True)
        if len(approx) <= 2:
            continue
        approxed_arr.append(approx)
        cv.drawContours(img, [approx], -1, (255, 0, 0), 1, cv.LINE_AA)
        for cn in approx:
            img[cn[0][1] - 1:cn[0][1] + 1, cn[0][0]-1:cn[0][0]+1] = starting_color
        starting_color = ((starting_color[0] + 50) % 255, (starting_color[1] + 5) % 255, (starting_color[2] + 30) % 255)


    # for eps in np.linspace(0.0001, 0.02, 2):
    #     img = img_new.copy()
    #     # print('eps=', eps)
    #     approxed_arr = []
    #     for cnt_arr in new_cntrs:
    #         epsilon = eps * cv.arcLength(cnt_arr, True)
    #         approx = cv.approxPolyDP(cnt_arr, epsilon, True)
    #         approxed_arr.append(approx)
    #         cv.drawContours(img, [approx], -1, (255, 0, 0), 1, cv.LINE_AA)
    #         for cn in approx:
    #             img[cn[0][1] - 2:cn[0][1] + 2, cn[0][0]-2:cn[0][0]+2] = starting_color
    #         starting_color = ((starting_color[0] + 50) % 255, (starting_color[1] + 5) % 255, (starting_color[2] + 30) % 255)

    # Плотность
    approxed_arr, density, average_grain_area, areas = count_area(np.shape(img)[0], np.shape(img)[1], approxed_arr)

    density_sum += density
    average_grain_area_sum += average_grain_area
    # Распределение углов
    angles = count_angles(np.shape(img)[0], np.shape(img)[1], approxed_arr)
    all_angle_amount += len(angles)

    all_grains_amount += len(angles)

    count_dist(angles, angle_dist)
    count_dist(areas, area_dist)


    # for ang in angles:
    #     if ang in angle_dist:
    #         angle_dist[ang] += 1
    #     else:
    #         angle_dist[ang] = 1

    if amount_images // 20 == 0:
        cv.imwrite(f'segmented_images/{amount_images}_segmented.png', img)
        cv.imwrite(f'segmented_images/{amount_images}.png', img_new)


print(f'Общие данные \n\tСредняя плотность зерен = {density_sum / amount_images}\n'
      f'\tСредний размер зерна = {average_grain_area_sum / amount_images}')
print(f'\tКол-во снимков = {amount_images}')
print(f'\n Epsilon = {eps}')
print(f'Общее время выполнения: {time.time() - t}')

plot_hist(angle_dist, all_angle_amount, 'angle_distribution', 50)
plot_hist(area_dist, all_grains_amount, 'area_distribution', 50)
# print(len(area_dist))

# index = sorted(angle_dist.keys())
# values = np.array([angle_dist[ang] for ang in index]) / all_angle_amount
#
# print('\n\n')
# for key in index:
#     angle_dist[key] = angle_dist[key] / all_angle_amount
#     if angle_dist[key] > 0.06:
#         print(f'угол = {key}, кол-во углов = {angle_dist[key] / all_angle_amount}')
# np.save('angle_distribution.npy', angle_dist)
# fig, ax = plt.subplots()
# ax.bar(index, values)
# ax.set_ylabel('Доля углов')
# ax.set_xlabel('значение угла, градусы')
# plt.show()
