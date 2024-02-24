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
    circle = [num for num in range(10, 90, 10)] + [num for num in range(89, 100)]
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


t = time.time()
density_sum = 0
average_grain_area_sum = 0
all_grains_amount = 0
all_angle_amount = 0
amount_images = 0

all_areas = []
all_angles = []
all_perimeters = []

CLSTRS = 8

angle_dist = {}

area_dist = {}

# цикл по изображениям



for img_name in name_files_30_density_colored():
    print(img_name)
    amount_images += 1 # подсчитать кол-во снимков
    # img_name = 'C:/Users/Dushese/files_for_jupiter/grains_density_30/img-30-49-2550-0.09-0.03/z5.8.png'
    image_base = cv.imread(img_name)
    img = image_base.reshape((image_base.shape[0] * image_base.shape[1], 3))
    common_params = {
        "init": 'k-means++',
        "n_init": "auto",
        "random_state": 4
    }
    y_pred = KMeans(n_clusters=CLSTRS, **common_params).fit_predict(img)#cv.medianBlur(image_base,5))

    clusters_matrix = y_pred.reshape(image_base.shape[0], image_base.shape[1])


    result_markers = np.zeros((image_base.shape[0], image_base.shape[1]))
    result_image = np.zeros((image_base.shape[0], image_base.shape[1]))
    kernel_erode = np.ones((5, 5), np.uint8)
    kernel_dilate = np.ones((3, 3), np.uint8)
    flag = 0
    for x in range(CLSTRS):
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
    # approxed_arr, density, average_grain_area, areas = count_area(np.shape(img)[0], np.shape(img)[1], approxed_arr, img)
    # Характеристики
    angles, approxed_arr, density, average_grain_area, areas, perimeters = count_characteristics(np.shape(img)[0], np.shape(img)[1], approxed_arr)

    all_angles += angles

    all_areas += areas

    all_perimeters += perimeters

    density_sum += density
    average_grain_area_sum += average_grain_area

    all_angle_amount += len(angles)

    all_grains_amount += len(areas)

    count_dist(angles, angle_dist)
    count_dist(areas, area_dist)


    # for ang in angles:
    #     if ang in angle_dist:
    #         angle_dist[ang] += 1
    #     else:
    #         angle_dist[ang] = 1
    if amount_images % 20 == 1:
        cv.imwrite(f'30_density_colored_images/{amount_images}_segmented.png', img)
        cv.imwrite(f'30_density_colored_images/{amount_images}.png', img_new)


print(f'Общие данные \n\tСредняя плотность зерен = {density_sum / amount_images}\n'
      f'\tСредний размер зерна = {average_grain_area_sum / amount_images}')
print(f'\n Epsilon = {eps}')
print(f'Кол-во обработанных изображений: {amount_images}')
print(f'Общее время выполнения: {time.time() - t}')

plot_hist(angle_dist, all_angle_amount, 'angle_distribution', '30_density_colored_distributions')
plot_hist(area_dist, all_grains_amount, 'area_distribution', '30_density_colored_distributions')

np.save(f'30_density_colored_distributions/all_areas.npy', np.array(all_areas))
np.save(f'30_density_colored_distributions/all_angles.npy', np.array(all_angles))
np.save(f'30_density_colored_distributions/all_perimeters.npy', np.array(all_perimeters))
