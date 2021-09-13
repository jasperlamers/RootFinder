#!/Udistance_transform_edtsers/Jasper/anaconda2/bin/python
import numpy as np
import cv2, os, math, itertools
from PIL import Image, ImageDraw
from scipy import ndimage
from operator import itemgetter
from xml.etree import ElementTree
from xml.dom import minidom
from xml.etree.ElementTree import Element, SubElement

def Start():
    #Script to find all tiff files in directory of .py file
    global file_path
    file_path = os.path.dirname(os.path.abspath(__file__))
    File_List = PathItems(file_path)
    File_List = sorted(File_List) #List of all tif files

    #Iterating over all the tif files in folder
    for file_name in File_List:
        #Creating filenames and result folder
        print(file_name)

        # Remove when finished with testing
        # file_result_path = file_path + '/Results/' + file_name[:-4]
        # if not os.path.exists(file_result_path):
        #     os.makedirs(file_result_path)

        total_list = [] #This is the list containing all root information of one .tif used to write .rsml
        image_read = cv2.imread(file_name) #loading the .tif into matrix
        make_grayscale(image_read, file_name)
        list_index, contours, Blue_channel_threshold = Contour_finding(image_read)

        #Here the program iterates over every selected contour. Final goal is to extend the total list, which is used for the
        #creation of rsml files
        for item in list_index:
            mask = np.zeros_like(image_read)  # Create mask where white is what we want, black otherwise
            cv2.drawContours(mask, contours, item, 255, -1)  # Draw filled contour in mask
            (y, x, z) = np.where(mask == 255)
            mask1 = np.zeros_like(image_read)
            image_read = cv2.imread(file_name)
            for i in range(len(x)):
                mask1[y[i]][x[i]] = Blue_channel_threshold[y[i]][x[i]]

            # Again setting up extremes, used to crop the image and analyse one root at a time
            extLeft = tuple(contours[item][contours[item][:, :, 0].argmin()][0])
            extRight = tuple(contours[item][contours[item][:, :, 0].argmax()][0])
            extTop = tuple(contours[item][contours[item][:, :, 1].argmin()][0])
            extBot = tuple(contours[item][contours[item][:, :, 1].argmax()][0])
            crop_img = mask1[extTop[1] - 20:extBot[1] + 20, extLeft[0] - 20:extRight[0] + 20]
            global Xrange_cropped, Yrange_cropped
            Xrange_cropped = len(crop_img[0])
            Yrange_cropped = len(crop_img)

            #name to save the images of every root (might be removed at a later stage)
            name = str(item) + '_cropped_bw_.jpg'
            distance, vector_init_list, factor = Euclidean_distance_mapping(crop_img)

            if [extBot[1] - extTop[1] + 20, extBot[0] - extLeft[0] + 20] not in vector_init_list:
                vector_init_list.append([extBot[1] - extTop[1] + 20, extBot[0] - extLeft[0] + 20])

            left = extTop[0] - extLeft[0] + 20
            start_point = (20,left)

            #Vectors are created from the extreme top of the root to the extremes of the LR and MR
            vector_list = Vector_initiation_and_refinement(vector_init_list, start_point)

            #By making use of Dijkstra's algorithm (used in GPS systems) the 'shortest' path is found
            #In this case shortest is the one with the highest values in the distance map
            #Hence the extreme low values of non-root pixels. I had to run this twice, from right to left and vice versa.
            #For every pixel in the image a route is created, starting from the Top pixel value of the segmented root
            #The output is a dictonary with the best route. For every extreme of the LR and MR, the route is placed in the list
            # 'new_vector_list'.
            if len(vector_list) > 0:
                new_vector_list = []

                path_dict, matrix_cost = Vector_Strechting_RLUD(distance, vector_list[0][0]) #left-right-down-up
                path_dict, matrix_cost = Vector_Strechting_LRUD(distance, vector_list[0][0], path_dict, matrix_cost) #right-left-down-up
                path_dict, matrix_cost = Vector_Strechting_RLDU(distance, vector_list[0][0], path_dict, matrix_cost) #right-left-up-down
                path_dict = Vector_Strechting_LRDU(distance, vector_list[0][0], path_dict, matrix_cost) #right-left-up-down
                for i in range(len(vector_list)):
                    string = path_dict['r' + str(vector_list[i][1][0]) + 'c' + str(vector_list[i][1][1])]
                    if string != 0:
                        string = string[2:].split()
                        vector = [[int(item.split('c')[0][1:]), int(item.split('c')[1])] for item in string]
                        vector.append(vector_list[i][1])
                        new_vector_list.append(vector)

                if len(new_vector_list) > 0:
                    main_root, laterals = Root_appointer(new_vector_list, distance, start_point)
                    main_root = Selection(distance, [main_root],factor)

                    if len(main_root) > 0:
                        main_root = main_root[0]
                        MR_DL = [distance[item[0]][item[1]][0] for item in main_root]
                        MR_DL_min = min(MR_DL[int(len(MR_DL)*0.3):])

                        main_root = list(filter(lambda a: distance[a[0]][a[1]][0] != 10 ** 10, main_root))  # remove all empty values

                        # Two checks here, one to remove highly similar laterals (if two extremes are found in one lateral)
                        # Two routes may be created. The other to remove laterals crossing large parts of negative values
                        if len(laterals) > 0:
                            laterals = Selection(distance, laterals, factor)
                        if len(laterals) > 0:
                            laterals = lateral_check(laterals)

                        laterals = [item for item in laterals if min([distance[subitem[0]][subitem[1]][0] for subitem in item]) > MR_DL_min]
                        laterals = [list(filter(lambda a: distance[a[0]][a[1]][0] != 10 ** 10, item)) for item in laterals]

                        # # Images are drawn (remove when done with testing software)
                        # crop_img_2 = image_read[extTop[1] - 20:extBot[1] + 20, extLeft[0] - 20:extRight[0] + 20]
                        # crop_img_2 = Image.fromarray(np.uint8(crop_img_2))
                        # draw_new = ImageDraw.Draw(crop_img_2)
                        # for i in range(len(main_root)):
                        #     draw_new.point((main_root[i][1], main_root[i][0]), 'rgb(255,255,0)')
                        #
                        # for i in range(len(laterals)):
                        #     for j in range(len(laterals[i])):
                        #         draw_new.point((laterals[i][j][1], laterals[i][j][0]), 'rgb(0,0,255)')
                        # crop_img_2.save(file_result_path + '/' + name, 'jpeg')
                        # del draw_new

                        total_list.append([main_root, laterals, extLeft, extTop, factor, distance])

        #Final creation of the .rsml
        XML_writer(total_list, file_name[:-4])

def PathItems(file_path): #Script to find all tiff files in directory of .py file
    return [item for item in os.listdir(file_path) if item[-3:] == 'tif']

def make_grayscale(image_read, file_name):
    SmartRootFolder = file_path + '/Improved_RSA/'
    name = SmartRootFolder + file_name + 'f'
    if not os.path.exists(SmartRootFolder):
        os.makedirs(SmartRootFolder)
    RSA_improve = image_read.astype(np.uint8)
    RSA_improve = cv2.cvtColor(RSA_improve, cv2.COLOR_BGR2GRAY)
    RSA_improve = np.invert(RSA_improve)
    RSA_improve = cv2.blur(RSA_improve, (3, 3))
    cv2.imwrite(name, RSA_improve)

def Contour_finding(image_read):
    # Extracting the blue channel
    Blue_channel = image_read
    Blue_channel[:, :, 1] = 0
    Blue_channel[:, :, 2] = 0

    # Setting an adaptive threshold. The hypocotyl is below the threshold, so image only contain roots
    # Next contours/objects are detected using the cv2 module.
    # https://docs.opencv.org/3.4.0/d9/d8b/tutorial_py_contours_hierarchy.html
    # Hierachy describes the relation between the contours

    imgray = cv2.cvtColor(Blue_channel, cv2.COLOR_BGR2GRAY)
    block_size = int(round(len(imgray)/3))
    if block_size % 2 == 0:
        block_size += 1
    block_size = int(block_size)
    Blue_channel_threshold = cv2.adaptiveThreshold(imgray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, -2)
    contours, hierarchy = cv2.findContours(Blue_channel_threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # Specific contours are selected based on size and location, to remove noise and contours of the square plate, respectively.
    minimal_area = float(len(image_read) * len(image_read[0])) * 0.00008  # 0.000115
    maximal_area = float(len(image_read) * len(image_read[0])) * 0.3

    pixel_threshold = float(len(image_read[0])) * 0.06

    contours = [item for item in contours if (cv2.contourArea(item) > minimal_area) and (cv2.contourArea(item) < maximal_area)]

    list_index = []
    for i in range(len(contours)):
        extLeft = tuple(contours[i][contours[i][:, :, 0].argmin()][0])
        extRight = tuple(contours[i][contours[i][:, :, 0].argmax()][0])
        extTop = tuple(contours[i][contours[i][:, :, 1].argmin()][0])
        extBot = tuple(contours[i][contours[i][:, :, 1].argmax()][0])
        if extTop[1] > pixel_threshold and extTop[1] < (len(image_read) / 2):
            if extBot[1] < (len(image_read) - pixel_threshold) and extBot[1] > pixel_threshold:
                if extRight[0] > pixel_threshold * 1.5 and extLeft[0] < (len(image_read[1]) - pixel_threshold * 1.5):
                    if float((extRight[0] - extLeft[0])) / float((extBot[1] - extTop[1])) < 1:
                        list_index.append(i)

    # Selected contours are selected again based on their shape and hierarchy.
    list_index_new = []
    for item in list_index:
        # Parent of contour must not be found in contour list. This removes inner contours of touching laterals.
        # Furthermore the circumference of the minimum Enclosing Circle, must be large. This to remove selected circular
        # contours (imaging artifacts, yeast contamination and so on). Roots have huge minimum Enclosing Circles due to
        # their length
        if hierarchy[0][item][3] not in list_index:
            (x, y), radius = cv2.minEnclosingCircle(contours[item])
            if (radius ** 2) * math.pi > cv2.contourArea(contours[item]) * 5:
                list_index_new.append(item)
    list_index = list_index_new  # Final list of indexes of selected contours, normally 4 in our setup

    # (works 100% on my dataset, no detection of other stuff)
    return list_index, contours, Blue_channel_threshold

def Euclidean_distance_mapping(image_read): #Core of the program.
    image_read[image_read < 140] = 0
    image_read[image_read >= 140] = 255
    image_read = cv2.blur(image_read, (3, 3))
    image_read[image_read < 110] = 0
    image_read[:, :, 1] = image_read[:, :, 0]
    image_read[:, :, 2] = image_read[:, :, 0]

    distance_map_normal = ndimage.distance_transform_edt(image_read)

    factor = np.amax(distance_map_normal)+1

    #Increase distances of 0 to 10^10, for optimal separation using Dijkstra's algorithm (Vector_Stretching)
    distance_map_normal[:, :] = (factor - distance_map_normal[:, :])**2
    distance_map_normal[distance_map_normal == (factor - 0)**2] = 10**10

    # Finding extremes within the root (tips of LR and MR). Requires optimization
    (y, x, z) = np.where(distance_map_normal == (factor - 1)**2)

    vector_init_list = []
    for i in range(0, len(x), 3):
        neighbour_list = neighbours(y[i], x[i], 'no')
        if len(list(filter(lambda a: distance_map_normal[a[0]][a[1]][0] == 10 ** 10, neighbour_list))) >= 5:
            vector_init_list.append([y[i], x[i]])

    return distance_map_normal,vector_init_list, factor

#Vectors are created from the extreme top of the root to the extremes of the LR and MR
def Vector_initiation_and_refinement(vector_init, start):
    return [[list(start),item] for item in vector_init]

#First run of Dijkstra's algorithm (from left to right)
def Vector_Strechting_RLUD(matrix,start):
    path_dict = {} #Dictonary to store the routes
    matrix_cost = [[float('Inf') for x in range(Xrange_cropped)] for y in range(Yrange_cropped)] #initiation of cost list. Infinite for all
    matrix_cost[start[0]][start[1]] = 0 #setting cost list for start point at 0
    path_dict['r'+str(start[0])+'c'+str(start[1])] = 0 #Start of dict

    #Core
    for i in range(1,max(Yrange_cropped,Xrange_cropped)):
        #Here all pixels to check are created. This starts with the outer pixels of a square of 3x3 and ends with a
        #square as large as the image
        area_list = area(start[0], start[1], i)
        for item in area_list:
            # Calculation of the costs. Value of the euclidean distance map * -1 + matrix_cost of neighbours
            # If the cost is lower than previously found cost, this pixel is selected
            # e.g. cost of starting point is set at 0. pixel next to it = 0+(pixel value) = cost to visit that pixel
            str1, str2, matrix_cost = Vector_Strechting_algorithm(item, matrix, matrix_cost)
            #Saving the route for a specific pixel. 'pixel = route lowest cost neighbour + lowest cost neighbour'
            path_dict['r'+str(item[0])+'c'+str(item[1])] = str(path_dict['r'+str(str1)+'c'+str(str2)]) + ' r'+str(str1)+'c'+str(str2)
    return path_dict, matrix_cost

#Pixel list to analyse (First run of Dijkstra's algorithm)
def area(y,x, area):
    area_list = []
    x_max = x+area+1
    area_list = [[y-area,item] for item in range(x-area,x_max) if -1 < y - area < Yrange_cropped and -1 < item < Xrange_cropped]

    for y1 in range(y-area+1,y+area):
        if -1 < y1 < Yrange_cropped:
            if -1 < x - area < Xrange_cropped:
                area_list.append([y1, x-area])
            if -1 < x + area < Xrange_cropped:
                area_list.append([y1, x+area])

    area_list += [[y+area,item] for item in range(x-area,x_max) if -1 < y+area < Yrange_cropped and -1 < item < Xrange_cropped]
    return area_list

#Neighbours to check (First and second run of Dijkstra's algorithm)
def neighbours(y,x, include_0):
    if include_0 == 'yes':
        neighbours_list = [[y2, x2] for y2 in range(y - 1, y + 2) for x2 in range(x - 1, x + 2) if (-1 < x2 < Xrange_cropped and -1 < y2 < Yrange_cropped and (0 <= x2 < Xrange_cropped) and (0 <= y2 < Yrange_cropped))]
    else:
        neighbours_list = [[y2, x2] for y2 in range(y - 1, y + 2) for x2 in range(x - 1, x + 2) if (-1 < x2 < Xrange_cropped and -1 < y2 < Yrange_cropped and (x != x2 or y != y2) and (0 <= x2 < Xrange_cropped) and (0 <= y2 < Yrange_cropped))]
    return neighbours_list

#Second run of Dijkstra's algorithm (from right to left)
def Vector_Strechting_LRUD(matrix,start, path_dict, matrix_cost):
    for i in (range(1,max(Yrange_cropped,Xrange_cropped))):
        area_list = area2(start[0], start[1], i)
        area_list = filter(lambda a: matrix[a[0]][a[1]][0] != 10 ** 10, area_list)

        for item in area_list:
            str1, str2, matrix_cost = Vector_Strechting_algorithm(item, matrix, matrix_cost)
            path_dict['r'+str(item[0])+'c'+str(item[1])] = str(path_dict['r'+str(str1)+'c'+str(str2)]) + ' r'+str(str1)+'c'+str(str2)
    return path_dict, matrix_cost

def Vector_Strechting_RLDU(matrix,start, path_dict, matrix_cost):
    for i in (range(1,max(Yrange_cropped,Xrange_cropped)))[::-1]:
        area_list = area(start[0], start[1], i)
        area_list = filter(lambda a: matrix[a[0]][a[1]][0] != 10 ** 10, area_list)

        for item in area_list:
            str1, str2, matrix_cost = Vector_Strechting_algorithm(item, matrix, matrix_cost)
            path_dict['r'+str(item[0])+'c'+str(item[1])] = str(path_dict['r'+str(str1)+'c'+str(str2)]) + ' r'+str(str1)+'c'+str(str2)
    return path_dict, matrix_cost

def Vector_Strechting_LRDU(matrix,start, path_dict, matrix_cost):
    for i in (range(1,max(Yrange_cropped,Xrange_cropped)))[::-1]:
        area_list = area2(start[0], start[1], i)
        area_list = filter(lambda a: matrix[a[0]][a[1]][0] != 10 ** 10, area_list)

        for item in area_list:
            str1, str2, matrix_cost = Vector_Strechting_algorithm(item, matrix, matrix_cost)
            path_dict['r'+str(item[0])+'c'+str(item[1])] = str(path_dict['r'+str(str1)+'c'+str(str2)]) + ' r'+str(str1)+'c'+str(str2)
    return path_dict

def Vector_Strechting_algorithm(item, matrix, matrix_cost):
    neighbors_list = neighbours(item[0], item[1], 'no')
    cost_list = [matrix[item[0]][item[1]][0] + matrix_cost[subitem[0]][subitem[1]] for subitem in neighbors_list]
    costs_of_neighbours = min(cost_list)
    index = cost_list.index(costs_of_neighbours)

    matrix_cost[item[0]][item[1]] = costs_of_neighbours
    str1 = neighbors_list[index][0]
    str2 = neighbors_list[index][1]
    return str1, str2, matrix_cost

#Neighbours to check (Second run of Dijkstra's algorithm)
def area2(y,x, area):
    x_max = x+area
    area_list = [[y-area,item] for item in range(x_max,x-area-1,-1) if -1 < y - area < Yrange_cropped and -1 < item < Xrange_cropped]
    for y1 in range(y-area+1,y+area):
        if -1 < y1 < Yrange_cropped:
            if -1 < x + area < Xrange_cropped:
                area_list.append([y1, x+area])
            if -1 < x - area < Xrange_cropped:
                area_list.append([y1, x-area])

    area_list += [[y+area,item] for item in range(x_max, x - area - 1, -1) if -1 < y+area < Yrange_cropped and -1 < item < Xrange_cropped]
    return area_list

def Root_appointer(new_vector_list, distance, start_point):
    main_root = sorted(new_vector_list, key=len)[-1]  # longest vector = MR
    new_vector_list.remove(main_root)  # Remove MR from list. Lateral remain

    # Here the laterals are found by removing all pixels overlapping with the main root. The leftover
    # is the route for every extreme - main root = lateral
    main_root_shape = [neighbour for item in main_root for neighbour in (neighbours(item[0], item[1], 'yes'))]
    main_root_shape = [list(item) for item in set(tuple(item) for item in main_root_shape)]

    # Extra security to remove some odd 'laterals'. (1) Laterals ending in MR. (2) High laterals
    # Mainly due to bad separation of root/shoot
    new_vector_list = [list(itertools.dropwhile(lambda x: x in main_root_shape, item)) for item in new_vector_list]
    new_vector_list = [item for item in new_vector_list if (len(item) > 5)]
    laterals = [item for item in new_vector_list if sum(neighbour not in main_root_shape for neighbour in neighbours(item[-1][0], item[-1][1], 'yes')) == 9]

    return main_root, laterals

#Simply counts all negative values in euclidean distance map. if less than 1% of pixels is negative, the vector is passed
def Selection(matrix, vector_list, factor):
    selection_count = [item for item in vector_list if len(list(filter(lambda a: matrix[a[0]][a[1]][0] == 10 ** 10, item))) <= (len(item) * 0.01)]
    selection_length = [item for item in selection_count if (len(item) >= 40)]
    selection_max = [item for item in selection_count if min([matrix[a[0]][a[1]][0] for a in item]) < ((factor - 1.5)**2) and item not in selection_length]
    return selection_length + selection_max

#Removal of similar laterals
def lateral_check(laterals):
    laterals_new = []
    check = True
    while check == True:
        laterals_new_temp = [laterals[0]]
        area_list = [laterals[0][0]]
        x_value = laterals[0][0][1]
        y_value = laterals[0][0][0]
        for i in range(1, 3):
            area_list += area(y_value, x_value, i)

        for k in range(len(laterals)):
            if 0 != k:
                if laterals[k][0] in area_list:
                    laterals_new_temp.append(laterals[k])
                else:
                    uniques = []
                    for i in range(len(laterals[0])):
                        if laterals[0][i] not in laterals[k]:
                            uniques.append(laterals[0][i])
                    if len(uniques) < float(min(len(laterals[0]), len(laterals[k])) * 0.6):
                        laterals_new_temp.append(laterals[k])

        laterals_new.append(sorted(laterals_new_temp, key=len)[-1])
        for item in laterals_new_temp:
            laterals.remove(item)
        if len(laterals) == 0:
            check = False
    return laterals_new

#Writing the rsml file
def XML_writer(total_list, file_name):
    root = Element('rsml')
    root.set('xmlns:po', 'http://www.plantontology.org/xml-dtd/po.dtd')
    metadata = SubElement(root, 'metadata')

    version = SubElement(metadata, 'version')
    version.text = '1'

    unit = SubElement(metadata, 'unit')
    unit.text = 'inch'

    unit = SubElement(metadata, 'resolution')
    unit.text = '300.0'

    unit = SubElement(metadata, 'last-modified')
    unit.text = 'today'

    unit = SubElement(metadata, 'software')
    unit.text = 'smartroot'

    unit = SubElement(metadata, 'user')
    unit.text = 'globet'

    unit = SubElement(metadata, 'file-key')
    unit.text = 'myimage'

    x = SubElement(metadata, 'property-definitions')

    list = [['diameter', 'float', 'cm'], ['length', 'float', 'cm'], ['pixel', 'float', 'none'],
            ['angle', 'float', 'degree'], ['insertion', 'float', 'cm'], ['lauz', 'float', 'cm'],
            ['lbuz', 'float', 'cm'], ['node-orientation', 'float', 'radian']]
    for i in range(len(list)):
        entry = SubElement(x, 'property-definition')
        label = SubElement(entry, 'label')
        label.text = list[i][0]
        label = SubElement(entry, 'type')
        label.text = list[i][1]
        label = SubElement(entry, 'unit')
        label.text = list[i][2]

    image = SubElement(metadata, 'image')
    label = SubElement(image, 'captured')
    label.text = 'today'
    label = SubElement(image, 'label')
    label.text = file_name

    scene = SubElement(root, 'scene')
    plant = SubElement(scene, 'plant')

    count = 0
    for roots in sorted(total_list, key=itemgetter(2)):
        main_root, laterals, extLeft, extTop, factor,distance_matrix = roots
        ID_string = file_name+'_root_'+str(count)+'_lat_x'
        label_string = 'root_' + str(count)

        main_root_xml = SubElement(plant, 'root')
        main_root_xml.set('ID', ID_string)
        main_root_xml.set('label', label_string)
        main_root_xml.set('po:accession', 'PO:0009005')

        #properties = SubElement(main_root_xml, 'properties')
        #label = SubElement(properties, 'rulerAtOrigin')
        #label.text = '0.0'

        geometry = SubElement(main_root_xml, 'geometry')
        polyline = SubElement(geometry, 'polyline')

        for i in range(1, len(main_root)-4, 10):
            item = main_root[i]
            root2 = SubElement(polyline, 'point')
            root2.set('x', str(item[1] - 20 + extLeft[0]))
            root2.set('y', str(item[0] - 20 + extTop[1]))

        item = main_root[-1]
        root2 = SubElement(polyline, 'point')
        root2.set('x', str(item[1] - 20 + extLeft[0]))
        root2.set('y', str(item[0] - 20 + extTop[1]))

        functions = SubElement(main_root_xml, 'functions')
        function1 = SubElement(functions, 'function')
        function1.set('name', 'diameter')
        function1.set('domain', 'polyline')

        for i in range(1, len(main_root)-4, 10):
            item = main_root[i]
            diameter = SubElement(function1, 'sample')
            diameter.text = str(2*float(factor-math.sqrt(distance_matrix[item[0]][item[1]][0])))

        item = main_root[-1]
        diameter = SubElement(function1, 'sample')
        diameter.text = str(2*float(factor-math.sqrt(distance_matrix[item[0]][item[1]][0])))

        for i in range(len(laterals)):
            ID_string = file_name+'root_'+str(count)+'_lat_'+str(i)
            Label_string = 'R'+str(count)+'lat_' + str(i)

            root1 = SubElement(main_root_xml, 'root')
            root1.set('ID', ID_string)
            root1.set('label', Label_string)
            root1.set('po:accession', 'PO:0009005')

            properties = SubElement(root1, 'properties')
            #label = SubElement(properties, 'rulerAtOrigin')
            #label.text = '0.0'

            geometry = SubElement(root1, 'geometry')
            polyline = SubElement(geometry, 'polyline')
            if len(laterals[i])-1 > 9:
                for j in range(1, len(laterals[i])-4, 3):
                    item = laterals[i][j]
                    root2 = SubElement(polyline, 'point')
                    root2.set('x', str(item[1] - 20 + extLeft[0]))
                    root2.set('y', str(item[0] - 20 + extTop[1]))
            else:
                item = laterals[i][1]
                root2 = SubElement(polyline, 'point')
                root2.set('x', str(item[1] - 20 + extLeft[0]))
                root2.set('y', str(item[0] - 20 + extTop[1]))

            item = laterals[i][-1]
            root2 = SubElement(polyline, 'point')
            root2.set('x', str(item[1] - 20 + extLeft[0]))
            root2.set('y', str(item[0] - 20 + extTop[1]))

            functions = SubElement(root1, 'functions')
            function = SubElement(functions, 'function')
            function.set('name', 'diameter')
            function.set('domain', 'polyline')

            for j in range(1, len(laterals[i])-4, 3):
                item = laterals[i][j]
                diameter = SubElement(function, 'sample')
                diameter.text = str(factor - math.sqrt(distance_matrix[item[0]][item[1]][0]))

            item = laterals[i][-1]
            diameter = SubElement(function, 'sample')
            diameter.text = str(factor - math.sqrt(distance_matrix[item[0]][item[1]][0]))

            annotations = SubElement(root1, 'annotations')

        count+= 1

    prettify(root,file_name)

#Creating a structured rsml
def prettify(elem,file_name):
    rough_string = ElementTree.tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough_string)

    reparsed = reparsed.toprettyxml(indent='  ')
    f = open('Improved_RSA/'+file_name+'.rsml', 'w')
    f.write(reparsed)
    f.close()

Start() #Start of the function
