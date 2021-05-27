import cv2 as cv2
import numpy as np
from os import environ
import scipy.cluster as cluster
import scipy.spatial as spatial
from collections import defaultdict
from statistics import mean
import math
import chess
import chess.svg
from svglib.svglib import svg2rlg
from reportlab.graphics import renderPM
import sys
from os import listdir
from os.path import isfile, join
from Model import transforms_array, get_prediction

def suppress_qt_warnings():
    environ["QT_DEVICE_PIXEL_RATIO"] = "0"
    environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
    environ["QT_SCREEN_SCALE_FACTORS"] = "1"
    environ["QT_SCALE_FACTOR"] = "1"

# Read image and do lite image processing
def read_img(file):
    img = cv2.imread(str(file), 1)

    W = 1000
    height, width, depth = img.shape
    imgScale = W / width
    newX, newY = img.shape[1] * imgScale, img.shape[0] * imgScale
    img = cv2.resize(img, (int(newX), int(newY)))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #gray_blur = cv2.blur(gray, (5, 5))              
    gray_blur = cv2.blur(gray, (4, 4))     
    return img, gray_blur
    #return img, gray

# Canny edge detection
def canny_edge(img, sigma=0.33, delta=-50.0):
    v = np.median(img) + delta
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edges = cv2.Canny(img, lower, upper, apertureSize=3)
    #edges = cv2.Canny(img, 220, 250, apertureSize=3)
    return edges

# Hough line detection
def hough_line(edges, min_line_length=100, max_line_gap=10):
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 125, min_line_length, max_line_gap)
    #lines = cv2.HoughLines(edges, 1, np.pi / 180, 180)
    #lines = np.reshape(lines, (-1, 2))
    return lines

# Separate line into horizontal and vertical
def h_v_lines(lines):
    h_lines, v_lines = [], []
    for rho, theta in lines:
        if theta < np.pi / 4 or theta > np.pi - np.pi / 4:
            v_lines.append([rho, theta])
        else:
            h_lines.append([rho, theta])
    h_lines = sorted(h_lines, key=lambda a: a[0])
    v_lines = sorted(v_lines, key=lambda a: a[0]/np.cos(a[1])  )

    to_remove = []
    for i in range(1, len(h_lines)):
        if h_lines[i][0] - h_lines[i-1][0] < 20:
            to_remove.append(i)
    to_remove.reverse()
    for index in to_remove:
        h_lines.remove(h_lines[index])

    to_remove = []
    for i in range(1, len(v_lines)):
        if v_lines[i][0]/np.cos(v_lines[i][1]) - v_lines[i-1][0]/np.cos(v_lines[i-1][1]) < 20:
            to_remove.append(i)
    to_remove.reverse()
    for index in to_remove:
        v_lines.remove(v_lines[index])

    return h_lines, v_lines

# Find the intersections of the lines
def line_intersections(h_lines, v_lines, corners):
    points = []
    points_not_recognized = []
    points_recognized = []
    h_lines_corners = np.zeros(len(h_lines)) 
    v_lines_corners = np.zeros(len(v_lines)) 
    i = 0
    j = 0
    n = 0
    #print( corners.shape)
    for r_h, t_h in h_lines:
        for r_v, t_v in v_lines:
            a = np.array([[np.cos(t_h), np.sin(t_h)], [np.cos(t_v), np.sin(t_v)]])
            b = np.array([r_h, r_v])
            inter_point = np.linalg.solve(a, b)
            points.append(inter_point)
            if inter_point[0] <= corners.shape[1] and inter_point[1] <= corners.shape[0] and corners[int(inter_point[1]), int(inter_point[0])][0] != 0:
                h_lines_corners[i] = h_lines_corners[i] + 1
                v_lines_corners[j] = v_lines_corners[j] + 1
                points_recognized.append(inter_point)
            j = j + 1
        i = i + 1
        j = 0
    return np.array(points), h_lines_corners, v_lines_corners, points_not_recognized, points_recognized

# Hierarchical cluster (by euclidean distance) intersection points
def cluster_points(points):
    dists = spatial.distance.pdist(points)  # odległości pomiędzy parami punktów
    single_linkage = cluster.hierarchy.single(dists)
    flat_clusters = cluster.hierarchy.fcluster(single_linkage, 15, 'distance')
    cluster_dict = defaultdict(list)
    for i in range(len(flat_clusters)):
        cluster_dict[flat_clusters[i]].append(points[i])
    cluster_values = cluster_dict.values()
    clusters = map(lambda arr: (np.mean(np.array(arr)[:, 0]), np.mean(np.array(arr)[:, 1])), cluster_values)
    return sorted(list(clusters), key=lambda k: [k[1], k[0]])

# Average the y value in each row and augment original point
def augment_points(points):
    points_shape = list(np.shape(points))
    augmented_points = []
    for row in range(int(points_shape[0] / 11)):
        start = row * 11
        end = (row * 11) + 10
        rw_points = points[start:end + 1]
        rw_y = []
        rw_x = []
        for point in rw_points:
            x, y = point
            rw_y.append(y)
            rw_x.append(x)
        y_mean = mean(rw_y)
        for i in range(len(rw_x)):
            point = (rw_x[i], y_mean)
            augmented_points.append(point)
    augmented_points = sorted(augmented_points, key=lambda k: [k[1], k[0]])
    return augmented_points

# Crop board into separate images
def write_crop_images(img, points, img_count, folder_path='C:\\Users\\Dell\\Desktop\\Python\\OpenCV\\ProgrammingKnowledge'):
    num_list = []
    shape = list(np.shape(points))
    start_point = shape[0] - 14

    if int(shape[0] / 11) >= 8:
        range_num = 8
    else:
        range_num = int((shape[0] / 11) - 2)

    for row in range(range_num):
        start = start_point - (row * 11)
        end = (start_point - 8) - (row * 11)
        num_list.append(range(start, end, -1))


    for row in num_list:
        for s in row:
            base_len = math.dist(points[s], points[s + 1])
            bot_left, bot_right = points[s], points[s + 1]
            start_x, start_y = int(bot_left[0]), int(bot_left[1] - (base_len * 2))
            end_x, end_y = int(bot_right[0]), int(bot_right[1])
            if start_y < 0:
                start_y = 0
            cropped = img[start_y: end_y, start_x: end_x]
            img_count += 1
            cv2.imwrite(folder_path + '\\raw_data\\alpha_data_image' + str(img_count) + '.jpeg', cropped)
            print(folder_path + '\\raw_data\\alpha_data_image' + str(img_count) + '.jpeg')
    return img_count

# Converts the FEN into a PNG file
def fen_to_image(fen):
    board = chess.Board(fen)
    current_board = chess.svg.board(board=board)

    output_file = open('current_board.svg', "w")
    output_file.write(current_board)
    output_file.close()

    svg = svg2rlg('current_board.svg')
    renderPM.drawToFile(svg, 'static\images\current_board.png', fmt="PNG")
    return board

# Resize the frame by scale by dimensions
def rescale_frame(frame, percent=75):
    dim = (1000, 750)
    return cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

def cornerDetector(img):
    kernel = np.ones((20,20), np.uint8)
    gray = np.float32(img)
    dst = cv2.cornerHarris(gray, 10, 3, 0.19)
    dst = cv2.dilate(dst, kernel)  # pogrubienie
    blank_image = np.zeros((img.shape[0],img.shape[1],3), np.uint8)
    blank_image[dst > 0.01 * dst.max()] = [255, 0, 0]
    return blank_image

def select_lines(h_lines, v_lines, h_lines_corners, v_lines_corners):
    while len(h_lines_corners) > 9:
        h_lines = np.delete(h_lines, np.argmin(h_lines_corners), 0)
        h_lines_corners = np.delete(h_lines_corners, np.argmin(h_lines_corners), 0)
    while len(v_lines_corners) > 9:
        v_lines = np.delete(v_lines, np.argmin(v_lines_corners), 0)
        v_lines_corners = np.delete(v_lines_corners, np.argmin(v_lines_corners), 0)
    return h_lines, v_lines

def processing(path):
    print(path)
    #img, gray_blur = read_img("chess_board6.jpg")
    #img_clear, gray_blur = read_img("chess_board6.jpg")
    img, gray_blur = read_img("static\images\chessboard.jpg")
    img_clear, gray_blur = read_img("static\images\chessboard.jpg")
    edges = canny_edge(gray_blur)
    lines = hough_line(edges)
    lines = np.reshape(lines, (-1, 2))
    h_lines, v_lines = h_v_lines(lines)
    corners = cornerDetector(gray_blur)
    intersection_points, h_lines_corners, v_lines_corners, points_not_recognized, points_recognized = line_intersections(h_lines, v_lines, corners)
    h_lines, v_lines = select_lines(h_lines, v_lines, h_lines_corners, v_lines_corners)
    board_corners, h_lines_corners, v_lines_corners, points_not_recognized, points_recognized = line_intersections(
        [ h_lines[0], h_lines[-1] ], [ v_lines[0], v_lines[-1] ], corners)
    pts1 = np.float32([board_corners[0], board_corners[2], board_corners[3], board_corners[1] ])
    pts2 = np.float32([[200, 200], [200, 600], [600, 600], [600, 200]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    result = cv2.warpPerspective(img_clear, matrix, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)

    class_names = ['b', 'k', 'n', 'p', 'q', 'r', '_', 
    'B', 'K', 'N', 'P', 'Q', 'R']
    figures_on_the_board = []
    for i in range(8):
        chessboard_row = []
        for j in range(8):
            crop_img = result[150+50*i:250+50*i, 200+50*j:250+50*j]
            tensor = transforms_array(crop_img)
            prediction = get_prediction(tensor)
            chessboard_row.append(int(prediction))
            #cv2.imshow('crop_img', crop_img)
        figures_on_the_board.append(chessboard_row)
    
    PGN_Postion = ""
    for i in range(8):
        chessboard_row = ""
        PGN_row = ""
        Empty_squares = 0
        for j in range(8):
            chessboard_row += class_names[figures_on_the_board[i][j]]
            if class_names[figures_on_the_board[i][j]] != '_':
                if Empty_squares != 0:
                    PGN_row += str(Empty_squares)
                    Empty_squares = 0
                PGN_row += class_names[figures_on_the_board[i][j]]
            else:
                Empty_squares += 1
        if Empty_squares != 0:
            PGN_row += str(Empty_squares)
        PGN_Postion += PGN_row + "/"
        PGN_row = ""
        print(chessboard_row)
    PGN_Postion = PGN_Postion[0:len(PGN_Postion)-1]
    print(PGN_Postion)
    fen_to_image(PGN_Postion)

    return True