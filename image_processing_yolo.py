import cv2 as cv2
import numpy as np
from os import environ
import math
import chess
import chess.svg
from reportlab.graphics import renderPM
from YOLOv4.YOLOv4.models import Yolo_predict

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
    gray_blur = cv2.blur(gray, (4, 4))     
    return img, gray_blur

# Canny edge detection
def canny_edge(img, sigma=0.33, delta=-50.0):
    v = np.median(img) + delta
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edges = cv2.Canny(img, lower, upper, apertureSize=3)
    return edges

# Hough line detection
def hough_line(edges, min_line_length=100, max_line_gap=10):
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 125, min_line_length, max_line_gap)

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

# Converts the FEN into a PNG file
def fen_to_image(fen):
    board = chess.Board(fen)
    current_board = chess.svg.board(board=board)

    output_file = open('static\images\current_board.svg', "w")
    output_file.write(current_board)
    output_file.close()
    return board

def cornerDetector(img):
    kernel = np.ones((20,20), np.uint8)
    gray = np.float32(img)
    dst = cv2.cornerHarris(gray, 10, 3, 0.19)
    dst = cv2.dilate(dst, kernel) 
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

def processing_yolo(path):

    img, gray_blur = read_img("static\images\chessboard.jpg")
    edges = canny_edge(gray_blur)
    lines = hough_line(edges)
    lines = np.reshape(lines, (-1, 2))
    h_lines, v_lines = h_v_lines(lines)
    corners = cornerDetector(gray_blur)
    intersection_points, h_lines_corners, v_lines_corners, points_not_recognized, points_recognized = line_intersections(h_lines, v_lines, corners)
    
    h_lines, v_lines = select_lines(h_lines, v_lines, h_lines_corners, v_lines_corners)

    class_names = ['_', 'b', 'k', 'n', 'p', 'q', 'r', 
    'B', 'K', 'N', 'P', 'Q', 'R']
    
    figures_on_the_board = [
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0]
    ]    
    
    boxes = Yolo_predict()
    for i in range(len(boxes)):
        box = boxes[i]
        x1 = int((box[0] - box[2] / 2.0) * img.shape[1])
        y2 = int((box[1] + box[3] / 2.0) * img.shape[0])
        
        distance = 9999
        closest_point = 0
        for j in range(len(intersection_points)):
            d = math.pow(intersection_points[j][0] - x1, 2) + math.pow(intersection_points[j][1] - y2, 2)
            if d<=distance:
                distance = d
                closest_point = j
        row = closest_point//9 - 1
        column = closest_point%9
        figures_on_the_board[row][column] = int(box[6])
    
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