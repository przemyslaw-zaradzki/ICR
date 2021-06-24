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
from models import Yolo_predict

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

    output_file = open('current_board.svg', "w")
    output_file.write(current_board)
    output_file.close()

    svg = svg2rlg('current_board.svg')
    renderPM.drawToFile(svg, 'current_board.png', fmt="PNG")
    return board

# Resize the frame by scale by dimensions
def rescale_frame(frame, percent=75):
    dim = (1000, 750)
    return cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

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

if __name__ == "__main__":

    suppress_qt_warnings()

    board_path = r"C:\pdf\pp\8_Semestr\Semestr VIII\ICR\Projekt\Dataset\Szachownice Chess Pieces.v23-raw.yolov4pytorch\valid"
    if len(sys.argv) > 1:
        n = int(sys.argv[1])
        f = listdir(board_path)[n]
        join(board_path, f)
        img, gray_blur = read_img(join(board_path, f))
        img_clear, gray_blur = read_img(join(board_path, f))
    else:
        image_path = "test/IMG_0170_JPG.rf.480e7164cb4727f6654402882f0ce942.jpg"
        img, gray_blur = read_img(image_path)
        img_clear, gray_blur = read_img(image_path)

    cv2.imshow('gray_blur', gray_blur)

    # Canny algorithm
    edges = canny_edge(gray_blur)
    cv2.imshow('edges', edges)
    
    # Hough Transform
    lines = hough_line(edges)
    
    for line in lines:
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho

        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))

        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 2)

    lines = np.reshape(lines, (-1, 2))

    # Separate the lines into vertical and horizontal lines
    h_lines, v_lines = h_v_lines(lines)

    corners = cornerDetector(gray_blur)

    # Find and cluster the intersecting
    intersection_points, h_lines_corners, v_lines_corners, points_not_recognized, points_recognized = line_intersections(h_lines, v_lines, corners)
    for point in points_recognized:
        cv2.circle(corners, (int(point[0]), int(point[1])), 4, (255, 255, 0), 4)
            
    cv2.imshow('corners', corners)

    h_lines, v_lines = select_lines(h_lines, v_lines, h_lines_corners, v_lines_corners)

    for line in h_lines:
        rho = line[0]
        theta = line[1]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho

        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))

        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

    for line in v_lines:
        rho = line[0]
        theta = line[1]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho

        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))

        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
 
    intersection_points, h_lines_corners, v_lines_corners, points_not_recognized, points_recognized = line_intersections(h_lines, v_lines, corners)
    # Locate points of the documents or object which you want to transform
    board_corners, h_lines_corners, v_lines_corners, points_not_recognized, points_recognized = line_intersections(
        [ h_lines[0], h_lines[-1] ], [ v_lines[0], v_lines[-1] ], corners)

    pts1 = np.float32([board_corners[0], board_corners[2], board_corners[3], board_corners[1] ])
    squares_cotners = []

    board_width = board_corners[3][1] - board_corners[0][0]

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

    class_names = ['_', 'b', 'k', 'n', 'p', 'q', 'r', 
    'B', 'K', 'N', 'P', 'Q', 'R']

    boxes = Yolo_predict()
    for i in range(len(boxes)):
        box = boxes[i]
        x1 = int((box[0] - box[2] / 2.0) * img.shape[0])
        y1 = int((box[1] - box[3] / 2.0) * img.shape[1])
        x2 = int((box[0] + box[2] / 2.0) * img.shape[0])
        y2 = int((box[1] + box[3] / 2.0) * img.shape[1])
        
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
        cv2.circle(img, (int(x1), y2), 4, (255, 0, 255), 4)
    cv2.imshow('lines', img)

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
    cv2.imshow('live', img)
    
    fen_to_image(PGN_Postion)
    board_png = cv2.imread("current_board.png", 1)
    cv2.imshow('FEN', board_png)  

    k = cv2.waitKey(0)
    cv2.destroyAllWindows()