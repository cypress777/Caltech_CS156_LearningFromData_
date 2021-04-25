import random
import numpy as np
from cvxopt import matrix, solvers


def generate_points(num):
    points = []
    while len(points) < num:
        x = random.uniform(-1, 1)
        y = random.uniform(-1, 1)
        points.append(np.array((x, y, 1)))
    return points


def generate_training_points(num):
    x1, y1, x2, y2 = 0, 0, 0, 0
    while x1 == x2 and y1 == y2:
        x1 = random.uniform(-1, 1)
        y1 = random.uniform(-1, 1)

        x2 = random.uniform(-1, 1)
        y2 = random.uniform(-1, 1)

    a = y1 - y2
    b = x2 - x1
    c = (x1 - x2) * y1 - (y1 - y2) * x1
    wr = np.array((a, b, c))

    training_points = generate_points(num)

    return wr, training_points


def evaluate(wr, w, training_points):
    miss = []

    for p in training_points:
        sign_r = np.dot(wr, p) >= 0
        sign = np.dot(w, p) >= 0
        if sign_r != sign:
            miss.append(p)

    return miss


def calc_pla(wr, training_points):
    w = np.array((0, 0, 0))
    miss = training_points
    cnt = 0

    while len(miss) > 0:
        p = random.choice(miss)
        d = np.dot(w, p)
        if d >= 0:
            w = w - p
        else:
            w = w + p
        cnt += 1

        miss = evaluate(wr, w, training_points)
    return w, cnt


def calc_svm(wr, training_points):
    X = []
    Y = []
    signX = []
    for point in training_points:
        y = 1
        if np.dot(wr, point) < 0:
            y = -1
        X.append([point[0], point[1]])
        Y.append([y])
        signX.append([y * point[0], y * point[1]])
    X = np.array(X, dtype=float)
    Y = np.array(Y, dtype=float)
    signX = np.array(signX, dtype=float)

    P = matrix([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
    q = matrix(np.zeros(3, dtype=float))
    G = -1 * np.concatenate((signX, Y), axis=1)
    G = matrix(G)
    h = -1 * np.ones((X.shape[0], 1), dtype=float)
    h = matrix(h)

    solvers.options['show_progress'] = False
    sol = solvers.qp(P, q, G, h)
    return sol


def get_sv(wr, w, points):
    cnt = 0
    for point in points:
        y = 1
        if np.dot(wr, point) < 0:
            y = -1

        if abs(y * np.dot(w, point) - 1) < 1e-5:
            cnt += 1
    return cnt


if __name__ == '__main__':
    training_num = 100
    evaluate_num = 1000
    times = 100
    
    better_cnt = 0
    equal_cnt = 0
    sv_cnt = 0
    for i in range(times):
        wr, training_points = generate_training_points(training_num)
        pla_w, cnt = calc_pla(wr, training_points)

        svm_sol = calc_svm(wr, training_points)
        svm_w = svm_sol["x"].T
        num = get_sv(wr, svm_w, training_points)
        sv_cnt += num

        evaluate_points = generate_points(evaluate_num)
        
        pla_miss = evaluate(wr, pla_w, evaluate_points)
        pla_error = len(pla_miss) / evaluate_num

        svm_miss = evaluate(wr, svm_w, evaluate_points)
        svm_error = len(svm_miss) / evaluate_num

        if svm_error == pla_error:
            equal_cnt += 1
        if svm_error < pla_error:
            better_cnt += 1

    print(equal_cnt, better_cnt)
    print(better_cnt / times)
    print(sv_cnt / times)
