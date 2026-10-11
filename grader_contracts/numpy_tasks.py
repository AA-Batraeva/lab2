from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class MatrixVectorBatchInput:
    matrices: Any
    vectors: Any


@dataclass(frozen=True)
class BinarizeInput:
    matrix: Any
    threshold: float = 0.5


@dataclass(frozen=True)
class MatrixInput:
    matrix: Any


@dataclass(frozen=True)
class RandomMatrixInput:
    rows: int
    columns: int
    mean: float = 0.0
    std: float = 1.0
    seed: int = 42


@dataclass(frozen=True)
class MatrixStatistics:
    matrix: Any
    row_means: Any
    column_means: Any
    row_variances: Any
    column_variances: Any


@dataclass(frozen=True)
class ChessInput:
    rows: int
    columns: int
    first: float
    second: float


@dataclass(frozen=True)
class RectangleInput:
    width: int
    height: int
    image_height: int
    image_width: int
    shape_color: tuple[int, int, int]
    background_color: tuple[int, int, int]


@dataclass(frozen=True)
class EllipseInput:
    semi_axis_x: int
    semi_axis_y: int
    image_height: int
    image_width: int
    shape_color: tuple[int, int, int]
    background_color: tuple[int, int, int]


@dataclass(frozen=True)
class TimeSeriesInput:
    values: Any
    window: int


@dataclass(frozen=True)
class TimeSeriesStatistics:
    mean: float
    variance: float
    std: float
    local_maxima_indices: Any
    local_minima_indices: Any
    moving_average: Any


@dataclass(frozen=True)
class OneHotInput:
    labels: Any
    class_count: int | None = None

import numpy as np
from grader_contracts.numpy_tasks import (
    BinarizeInput, ChessInput, EllipseInput, MatrixInput, MatrixStatistics,
    MatrixVectorBatchInput, OneHotInput, RandomMatrixInput, RectangleInput,
    TimeSeriesInput, TimeSeriesStatistics,)




def sum_prod(data: MatrixVectorBatchInput) -> np.ndarray:
    matrices, vectors = data.matrices, data.vectors
    s = np.zeros_like(vectors[0])
    for A, x in zip(matrices, vectors):
        p = A @ x
        s = s + p
    return s

'''
def binarize(data: BinarizeInput) -> np.ndarray:
    matrix, threshold = data.matrix, data.threshold
    res =[]
    for A in matrix:
        new = []
        for x in A:
            if x > threshold:
                new.append(1)
            else:
                new.append(0)
        res.append(new)
    return res'''
'''
def unique_rows(data: MatrixInput) -> list[list[float]]:
    matrix = data.matrix
    res = []
    for row in matrix:
        unique_elements = np.unique(row)
        res.append(unique_elements.tolist())
    return res'''

'''
def unique_columns(data: MatrixInput) -> list[list[float]]:
    matrix = data.matrix
    m = np.transpose(matrix)
    res = []
    for A in m:
        u = np.unique(A)
        l = u.tolist()
        res.append(l)
    return res'''
'''
def matrix_statistics(data: RandomMatrixInput) -> MatrixStatistics:
    rows, columns, mean_val, std_val, seed = data.rows, data.columns, data.mean, data.std, data.seed

    np.random.seed(seed)
    matrix = np.random.normal(loc=mean_val, scale=std_val, size=(rows, columns))

    row_means = np.mean(matrix, axis=1)
    row_vars = np.var(matrix, axis=1)

    col_means = np.mean(matrix, axis=0)
    col_vars = np.var(matrix, axis=0)

    return MatrixStatistics(
        matrix=matrix,
        row_means=row_means,
        col_means=col_means,
        row_vars=row_vars,
        col_vars=col_vars)'''

''' 
def chess(data: ChessInput) -> np.ndarray:
    m, n, a, b = data.m, data.n, data.a, data.b
    rows_grid = np.arange(m) [:, np.newaxis]
    cols_grid = np.arange(n)
    index_sums = rows_grid + cols_grid
    res = np.where(index_sums % 2 == 0, a, b)
    return np.array(res)'''


'''
def draw_rectangle(data: RectangleInput) -> np.ndarray:
    width, height = data.width, data.height
    img_h, img_w = data.image_height, data.image_width
    shape_color = data.shape_color
    bg_color = data.background_color

    y0 = img_h / 2.0
    x0 = img_w / 2.0

    y_min = y0 - height / 2.0
    y_max = y0 + height / 2.0
    x_min = x0 - width / 2.0
    x_max = x0 + width / 2.0
    img = np.zeros((img_h, img_w, 3), dtype=np.uint8)

    for y in range(img_h):
        row = []
        for x in range(img_w):
            if y_min <= y < y_max and x_min <= x < x_max:
                row.append(shape_color)
            else:
                row.append(bg_color)
        img.append(row)
    return np.array(img)'''

'''
def draw_ellipse(data: EllipseInput) -> np.ndarray:
    semi_axis_x, semi_axis_y = data.semi_axis_x, data.semi_axis_y
    image_height, image_width = data.image_height, data.image_width
    shape_color, background_color = data.shape_color, data.background_color
    y0 = image_height / 2.0
    x0 = image_width / 2.0
    img = []
    for y in range(image_height):
        row = []
        for x in range(image_width):
            value = ((x - x0) ** 2) / (a ** 2) + ((y - y0) ** 2) / (b ** 2)
            if value <= 1.0:
                row.append(shape_color)
            else:
                row.append(background_color)

        img.append(row)
    return np.array(img, dtype=np.uint8)'''






def analyze_time_series(data: TimeSeriesInput) -> TimeSeriesStatistics:
    values, window = data.values, data.window
    n = len(values)
    expected_value = np.mean(values)
    std_value = np.std(values)
    variance_value = np.var(values)
    local_maxima = []
    local_minima = []
    for i in range(1, n - 1):
        # локальный максимум
        if values[i] > values[i - 1] and values[i] > values[i + 1]:
            local_maxima.append(i)

        # локальный минимум
        if values[i] < values[i - 1] and values[i] < values[i + 1]:
            local_minima.append(i)

    moving_average = []
    for i in range(n - window + 1):
        current_window = values[i: i + window]
        w_mean = np.mean(current_window)
        moving_average.append(w_mean)

    local_maxima = np.array(local_maxima)
    local_minima = np.array(local_minima)
    moving_average = np.array(moving_average)

    return TimeSeriesStatistics(expected_value=expected_value, variance=variance_value, std_value=std_value,
         local_maxima=local_maxima, local_minima=local_minima, moving_average=moving_average)


'''
def one_hot(data: OneHotInput) -> np.ndarray:
    labels, class_count = data.labels, data.class_count
    if class_count is None:
        class_count = int(np.max(labels)) + 1
    res = []
    for val in labels:
        row = [0] * class_count
        row[val] = 1
        res.append(row)
    return np.array(res, dtype=np.int32)'''
