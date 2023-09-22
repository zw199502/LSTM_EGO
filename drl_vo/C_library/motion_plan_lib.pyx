## python setup.py build_ext --inplace ####
from libc.stdlib cimport malloc, free

cdef extern from "math.h":
    float cos(float theta)
    float sin(float theta)
    float sqrt(float item)
    float atan2(float y, float x)
    float hypot(float dx, float dy)
    float fabs(float item)

cdef int num_line
cdef int num_circle
cdef int num_scan
cdef int map_size
cdef float map_resolution
cdef float margin
cdef float *lines
cdef float *circles
cdef float *scans
cdef float *scan_lines
cdef float *robot_pose
cdef int *cost_map
cdef int *cost_map_output
cdef int *inflate
cdef int inflate_row
cdef int inflate_col
cdef float laser_resolution
cdef float PI = 3.141592654

# transform previous lasers into current robot frame
cdef float *scan_last_1
cdef float *scan_last_2
cdef float *scan_last_3
cdef float *scan_end_last_1
cdef float *scan_end_last_2
cdef float *scan_end_last_3

def InitializeEnv(_num_line: int, _num_circle: int, _num_scan: int, _laser_resolution: float, _map_size: int, _map_resolution: float, _margin: float):
    global num_line, num_circle, num_scan
    global map_size, map_resolution, margin, cost_map, cost_map_output
    global inflate_col, inflate_row, inflate
    global lines, circles, scans, scan_lines, robot_pose
    global laser_resolution
    global scan_last_1, scan_last_2, scan_last_3
    global scan_end_last_1, scan_end_last_2, scan_end_last_3

    num_line = _num_line
    num_circle = _num_circle
    num_scan = _num_scan
    laser_resolution = _laser_resolution

    map_size = _map_size
    map_resolution = _map_resolution
    margin = _margin

    lines = <float*>malloc(num_line * 4 * sizeof(float))
    circles = <float*>malloc(num_circle * 3 * sizeof(float))
    scans = <float*>malloc(num_scan * sizeof(float))
    scan_lines = <float*>malloc(num_scan * 4 * sizeof(float))
    robot_pose = <float*>malloc(3 * sizeof(float))
    cost_map = <int*>malloc(map_size * map_size * sizeof(int))
    cost_map_output = <int*>malloc(map_size * map_size * sizeof(int))

    scan_last_1 = <float*>malloc(num_scan * sizeof(float))
    scan_last_2 = <float*>malloc(num_scan * sizeof(float))
    scan_last_3 = <float*>malloc(num_scan * sizeof(float))
    for i in range(num_scan):
        scan_last_1[i] = 999.9
        scan_last_2[i] = 999.9
        scan_last_3[i] = 999.9
    scan_end_last_1 = <float*>malloc(num_scan * 2 * sizeof(float))
    scan_end_last_2 = <float*>malloc(num_scan * 2 * sizeof(float))
    scan_end_last_3 = <float*>malloc(num_scan * 2 * sizeof(float))

    for i in range(map_size * map_size):
        cost_map[i] = 255
        cost_map_output[i] = 255

    inflate_row = 6
    inflate_col = 10
    inflate = <int*>malloc(inflate_row * inflate_col * sizeof(int))

    for i in range(5):
        inflate[i] = 0
    inflate[5] = 3
    inflate[6] = 4
    for i in range(7, inflate_col):
        inflate[i] = 5

    for i in range(4):
        inflate[i + inflate_col * 1] = 0
    inflate[4 + inflate_col * 1] = 3
    inflate[5 + inflate_col * 1] = 4
    inflate[6 + inflate_col * 1] = 5
    for i in range(7, inflate_col):
        inflate[i + inflate_col * 1] = 6

    for i in range(3):
        inflate[i + inflate_col * 2] = 0
    inflate[3 + inflate_col * 2] = 3
    inflate[4 + inflate_col * 2] = 5
    inflate[5 + inflate_col * 2] = 6
    inflate[6 + inflate_col * 2] = 6
    for i in range(7, inflate_col):
        inflate[i + inflate_col * 2] = 7

    inflate[0 + inflate_col * 3] = 0
    inflate[1 + inflate_col * 3] = 0
    inflate[2 + inflate_col * 3] = 4
    inflate[3 + inflate_col * 3] = 5
    inflate[4 + inflate_col * 3] = 6
    inflate[5 + inflate_col * 3] = 7
    for i in range(6, inflate_col):
        inflate[i + inflate_col * 3] = 8
    
    inflate[0 + inflate_col * 4] = 0
    inflate[1 + inflate_col * 4] = 4
    inflate[2 + inflate_col * 4] = 5
    inflate[3 + inflate_col * 4] = 7
    inflate[4 + inflate_col * 4] = 7
    inflate[5 + inflate_col * 4] = 8
    for i in range(6, inflate_col):
        inflate[i + inflate_col * 4] = 9

    inflate[0 + inflate_col * 5] = 4
    inflate[1 + inflate_col * 5] = 6
    inflate[2 + inflate_col * 5] = 7
    inflate[3 + inflate_col * 5] = 8
    inflate[4 + inflate_col * 5] = 9
    inflate[5 + inflate_col * 5] = 9
    for i in range(6, inflate_col):
        inflate[i + inflate_col * 5] = 10

def set_lines(index: int, item: float):
    global lines
    lines[index] = item

def set_circles(index: int, item: float):
    global circles
    circles[index] = item

def set_robot_pose(x: float, y: float, yaw: float):
    global robot_pose
    robot_pose[0] = x
    robot_pose[1] = y
    robot_pose[2] = yaw

def set_scan_end_last(x: float, y: float, index: int, scan: int):
    global scan_end_last_1, scan_end_last_2, scan_end_last_3
    if scan == 1:
        scan_end_last_1[2 * index] = x
        scan_end_last_1[2 * index + 1] = y
    elif scan == 2:
        scan_end_last_2[2 * index] = x
        scan_end_last_2[2 * index + 1] = y
    else:
        scan_end_last_3[2 * index] = x
        scan_end_last_3[2 * index + 1] = y


def cal_laser():
    global num_line, num_circle, num_scan, lines, circles, scans, scan_lines, robot_pose, laser_resolution
    cdef float _intersection_x = 0.
    cdef float _intersection_y = 0.
    cdef float scan_range = 0.
    cdef float min_range = 999.9
    cdef float float_num_scan = num_scan
    cdef float float_i = 0.
    cdef float angle_rel = 0.
    cdef float angle_abs = 0.
    cdef float line_unit_vector[2]
    cdef float a1, b1, c1, x0, y0, x1, y1, f0, f1, dx, dy, a2, b2, c2, intersection_x, intersection_y
    cdef float line_vector[2]
    cdef float x2, y2, r, f2, d, a3, b3, c3, intermediate_x, intermediate_y
    cdef float temp, l_vector, intersection_x_1, intersection_x_2, intersection_y_1, intersection_y_2
    cdef: 
        float *line_vector_1 
        float *line_vector_2
    line_vector_1 = <float*>malloc(2 * sizeof(float))
    line_vector_2 = <float*>malloc(2 * sizeof(float))
    for i in range(num_scan):
        min_range = 999.9
        float_i = i
        angle_rel = (float_i - (float_num_scan - 1.0) / 2.0) * laser_resolution
        # laser angle range: [-pi, pi]
        if angle_rel < -PI:
            angle_rel = -PI
        if angle_rel > PI:
            angle_rel = PI
        angle_abs = angle_rel + robot_pose[2]
        line_unit_vector[:] = [cos(angle_abs), sin(angle_abs)]
        # a1,b1,c1 are the parameters of line
        a1 = line_unit_vector[1]
        b1 = -line_unit_vector[0]
        c1 = robot_pose[1] * line_unit_vector[0] - robot_pose[0] * line_unit_vector[1]

        for j in range(num_line):
            x0 = lines[j * 4 + 0]
            y0 = lines[j * 4 + 1]
            x1 = lines[j * 4 + 2]
            y1 = lines[j * 4 + 3]
            f0 = a1 * x0 + b1 * y0 + c1
            f1 = a1 * x1 + b1 * y1 + c1
            if f0 * f1 > 0: # the two points are located on the same side
                continue
            else:
                dx = x1 - x0
                dy = y1 - y0
                a2 = dy
                b2 = -dx
                c2 = y0 * dx - x0 * dy
                intersection_x = (b1 * c2 - b2 * c1) / (a1 * b2 - a2 * b1)
                intersection_y = (a2 * c1 - a1 * c2) / (a1 * b2 - a2 * b1)
                # intersection is always in front of the robot
                line_vector[:] = [intersection_x - robot_pose[0], intersection_y - robot_pose[1]]
                # the intersection point must be located in the one direction of the line
                if (line_vector[0] * line_unit_vector[0] > 0) or (line_vector[1] * line_unit_vector[1] > 0):
                    scan_range = hypot(intersection_x - robot_pose[0], intersection_y - robot_pose[1])
                    if scan_range < min_range:
                        _intersection_x = intersection_x
                        _intersection_y = intersection_y
                        min_range = scan_range

        for k in range(num_circle):
            x2 = circles[3 * k + 0]
            y2 = circles[3 * k + 1]
            r = circles[3 * k + 2]
            f2 = a1 * x2 + b1 * y2 + c1
            
            d = fabs(f2) / hypot(a1, b1)
            if d > r:
                continue
            else:
                a3 = b1
                b3 = - a1
                c3 = -(a3 * x2 + b3 * y2)
                intermediate_x = (b1 * c3 - b3 * c1) / (a1 * b3 - a3 * b1)
                intermediate_y = (a3 * c1 - a1 * c3) / (a1 * b3 - a3 * b1)
                if d == r:
                    _intersection_x = intermediate_x
                    _intersection_y = intermediate_y
                else:
                    if d >= 0 and d < r:
                        temp = r * r - d * d
                        if temp < 0.0 or temp > 999.0:
                            continue
                        l_vector = sqrt(temp)
                        intersection_x_1 = l_vector * line_unit_vector[0] + intermediate_x
                        intersection_y_1 = l_vector * line_unit_vector[1] + intermediate_y
                        intersection_x_2 = -l_vector * line_unit_vector[0] + intermediate_x
                        intersection_y_2 = -l_vector * line_unit_vector[1] + intermediate_y
                        line_vector_1[0] = intersection_x_1 - robot_pose[0]
                        line_vector_1[1] = intersection_y_1 - robot_pose[1]
                        # the intersection point must be located in the one direction of the line
                        if (line_vector_1[0] * line_unit_vector[0]) > 0 or (line_vector_1[1] * line_unit_vector[1] > 0):
                            scan_range = hypot(intersection_x_1 - robot_pose[0], intersection_y_1 - robot_pose[1])
                            if scan_range < min_range:
                                min_range = scan_range
                                _intersection_x = intersection_x_1
                                _intersection_y = intersection_y_1
                        line_vector_2[0] = intersection_x_2 - robot_pose[0]
                        line_vector_2[1] = intersection_y_2 - robot_pose[1]
                        if (line_vector_2[0] * line_unit_vector[0]) > 0 or (line_vector_2[1] * line_unit_vector[1] > 0):
                            scan_range = hypot(intersection_x_2 - robot_pose[0], intersection_y_2 - robot_pose[1])
                            if scan_range < min_range:
                                min_range = scan_range
                                _intersection_x = intersection_x_2
                                _intersection_y = intersection_y_2

        scans[i] = min_range
        scan_lines[4 * i + 0] = robot_pose[0]
        scan_lines[4 * i + 1] = robot_pose[1]
        scan_lines[4 * i + 2] = _intersection_x
        scan_lines[4 * i + 3] = _intersection_y
    free(line_vector_1)
    free(line_vector_2)

def create_cost_map():
    global robot_pose, num_scan, laser_resolution
    global margin, map_resolution, map_size
    global inflate, inflate_col, inflate_row 
    global cost_map, cost_map_output
    global scans
    cdef float float_num_scan = num_scan
    cdef float float_i = 0.
    cdef float angle_rel = 0.
    cdef float dx, dy, theta, scan_x, scan_y
    cdef int scan_index
    cdef int x_grid, y_grid
    cdef int grid_value
    cdef int idx1, idx2, idx3
    theta = robot_pose[2]
    for i in range(num_scan):
        float_i = i
        angle_rel = (float_i - (float_num_scan - 1.0) / 2.0) * laser_resolution
        # laser angle range: [-pi, pi]
        if angle_rel < -PI:
            angle_rel = -PI
        if angle_rel > PI:
            angle_rel = PI
        dx = scans[i] * cos(angle_rel)
        dy = scans[i] * sin(angle_rel)
        scan_x = dx * cos(theta) - dy * sin(theta) + robot_pose[0]
        scan_y = dx * sin(theta) + dy * cos(theta) + robot_pose[1]
        if scan_x <= margin and scan_x >= -margin and scan_y <= margin and scan_y >= -margin:
            x_grid = int((margin - scan_x) / map_resolution)
            y_grid = int((margin - scan_y) / map_resolution)
            if 0 <= x_grid and x_grid < map_size and 0 <= y_grid and y_grid < map_size:
                cost_map[y_grid + x_grid * map_size] = 0
    for i in range(map_size): # rows
        for j in range(map_size): # cols
            if cost_map[j + i * map_size] == 0:
                for k in range(j - 6, j + 6 + 1): # horizon, fatal collision
                    if k >= 0 and k < map_size:
                        cost_map_output[k + i * map_size] = 0
                for m in range(i - 6, i + 6 + 1): # verticle, fatal collision
                    if m >= 0 and m < map_size:
                        cost_map_output[j + m * map_size] = 0
                
                for n in range(7, 12): # extension, discomfort zone
                    grid_value = n * n
                    if i - n >= 0:
                        if cost_map_output[j + (i - n) * map_size] > grid_value:
                            cost_map_output[j + (i - n) * map_size] = grid_value
                    if i + n < map_size:
                        if cost_map_output[j + (i + n) * map_size] > grid_value:
                            cost_map_output[j + (i + n) * map_size] = grid_value
                    if j - n >= 0:
                        if cost_map_output[j - n + i * map_size] > grid_value:
                            cost_map_output[j - n + i * map_size] = grid_value
                    if j + n < map_size:
                        if cost_map_output[j + n + i * map_size] > grid_value:
                            cost_map_output[j + n + i * map_size] = grid_value
                            
                for i1 in range(inflate_row):
                    for j1 in range(inflate_col):
                        idx1 = j1 + i1 * inflate_col
                        if i1 == 0 and inflate[idx1] > 0:
                            for m in range(i - inflate[idx1], i + inflate[idx1] + 1):
                                if m >= 0 and m < map_size:
                                    idx2 = j - (inflate_col - j1)
                                    if idx2 >= 0:
                                        cost_map_output[idx2 + m * map_size] = 0
                                    idx2 = j + (inflate_col - j1)
                                    if idx2 < map_size:
                                        cost_map_output[idx2 + m * map_size] = 0
                        if i1 > 0 and inflate[idx1] > 0:
                            idx3 = j1 + (i1 - 1) * inflate_col
                            for m in range(inflate[idx3] + 1, inflate[idx1] + 1):
                                grid_value = (6 + i1) * (6 + i1)
                                idx2 = j - (inflate_col - j1)
                                if i - m >= 0 and idx2 >= 0:
                                    if cost_map_output[idx2 + (i - m) * map_size] > grid_value:
                                        cost_map_output[idx2 + (i - m) * map_size] = grid_value
                                if i + m < map_size and idx2 >= 0:
                                    if cost_map_output[idx2 + (i + m) * map_size] > grid_value:
                                        cost_map_output[idx2 + (i + m) * map_size] = grid_value
                                idx2 = j + (inflate_col - j1)
                                if i - m >= 0 and idx2 < map_size:
                                    if cost_map_output[idx2 + (i - m) * map_size] > grid_value:
                                        cost_map_output[idx2 + (i - m) * map_size] = grid_value
                                if i + m < map_size and idx2 < map_size:
                                    if cost_map_output[idx2 + (i + m) * map_size] > grid_value:
                                        cost_map_output[idx2 + (i + m) * map_size] = grid_value


def transform_scan_last():
    global robot_pose, num_scan, laser_resolution
    global scan_last_1, scan_last_2, scan_last_3
    global scan_end_last_1, scan_end_last_2, scan_end_last_3
    cdef float dx, dy, theta, scan_x, scan_y, scan_angle
    cdef int scan_index
    theta = robot_pose[2]
    for i in range(num_scan):
        dx = scan_end_last_1[2 * i] - robot_pose[0]
        dy = scan_end_last_1[2 * i + 1] - robot_pose[1]
        scan_x = dy * sin(theta) + dx * cos(theta)
        scan_y = dy * cos(theta) - dx * sin(theta)
        scan_angle = atan2(scan_y, scan_x)
        if scan_angle >= -PI and scan_angle <= PI:
            scan_index = int((scan_angle + (PI - laser_resolution / 2.0)) / laser_resolution)
            if scan_index >=0 and scan_index < num_scan:
                scan_last_1[scan_index] = hypot(scan_x, scan_y)
        
        dx = scan_end_last_2[2 * i] - robot_pose[0]
        dy = scan_end_last_2[2 * i + 1] - robot_pose[1]
        scan_x = dy * sin(theta) + dx * cos(theta)
        scan_y = dy * cos(theta) - dx * sin(theta)
        scan_angle = atan2(scan_y, scan_x)
        if scan_angle >= -PI and scan_angle <= PI:
            scan_index = int((scan_angle + (PI - laser_resolution / 2.0)) / laser_resolution)
            if scan_index >=0 and scan_index < num_scan:
                scan_last_2[scan_index] = hypot(scan_x, scan_y)

        dx = scan_end_last_3[2 * i] - robot_pose[0]
        dy = scan_end_last_3[2 * i + 1] - robot_pose[1]
        scan_x = dy * sin(theta) + dx * cos(theta)
        scan_y = dy * cos(theta) - dx * sin(theta)
        scan_angle = atan2(scan_y, scan_x)
        if scan_angle >= -PI and scan_angle <= PI:
            scan_index = int((scan_angle + (PI - laser_resolution / 2.0)) / laser_resolution)
            if scan_index >=0 and scan_index < num_scan:
                scan_last_3[scan_index] = hypot(scan_x, scan_y)    

    # for i in range(num_scan):
    #     if i == 0:
    #         if scan_last_1[i] > 999:
    #             scan_last_1[i] = min([scan_last_1[num_scan - 1], scan_last_1[num_scan - 2],\
    #                 scan_last_1[1], scan_last_1[2]])   
    #         if scan_last_2[i] > 999:
    #             scan_last_2[i] = min([scan_last_2[num_scan - 1], scan_last_2[num_scan - 2],\
    #                 scan_last_2[1], scan_last_2[2]]) 
    #         if scan_last_3[i] > 999:
    #             scan_last_3[i] = min([scan_last_3[num_scan - 1], scan_last_3[num_scan - 2],\
    #                 scan_last_3[1], scan_last_3[2]])
    #     elif i == 1:
    #         if scan_last_1[i] > 999:
    #             scan_last_1[i] = min([scan_last_1[num_scan - 1], scan_last_1[0],\
    #                 scan_last_1[3], scan_last_1[2]])   
    #         if scan_last_2[i] > 999:
    #             scan_last_2[i] = min([scan_last_2[num_scan - 1], scan_last_2[0],\
    #                 scan_last_2[3], scan_last_2[2]]) 
    #         if scan_last_3[i] > 999:
    #             scan_last_3[i] = min([scan_last_3[num_scan - 1], scan_last_3[0],\
    #                 scan_last_3[3], scan_last_3[2]])
        
    #     else:
    #         if scan_last_1[i] > 999:
    #             scan_last_1[i] = min([scan_last_1[i - 1], scan_last_1[i - 2], \
    #                 scan_last_1[(i + 1) % num_scan], scan_last_1[(i + 2) % num_scan]])   
    #         if scan_last_2[i] > 999:
    #             scan_last_2[i] = min([scan_last_2[i - 1], scan_last_2[i - 2], \
    #                 scan_last_2[(i + 1) % num_scan], scan_last_2[(i + 2) % num_scan]])   
    #         if scan_last_3[i] > 999:
    #             scan_last_3[i] = min([scan_last_3[i - 1], scan_last_3[i - 2], \
    #                 scan_last_3[(i + 1) % num_scan], scan_last_3[(i + 2) % num_scan]])   

def get_scan(index: int):
    global scans
    return scans[index]

def get_cost_map(index: int):
    global cost_map_output
    return cost_map_output[index]

def get_scan_line(index: int):
    global scan_lines
    return scan_lines[index]

def get_last_scan(index: int, scan: int):
    global scan_last_1, scan_last_2, scan_last_3
    if scan == 1:
        return scan_last_1[index]
    elif scan == 2:
        return scan_last_2[index]
    else:
        return scan_last_3[index]

def ReleaseEnv():
    global lines, circles, scans, scan_lines, robot_pose
    global scan_last_1, scan_last_2, scan_last_3
    global cost_map, cost_map_output, inflate
    global scan_end_last_1, scan_end_last_2, scan_end_last_3
    free(lines)
    free(circles)
    free(scans)
    free(scan_lines)
    free(robot_pose)
    free(scan_last_1)
    free(scan_last_2)
    free(scan_last_3)
    free(cost_map)
    free(cost_map_output)
    free(inflate)
    free(scan_end_last_1)
    free(scan_end_last_2)
    free(scan_end_last_3)