import numpy as np
import math
from math import fabs, sin, cos, pi, tan, atan2, log
import threading

import matplotlib.pyplot as plt
from grid_map import OccupancyGridMap
from a_star import a_star


import cv2

###################### create cost map ##########################
HEIGHT = 200
WIDTH = 200
RESOLUTION = 0.05
ORIGIN = [5.0, 5.0]
###################### create cost map ##########################

class CostMap():
    def __init__(self, height, width, resolution, origin):
        self.height = height
        self.width = width
        self.resolution = resolution
        self.origin = origin
        self.angle
    
        

    def create_costmap(self, scan, current_robot_pose):
        grid_map = 255 * np.ones((HEIGHT, WIDTH), dtype=np.uint8)
        scan_data = scan.ranges
        angle_min = scan.angle_min
        angle_increment=scan.angle_increment
        scan_range_max = scan.range_max
        scan_dim = len(scan_data)
        for i in range(scan_dim):
            if (not math.isnan(scan_data[i])) and (scan_data[i] <= scan_range_max):
                beta = (angle_min + i * angle_increment)
                x_robot_frame = scan_data[i] * cos(beta)
                y_robot_frame = scan_data[i] * sin(beta)
                c_theta = cos(current_robot_pose[2])
                s_theta = sin(current_robot_pose[2])
                x_odom_frame = current_robot_pose[0] + x_robot_frame * c_theta - y_robot_frame * s_theta
                y_odom_frame = current_robot_pose[1] + x_robot_frame * s_theta + y_robot_frame * c_theta
                x_grid_map_frame = ORIGIN[0] - x_odom_frame
                y_grid_map_frame = ORIGIN[1] - y_odom_frame
                x_grid = int(x_grid_map_frame / RESOLUTION)
                y_grid = int(y_grid_map_frame / RESOLUTION)
                if (0 <= x_grid < WIDTH) and (0 <= y_grid < HEIGHT):
                    grid_map[x_grid, y_grid] = 0
        # dilate
        dst = 255 * np.ones((HEIGHT + 22, WIDTH + 22), dtype=np.uint8)
        
        layer = np.array([[ 0,  0,  0,  0,  0,  3,  4,  5,  5,  5],
                          [ 0,  0,  0,  0,  3,  4,  5,  6,  6,  6],
                          [ 0,  0,  0,  3,  5,  6,  6,  7,  7,  7],
                          [ 0,  0,  4,  5,  6,  7,  8,  8,  8,  8],
                          [ 0,  4,  5,  7,  7,  8,  9,  9,  9,  9],
                          [ 4,  6,  7,  8,  9,  9, 10, 10, 10, 10]])
        for i in range(HEIGHT):
            for j in range(WIDTH):
                if grid_map[i, j] == 0:
                    # up, dowm, left, right
                    dst[i + 11 - 6 : i + 11 + 6 + 1, j + 11] = 0
                    dst[i + 11, j + 11 - 6 : j + 11 + 6 + 1] = 0
                    for m in range(7, 12):
                        grid_value = m * m
                        if dst[i + 11 - m, j + 11] > grid_value:
                            dst[i + 11 - m, j + 11] = grid_value
                        if dst[i + 11 + m, j + 11] > grid_value:
                            dst[i + 11 + m, j + 11] = grid_value
                        if dst[i + 11, j + 11 - m] > grid_value:
                            dst[i + 11, j + 11 - m] = grid_value
                        if dst[i + 11, j + 11 + m] > grid_value:
                            dst[i + 11, j + 11 + m] = grid_value
                    # others
                    for i1 in range(layer.shape[0]):
                        for j1 in range(layer.shape[1]):
                            if i1 == 0 and layer[i1, j1] != 0:
                                dst[i + 11 - layer[i1, j1] : i + 11 + layer[i1, j1] + 1, j + 11 - (layer.shape[1] - j1)] = 0
                                dst[i + 11 - layer[i1, j1] : i + 11 + layer[i1, j1] + 1, j + 11 + (layer.shape[1] - j1)] = 0
                            if i1 > 0 and layer[i1, j1] != 0:
                                for n in range(layer[i1 - 1, j1] + 1, layer[i1, j1] + 1):
                                    grid_value = (6 + i1) * (6 + i1)
                                    if dst[i + 11 - n, j + 11 - (layer.shape[1] - j1)] > grid_value:
                                        dst[i + 11 - n, j + 11 - (layer.shape[1] - j1)] = grid_value
                                    if dst[i + 11 + n, j + 11 - (layer.shape[1] - j1)] > grid_value:
                                        dst[i + 11 + n, j + 11 - (layer.shape[1] - j1)] = grid_value
                                    if dst[i + 11 - n, j + 11 + (layer.shape[1] - j1)] > grid_value:
                                        dst[i + 11 - n, j + 11 + (layer.shape[1] - j1)] = grid_value
                                    if dst[i + 11 + n, j + 11 + (layer.shape[1] - j1)] > grid_value:
                                        dst[i + 11 + n, j + 11 + (layer.shape[1] - j1)] = grid_value
                       
        _dst = dst[11 : 11 + HEIGHT, 11 : 11 + WIDTH]
        # ret, binary = cv2.threshold(grid_map, 0, 255, cv2.THRESH_BINARY)
        # # why 23: 0.3 / RESOLUTION * 2 + 1, 0.3m means the robot would obviously be in collision.
        # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
        # dst = cv2.erode(binary, kernel)
        # cv2.imwrite('image2.jpg', _dst)     
        gmap = OccupancyGridMap(_dst, RESOLUTION)  
        return gmap
    
    def plot_path(sefl, path):
        start_x, start_y = path[0]
        goal_x, goal_y = path[-1]

        # plot path
        path_arr = np.array(path)
        plt.plot(path_arr[:, 1], path_arr[:, 0], 'y')

        # plot start point
        plt.plot(start_y, start_x, 'ro')

        # plot goal point
        plt.plot(goal_y, goal_x, 'go')
        # print(start_x, start_y)
        # print(goal_x, goal_y)

        # plt.show()

    def getGoalDistace(self, map, current_robot_pose):
        start_node = (ORIGIN[0] - current_robot_pose[0], ORIGIN[1] - current_robot_pose[1])
        goal_node = (ORIGIN[0] - self.goal_position.position.x, ORIGIN[1] - self.goal_position.position.y)
        path, path_px, cost, flag = a_star(start_node, goal_node, map, movement='8N')
        
        # map.plot()

        # if path:
        #     # plot resulting path in pixels over the map
        #     self.plot_path(path_px)
        # else:
        #     print('Goal is not reachable')

        #     # plot start and goal points over the map (in pixels)
        #     start_node_px = map.get_index_from_coordinates(start_node[0], start_node[1])
        #     goal_node_px = map.get_index_from_coordinates(goal_node[0], goal_node[1])

        #     plt.plot(start_node_px[0], start_node_px[1], 'ro')
        #     plt.plot(goal_node_px[0], goal_node_px[1], 'go')
        # plt.pause(0.001)
        # plt.clf()
      
        # print(cost)
        
        return cost, flag

    def getOdometry(self, odom):
        
        orientation = odom.pose.pose.orientation
        q_x, q_y, q_z, q_w = orientation.x, orientation.y, orientation.z, orientation.w
        self.yaw = atan2(2 * (q_x * q_y + q_w * q_z), 1 - 2 * (q_y * q_y + q_z * q_z))
        dx = self.goal_position.position.x - odom.pose.pose.position.x
        dy = self.goal_position.position.y - odom.pose.pose.position.y
        c_theta = cos(self.yaw)
        s_theta = sin(self.yaw)
        rel_dis_x = dy * s_theta + dx * c_theta
        rel_dis_y = dy * c_theta - dx * s_theta
        rel_theta = atan2(rel_dis_y, rel_dis_x)

        self.state_lock.acquire()
        self.velocity_current[0] = odom.twist.twist.linear.x
        self.velocity_current[1] = odom.twist.twist.angular.z
        self.rel_theta = rel_theta
        self.rel_distance = math.hypot(dx, dy)

        self.robot_pose = [odom.pose.pose.position.x, odom.pose.pose.position.y, self.yaw]
        self.state_lock.release()

    def getState(self, scan):
        scan_range = []
        done = False
        arrive = False

        self.state_lock.acquire()
        velocity = self.velocity_current
        rel_theta = self.rel_theta
        ref_distance = self.rel_distance
        current_robot_pose = self.robot_pose
        self.state_lock.release()

        self.min_scan_range = min(scan.ranges)
        # print('min_range: ', self.min_scan_range)
        if (self.collision_min_range > self.min_scan_range):
            done = True
        for i in range(len(scan.ranges)):
            if scan.ranges[i] == float('Inf') or scan.ranges[i] > 3.5:
                scan_range.append(3.5)
            elif np.isnan(scan.ranges[i]) or scan.ranges[i] < 0.5:
                scan_range.append(0.5)
            else:
                scan_range.append(scan.ranges[i])
        
        scan_range_reduce = []
        j = 0
        # original scan: 901 dimensions
        # reduced scan: 37 dimnensions
        while (j < len(scan_range)):
            scan_range_reduce.append(scan_range[j])
            j = j + 25
        
        if ref_distance <= self.threshold_arrive:
            # done = True
            arrive = True
        # for this condition, the robot can arrive at the target free of collision if using traditional method like dwa and ftc
        # such processing way can avoid sparse reward
        # if (current_distance < (min(scan_range) - min_range)) and (math.fabs(rel_theta) < math.pi / 4.0):
        #     arrive = True 
        
        return scan_range_reduce, velocity, ref_distance, rel_theta, done, arrive, current_robot_pose

    def setReward(self, done, arrive, map, current_robot_pose):
   
        if done:
            reward = -100.
            self.pub_cmd_vel.publish(Twist())
            return reward

        if arrive:
            reward = 120.
            self.pub_cmd_vel.publish(Twist())
            rospy.wait_for_service('/gazebo/delete_model')
            self.del_model('target')

            # Build the target
            rospy.wait_for_service('/gazebo/spawn_sdf_model')
            try:
                goal_urdf = open(goal_model_dir, "r").read()
                target = SpawnModel
                target.model_name = 'target'  # the same with sdf name
                target.model_xml = goal_urdf
                index = random.randint(0,(self.target_position_set.shape[0]-1))
                self.goal_position.position.x = self.target_position_set[index, 0]
                self.goal_position.position.y = self.target_position_set[index, 1]
                self.goal(target.model_name, target.model_xml, 'namespace', self.goal_position, 'world')
            except (rospy.ServiceException) as e:
                print("/gazebo/failed to build the target")
            rospy.wait_for_service('/gazebo/unpause_physics')
           
            self.past_distance, flag = self.getGoalDistace(map, current_robot_pose)
            if flag == False:
                print('reset goal false')
           
            arrive = False

            return reward

        current_distance, flag = self.getGoalDistace(map, current_robot_pose)
        if flag == False:
            print('calculate path length false')
            reward = -100.
            return reward
        distance_rate = (self.past_distance - current_distance)
        
        reward = 500.*distance_rate
        self.past_distance = current_distance

        return reward

    def step(self, action):
        # index = random.randint(0,(self.target_position_set.shape[0]-1))
        # self.goal_position.position.x = self.target_position_set[index, 0]
        # self.goal_position.position.y = self.target_position_set[index, 1]
        # print(self.goal_position.position.x, self.goal_position.position.y)
        linear_vel = action[0]
        ang_vel = action[1]

        vel_cmd = Twist()
        vel_cmd.linear.x = linear_vel / 4.0
        vel_cmd.angular.z = ang_vel / 2.0
        self.pub_cmd_vel.publish(vel_cmd)

        for i in range(self.scan_interval):
            data = None
            # t1 = rospy.get_time()
            while data is None:
                try:
                    data = rospy.wait_for_message('scan', LaserScan, timeout=5)
                except:
                    pass
        # t2 = rospy.get_time()
        # print(t2 - t1)  # from 0.174 to 0.198s
        
        state, velocity, rel_dis, rel_theta, done, arrive, current_robot_pose = self.getState(data)
        map = self.create_costmap(data, current_robot_pose)
        state = [i / 3.5 for i in state]

        state_ = np.array([self.velocity_last[0], self.velocity_last[1], rel_dis / diagonal_dis, rel_theta / math.pi])
        self.velocity_last[0] = velocity[0]
        self.velocity_last[1] = velocity[1]

        reward = self.setReward(done, arrive, map, current_robot_pose)
        # reward = 0

        return np.asarray(state), state_, reward, done, arrive

    def reset(self, episode):
        # Reset the env #
        if episode > 0:
            rospy.wait_for_service('/gazebo/delete_model')
            self.del_model('target')

        rospy.wait_for_service('gazebo/reset_simulation')
        try:
            self.reset_proxy()
        except (rospy.ServiceException) as e:
            print("gazebo/reset_simulation service call failed")

        # Build the targetz
        rospy.wait_for_service('/gazebo/spawn_sdf_model')
        try:
            goal_urdf = open(goal_model_dir, "r").read()
            target = SpawnModel
            target.model_name = 'target'  # the same with sdf name
            target.model_xml = goal_urdf
            index = random.randint(0,(self.target_position_set.shape[0]-1))
            self.goal_position.position.x = self.target_position_set[index, 0]
            self.goal_position.position.y = self.target_position_set[index, 1]

            # if -0.3 < self.goal_position.position.x < 0.3 and -0.3 < self.goal_position.position.y < 0.3:
            #     self.goal_position.position.x += 1
            #     self.goal_position.position.y += 1

            self.goal(target.model_name, target.model_xml, 'namespace', self.goal_position, 'world')
        except (rospy.ServiceException) as e:
            print("/gazebo/failed to build the target")
        rospy.wait_for_service('/gazebo/unpause_physics')
        data = None
        for i in range(self.scan_interval):
            data = None
            # t1 = rospy.get_time()
            while data is None:
                try:
                    data = rospy.wait_for_message('scan', LaserScan, timeout=5)
                except:
                    pass
        
        state, velocity, rel_dis, rel_theta, done, arrive, current_robot_pose = self.getState(data)
        map = self.create_costmap(data, current_robot_pose)
        self.past_distance, flag = self.getGoalDistace(map, current_robot_pose)
        if flag == False:
            print('reset environment false')
        state = [i / 3.5 for i in state]

        state_ = np.array([0.0, 0.0, rel_dis / diagonal_dis, rel_theta / math.pi])

        return np.asarray(state), state_

# if __name__ == '__main__':
#     rospy.init_node('env')
#     env = Env()
#     a = np.array([0.0, 0.0])
#     env.step(a)