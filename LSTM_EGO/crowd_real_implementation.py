import logging
import rospy
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import PoseArray
from geometry_msgs.msg import Twist 
from threading import Lock
import numpy as np
from numpy.linalg import norm
from utils.human import Human
from utils.robot import Robot
from utils.state import *
from policy.policy_factory import policy_factory
from info import *
from math import atan2, hypot, sqrt, cos, sin, fabs, inf, ceil
from time import sleep, time
from C_library.motion_plan_lib import *


class CrowdSim:
    def __init__(self, args):
        self.only_dynamic = args.only_dynamic
        self.n_laser = args.lidar_dim
        self.laser_angle_resolute = args.laser_angle_resolute
        self.laser_min_range = args.laser_min_range
        self.laser_max_range = args.laser_max_range
        self.square_width = args.square_width
        self.human_policy_name = 'orca' # human policy is fixed orca policy
        
        
        # last-time distance from the robot to the goal
        self.goal_distance_last = None

        
        # scan_intersection, each line connects the robot and the end of each laser beam
        self.scan_intersection = None # used for visualization

        # laser state
        self.scan_current = np.zeros(self.n_laser, dtype=np.float32)
        
        self.global_time = None
        self.time_limit = 20
        self.time_step = 0.2
        self.randomize_attributes = False
        self.success_reward = 1.0
        self.collision_penalty = -0.3
        self.discomfort_dist = 0.2
        self.discomfort_penalty_factor = 0.5
        self.goal_distance_factor = 0.1
       

        # here, more lines can be added to simulate obstacles
        self.circles = None # human margin
        self.area_size = [6.0, 2.8]
     
        self.obstacle_x_margin = 2.5
        self.obstacle_y_margin = 1.3
        self.obstacle_num_max = 3

        self.human_obstacle_num_max = 3
        self.static_obstacle_num_max = 3
        self.static_obstacle_shape_range = [0.3, 0.6]

        self.human_v_pref = 0.3
        self.human_radius = 0.25

        self.human_num = None
        self.static_obstacle_num = None

        
        self.humans = None
        self.static_obstacles = None
        self.rectangles = None
        self.robot = Robot()
        self.robot.radius = 0.25
        self.robot.v_pref = 0.3
        self.robot_goal = [2.0, 0.0]
        self.robot_initial_position_area = [(-2.0, 0.0), (-1.0, 1.0)]
        self.robot.time_step = self.time_step

        self.log_env = {}
        self.goal_reached = False
        self.lock_robot_pose = Lock()
        self.lock_obstacle_pose = Lock()
        self.pub_cmd_vel = rospy.Publisher('cmd_vel', Twist, queue_size=5)
        sub_robot_pose = rospy.Subscriber('/robot/pose', PoseStamped, self.robot_pose_callback)
        sub_obstacle_pose = rospy.Subscriber('/adaptive_clustering/poses', PoseArray, self.obstacle_pose_callback)
        sleep(2.0)

    def cal_yaw_from_quaternion(self, q_0, q_1, q_2, q_3):
        # roll = atan2(2.0 * (q_0 * q_1 + q_2 * q_3), 1.0 - 2.0 * (q_1 * q_1 + q_2 * q_2))
        # pitch = asin(2.0 * (q_0 * q_2 - q_1 * q_3))
        yaw = atan2(2.0 * (q_0 * q_3 + q_1 * q_2), 1.0 - 2.0 * (q_2 * q_2 + q_3 * q_3))
        return yaw

    def robot_pose_callback(self, msg):
        position = msg.pose.position
        orientation = msg.pose.orientation
        yaw = self.cal_yaw_from_quaternion(orientation.w, orientation.x, orientation.y, orientation.z)
        self.lock_robot_pose.acquire()
        self.robot.px = position.x
        self.robot.py = position.y
        self.robot.theta = yaw
        self.lock_robot_pose.release()
        # print('robot pose: ', self.robot.px, self.robot.py, self.robot.theta)

    def obstacle_pose_callback(self, msg):
        obs_num = len(msg.poses)
        self.lines = [
                      [( self.area_size[0] / 2.0,  self.area_size[1] / 2.0), (-self.area_size[0] / 2.0,  self.area_size[1] / 2.0)], 
                      [(-self.area_size[0] / 2.0,  self.area_size[1] / 2.0), (-self.area_size[0] / 2.0, -self.area_size[1] / 2.0)], 
                      [(-self.area_size[0] / 2.0, -self.area_size[1] / 2.0), ( self.area_size[0] / 2.0, -self.area_size[1] / 2.0)], 
                      [( self.area_size[0] / 2.0, -self.area_size[1] / 2.0), ( self.area_size[0] / 2.0,  self.area_size[1] / 2.0)]  
                     ]
        self.lock_obstacle_pose.acquire()
        self.static_obstacle_num = 0
        if obs_num == 0:
            return
        self.human_num = obs_num
        self.humans = []
        for i in range(obs_num):
            human = Human()
            human.radius = self.human_radius
            human.px = msg.poses[i].position.x
            human.py = msg.poses[i].position.y
            self.humans.append(human)
            # print('human: ', i, human.px, human.py)
        self.lock_obstacle_pose.release()
        

    def get_lidar(self):
        scan = np.zeros(self.n_laser, dtype=np.float32)
        scan_end = np.zeros((self.n_laser, 2), dtype=np.float32)
        self.circles = np.zeros((self.human_num, 3), dtype=np.float32)
        # here, more circles can be added to simulate obstacles
        for i in range(self.human_num):
            self.circles[i, :] = np.array([self.humans[i].px, self.humans[i].py, self.humans[i].radius])
        robot_pose = np.array([self.robot.px, self.robot.py, 0.0])
        num_line = len(self.lines)
        num_circle = self.human_num
        InitializeEnv(num_line, num_circle, self.n_laser, self.laser_angle_resolute)
        for i in range (num_line):
            set_lines(4 * i    , self.lines[i][0][0])
            set_lines(4 * i + 1, self.lines[i][0][1])
            set_lines(4 * i + 2, self.lines[i][1][0])
            set_lines(4 * i + 3, self.lines[i][1][1])
        for i in range (num_circle):
            set_circles(3 * i    , self.humans[i].px)
            set_circles(3 * i + 1, self.humans[i].py)
            set_circles(3 * i + 2, self.humans[i].radius)
        set_robot_pose(robot_pose[0], robot_pose[1], robot_pose[2])
        cal_laser()
        self.scan_intersection = []
        for i in range(self.n_laser):
            scan[i] = get_scan(i)
            scan_end[i, :] = np.array([get_scan_line(4 * i + 2), get_scan_line(4 * i + 3)])
            ### used for visualization
            self.scan_intersection.append([(get_scan_line(4 * i + 0), get_scan_line(4 * i + 1)), \
                                           (get_scan_line(4 * i + 2), get_scan_line(4 * i + 3))])
            ### used for visualization
        
        self.scan_current = np.clip(scan, self.laser_min_range, self.laser_max_range) / self.laser_max_range
        ReleaseEnv()

    def reset(self, phase='test'):
        assert phase in ['train', 'val', 'test']
        self.global_time = 0
        self.log_env = {}
        # px, py, gx, gy, vx, vy, theta
        robot_position_x = self.robot.px
        robot_position_y = self.robot.py
        self.robot.set(robot_position_x, robot_position_y, self.robot_goal[0], self.robot_goal[1], 0, 0, 0)
        self.goal_distance_last = self.robot.get_goal_distance()

        self.get_lidar()

        # get the observation
        dx = self.robot.gx - self.robot.px
        dy = self.robot.gy - self.robot.py
        theta = self.robot.theta
        y_rel = dy * cos(theta) - dx * sin(theta)
        x_rel = dy * sin(theta) + dx * cos(theta)
        r = hypot(x_rel, y_rel) / self.square_width
        t = atan2(y_rel, x_rel) / np.pi
        ob_position = np.array([r, t], dtype=np.float32)

        self.log_env['robot'] = [np.array([self.robot.px, self.robot.py])]
        self.log_env['goal'] = [np.array([self.robot.gx, self.robot.gy])]
        humans_position = []
        for human in self.humans:
            humans_position.append(np.array([human.px, human.py]))
        self.log_env['humans'] = [np.array(humans_position)]
        static_obstacles_info = []
        for i in range (self.static_obstacle_num):
            static_obstacles_info.append(np.array([self.static_obstacles['positions'][i][0], 
                                                   self.static_obstacles['positions'][i][1],
                                                   self.static_obstacles['shapes'][i][0],  
                                                   self.static_obstacles['shapes'][i][1]]))
        self.log_env['static_obstacles'] = [np.array(static_obstacles_info)] 
        lasers = []
        for laser in self.scan_intersection:
            lasers.append(np.array([laser[0][0], laser[0][1], laser[1][0], laser[1][1]]))
        self.log_env['laser'] = [np.array(lasers)]
        sleep(self.time_step)
        return self.scan_current, ob_position

    def step(self, action):
        # uodate states
        action = action * self.robot.v_pref
        dx_world = action[0] * self.time_step
        dy_world = action[1] * self.time_step
        ct = cos(self.robot.theta)
        st = sin(self.robot.theta)
        dx_robot = dy_world * st + dx_world * ct
        dy_robot = dy_world * ct - dx_world * st
        move_cmd = Twist()
        if self.goal_reached:
            move_cmd.linear.x = 0.0
            move_cmd.linear.y = 0.0
        else:
            move_cmd.linear.x = dx_robot / self.time_step
            move_cmd.linear.y = dy_robot / self.time_step
            # avoid moving outside
            if self.robot.py > 1.1:
                move_cmd.linear.y = -0.1
            elif self.robot.py < -1.1:
                move_cmd.linear.y = 0.1
        self.pub_cmd_vel.publish(move_cmd)
        sleep(self.time_step)
        # get new laser scan and grid map
        self.get_lidar()  
        self.global_time += self.time_step
        
        # if reaching goal
        goal_dist = hypot(self.robot.px - self.robot.gx, self.robot.py - self.robot.gy)
        reaching_goal = goal_dist < self.robot.radius

    
        # collision detection between the robot and humans
        collision = False
        dmin = (self.scan_current * self.laser_max_range).min()
        if dmin <= 0.3:
            collision = True

        reward = 0
        if self.global_time >= self.time_limit:
            reward = 0
            done = True
            info = Timeout()
        elif collision:
            reward = self.collision_penalty
            done = False
            info = Collision()
        elif ((dmin - self.robot.radius) < self.discomfort_dist):
            # penalize agent for getting too close 
            reward = (dmin - self.robot.radius - self.discomfort_dist) * self.discomfort_penalty_factor
            done = False
            info = Danger(dmin)
        else:
            reward = 0
            done = False
            info = Nothing()

        if reaching_goal:
            self.goal_reached = True
            reward = reward + self.success_reward
            done = True
            info = ReachGoal()
        else:
            reward = reward + self.goal_distance_factor * (self.goal_distance_last - goal_dist)
        self.goal_distance_last = goal_dist
  

        # get the observation
        dx = self.robot.gx - self.robot.px
        dy = self.robot.gy - self.robot.py
        theta = self.robot.theta
        y_rel = dy * cos(theta) - dx * sin(theta)
        x_rel = dy * sin(theta) + dx * cos(theta)
        r = hypot(x_rel, y_rel) / self.square_width
        t = atan2(y_rel, x_rel) / np.pi
        ob_position = np.array([r, t], dtype=np.float32)

        self.log_env['robot'].append(np.array([self.robot.px, self.robot.py]))
        self.log_env['goal'].append(np.array([self.robot.gx, self.robot.gy])) 
        humans_position = []
        for human in self.humans:
            humans_position.append(np.array([human.px, human.py]))
        self.log_env['humans'].append(np.array(humans_position)) 
        static_obstacles_info = []
        for i in range (self.static_obstacle_num):
            static_obstacles_info.append(np.array([self.static_obstacles['positions'][i][0], 
                                                   self.static_obstacles['positions'][i][1],
                                                   self.static_obstacles['shapes'][i][0],  
                                                   self.static_obstacles['shapes'][i][1]]))
        self.log_env['static_obstacles'].append(np.array(static_obstacles_info)) 
        lasers = []
        for laser in self.scan_intersection:
            lasers.append(np.array([laser[0][0], laser[0][1], laser[1][0], laser[1][1]]))
        self.log_env['laser'].append(np.array(lasers))
        return self.scan_current, ob_position, reward, done, info
