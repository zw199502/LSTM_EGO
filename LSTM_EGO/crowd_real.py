import logging
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import collections as mc
from matplotlib.patches import Rectangle
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
        self.discomfort_dist = 0.1
        self.discomfort_penalty_factor = 0.5
        self.goal_distance_factor = 0.3
       

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


        plt.ion()
        plt.show()
        self.fig, self.ax = plt.subplots(figsize=(7, 7))

        self.log_env = {}

    def generate_random_static_obstacle(self):
        self.static_obstacles = {}
        self.rectangles = []
        if self.only_dynamic:
            self.static_obstacle_num = 0
        else:
            self.static_obstacle_num = np.random.randint(self.obstacle_num_max - self.human_num + 1,  size=1)[0]
        
        self.lines = [
                      [( self.area_size[0] / 2.0,  self.area_size[1] / 2.0), (-self.area_size[0] / 2.0,  self.area_size[1] / 2.0)], 
                      [(-self.area_size[0] / 2.0,  self.area_size[1] / 2.0), (-self.area_size[0] / 2.0, -self.area_size[1] / 2.0)], 
                      [(-self.area_size[0] / 2.0, -self.area_size[1] / 2.0), ( self.area_size[0] / 2.0, -self.area_size[1] / 2.0)], 
                      [( self.area_size[0] / 2.0, -self.area_size[1] / 2.0), ( self.area_size[0] / 2.0,  self.area_size[1] / 2.0)]  
                     ]
        if self.static_obstacle_num > 0:
            while True:
                shapes = np.random.uniform(self.static_obstacle_shape_range[0], self.static_obstacle_shape_range[1], size=(self.static_obstacle_num, 2))
                positions = np.zeros((self.static_obstacle_num, 2))
                for i in range(self.static_obstacle_num):
                    positions[i][0] = np.random.uniform(-(self.area_size[0] / 2.0 - self.robot.radius * 3), self.area_size[0] / 2.0 - shapes[i][0] - 0.01, size=1)[0]
                    positions[i][1] = np.random.uniform(-(self.area_size[1] / 2.0 - self.robot.radius * 3), self.area_size[1] / 2.0 - shapes[i][1] - 0.01, size=1)[0]
                    
                
                collision = False
                for i in range(self.static_obstacle_num):
                    if self.robot.px >= positions[i][0] - self.robot.radius * 2 and \
                       self.robot.px <= positions[i][0] + shapes[i][0] + self.robot.radius * 2 and \
                       self.robot.py >= positions[i][1] - self.robot.radius * 2 and \
                       self.robot.py <= positions[i][1] + shapes[i][1] + self.robot.radius * 2:
                        collision = True
                        break
                    if self.robot.gx >= positions[i][0] - self.robot.radius * 2 and \
                       self.robot.gx <= positions[i][0] + shapes[i][0] + self.robot.radius * 2 and \
                       self.robot.gy >= positions[i][1] - self.robot.radius * 2 and \
                       self.robot.gy <= positions[i][1] + shapes[i][1] + self.robot.radius * 2:
                        collision = True
                        break
                    temp = False
                    for j in range(i+1, self.static_obstacle_num):
                        if fabs(positions[i][0] + shapes[i][0] / 2.0 - positions[j][0] - shapes[j][0] / 2.0) <= (shapes[i][0] + shapes[j][0]) / 2.0 and \
                           fabs(positions[i][1] + shapes[i][1] / 2.0 - positions[j][1] - shapes[j][1] / 2.0) <= (shapes[i][1] + shapes[j][1]) / 2.0:
                            collision = True
                            temp = True
                            break
                    if temp:
                        break
                if not collision:
                    centers = positions + shapes / 2.0
                    radiuses = np.sqrt(np.sum(np.square(shapes), axis=1)) / 2.0
                    self.static_obstacles['positions'] = positions
                    self.static_obstacles['shapes'] = shapes
                    self.static_obstacles['centers'] = centers
                    self.static_obstacles['radiuses'] = radiuses
                    self.rectangles = [positions, shapes]
                    # add lines
                    for k in range(self.static_obstacle_num):
                        self.lines.append([(positions[k][0], positions[k][1]), 
                                        (positions[k][0] + shapes[k][0], positions[k][1])])
                        self.lines.append([(positions[k][0], positions[k][1]), 
                                        (positions[k][0], positions[k][1] + shapes[k][1])])
                        self.lines.append([(positions[k][0] + shapes[k][0], positions[k][1]), 
                                        (positions[k][0] + shapes[k][0], positions[k][1] + shapes[k][1])])
                        self.lines.append([(positions[k][0], positions[k][1] + shapes[k][1]),
                                        (positions[k][0] + shapes[k][0], positions[k][1] + shapes[k][1])])
                        # print(self.lines)
                    break

        
    def generate_random_human_position(self):
        
        if self.static_obstacle_num == 0 and self.human_num == 0:
            self.human_num = 1
        self.humans = []
        if self.human_num > 0:
            for i in range(self.human_num):
                self.humans.append(self.generate_circle_crossing_human())

            for i in range(len(self.humans)):
                human_policy = policy_factory[self.human_policy_name]()
                human_policy.max_speed = self.humans[i].v_pref
                human_policy.radius = self.human_radius
                human_policy.time_step = self.time_step
                self.humans[i].set_policy(human_policy)

    def generate_circle_crossing_human(self):
        human = Human()
        human.v_pref = np.random.uniform(self.human_v_pref / 3.0 * 2.0, self.human_v_pref, size=[1]) / 2.0
        human.time_step = self.time_step
        human.radius = self.human_radius

        while True:
            px = np.random.uniform(-(self.area_size[0] / 2.0 - human.radius * 2), self.area_size[0] / 2.0 - human.radius * 2, size=1)[0]
            py = np.random.uniform(-(self.area_size[1] / 2.0 - human.radius * 2), self.area_size[1] / 2.0 - human.radius * 2, size=1)[0]
            
            g_side = np.random.randint(2, size=1)[0]
            if px >=0 and py > 0:
                if g_side == 0:
                    gx = np.random.uniform(-(self.area_size[0] / 2.0 - human.radius * 2), self.area_size[0] / 2.0 - human.radius * 2, size=1)[0]
                    gy = -(self.area_size[1] / 2.0 - human.radius * 2)
                else:
                    gx = -(self.area_size[0] / 2.0 - human.radius * 2)
                    gy = np.random.uniform(-(self.area_size[1] / 2.0 - human.radius * 2), self.area_size[1] / 2.0 - human.radius * 2, size=1)[0]
            elif px >= 0 and py <= 0:
                if g_side == 0:
                    gx = np.random.uniform(-(self.area_size[0] / 2.0 - human.radius * 2), self.area_size[0] / 2.0 - human.radius * 2, size=1)[0]
                    gy = self.area_size[1] / 2.0 - human.radius * 2
                else:
                    gx = -(self.area_size[0] / 2.0 - human.radius * 2)
                    gy = np.random.uniform(-(self.area_size[1] / 2.0 - human.radius * 2), self.area_size[1] / 2.0 - human.radius * 2, size=1)[0]
            elif px < 0 and py > 0:
                if g_side == 0:
                    gx = np.random.uniform(-(self.area_size[0] / 2.0 - human.radius * 2), self.area_size[0] / 2.0 - human.radius * 2, size=1)[0]
                    gy = -(self.area_size[1] / 2.0 - human.radius * 2)
                else:
                    gx = self.area_size[0] / 2.0 - human.radius * 2
                    gy = np.random.uniform(-(self.area_size[1] / 2.0 - human.radius * 2), self.area_size[1] / 2.0 - human.radius * 2, size=1)[0]
            else:
                if g_side == 0:
                    gx = np.random.uniform(-(self.area_size[0] / 2.0 - human.radius * 2), self.area_size[0] / 2.0 - human.radius * 2, size=1)[0]
                    gy = self.area_size[1] / 2.0 - human.radius * 2
                else:
                    gx = self.area_size[0] / 2.0 - human.radius * 2
                    gy = np.random.uniform(-(self.area_size[1] / 2.0 - human.radius * 2), self.area_size[1] / 2.0 - human.radius * 2, size=1)[0]

            collide = False
            for agent in [self.robot] + self.humans:
                min_dist = human.radius * 2 + agent.radius
                if norm((px - agent.px, py - agent.py)) < min_dist or \
                        norm((gx - agent.gx, gy - agent.gy)) < min_dist:
                    collide = True
                    break
            if self.static_obstacle_num > 0:
                for static_obs in range(self.static_obstacle_num):
                    min_dist = human.radius * 2 + self.static_obstacles['radiuses'][static_obs]
                    if norm((px - self.static_obstacles['centers'][static_obs][0], 
                            py - self.static_obstacles['centers'][static_obs][1])) < min_dist or \
                       norm((gx - self.static_obstacles['centers'][static_obs][0], 
                            gy - self.static_obstacles['centers'][static_obs][1])) < min_dist:
                        collide = True
                        break
            if not collide:
                break
        # px, py, gx, gy, vx, vy, theta
        human.set(px, py, gx, gy, 0, 0, 0)
        return human
        

    def get_lidar(self):
        scan = np.zeros(self.n_laser, dtype=np.float32)
        scan_end = np.zeros((self.n_laser, 2), dtype=np.float32)
        self.circles = np.zeros((self.human_num, 3), dtype=np.float32)
        # here, more circles can be added to simulate obstacles
        for i in range(self.human_num):
            self.circles[i, :] = np.array([self.humans[i].px, self.humans[i].py, self.humans[i].radius])
        robot_pose = np.array([self.robot.px, self.robot.py, self.robot.theta])
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
        robot_position_x = np.random.uniform(self.robot_initial_position_area[0][0], self.robot_initial_position_area[0][1], size=1)[0]
        robot_position_y = np.random.uniform(self.robot_initial_position_area[1][0], self.robot_initial_position_area[1][1], size=1)[0]
        self.robot.set(robot_position_x, robot_position_y, self.robot_goal[0], self.robot_goal[1], 0, 0, 0)
        self.goal_distance_last = self.robot.get_goal_distance()

        self.human_num = np.random.randint(self.human_obstacle_num_max + 1,  size=1)[0]
        
        
        self.generate_random_static_obstacle()

        self.generate_random_human_position()

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
        return self.scan_current, ob_position

    def step(self, action):
        human_actions = []
        for human in self.humans:
            # observation for humans is always coordinates
            ob = [other_human.get_observable_state() for other_human in self.humans if other_human != human]
            for k in range(self.static_obstacle_num):
                ob.append(ObservableState(
                           self.static_obstacles['centers'][k][0], 
                           self.static_obstacles['centers'][k][1], 
                           0.0, 0.0, self.static_obstacles['radiuses'][k])
                         )
            human_actions.append(human.act(ob))

        # uodate states
        action = action * self.robot.v_pref
        
        robot_x, robot_y, robot_theta = self.robot.compute_pose(action)
        self.robot.update_states(robot_x, robot_y, robot_theta, action)
        for i, human_action in enumerate(human_actions):
            self.humans[i].update_states(human_action)

        # get new laser scan and grid map
        self.get_lidar()  
        self.global_time += self.time_step
        
        # if reaching goal
        goal_dist = hypot(robot_x - self.robot.gx, robot_y - self.robot.gy)
        reaching_goal = goal_dist < self.robot.radius

        # collision detection between humans
        for i in range(self.human_num):
            for j in range(i + 1, self.human_num):
                dx = self.humans[i].px - self.humans[j].px
                dy = self.humans[i].py - self.humans[j].py
                dist = (dx ** 2 + dy ** 2) ** (1 / 2) - self.humans[i].radius - self.humans[j].radius
                if dist < 0:
                    # detect collision but don't take humans' collision into account
                    logging.debug('Collision happens between humans in step()')

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
            done = True
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

    def render(self, mode='laser'):
        if mode == 'laser':
            self.ax.set_xlim(-3.0, 3.0)
            self.ax.set_ylim(-3.0, 3.0)
            for human in self.humans:
                human_circle = plt.Circle(human.get_position(), human.radius, fill=False, color='b')
                self.ax.add_artist(human_circle)
            self.ax.add_artist(plt.Circle(self.robot.get_position(), self.robot.radius, fill=True, color='r'))
            self.ax.add_artist(plt.Circle(self.robot_goal, self.robot.radius, fill=True, color='g'))
            for i in range(self.static_obstacle_num):
                self.ax.add_patch(Rectangle(self.rectangles[0][i], 
                                            self.rectangles[1][i][0], self.rectangles[1][i][1],
                                            facecolor='c',
                                            fill=True))
            plt.text(-4.5, -4.5, str(round(self.global_time, 2)), fontsize=20)
            plt.plot([-self.area_size[0] / 2, self.area_size[0] / 2], [-self.area_size[1] / 2, -self.area_size[1] / 2], color='k')
            plt.plot([-self.area_size[0] / 2, self.area_size[0] / 2], [self.area_size[1] / 2, self.area_size[1] / 2], color='k')
            # x, y, theta = self.robot.px, self.robot.py, self.robot.theta
            # dx = cos(theta)
            # dy = sin(theta)
            # self.ax.arrow(x, y, dx, dy,
            #     width=0.01,
            #     length_includes_head=True, 
            #     head_width=0.15,
            #     head_length=1,
            #     fc='r',
            #     ec='r')
            ii = 0
            lines = []
            while ii < self.n_laser:
                lines.append(self.scan_intersection[ii])
                ii = ii + 36
            lc = mc.LineCollection(lines)
            self.ax.add_collection(lc)
            plt.draw()
            plt.pause(0.001)
            plt.cla()
