// Copyright (C) 2018  Zhi Yan and Li Sun

// This program is free software: you can redistribute it and/or modify it
// under the terms of the GNU General Public License as published by the Free
// Software Foundation, either version 3 of the License, or (at your option)
// any later version.

// This program is distributed in the hope that it will be useful, but WITHOUT
// ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
// FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for
// more details.

// You should have received a copy of the GNU General Public License along
// with this program.  If not, see <http://www.gnu.org/licenses/>.

// ROS
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/PoseArray.h>
#include <visualization_msgs/MarkerArray.h>
#include "aloam_velodyne/ClusterArray.h"

// PCL
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/passthrough.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/common/common.h>
#include <pcl/common/centroid.h>

#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Core>
#include <mutex>
#include <iostream>

//#define LOG
ros::Publisher cloud_point_global_pub_;
ros::Publisher cluster_array_pub_;
ros::Publisher cloud_filtered_pub_;
ros::Publisher pose_array_pub_;
ros::Publisher marker_array_pub_;

// robot pose in the world frame
double parameters[7] = {0, 0, 0, 1, 0, 0, 0};
Eigen::Map<Eigen::Quaterniond> q_w_curr(parameters);
Eigen::Map<Eigen::Vector3d> t_w_curr(parameters + 4);
std::mutex RobotPoseMutex;

bool print_fps_;
float x_axis_min_;
float x_axis_max_;
float y_axis_min_;
float y_axis_max_;
float z_axis_min_;
float z_axis_max_;
bool human_size_limit_;
int cluster_size_min_;
int cluster_size_max_;

const int region_max_ = 6; // Change this value to match how far you want to detect.
int regions_[100];

int frames; clock_t start_time; bool reset = true;//fps

std::string frame_id = "aft_mapped";

void robotPoseCallback(const geometry_msgs::PoseStampedConstPtr &robotPoseMsg)
{
    RobotPoseMutex.lock();
    t_w_curr.x() = robotPoseMsg->pose.position.x;
    t_w_curr.y() = robotPoseMsg->pose.position.y;
    t_w_curr.z() = robotPoseMsg->pose.position.z;
    q_w_curr.x() = robotPoseMsg->pose.orientation.x;
    q_w_curr.y() = robotPoseMsg->pose.orientation.y;
    q_w_curr.z() = robotPoseMsg->pose.orientation.z;
    q_w_curr.w() = robotPoseMsg->pose.orientation.w;
    RobotPoseMutex.unlock();
}

void pointAssociateToMap(pcl::PointXYZI const *const pi, pcl::PointXYZI *const po)
{
	Eigen::Vector3d point_curr(pi->x, pi->y, pi->z);
	Eigen::Vector3d point_w = q_w_curr * point_curr + t_w_curr;
	po->x = point_w.x();
	po->y = point_w.y();
	po->z = point_w.z();
	po->intensity = pi->intensity;
	//po->intensity = 1.0;
}


void pointCloudCallback(const sensor_msgs::PointCloud2::ConstPtr& ros_pc2_in) {
  if(print_fps_)if(reset){frames=0;start_time=clock();reset=false;}//fps
  
  /*** Convert ROS message to PCL ***/
  pcl::PointCloud<pcl::PointXYZI>::Ptr pcl_pc_in(new pcl::PointCloud<pcl::PointXYZI>);
  pcl::PointCloud<pcl::PointXYZI>::Ptr pcl_pc_in_transformed(new pcl::PointCloud<pcl::PointXYZI>);
  pcl::PointCloud<pcl::PointXYZI>::Ptr pcl_pc_in_filter(new pcl::PointCloud<pcl::PointXYZI>);
  pcl::PointCloud<pcl::PointXYZI>::Ptr PedestrianPointCloudGloubalMap(new pcl::PointCloud<pcl::PointXYZI>);
  pcl::fromROSMsg(*ros_pc2_in, *pcl_pc_in);

  pcl_pc_in_transformed->header = pcl_pc_in->header;
  pcl_pc_in_filter->header = pcl_pc_in->header;

  // transform the local point to into the world frame
  int point_num = pcl_pc_in->points.size();
  std::cout<<"point_num: "<<point_num<<std::endl;
  boost::array<std::vector<int>, region_max_> indices_array;
  std::vector<int> outside_point;
  for (int i = 0; i < point_num; i++)
  {
    pcl::PointXYZI po;
    float x = pcl_pc_in->points[i].x;
    float y = pcl_pc_in->points[i].y;
    float z = pcl_pc_in->points[i].z;
    if(x > x_axis_min_ && x < x_axis_max_ && y > y_axis_min_ && y < y_axis_max_ && z > z_axis_min_ && z < z_axis_max_){
      pointAssociateToMap(&pcl_pc_in->points[i], &po);
      pcl_pc_in_transformed->push_back(po);
      if (po.x >= -3.5 && po.x <= 3.5 && po.y >= -1.2 && po.y <= 1.2){
        float range = 0.0;
        float d2 = x * x + y * y + z * z;
        for(int j = 0; j < region_max_; j++) {
          if(d2 > range * range && d2 <= (range+regions_[j]) * (range+regions_[j])) {
            indices_array[j].push_back(i);
            break;
          }
          range += regions_[j];
        }
        pcl_pc_in_filter->push_back(pcl_pc_in->points[i]);
      }
    }
  }
  // cost time !!!
  // sensor_msgs::PointCloud2 pcl_pc_in_transformed_msg;
  // pcl::toROSMsg(*pcl_pc_in_transformed, pcl_pc_in_transformed_msg);
  // pcl_pc_in_transformed_msg.header.stamp = ros_pc2_in->header.stamp;
  // pcl_pc_in_transformed_msg.header.frame_id = "camera_init";
  // cloud_point_global_pub_.publish(pcl_pc_in_transformed_msg);
  // cost time !!!
  
  
  /*** Euclidean clustering ***/
  float tolerance = 0.0;
  std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr, Eigen::aligned_allocator<pcl::PointCloud<pcl::PointXYZI>::Ptr > > clusters;
  
  for(int i = 0; i < region_max_; i++) {
    tolerance += 0.1;
    if(indices_array[i].size() > cluster_size_min_) {
      boost::shared_ptr<std::vector<int> > indices_array_ptr(new std::vector<int>(indices_array[i]));
      pcl::search::KdTree<pcl::PointXYZI>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZI>);
      tree->setInputCloud(pcl_pc_in, indices_array_ptr);
      
      std::vector<pcl::PointIndices> cluster_indices;
      pcl::EuclideanClusterExtraction<pcl::PointXYZI> ec;
      ec.setClusterTolerance(tolerance);
      ec.setMinClusterSize(cluster_size_min_);
      ec.setMaxClusterSize(cluster_size_max_);
      ec.setSearchMethod(tree);
      ec.setInputCloud(pcl_pc_in);
      ec.setIndices(indices_array_ptr);
      ec.extract(cluster_indices);
      
      for(std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin(); it != cluster_indices.end(); it++) {
      	pcl::PointCloud<pcl::PointXYZI>::Ptr cluster(new pcl::PointCloud<pcl::PointXYZI>);
      	for(std::vector<int>::const_iterator pit = it->indices.begin(); pit != it->indices.end(); ++pit) {
      	  cluster->points.push_back(pcl_pc_in->points[*pit]);
  	    }
      	cluster->width = cluster->size();
      	cluster->height = 1;
      	cluster->is_dense = true;

        Eigen::Vector4f min_box, max_box;
      	pcl::getMinMax3D(*cluster, min_box, max_box);
	
      	// Size limitation is not reasonable, but it can increase fps.
      	// if(human_size_limit_ &&
        //    (max_box[0]-min_box[0] < 0.1 || max_box[0]-min_box[0] > 0.8 ||
        //     max_box[1]-min_box[1] < 0.1 || max_box[1]-min_box[1] > 0.8 ||
        //     max_box[2]-min_box[2] < 0.1 || max_box[2]-min_box[2] > 2.0))
        // {
        //   continue;
        // }
          

	      clusters.push_back(cluster);
      }
    }
  }
  
  /*** Output ***/
  // cost time
  // if(cloud_filtered_pub_.getNumSubscribers() > 0) {
  //   pcl::PointCloud<pcl::PointXYZI>::Ptr pcl_pc_out(new pcl::PointCloud<pcl::PointXYZI>);
  //   sensor_msgs::PointCloud2 ros_pc2_out;
  //   pcl::toROSMsg(*pcl_pc_in_transformed, ros_pc2_out);
  //   cloud_filtered_pub_.publish(ros_pc2_out);
  // }
  // cost time
  
  aloam_velodyne::ClusterArray cluster_array;
  geometry_msgs::PoseArray pose_array;
  visualization_msgs::MarkerArray marker_array;
  
  for(int i = 0; i < clusters.size(); i++) {
    Eigen::Vector4f centroid;
    pcl::compute3DCentroid(*clusters[i], centroid);

    // transform local centroid into the global frame
    pcl::PointXYZI pi_1;
    pi_1.x = centroid[0];
    pi_1.y = centroid[1];
    pi_1.z = centroid[2];
    pi_1.intensity = 1.0;
    pcl::PointXYZI po_1;
    pointAssociateToMap(&pi_1, &po_1);
    centroid[0] = po_1.x;
    centroid[1] = po_1.y;
    centroid[2] = po_1.z;

    if(centroid[0] > 2.5 || centroid[0] < -2.5){
      continue;
    }
    if(cluster_array_pub_.getNumSubscribers() > 0) {
      sensor_msgs::PointCloud2 ros_pc2_out;
      pcl::toROSMsg(*clusters[i], ros_pc2_out);
      cluster_array.clusters.push_back(ros_pc2_out);
    }
    
    if(pose_array_pub_.getNumSubscribers() > 0) {
      
      
      geometry_msgs::Pose pose;
      pose.position.x = centroid[0];
      pose.position.y = centroid[1];
      pose.position.z = centroid[2];
      pose.orientation.w = 1;
      pose_array.poses.push_back(pose);
      
#ifdef LOG
      Eigen::Vector4f min, max;
      pcl::getMinMax3D(*clusters[i], min, max);
      std::cerr << ros_pc2_in->header.seq << " "
		<< ros_pc2_in->header.stamp << " "
		<< min[0] << " "
		<< min[1] << " "
		<< min[2] << " "
		<< max[0] << " "
		<< max[1] << " "
		<< max[2] << " "
		<< std::endl;
#endif
    }
    
    if(marker_array_pub_.getNumSubscribers() > 0) {
      Eigen::Vector4f min, max;
      pcl::getMinMax3D(*clusters[i], min, max);

      // transform local centroid into the global frame
      pcl::PointXYZI pi_2;
      pi_2.x = min[0];
      pi_2.y = min[1];
      pi_2.z = min[2];
      pi_2.intensity = 1.0;
      pcl::PointXYZI po_2;
      pointAssociateToMap(&pi_2, &po_2);
      min[0] = po_2.x;
      min[1] = po_2.y;
      min[2] = po_2.z;

      pcl::PointXYZI pi_3;
      pi_3.x = max[0];
      pi_3.y = max[1];
      pi_3.z = max[2];
      pi_3.intensity = 1.0;
      pcl::PointXYZI po_3;
      pointAssociateToMap(&pi_3, &po_3);
      max[0] = po_3.x;
      max[1] = po_3.y;
      max[2] = po_3.z;
      
      visualization_msgs::Marker marker;
      marker.header = ros_pc2_in->header;
      marker.header.frame_id = "camera_init";
      marker.ns = "adaptive_clustering";
      marker.id = i;
      marker.type = visualization_msgs::Marker::LINE_LIST;
      
      marker.pose.orientation.x = 0.0;
      marker.pose.orientation.y = 0.0;
      marker.pose.orientation.z = 0.0;
      marker.pose.orientation.w = 1.0;
      geometry_msgs::Point p[24];
      p[0].x = max[0];  p[0].y = max[1];  p[0].z = max[2];
      p[1].x = min[0];  p[1].y = max[1];  p[1].z = max[2];
      p[2].x = max[0];  p[2].y = max[1];  p[2].z = max[2];
      p[3].x = max[0];  p[3].y = min[1];  p[3].z = max[2];
      p[4].x = max[0];  p[4].y = max[1];  p[4].z = max[2];
      p[5].x = max[0];  p[5].y = max[1];  p[5].z = min[2];
      p[6].x = min[0];  p[6].y = min[1];  p[6].z = min[2];
      p[7].x = max[0];  p[7].y = min[1];  p[7].z = min[2];
      p[8].x = min[0];  p[8].y = min[1];  p[8].z = min[2];
      p[9].x = min[0];  p[9].y = max[1];  p[9].z = min[2];
      p[10].x = min[0]; p[10].y = min[1]; p[10].z = min[2];
      p[11].x = min[0]; p[11].y = min[1]; p[11].z = max[2];
      p[12].x = min[0]; p[12].y = max[1]; p[12].z = max[2];
      p[13].x = min[0]; p[13].y = max[1]; p[13].z = min[2];
      p[14].x = min[0]; p[14].y = max[1]; p[14].z = max[2];
      p[15].x = min[0]; p[15].y = min[1]; p[15].z = max[2];
      p[16].x = max[0]; p[16].y = min[1]; p[16].z = max[2];
      p[17].x = max[0]; p[17].y = min[1]; p[17].z = min[2];
      p[18].x = max[0]; p[18].y = min[1]; p[18].z = max[2];
      p[19].x = min[0]; p[19].y = min[1]; p[19].z = max[2];
      p[20].x = max[0]; p[20].y = max[1]; p[20].z = min[2];
      p[21].x = min[0]; p[21].y = max[1]; p[21].z = min[2];
      p[22].x = max[0]; p[22].y = max[1]; p[22].z = min[2];
      p[23].x = max[0]; p[23].y = min[1]; p[23].z = min[2];
      for(int i = 0; i < 24; i++) {
  	    marker.points.push_back(p[i]);
      }
      
      marker.scale.x = 0.02;
      marker.color.a = 1.0;
      marker.color.r = 0.0;
      marker.color.g = 1.0;
      marker.color.b = 0.5;
      marker.lifetime = ros::Duration(0.1);
      marker_array.markers.push_back(marker);
    }
  }
  
  if(cluster_array.clusters.size()) {
    cluster_array.header = ros_pc2_in->header;
    cluster_array.header.frame_id = frame_id;
    cluster_array_pub_.publish(cluster_array);
  }

  if(pose_array.poses.size()) {
    pose_array.header = ros_pc2_in->header;
    pose_array.header.frame_id = frame_id;
    pose_array_pub_.publish(pose_array);
  }
  
  if(marker_array.markers.size()) {
    marker_array_pub_.publish(marker_array);
  }
  
  if(print_fps_)if(++frames>10){std::cerr<<"[adaptive_clustering] fps = "<<float(frames)/(float(clock()-start_time)/CLOCKS_PER_SEC)<<", timestamp = "<<clock()/CLOCKS_PER_SEC<<std::endl;reset = true;}//fps
}

int main(int argc, char **argv) {
  ros::init(argc, argv, "adaptive_clustering");
  
  /*** Subscribers ***/
  ros::NodeHandle nh;
  ros::Subscriber point_cloud_sub = nh.subscribe<sensor_msgs::PointCloud2>("velodyne_points", 1, pointCloudCallback);
  ros::Subscriber robot_pose_sub = nh.subscribe<geometry_msgs::PoseStamped>("/robot/pose", 100, robotPoseCallback);

  /*** Publishers ***/
  ros::NodeHandle private_nh("~");
  cloud_point_global_pub_ = nh.advertise<sensor_msgs::PointCloud2>("/velodyne_cloud_global", 100);
  cluster_array_pub_ = private_nh.advertise<aloam_velodyne::ClusterArray>("clusters", 100);
  cloud_filtered_pub_ = private_nh.advertise<sensor_msgs::PointCloud2>("cloud_filtered", 100);
  pose_array_pub_ = private_nh.advertise<geometry_msgs::PoseArray>("poses", 100);
  marker_array_pub_ = private_nh.advertise<visualization_msgs::MarkerArray>("markers", 100);
  
  /*** Parameters ***/
  std::string sensor_model;
  
  private_nh.param<std::string>("sensor_model", sensor_model, "VLP-16"); // VLP-16, HDL-32E, HDL-64E
  private_nh.param<bool>("print_fps", print_fps_, true);
  private_nh.param<float>("x_axis_min", x_axis_min_, -6.0);
  private_nh.param<float>("x_axis_max", x_axis_max_, 6.0);
  private_nh.param<float>("y_axis_min", y_axis_min_, -3.5);
  private_nh.param<float>("y_axis_max", y_axis_max_, 3.5);
  private_nh.param<float>("z_axis_min", z_axis_min_, -0.25);
  private_nh.param<float>("z_axis_max", z_axis_max_, 1.6);
  private_nh.param<bool>("human_size_limit", human_size_limit_, true);
  private_nh.param<int>("cluster_size_min", cluster_size_min_, 50);
  private_nh.param<int>("cluster_size_max", cluster_size_max_, 50000);
  
  // Divide the point cloud into nested circular regions centred at the sensor.
  // For more details, see our IROS-17 paper "Online learning for human classification in 3D LiDAR-based tracking"
  if(sensor_model.compare("VLP-16") == 0) {
    regions_[0] = 2; regions_[1] = 3; regions_[2] = 3; regions_[3] = 3; regions_[4] = 3;
    regions_[5] = 3; regions_[6] = 3; regions_[7] = 2; regions_[8] = 3; regions_[9] = 3;
    regions_[10]= 3; regions_[11]= 3; regions_[12]= 3; regions_[13]= 3;
  } else if (sensor_model.compare("HDL-32E") == 0) {
    regions_[0] = 4; regions_[1] = 5; regions_[2] = 4; regions_[3] = 5; regions_[4] = 4;
    regions_[5] = 5; regions_[6] = 5; regions_[7] = 4; regions_[8] = 5; regions_[9] = 4;
    regions_[10]= 5; regions_[11]= 5; regions_[12]= 4; regions_[13]= 5;
  } else if (sensor_model.compare("HDL-64E") == 0) {
    regions_[0] = 14; regions_[1] = 14; regions_[2] = 14; regions_[3] = 15; regions_[4] = 14;
  } else {
    ROS_FATAL("Unknown sensor model!");
  }
  
  ros::spin();

  return 0;
}