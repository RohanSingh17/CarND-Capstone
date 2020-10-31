#!/usr/bin/env python
import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, Pose
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from light_classification.tl_classifier import TLClassifier
from scipy.spatial import KDTree
import tf
import cv2
import yaml

STATE_COUNT_THRESHOLD = 1

class TLDetector(object):
    def __init__(self):
        rospy.init_node('tl_detector')

        self.pose = None
        self.waypoints = None
        self.waypoints_2d = None
        self.camera_image = None
        self.lights = []

        sub1 = rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        sub2 = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        '''
        /vehicle/traffic_lights provides you with the location of the traffic light in 3D map space and
        helps you acquire an accurate ground truth data source for the traffic light
        classifier by sending the current color state of all traffic lights in the
        simulator. When testing on the vehicle, the color state will not be available. You'll need to
        rely on the position of the light and the camera image to predict it.
        '''
        sub3 = rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cb)
        sub6 = rospy.Subscriber('/image_color', Image, self.image_cb)

        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)

        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)

        self.bridge = CvBridge()
        self.Train = False
        self.Train_model = True
        self.light_classifier = TLClassifier()
        self.listener = tf.TransformListener()

        self.state = TrafficLight.UNKNOWN
        self.last_state = TrafficLight.UNKNOWN
        self.last_wp = -1
        self.state_count = 0
        self.save_count = 1
        self.pred_state = -1

        # if self.Train_model:
        #     self.light_classifier.train_model()

        rospy.spin()

    def pose_cb(self, msg):
        self.pose = msg

    def waypoints_cb(self, waypoints):
        self.waypoints = waypoints
        if not self.waypoints_2d:
            self.waypoints_2d = [[waypoint.pose.pose.position.x, waypoint.pose.pose.position.y] for waypoint in waypoints.waypoints]
            self.waypoint_tree = KDTree(self.waypoints_2d)

    def traffic_cb(self, msg):
        self.lights = msg.lights
        # rospy.loginfo("Current Light State %s", self.lights.state)

    def image_cb(self, msg):
        """Identifies red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light's stop line to /traffic_waypoint

        Args:
            msg (Image): image from car-mounted camera

        """
        self.has_image = True
        self.camera_image = msg

        light_wp, state = self.process_traffic_lights()


        '''
        Publish upcoming red lights at camera frequency.
        Each predicted state has to occur `STATE_COUNT_THRESHOLD` number
        of times till we start using it. Otherwise the previous stable state is
        used.
        '''
        if self.state != state:
            self.state_count = 0
            self.state = state
        elif self.state_count >= STATE_COUNT_THRESHOLD:
            # rospy.loginfo("COUNT STATE_COUNT_THRESHOLD")
            self.last_state = self.state
            light_wp = light_wp if state == TrafficLight.RED else -1
            self.last_wp = light_wp
            # rospy.loginfo("Current Light State %s", state)
            # rospy.loginfo(" ")
            # rospy.loginfo("Closest Light WP Index %s", light_wp)
            # rospy.loginfo("Car WP Index %s", car_wp)

            self.upcoming_red_light_pub.publish(Int32(light_wp))
        else:
            self.upcoming_red_light_pub.publish(Int32(self.last_wp))
        self.state_count += 1

    def get_closest_waypoint(self, x, y):
        """Identifies the closest path waypoint to the given position
            https://en.wikipedia.org/wiki/Closest_pair_of_points_problem
        Args:
            pose (Pose): position to match a waypoint to

        Returns:
            int: index of the closest waypoint in self.waypoints

        """
        #TODO implement
        closest_idx = self.waypoint_tree.query([x, y], 1)[1]
        return closest_idx

    def get_training_data(self,state):
        # if(not self.has_image):
        #     self.prev_light_loc = None
        #     return False

        cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")

        time = rospy.Time().now().to_sec()
        #cv2.imwrite('/home/student/CarND-Capstone/imgs/Train_Imgs/'+str(time)+'.jpeg', cv_image)
        cv2.imwrite('/home/student/CarND-Capstone/imgs/Train_Imgs/Image_'+str(self.save_count)+'.jpeg', cv_image)
        # rospy.loginfo('image saved')
        # rospy.loginfo("Saved Image State %s, %s",self.save_count, state)
        self.save_count = self.save_count + 1
        


    def get_light_state(self, light):
        """Determines the current color of the traffic light

        Args:
            light (TrafficLight): light to classify

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        if(not self.has_image):
            self.prev_light_loc = None
            return False

        cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")
        # rospy.loginfo("Current ---- %s", light.state)
        # rospy.loginfo(" ") 

        #Get classification
        if self.save_count%3 == 0:
            pr_state = self.light_classifier.get_classification(cv_image)
            self.pred_state = pr_state

            rospy.loginfo("Current State: %s, Prediction State: %s ", light.state, pr_state)
        #     rospy.loginfo("Predicted ---- %s", pr_state)
        # rospy.loginfo(" ")  
        self.save_count = self.save_count + 1
        # rospy.loginfo("Predicted Light State %s",self.light_classifier.get_classification(cv_image))
        # rospy.loginfo("Current Light State %s", light.state)
        #rospy.loginfo("Mobile net Model ", self.light_classifier.train_model())
        return self.pred_state #light.state

    def process_traffic_lights(self):
        """Finds closest visible traffic light, if one exists, and determines its
            location and color

        Returns:
            int: index of waypoint closes to the upcoming stop line for a traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        closest_light = None
        line_wp_idx = None

        # List of positions that correspond to the line to stop in front of for a given intersection
        stop_line_positions = self.config['stop_line_positions']

        if(self.pose):

            car_wp_idx = self.get_closest_waypoint(self.pose.pose.position.x, self.pose.pose.position.y)
            # print("Waypoint velocity--",self.waypoints.waypoints[car_wp_idx].twist.twist.linear.x)

        #TODO find the closest visible traffic light (if one exists)
            diff = len(self.waypoints.waypoints)
            # print(diff)
            for i, light in enumerate(self.lights):
                #Get Stop line waypoint index
                line = stop_line_positions[i]
                temp_wp_idx = self.get_closest_waypoint(line[0], line[1])
                
                #Find closest stop line waypoint
                d = temp_wp_idx - car_wp_idx

                if d >= 0 and d < diff:
                    diff = d
                    closest_light = light
                    line_wp_idx = temp_wp_idx

            # Used to append the update the light waypoints after the last light waypoint
            
            if line_wp_idx == None:
                dist_to_closest_light = diff - car_wp_idx
                diff = len(self.waypoints.waypoints)

                for i, light in enumerate(self.lights):
                    line = stop_line_positions[i]
                    temp_wp_idx = self.get_closest_waypoint(line[0], line[1])
                    d = temp_wp_idx - 0 + dist_to_closest_light

                    if d >= 0 and d < diff:
                        diff = d
                        closest_light = light
                        line_wp_idx = temp_wp_idx

                dist_to_closest_light = d     

            else:
                
                dist_to_closest_light = line_wp_idx - car_wp_idx
                

        # rospy.loginfo("Line_WP_Index %s Car WP Index %s",line_wp_idx, car_wp_idx)
        state = -1

        if closest_light and dist_to_closest_light < 180:
            state = self.get_light_state(closest_light)

            if self.Train:
                self.get_training_data(state)
            return line_wp_idx, state

        return -1, TrafficLight.UNKNOWN

if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
