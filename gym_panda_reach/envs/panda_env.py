import gym
from gym import error, spaces, utils
from gym.utils import seeding

import os
import pybullet as p
import pybullet_data
import math
import numpy as np
import random


MAX_EPISODE_LEN = 20*100


def goal_distance(goal_a, goal_b):
    return np.linalg.norm(np.array(goal_a) - np.array(goal_b), axis=-1)

class PandaEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.step_counter = 0
        p.connect(p.GUI,options='--background_color_red=0.0 --background_color_green=0.93--background_color_blue=0.54')
        p.resetDebugVisualizerCamera(cameraDistance=1.25, cameraYaw=45, cameraPitch=-20, cameraTargetPosition=[0.55,-0.35,0.15])
        self.action_space = spaces.Box(np.array([-1]*4), np.array([1]*4))
#         self.observation_space = spaces.Box(np.array([-1]*5), np.array([1]*5))
        self.observation_space = spaces.Box(np.array([-1]*24), np.array([1]*24))
        
    def compute_reward(self, achieved_goal, goal):
        # Compute distance between goal and the achieved goal.
        d = goal_distance(achieved_goal, goal)
        if self.reward_type == 'sparse':
            return -(d > self.distance_threshold).astype(np.float32)
        else:
            return -d


    def step(self, action):
        p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING)
        orientation = p.getQuaternionFromEuler([0.,-math.pi,-math.pi])
#         dv = 0.005
        dv = 0.05
        dx = action[0] * dv
        dy = action[1] * dv
        dz = action[2] * dv
        fingers = action[3]

        currentPose = p.getLinkState(self.pandaUid, 11)
        currentPosition = currentPose[0]
        newPosition = [currentPosition[0] + dx,
                       currentPosition[1] + dy,
                       currentPosition[2] + dz]
        jointPoses = p.calculateInverseKinematics(self.pandaUid,11,newPosition, orientation)[0:7]

        p.setJointMotorControlArray(self.pandaUid, list(range(7))+[9,10], p.POSITION_CONTROL, list(jointPoses)+2*[fingers])

        p.stepSimulation()

        state_object, state_object_orienation = p.getBasePositionAndOrientation(self.objectUid)
        twist_object, twist_object_orienation = p.getBaseVelocity(self.objectUid)
        state_robot = p.getLinkState(self.pandaUid, 11)[0]
        state_fingers = (p.getJointState(self.pandaUid,9)[0], p.getJointState(self.pandaUid, 10)[0])
        
#         if state_object[2]>0.45:
#             reward = 1
#             done = True
#         else:
#             reward = 0
#             done = False

        # Compute reward and completition based: the reward is either dense or sparse
        self.reward_type = "dense"
        self.distance_threshold = 0.05
        d = goal_distance(state_robot, state_object)
        if d<self.distance_threshold:
            reward = self.compute_reward(state_robot, state_object)
            done = True
        else:
            reward = self.compute_reward(state_robot, state_object)
            done = False


        self.step_counter += 1

        if self.step_counter > MAX_EPISODE_LEN:
            #reward = 0
            done = True

        info = {'object_position': state_object}
#         self.observation = state_robot + state_fingers
#         return np.array(self.observation).astype(np.float32), reward, done, info
        
        # src -> https://github.com/openai/gym/issues/1503
        grip_pos = np.array(state_robot)
        object_pos = np.array(state_object)
        object_rel_pos = object_pos - grip_pos
        gripper_state = np.array([0.5*(state_fingers[0]+state_fingers[1])])#this is the gripper q
        object_rot = np.array(state_object_orienation) # quaternions?
        object_velp = np.array(twist_object)
        object_velr = np.array(twist_object_orienation)
        grip_velp = np.array([dx,dy,dz])#The velocity of gripper moving
        gripper_vel = np.array([fingers*dv]) #The velocity of gripper opening/closing
    
        obs = np.concatenate([
                    grip_pos.ravel(), object_pos.ravel(), object_rel_pos.ravel(), gripper_state, object_rot.ravel(),
                    object_velp.ravel(), object_velr.ravel(), grip_velp.ravel(), gripper_vel.ravel(),
                ])
        #print(np.shape(obs))
        return obs.copy(), reward, done, info

    def reset(self):
        self.step_counter = 0
        p.resetSimulation()
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,0) # we will enable rendering after we loaded everything
        urdfRootPath=pybullet_data.getDataPath()
        p.setGravity(0,0,-10)

#         planeUid = p.loadURDF(os.path.join(urdfRootPath,"plane.urdf"), basePosition=[0,0,-0.65])
        dir_path = os.path.dirname(os.path.realpath(__file__))
        planeUid = p.loadURDF(os.path.join(dir_path, "floor.urdf"), basePosition=[0,0,-0.65],useFixedBase=True)

        rest_poses = [0, -0.215, 0, -2.57, 0.0, 2.356, math.pi/4, 0.0, 0.0, 0.0]
#         rest_poses = [0, 0, 0, 0, 0.0, 0, 0, 0.0, 0.0, 0.0]
        self.pandaUid = p.loadURDF(os.path.join(urdfRootPath, "franka_panda/panda.urdf"),useFixedBase=True)
        
        for i in range(len(rest_poses)):
            p.resetJointState(self.pandaUid,i, rest_poses[i])
            
#         tableUid = p.loadURDF(os.path.join(urdfRootPath, "table/table.urdf"),basePosition=[0.5,0,-0.65])
        dir_path = os.path.dirname(os.path.realpath(__file__))
        tableUid = p.loadURDF(os.path.join(dir_path, "flat_table.urdf"),basePosition=[0.5,0,-0.65/2],useFixedBase=True)

        #trayUid = p.loadURDF(os.path.join(urdfRootPath, "tray/traybox.urdf"),basePosition=[0.65,0,0])
        
        p.addUserDebugLine([-1,0,0.05],[1,0,0.05],[0.9,0.9,0.9],parentObjectUniqueId=self.pandaUid, parentLinkIndex=8)
        p.addUserDebugLine([0,-1,0.05],[0,1,0.05],[0.9,0.9,0.9],parentObjectUniqueId=self.pandaUid, parentLinkIndex=8)
        p.addUserDebugLine([0,0,-1],[0,0,1],[0.9,0.9,0.9],parentObjectUniqueId=self.pandaUid, parentLinkIndex=8)

        state_object= [random.uniform(0.5,0.8),random.uniform(-0.2,0.2),random.uniform(0.0,0.2)]
#         self.objectUid = p.loadURDF(os.path.join(urdfRootPath, "random_urdfs/000/000.urdf"), basePosition=state_object)
        self.objectUid = p.loadURDF(os.path.join(dir_path, "goal.urdf"), basePosition=state_object,useFixedBase=True)
        state_robot = p.getLinkState(self.pandaUid, 11)[0]
        state_fingers = (p.getJointState(self.pandaUid,9)[0], p.getJointState(self.pandaUid, 10)[0])
        self.observation = state_robot + state_fingers
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,1)
        return np.array(self.observation).astype(np.float32)

    def render(self, mode='human'):
        view_matrix = p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=[0.7,0,0.05],
                                                            distance=.7,
                                                            yaw=90,
                                                            pitch=-50,
                                                            roll=0,
                                                            upAxisIndex=2)
        proj_matrix = p.computeProjectionMatrixFOV(fov=60,
                                                     aspect=float(960) /720,
                                                     nearVal=0.1,
                                                     farVal=100.0)
        (_, _, px, _, _) = p.getCameraImage(width=960,
                                              height=720,
                                              viewMatrix=view_matrix,
                                              projectionMatrix=proj_matrix,
                                              renderer=p.ER_BULLET_HARDWARE_OPENGL)

        rgb_array = np.array(px, dtype=np.uint8)
        rgb_array = np.reshape(rgb_array, (720,960, 4))

        rgb_array = rgb_array[:, :, :3]
        return rgb_array

    def _get_state(self):
        return self.observation

    def close(self):
        p.disconnect()

# import gym
# from gym import error, spaces, utils
# from gym.utils import seeding

# import os
# import pybullet as p
# import pybullet_data
# import math
# import numpy as np
# import random

# class PandaEnv(gym.Env):
#     metadata = {'render.modes': ['human']}

#     def __init__(self):
#         p.connect(p.GUI,options='--background_color_red=0.0 --background_color_green=0.929--background_color_blue=0.549')
#         p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=0, cameraPitch=-40, cameraTargetPosition=[0.55,-0.35,0.2])
#         self.action_space = spaces.Box(np.array([-1]*4), np.array([1]*4))
#         self.observation_space = spaces.Box(np.array([-1]*5), np.array([1]*5))

#     def step(self, action):
#         p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING)
#         orientation = p.getQuaternionFromEuler([0.,-math.pi,math.pi/2.])
#         dv = 0.005
#         dx = action[0] * dv
#         dy = action[1] * dv
#         dz = action[2] * dv
#         fingers = action[3]

#         currentPose = p.getLinkState(self.pandaUid, 11)
#         currentPosition = currentPose[0]
#         newPosition = [currentPosition[0] + dx,
#                        currentPosition[1] + dy,
#                        currentPosition[2] + dz]
#         jointPoses = p.calculateInverseKinematics(self.pandaUid,11,newPosition, orientation)

#         p.setJointMotorControlArray(self.pandaUid, list(range(7))+[9,10], p.POSITION_CONTROL, list(jointPoses)+2*[fingers])

#         p.stepSimulation()

#         state_object, _ = p.getBasePositionAndOrientation(self.objectUid)
#         state_robot = p.getLinkState(self.pandaUid, 11)[0]
#         state_fingers = (p.getJointState(self.pandaUid,9)[0], p.getJointState(self.pandaUid, 10)[0])
#         if state_object[2]>0.45:
#             reward = 1
#             done = True
#         else:
#             reward = 0
#             done = False
#         info = state_object
#         observation = state_robot + state_fingers
#         return observation, reward, done, info

        
#     def reset(self):
#         p.resetSimulation()
#         p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,0) # we will enable rendering after we loaded everything
#         p.setGravity(0,0,-10)
#         urdfRootPath=pybullet_data.getDataPath()
        
#         planeUid = p.loadURDF(os.path.join(urdfRootPath,"plane.urdf"), basePosition=[0,0,-0.65])

#         rest_poses = [0,-0.215,0,-2.57,0,2.356,2.356,0.08,0.08]
#         self.pandaUid = p.loadURDF(os.path.join(urdfRootPath, "franka_panda/panda.urdf"),useFixedBase=True)
#         for i in range(7):
#             p.resetJointState(self.pandaUid,i, rest_poses[i])

#         tableUid = p.loadURDF(os.path.join(urdfRootPath, "table/table.urdf"),basePosition=[0.5,0,-0.65])

#         trayUid = p.loadURDF(os.path.join(urdfRootPath, "tray/traybox.urdf"),basePosition=[0.65,0,0])

#         state_object = [random.uniform(0.5,0.8),random.uniform(-0.2,0.2),0.05]
        
#         self.objectUid = p.loadURDF(os.path.join(urdfRootPath, "random_urdfs/000/000.urdf"), basePosition=state_object)
#         state_robot = p.getLinkState(self.pandaUid, 11)[0]
#         state_fingers = (p.getJointState(self.pandaUid,9)[0], p.getJointState(self.pandaUid, 10)[0])
#         observation = state_robot + state_fingers
#         p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,1) # rendering's back on again
#         return observation

#     def render(self, mode='human'):
#         view_matrix = p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=[0.7,0,0.05],
#                                                             distance=.7,
#                                                             yaw=90,
#                                                             pitch=-70,
#                                                             roll=0,
#                                                             upAxisIndex=2)
#         proj_matrix = p.computeProjectionMatrixFOV(fov=60,
#                                                      aspect=float(960) /720,
#                                                      nearVal=0.1,
#                                                      farVal=100.0)
#         (_, _, px, _, _) = p.getCameraImage(width=960,
#                                               height=720,
#                                               viewMatrix=view_matrix,
#                                               projectionMatrix=proj_matrix,
#                                               renderer=p.ER_BULLET_HARDWARE_OPENGL)

#         rgb_array = np.array(px, dtype=np.uint8)
#         rgb_array = np.reshape(rgb_array, (720,960, 4))

#         rgb_array = rgb_array[:, :, :3]
#         return rgb_array

#     def close(self):
#         p.disconnect()
