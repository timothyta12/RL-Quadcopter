import numpy as np
from physics_sim import PhysicsSim

class Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None, 
        init_angle_velocities=None, runtime=5., target_pos=None):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime) 
        self.action_repeat = 3

        self.state_size = self.action_repeat * 6
        self.action_low = 420
        self.action_high = 420
        self.action_size = 4
        
        self.init_pose = self.sim.init_pose

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.]) 
        
        self.count = 0
        self.reward_avg = 0

    def get_reward(self):
        """Uses current pose of sim to return reward.""" 
#         print("\tx = {:7.3f}, y = {:7.3f}, z = {:7.3f}, x^ = {:7.3f}, y^ = {:7.3f}, z^ = {:7.3f}".format(
#             *self.sim.pose[:3], *self.sim.v))
        rewards = [1]
    
#         if self.sim.pose[2] > self.target_pos[2]:
#             rewards.append( -0.5 )
#         else:
#             rewards.append( -0.5 ) 
        # Absolute Distance
        current_distance = np.linalg.norm(self.sim.pose[:3] - self.target_pos)
        original_distance = np.linalg.norm(self.target_pos - self.init_pose[:3])
        rewards.append( -1*(current_distance/original_distance if original_distance != 0 else current_distance) )
        
#         # Velocity Reward
#         v = self.sim.v
#         norm_v = v / np.linalg.norm(v) if np.linalg.norm(v) != 0 else v
        
#         dist_diff = self.target_pos - self.sim.pose[:3]
#         norm_v_dist = np.linalg.norm( norm_v * dist_diff ) 
        
#         rewards.append( -0.1*norm_v_dist )

        # Crash
        if self.sim.done and self.sim.time < self.sim.runtime:
            rewards.append ( -1 )
#         print(rewards)
        
        self.count += 1
        self.reward_avg = self.reward_avg - self.reward_avg/self.count + np.sum(rewards)/self.count

#         reward = np.tanh(np.sum(rewards))
        reward = np.clip(np.sum(rewards), -1, 1)

        return reward

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
#         print("\t Rotor speeds: {}".format(rotor_speeds))
#         print("\tx = {:7.3f}, y = {:7.3f}, z = {:7.3f}, x^ = {:7.3f}, y^ = {:7.3f}, z^ = {:7.3f}".format(
#             *self.sim.pose[:3], *self.sim.v))
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward() 
            pose_all.append(self.sim.pose)
        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.count = 0
        self.reward_avg = 0
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat) 
        return state