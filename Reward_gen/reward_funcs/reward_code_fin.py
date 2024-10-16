import numpy as np
def compute_reward(self): # defined in the Env class
    # ------ PARAMETERS ------
    w_collision;
    w_border;
    w_service;
    w_overflow;
    w_energy;
    
    # --------- Code ---------
    reward = np.zeros(self.N_AUV)
    
    # Penalty for collisions (sparse)
    for i in range(self.N_AUV):
        if self.crash[i]:
            reward[i] += w_collision
    
    # Penalty for border crossing (sparse)
    for i in range(self.N_AUV):
        if self.FX[i]:
            reward[i] += w_border
    
    # Reward for successful connection to SN (sparse)
    for i in range(self.N_AUV):
        if self.TL[i]:
            reward[i] += w_service
    
    # Penalty for data overflow (sparse)
    reward += w_overflow * self.N_DO
    
    # Penalty for energy consumption (dense)
    reward += w_energy * self.ec
    
    # Avoiding collisions (dense)
    for i in range(self.N_AUV):
        for j in range(i+1, self.N_AUV):
            dist = np.linalg.norm(self.xy[i] - self.xy[j]) / np.linalg.norm(self.border)
            if dist < self.safe_dist / np.linalg.norm(self.border):
                reward[i] -= w_collision * (dist - self.safe_dist / np.linalg.norm(self.border))
    
    # Avoiding border crossing (dense)
    for i in range(self.N_AUV):
        dist_to_border = np.min([self.xy[i][0], self.border[0] - self.xy[i][0], self.xy[i][1], self.border[1] - self.xy[i][1]]) / np.linalg.norm(self.border)
        if dist_to_border < self.safe_dist / np.linalg.norm(self.border):
            reward[i] -= w_border * (dist_to_border - self.safe_dist / np.linalg.norm(self.border))
    
    # Reward for timely service (dense)
    for i in range(self.N_AUV):
        dist_to_target = np.linalg.norm(self.xy[i] - self.target_Pcenter[i]) / np.linalg.norm(self.border)
        reward[i] += w_service * (1 - dist_to_target)
    
    return reward