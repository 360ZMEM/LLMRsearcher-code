import numpy as np # automatically added

def compute_reward(self):
    reward = np.zeros(self.N_AUV)
    for i in range(self.N_AUV):
        dist_to_target = np.linalg.norm(self.xy[i] - self.target_Pcenter[i])
        reward[i] += (-0.6 * dist_to_target - self.FX[i] * 0.1-self.N_DO * 0.05)
        # collision
        for j in range(i+1,self.N_AUV):
            dist_between_auvs = np.linalg.norm(self.xy[j] - self.xy[i])
            if dist_between_auvs < 12:
                reward[i] -= 6 * (12 - dist_between_auvs)
        if self.TL[i] > 0:
            reward[i] += 12
        reward[i] -= 0.01 * self.ec[i]
    return reward