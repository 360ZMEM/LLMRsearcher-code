env_wo_desc_prompt = """
## Task Overview

Key code of the environment in python are provided here, followed by a short description of the task.

```python
import numpy as np
from gym import spaces
class Env:
    def __init_(self, **kwargs):
        # --------------------
        #  Environment Parameters
        # --------------------
        self.N_POI = kwargs['NOO']  # Number of SNs, e.g. 50
        # Area position range (0,0) ~ (X_max,Y_max)
        self.X_max = kwargs['X_max'];
        self.Y_max = kwargs['Y_max']  # e.g. 120m
        self.border = np.array([self.X_max, self.Y_max])
        # Serving radius
        self.r_dc = kwargs['r_dc']  # e.g. 6m
        # Safe distance between AUVs
        self.safe_dist = kwargs['safe_dist']  # Safe distance between AUVs for avoiding collision, e.g. 8m
        # --------------------
        #  Variables
        # --------------------
        self.N_AUV = N  # Number of AUVs
        self.ec = np.zeros(self.N_AUV)  # Energy comsuption of AUVs
        self.SoPcenter = np.random.randint(10, self.X_max - 10, size=[self.N_POI, 2])  # Position of SNs
        self.target_Pcenter = np.zeros((self.N_UAV, 2))  # Position of target SNs (for AUV serving)
        self.Vxy = np.zeros(self.N_AUV, 2)  # x/y Velocity of AUVs
        self.xy = np.zeros((self.N_UAV, 2))  # Position of AUVs
        # --------------------
        # Some Metrics
        # --------------------
        self.crash = np.zeros(self.N_AUV, dtype=np.bool_)  # Collision between AUVs
        self.N_DO = 0  # Number of SNs data overflow (not served timely from AUVs)
        self.FX = np.zeros(self.N_AUV, dtype=np.bool_)  # Set to True when AUV crossing the border, else False
        self.TL = np.zeros(self.N_AUV,
                           dtype=np.bool_)  # Set to True when AUV successfully establish connection to the target SN, else False

    def compute_energy_consumption(self):  # Compute energy cosumption of AUVs
        V = np.linalg.norm(self.V, axis=1)
        S = 63;
        F = (0.7 * S * (V ** 2)) / 2
        self.ec = (F * V) / (-0.081 * (V ** 3) + 0.215 * (V ** 2) - 0.01 * V + 0.541)  # ~200W for 2m/s, ~70W for 1.3m/s

    def step(self, actions):  # A simplified pseudo-code for task executing
        # Actions contain velocity information of AUVs
        # actions.shape == (N_AUV,2) -> True
        for i in range(self.N_AUV):
            self.TL[i] = False;  # Same for self.crash/self.FX ...
            self.Vxy[:, 0] = (actions[0] * (self.max_speed)) * math.cos(actions[1] * math.pi)
            self.Vxy[:, 1] = (actions[0] * (self.max_speed)) * math.sin(actions[1] * math.pi)
            self.xy += self.Vxy
            # When distance constriant is met, AUV can establish the connection with the target SN
            if np.linalg.norm(self.xy[i] - self.target_Pcenter) <= self.r_dc:
                # Something TODO: hovering for data transmission, SN's replay buffer is cleared, ready for serving the next SN
                self.TL[i] = True
                pass
```

Description in Natural Language: `self.N_AUV` AUVs are deployed in an area measuring `self.border[0]` × `self.border[1]`. Each AUV must respond to data transfer requests from its corresponding target SN and navigate to the target device to prevent data overflow of SNs. After establishing a connection with an SN, the AUV will proceed to serve the next SN.
"""

obj_desc_prompt = """
## Objectives of the task

Here are the main objectives to be optimized for the task: Safety requirements must be met first, followed by meeting performance requirements, and then optimizing energy consumption.

(Safety requirements) The number of both **collisions** and **border crossings** should be **reduced to zero**.

(Performance requirements) The number of data overflows **should be reduced to zero as much as possible**. This can be achieved by **responding promptly** to SNs and **increasing** the number of served SNs.

(Performance objective) The energy consumption of AUVs may be optimized (lower is better) without violating the **aforementioned** requirements.
"""

desc_short_name = "underwater information collection task"

env_desc_prompt = env_wo_desc_prompt + obj_desc_prompt
