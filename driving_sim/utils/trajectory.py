import numpy as np

# defines the trajectory of the ego car
# todo: what does s, t, theta, curv mean?
class EgoTrajectory:
    def xy_to_traj(self, pos):
        x, y = pos[0], pos[1]
        r = 6 # 4.5
        if y < 0.:
            s = y
            t = -x
            theta = np.pi/2
            curv = 0.
        elif x > r:
            s = r*np.pi/2. + x - r
            t = y - r
            theta = 0.
            curv = 0.
        else:
            theta = np.arctan2(r-x ,y)
            curv = 1./r
            s = r*(np.pi/2.-theta)
            t = np.sqrt((r-x)**2+(y)**2) - r

        return s, t, theta, curv

    def traj_to_xy(self, pos):
        s, t = pos[0], pos[1]
        r = 6 # 4.5
        if s < 0.:
            x = -t
            y = s
            theta = np.pi/2
            curv = 0.
        elif s > r*np.pi/2.:
            x = r + s - r*np.pi/2.
            y = r + t
            theta = 0.
            curv = 0.
        else:
            theta = np.pi/2 - s/r
            curv = 1./r
            x = r - (r+t)*np.sin(theta)
            y = (r+t)*np.cos(theta)

        return x, y, theta, curv