import numpy as np

class GripperModel:
    """Utility class for ALOHA-style gripper kinematics & conversions."""

    def __init__(self):
        # geometry / offsets
        self.THRESH_PARTIALLY_CLOSED = 85.0
        self.TAU = 64.0
        self.L1 = 0.038   # m
        self.L2 = 0.0444  # m
        self.eps = 1e-9

        # current→torque constants (N·m/A)
        self.k1_const = 1.769
        self.k2_const = -0.2214

        # CAD constants (for contact points)
        self.ry_helper = 2*0.007071
        self.rz_helper = 0.144

        # calibration data (edges in mA, per-bin η)
        self.edges_mA = np.array([
            -200.0, -130.0, -60.0, 10.0, 80.0, 150.0, 220.0, 290.0, 360.0,
            430.0, 500.0, 570.0, 640.0, 710.0, 780.0, 850.0, 920.0, 990.0,
            1060.0, 1130.0, 1200.0
        ], dtype=float)

        self.eta_bins = np.array([
            0.03186942920960295, 0.03186942920960295, 0.025132572541536807,
            0.09254881657478439, 1.5, 1.5, 0.9738673713505731, 0.6192106650584402,
            0.46332143648903246, 0.48681474522139245, 0.4833029744117554,
            0.4172706221064947, 0.3740295235488369, 0.3874455458009995,
            0.4275646686392278, 0.38496224305917287, 0.33775737967135905,
            0.33775737967135905, 0.33775737967135905, 0.33775737967135905
        ], dtype=float)

        # compute bin centers for interpolation
        self.centers_mA = 0.5 * (self.edges_mA[:-1] + self.edges_mA[1:])

    # ---------------- basic kinematics ----------------
    def x_of_alpha(self, alpha):
        R = self.L2**2 - (self.L1*np.sin(alpha))**2
        R = np.maximum(R, 0.0)
        return self.L1*np.cos(alpha) + np.sqrt(R)

    def dx_dalpha(self, alpha):
        R = self.L2**2 - (self.L1*np.sin(alpha))**2
        R = np.maximum(R, 1e-12)
        return -self.L1*np.sin(alpha) - (self.L1**2*np.sin(alpha)*np.cos(alpha))/np.sqrt(R)

    def dg_dalpha(self, alpha):
        return 2.0 * self.dx_dalpha(alpha)

    # ---------------- motor conversions ----------------
    def current_to_torque(self, current_mA):
        """Convert motor current [mA] to torque [N·m]."""
        current_A = current_mA / 1000.0
        return self.k1_const * current_A + self.k2_const
    
    def clamp_force(self, alpha, current_mA):
        """Directly get per-finger clamp force from motor state."""
        tau = self.current_to_torque(current_mA)
        J = np.maximum(np.abs(self.dg_dalpha(alpha)), self.eps)
        return np.abs(tau) / J
    

    def eta_of_current(self, current_mA: float | np.ndarray) -> float | np.ndarray:
        """Interpolate efficiency η for a given motor current [mA]."""
        return np.interp(current_mA,
                         self.centers_mA,
                         self.eta_bins,
                         left=self.eta_bins[0],
                         right=self.eta_bins[-1])
    
    
def x_of_alpha(alpha, l1, l2):
    R = l2**2 - (l1*np.sin(alpha))**2
    if np.any(R < 0):
        # mask small negatives from numeric jitter
        R = np.maximum(R, 0.0)
    return l1*np.cos(alpha) + np.sqrt(R)

def dx_dalpha(alpha, l1, l2):
    R = l2**2 - (l1*np.sin(alpha))**2
    if np.any(R <= 0):
        # near singularity; return very small derivative to avoid blow-up
        R = np.maximum(R, 1e-12)
    return -l1*np.sin(alpha) - (l1**2*np.sin(alpha)*np.cos(alpha))/np.sqrt(R)

def dg_dalpha(alpha, l1, l2):
    return 2.0 * dx_dalpha(alpha, l1, l2)

def current_to_torque(current, k1_const, k2_const):
    return k1_const * current + k2_const