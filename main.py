import astropy.constants as cnst
import astropy.units as u
import numpy as np

from scipy.special import lambertw

class HawkingRadiationCalculator:
    def __init__(
        self, mass, 
        c = cnst.c,
        G = cnst.G,
        hbar = cnst.hbar,
        k_b = cnst.k_B,
        sigma_sb = cnst.sigma_sb
    ):
        self.mass = mass*u.kg
        self.G = G
        self.c = c
        self.hbar = hbar
        self.k_b = k_b
        self.sigma_sb = sigma_sb
        self.update()

    def update(self):
        self.schwarzschild_radius = self.calculate_schwarzschild_radius()
        self.surface_area = self.calculate_surface_area()
        self.effective_density = self.calculate_effective_density()
        self.surface_gravity = self.calculate_surface_gravity()
        self.surface_tides = self.calculate_surface_tides()
        self.time_to_singularity = self.calculate_time_to_singularity()
        self.entropy = self.calculate_entropy()
        self.temperature = self.calculate_temperature()
        self.peak_photons = self.calculate_peak_photons()
        self.nominal_luminosity = self.calculate_nominal_luminosity()
        self.lifetime = self.calculate_lifetime()

    def calculate_schwarzschild_radius(self):
        return (2*self.G*self.mass) / (self.c**2)

    def calculate_surface_area(self):
        # schwarzschild radius
        Rs = self.calculate_schwarzschild_radius()
        
        return 4*np.pi*Rs**2
    
    def calculate_effective_density(self):
        scaling_constant = (3*self.c**6) / (32*np.pi*self.G**3)
        
        return scaling_constant / self.mass**2

    def calculate_surface_gravity(self):
        scaling_constant = (self.c**4) / (4*self.G)
        return scaling_constant / self.mass

    def calculate_surface_tides(self):
        scaling_constant = (self.c**6) / (4*self.G**2)
        return scaling_constant / self.mass**2

    def calculate_time_to_singularity(self):
        # free-fall time
        return np.pi*self.G*self.mass/self.c**3

    def calculate_entropy(self):
        A = self.calculate_surface_area()
        # dimensionless Bekenstein-Hawking entropy
        S_dimless = A*self.c**3 / (4*self.G*self.hbar)
        
        return S_dimless*self.k_b

    def calculate_temperature(self):
        kappa = (self.hbar*self.c**3)/(4*self.k_b*self.G*self.mass)
        
        return kappa / 2*np.pi

    def calculate_peak_photons(self):
        h = 2*np.pi*self.hbar
        T = self.calculate_temperature()
        E_therm = self.k_b*T
        
        # argument of Lambert W-function
        arg = -4*np.exp(-4)
        W = lambertw(arg)
        
        return h*self.c / (E_therm*(W + 4))

    def calculate_nominal_luminosity(self):
        A = self.calculate_surface_area()
        T = self.calculate_temperature()
        
        return A*self.sigma_sb*T**4

    def calculate_lifetime(self):
        numerator = 5120*np.pi*self.G**2
        denominator = 1.8083*self.hbar*self.c**4
        
        lifetime = (numerator/denominator)*self.mass**3
        
        return lifetime

mass_of_object = 2*10**(33)

calculator = HawkingRadiationCalculator(mass_of_object)

print("Schwarzschild radius:", calculator.schwarzschild_radius)
print("Surface area:", calculator.surface_area)