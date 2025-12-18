import random
import numpy as np


class ParticleSystem:
    def __init__(self):
        self.particles = []

    def explode(self, snr):
        """_summary_

        Args:
            snr (_type_): _description_
        """
        count = int(max(10, snr * 20))

        for _ in range(count):
            theta = random.uniform(0, 2 * np.pi)
            r = random.uniform(0, 90)

            v_theta = random.uniform(-0.005, 0.005)
            v_r = random.uniform(-0.05, 0.05)

            life = snr * 10

            self.particles.append({
                'theta': theta,
                'r': r,
                'v_theta': v_theta,
                'v_r': v_r,
                'life': life,
                'max_life': life,
                'size': random.uniform(2, 5)
            })

    def update(self):
        """Update physics for all particles"""
        alive = []
        for p in self.particles:
            p['life'] -= 1
            if p['life'] > 0:
                p['theta'] += p['v_theta']
                p['r'] += p['v_r']
                alive.append(p)
        self.particles = alive

    def get_data(self):
        if not self.particles:
            return [], [], [], []

        thetas = [p['theta'] for p in self.particles]
        rs = [p['r'] for p in self.particles]
        sizes = [p['size'] * (0.5 + 0.5 * (p['life'] / p['max_life']))
                 for p in self.particles]
        alphas = [0.4 * np.sin(np.pi * (p['life'] / p['max_life']))
                  for p in self.particles]

        return thetas, rs, sizes, alphas