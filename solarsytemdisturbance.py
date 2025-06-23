# Solar System Disturbance Simulation with Terminal Inputs

import numpy as np
import matplotlib.pyplot as plt

# Constants
G = 6.67430e-11  # Gravitational constant
M_sun = 1.989e30  # Sun's mass

# Planetary data
planets = {
    'Mercury': {'radius': 5.79e10, 'mass': 3.30e23},
    'Venus': {'radius': 1.08e11, 'mass': 4.87e24},
    'Earth': {'radius': 1.496e11, 'mass': 5.97e24},
    'Mars': {'radius': 2.28e11, 'mass': 6.42e23},
    'Jupiter': {'radius': 7.78e11, 'mass': 1.90e27},
    'Saturn': {'radius': 1.43e12, 'mass': 5.68e26},
    'Uranus': {'radius': 2.87e12, 'mass': 8.68e25},
    'Neptune': {'radius': 4.5e12, 'mass': 1.02e26},
    'Pluto': {'radius': 5.9e12, 'mass': 1.31e22}
}

# Initialize planets
planet_data = []
for name, props in planets.items():
    radius = props['radius']
    mass = props['mass']
    x = radius
    y = 0
    vx = 0
    vy = np.sqrt(G * M_sun / radius)
    planet_data.append({'name': name, 'mass': mass, 'x': x, 'y': y, 'vx': vx, 'vy': vy})

# --- Input new object details ---
print("Enter the new object's properties:")

new_name = input("Name of the object: ")
new_mass = float(input("Mass of the object (in kg): "))
new_x = float(input("Initial x-position (in meters): "))
new_y = float(input("Initial y-position (in meters): "))
new_vx = float(input("Initial x-velocity (in meters/second): "))
new_vy = float(input("Initial y-velocity (in meters/second): "))

new_object = {
    'name': new_name,
    'mass': new_mass,
    'x': new_x,
    'y': new_y,
    'vx': new_vx,
    'vy': new_vy
}

# Combine planets + new object
all_bodies = planet_data + [new_object]

# Save original positions
original_positions = [(body['x'], body['y']) for body in all_bodies]

# Gravitational acceleration function
def compute_acceleration(mass_source, x_source, y_source, x_target, y_target):
    dx = x_source - x_target
    dy = y_source - y_target
    r_squared = dx**2 + dy**2 + 1e6
    r = np.sqrt(r_squared)
    a = G * mass_source / r_squared
    ax = a * dx / r
    ay = a * dy / r
    return ax, ay

# Simulation settings
num_steps = 100
dt = 86400  # 1 day per step

# Record trajectories
trajectories = {body['name']: {'x': [], 'y': []} for body in all_bodies}

for step in range(num_steps):
    for body in all_bodies:
        trajectories[body['name']]['x'].append(body['x'])
        trajectories[body['name']]['y'].append(body['y'])
    
    for body in all_bodies:
        if body['name'] == new_object['name']:
            continue  # New object affects others, but not itself here

        # Gravity from Sun
        ax_sun, ay_sun = compute_acceleration(M_sun, 0, 0, body['x'], body['y'])

        # Gravity from New Object
        ax_obj, ay_obj = compute_acceleration(new_object['mass'], new_object['x'], new_object['y'], body['x'], body['y'])

        # Total acceleration
        ax_total = ax_sun + ax_obj
        ay_total = ay_sun + ay_obj

        # Update velocity
        body['vx'] += ax_total * dt
        body['vy'] += ay_total * dt

    # Update all positions
    for body in all_bodies:
        body['x'] += body['vx'] * dt
        body['y'] += body['vy'] * dt

# --- Plot results ---

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
colors = plt.cm.get_cmap('tab10', len(all_bodies))

# Before plot
for i, pos in enumerate(original_positions):
    ax1.scatter(pos[0], pos[1], label=all_bodies[i]['name'], color=colors(i))
    ax1.annotate(all_bodies[i]['name'], (pos[0], pos[1]), fontsize=8)

ax1.scatter(0, 0, color='yellow', edgecolors='black', s=300, marker='o', label='Sun')
ax1.set_title('Original Solar System (Before Disturbance)', fontsize=16)
ax1.set_xlabel('X position (meters)')
ax1.set_ylabel('Y position (meters)')
ax1.grid(True)
ax1.axis('equal')
ax1.legend()

# After plot with Trails
for i, body in enumerate(all_bodies):
    name = body['name']
    ax2.plot(trajectories[name]['x'], trajectories[name]['y'], color=colors(i), label=name)
    ax2.scatter(trajectories[name]['x'][-1], trajectories[name]['y'][-1], color=colors(i))  # Final position
    ax2.annotate(name, (trajectories[name]['x'][-1], trajectories[name]['y'][-1]), fontsize=8)

ax2.scatter(0, 0, color='yellow', edgecolors='black', s=300, marker='o', label='Sun')
ax2.set_title('Solar System After Disturbance (Trails Shown)', fontsize=16)
ax2.set_xlabel('X position (meters)')
ax2.set_ylabel('Y position (meters)')
ax2.grid(True)
ax2.axis('equal')
ax2.legend()

plt.tight_layout()
plt.show()
