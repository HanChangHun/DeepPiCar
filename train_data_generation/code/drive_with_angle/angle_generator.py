import json
import numpy as np

if __name__ == "__main__":
    angles = []

    for angle in np.linspace(90, 90, 25):
        angles.append(float(angle))

    for angle in np.linspace(90, 120, 10):
        angles.append(float(angle))

    for angle in np.linspace(120, 120, 30):
        angles.append(float(angle))

    for angle in np.linspace(120, 100, 15):
        angles.append(float(angle))

    for angle in np.linspace(100, 90, 5):
        angles.append(float(angle))

    for angle in np.linspace(90, 75, 20):
        angles.append(float(angle))

    for angle in np.linspace(75, 70, 30):
        angles.append(float(angle))

    for angle in np.linspace(70, 70, 67):
        angles.append(float(angle))

    for angle in np.linspace(70, 80, 20):
        angles.append(float(angle))

    for angle in np.linspace(80, 110, 10):
        angles.append(float(angle))

    for angle in np.linspace(110, 120, 10):
        angles.append(float(angle))

    for angle in np.linspace(120, 100, 20):
        angles.append(float(angle))

    for angle in np.linspace(100, 95, 5):
        angles.append(float(angle))

    for angle in np.linspace(95, 90, 10):
        angles.append(float(angle))

    for angle in np.linspace(90, 90, 5):
        angles.append(float(angle))

    with open("train_data_generation/code/drive_with_angle/steer_angles.json", "w") as f:
        json.dump(angles, f, indent=4)
