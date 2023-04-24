import numpy as np;
import matplotlib.pyplot as plt;
import json;

from Model import Model;
from Sensor import Sensor;
from KF import KF;

model = Model();

# Constant error!

# Settings
CPS = 20;
dt = 1/CPS;
time = 60;
iterations = round(time / dt)
system_dev = 0.15;

camera_off_intervals = [[0.2, 0.7]]

pos_sensor_dev = 1;
vel_sensor_dev = 1;

# Matrices
posH = np.array([
    [1, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0]
])

velH = np.array([
    [0, 0, 1, 0, 0, 0],
    [0, 0, 0, 1, 0, 0]
])

F = np.array([
    [1, 0, dt, 0, 0.5*pow(dt,2), 0],
    [0, 1, 0, dt, 0, 0.5*pow(dt,2)],
    [0, 0, 1, 0, dt, 0],
    [0, 0, 0, 1, 0, dt],
    [0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 1]
])

Q = np.array([
    [pow(dt, 4)/4, 0, pow(dt,3)/2, 0, pow(dt,2)/2, 0],
    [0, pow(dt, 4)/4, 0, pow(dt,3)/2, 0, pow(dt,2)/2],
    [pow(dt,3)/2, 0, pow(dt, 2), 0, dt, 0],
    [0, pow(dt,3)/2, 0, pow(dt,2 ), 0, dt],
    [pow(dt,2)/2, 0, dt, 0, 1, 0],
    [0, pow(dt,2)/2, 0, dt, 0, 1]
]) * pow(system_dev, 2)

posR = np.array([
    [pow(pos_sensor_dev, 2), 0],
    [0, pow(pos_sensor_dev, 2)]
])

velR = np.array([
    [pow(vel_sensor_dev, 2), 0],
    [0, pow(vel_sensor_dev, 2)]
])

P = np.array([
    [0.1, 0, 0, 0, 0, 0],
    [0, 0.1, 0, 0, 0, 0],
    [0, 0, 500, 0, 0, 0],
    [0, 0, 0, 500, 0, 0],
    [0, 0, 0, 0, 500, 0],
    [0, 0, 0, 0, 0, 500]
])

x0 = np.array([
    [0.],
    [0.],
    [0.],
    [0.],
    [0.],
    [0.]
])

rawX = x0;

posSensor = Sensor(model, posH, pos_sensor_dev, 0, 0)
velSensor = Sensor(model, velH, vel_sensor_dev, 0.1, 0.5)

kf1 = KF(F, 0, Q, P, x0)
kf2 = KF(F, 0, Q, P, x0)

groundTruths = []
posMeasurements = []
velMeasurements = []

rawEstimates = []
estimates1 = []
covariances1 = []
estimates2 = []
covariances2 = []

pastZ = np.array([[0], [0]]);

for i in range(iterations):
    model.update(dt)
    
    z = posSensor.get_reading()
    z2 = velSensor.get_reading();

    rawX = rawX + np.dot(F, np.dot(velH.T, z2));
    rawEstimates.append(rawX)

    groundTruths.append(model.get_ground_truth())
    posMeasurements.append(z)
    velMeasurements.append(z2)
    
    estimates1.append(kf1.predict())
    estimates2.append(kf2.predict())
    
    kf1.update(z2, velH, velR)
    
    camera_is_off = False;
    
    for interval in camera_off_intervals:
        if (i > interval[0] * iterations and i < interval[1] * iterations):
            camera_is_off = True;
    
    if not camera_is_off:
        kf2.update(z, posH, posR)
    kf2.update(z2, velH, velR)
    
def generate1DArray(array, index):
    return list(zip(*list(zip(*array))[index]))[0]

# Save data
class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

data = {
    "groundTruths": np.array(groundTruths).tolist(),
    "posMeasurements": np.array(posMeasurements).tolist(),
    "velMeasurements": np.array(velMeasurements).tolist()
}

with open("output.json", "w") as json_file:
    json_file.write(json.dumps(data))


# Upper 3
plt.subplot(2, 3, 1)
plt.title("X Pos")
plt.xlabel("Time (sec)")
plt.ylabel("Pos (m)")
plt.plot(np.arange(0, len(groundTruths) * dt, dt), generate1DArray(groundTruths, 0), label="Ground Truth")
plt.plot(np.arange(0, len(posMeasurements) * dt, dt), generate1DArray(posMeasurements, 0), label="Pos Measurement")
# plt.plot(np.arange(0, len(rawEstimates) * dt, dt), genrate1DArray(rawEstimates, 0), label="Estimates (raw)")
plt.plot(np.arange(0, len(estimates1) * dt, dt), generate1DArray(estimates1, 0), label="Estimates (vel)")
plt.plot(np.arange(0, len(estimates2) * dt, dt), generate1DArray(estimates2, 0), label="Estimates (pos & vel)")

ax = plt.gca()
for interval in camera_off_intervals:
    ax.axvspan(i * interval[0] * dt, i * interval[1] * dt, alpha=0.3, color='red', label="Camera Off")
plt.legend()

plt.subplot(2, 3, 2)
plt.title("X Vel")
plt.xlabel("Time (sec)")
plt.ylabel("vel (ms^-1)")
plt.plot(np.arange(0, len(groundTruths) * dt, dt), generate1DArray(groundTruths, 2), label="Ground Truth")
plt.plot(np.arange(0, len(velMeasurements) * dt, dt), generate1DArray(velMeasurements, 0), label="Vel Measurement")
plt.plot(np.arange(0, len(estimates1) * dt, dt), generate1DArray(estimates1, 2), label="Estimates (vel)")
plt.plot(np.arange(0, len(estimates2) * dt, dt), generate1DArray(estimates2, 2), label="Estimates (pos & vel)")
plt.legend()

plt.subplot(2, 3, 3)
plt.title("X Err")
plt.xlabel("Time (sec)")
plt.ylabel("Err (m)")

plt.plot(np.arange(0, len(estimates2) * dt, dt), np.abs(np.subtract(generate1DArray(estimates2, 0), generate1DArray(groundTruths, 0))), label="Error")
plt.ylim(0)
ax = plt.gca()
for interval in camera_off_intervals:
    ax.axvspan(i * interval[0] * dt, i * interval[1] * dt, alpha=0.3, color='red', label="Camera Off")
plt.legend()

# Lower 3
plt.subplot(2, 3, 4)
plt.title("Y Pos")
plt.xlabel("Time (sec)")
plt.ylabel("Pos (m)")
plt.plot(np.arange(0, len(groundTruths) * dt, dt), generate1DArray(groundTruths, 1), label="Ground Truth")
plt.plot(np.arange(0, len(posMeasurements) * dt, dt), generate1DArray(posMeasurements, 1), label="Pos Measurement")
# plt.plot(np.arange(0, len(rawEstimates) * dt, dt), genrate1DArray(rawEstimates, 1), label="Estimates (raw)")
plt.plot(np.arange(0, len(estimates1) * dt, dt), generate1DArray(estimates1, 1), label="Estimates (vel)")
plt.plot(np.arange(0, len(estimates2) * dt, dt), generate1DArray(estimates2, 1), label="Estimates (pos & vel)")

ax = plt.gca()
for interval in camera_off_intervals:
    ax.axvspan(i * interval[0] * dt, i * interval[1] * dt, alpha=0.3, color='red', label="Camera Off")
plt.legend()

plt.subplot(2, 3, 5)
plt.title("Y Vel")
plt.xlabel("Time (sec)")
plt.ylabel("vel (ms^-1)")
plt.plot(np.arange(0, len(groundTruths) * dt, dt), generate1DArray(groundTruths, 3), label="Ground Truth")
plt.plot(np.arange(0, len(velMeasurements) * dt, dt), generate1DArray(velMeasurements, 1), label="Vel Measurement")
plt.plot(np.arange(0, len(estimates1) * dt, dt), generate1DArray(estimates1, 3), label="Estimates (vel)")
plt.plot(np.arange(0, len(estimates2) * dt, dt), generate1DArray(estimates2, 3), label="Estimates (pos & vel)")
plt.legend()

plt.subplot(2, 3, 6)
plt.title("Y Err")
plt.xlabel("Time (sec)")
plt.ylabel("Err (m)")

plt.plot(np.arange(0, len(estimates2) * dt, dt), np.abs(np.subtract(generate1DArray(estimates2, 1), generate1DArray(groundTruths, 1))), label="Error")
plt.ylim(0)
ax = plt.gca()
for interval in camera_off_intervals:
    ax.axvspan(i * interval[0] * dt, i * interval[1] * dt, alpha=0.3, color='red', label="Camera Off")
plt.legend()

plt.show()