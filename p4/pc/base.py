import serial
import numpy
import matplotlib.pyplot as plt
import json

# Graphing setup
plt.ion()
fig, ax = plt.subplots()
x, y = [],[]
sc = ax.scatter(x,y)
plt.title("Position of Mobile Node")
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.xlim(0,10)
plt.ylim(0,10)
plt.draw()

dt = 0.1 #update interval (seconds)
#beacons and their physical coordinates (x, y, d) in meters
# ibeacons = [[0,0,0], [4,0,0], [4,4,0], [0,4,0]]
ibeacons = [[0,0,0], [4,0,0]]
#Kalman filter settings
A = numpy.array([(1, 0, dt, 0), (0, 1, 0, dt), (0, 0, 1, 0), (0, 0, 0, 1)]) #[x, y, vx, vy]
#observation vector [x_rs, y_rs, x_us, y_us] - fusion = (0.5x_rs + 0.5x_us, 0.5y_rs + 0.5y_us)
H = numpy.array([(0.5, 0, 0, 0), (0, 0.5, 0, 0), (0.5, 0, 0, 0), (0, 0.5, 0, 0)])
#process error covariance matrix (emprical)
Q = numpy.eye(4)*0.1 #Q=eye(n)*process_error % n: number of variables i.e. x, y, vx, vy
#observation noise covariance matrix (emprical)
R = numpy.diag([0.5, 0.5, 0.1, 0.1]) #emperical means of measurement errors: RSSI (0.5m) and Ultrasonic (0.1m)
loclog_kalman = []
loclog_rssi = []
loclog_unic = []

# Training data for rssi
training_data_rssi = [
        [-56.8, (0, 0)], [-59.8, (0, 0)], [-52.18, (0, 0)], [-70.1, (0, 0)], [-62.0, (0, 0)], [-68.68, (0, 0)], [-77.74, (0, 0)],
        [-55.54, (4, 0)], [-68.08, (4, 0)], [-57.4, (4, 0)], [-65.02, (4, 0)], [-87.46, (4, 0)],  [-62.0, (4, 0)], [-67.64, (4, 0)], [-83.02, (4, 0)],
        [-72.04, (4, 3.5)], [-66.7, (4, 3.5)], [-48.02, (4, 3.5)], [-70.78, (4, 3.5)], [-84.86, (4, 3.5)], [-62, (4, 3.5)], [-64.62, (4, 3.5)], [-79.78, (4, 3.5)],
        [-80.2, (4, 4.5)], [-68.14, (4, 4.5)], [-53.44, (4, 4.5)], [-63.98, (4, 4.5)], [-87.84, (4, 4.5)], [-62, (4, 4.5)], [-62.42, (4, 4.5)], [-81.5, (4, 4.5)],
        [-64.62, (4, 8)], [-71.58, (4, 8)], [-60.46, (4, 8)], [-63.98, (4, 8)], [-87.26, (4, 8)], [-62.0, (4, 8)], [-56.44, (4, 8)], [-78.04, (4, 8)],
        [-70.26, (0, 8)], [-61.56, (0, 8)], [-57.94, (0, 8)], [-63.48, (0, 8)], [-87.2, (0, 8)], [-62.0, (0, 8)], [-62.72, (0, 8)], [-70.18, (0, 8)],
        [-55.06, (0, 4.5)], [-64.52, (0, 4.5)], [-58.56, (0, 4.5)], [-49.9, (0, 4.5)], [-74.06, (0, 4.5)], [-62.0, (0, 4.5)], [-69.1, (0, 4.5)], [-73.38, (0, 4.5)],
        [-64.38, (0, 3.5)], [-60.64, (0, 3.5)], [-54.56, (0, 3.5)], [-39.82, (0, 3.5)], [-80.2, (0, 3.5)], [-62.0, (0, 3.5)], [-58.72, (0, 3.5)], [-73.74, (0, 3.5)],
        [-69.44, (4, 1.75)], [-56.02, (4, 1.75)], [-56.88, (4, 1.75)], [-67.38, (4, 1.75)], [-85.24, (4, 1.75)], [-62.0, (4, 1.75)], [-60.58, (4, 1.75)], [-78.5, (4, 1.75)],
        [-66.64, (2, 0)], [-69.42, (2, 0)], [-59.24, (2, 0)], [-66.26, (2, 0)], [-82.2, (2, 0)], [-62.0, (2, 0)], [-69.76, (2, 0)], [-74.88, (2, 0)],
        [-66.64, (0, 1.75)], [-68.84, (0, 1.75)], [-55.4, (0, 1.75)], [-60.48, (0, 1.75)], [-86.08, (0, 1.75)], [-62.0, (0, 1.75)], [-65.52, (0, 1.75)], [-68.44, (0, 1.75)],
        [-60.84, (2, 1.75)], [-50.78, (2, 1.75)], [-53.92, (2, 1.75)], [-56.52, (2, 1.75)], [-83.74, (2, 1.75)], [-62.0, (2, 1.75)], [-64.82, (2, 1.75)], [-81.64, (2, 1.75)],
        [-51.34, (2, 3.5)], [-69.7, (2, 3.5)], [-55.02, (2, 3.5)], [-61.24, (2, 3.5)]
    # [[2, 3.5], [-51.34, -69.7, -55.02, -61.24, -83.16, -62.0, -68.14, -70.82]],
    # [[4, 6.25], [-67.78, -65.6, -63.66, -61.42, -88.56, -62.0, -65.48, -76.56]],
    # [[2, 4.5], [-67.14, -55.16, -55.54, -62.06, -77.06, -62.0, -67.62, -76.62]],
    # [[0, 6.25], [-59.04, -63.8, -60.0, -48.88, -79.26, -62.0, -76.8, -66.74]],
    # [[2, 8], [-65.4, -64.42, -60.72, -55.58, -84.5, -62.0, -57.36, -76.18]],
    # [[2, 6.25], [-63.78, -59.12, -47.4, -61.9, -83.5, -62.0, -72.34, -76.66]]

    ] #training data collected [rssi, (x, y)]



MY_DEVICE_TOKEN = '51ae8dcd-f076-402e-8f98-5ed3222de341'
my_device = tago.Device(MY_DEVICE_TOKEN)

# This function sends a complete set of RSSI and ultrasonic ranging data to the web dashboard
def send_complete_data_to_dashboard(rssi1, rssi2, rssi3, rssi4, rssi5, rssi6, rssi7, rssi8, us1, us2, us3, us4, mobile1x, mobile1y, mobile2x, mobile2y):

  data =  [

    {
      "variable": "rssi1",
      "value": rssi1
    },
    {
      "variable": "rssi2",
      "value": rssi2
    },
    {
      "variable": "rssi3",
      "value": rssi3
    },
    {
      "variable": "rssi4",
      "value": rssi4
    },
    {
      "variable": "rssi5",
      "value": rssi5
    },
    {
      "variable": "rssi6",
      "value": rssi6
    },
    {
      "variable": "rssi7",
      "value": rssi7
    },
    {
      "variable": "rssi8",
      "value": rssi8
    },
    {
      "variable": "us1",
      "value": us1
    },
    {
      "variable": "us2",
      "value": us2
    },
    {
      "variable": "us3",
      "value": us3
    },
    {
      "variable": "us4",
      "value": us4
    },
    {
      "variable": "us3",
      "value": us3
    },
    {
      "variable": "us4",
      "value": us4
    },
    {
      "variable": "mobile1x",
      "value": mobile1x
    },
    {
      "variable": "mobile1y",
      "value": mobile1y
    },
    {
      "variable": "mobile2x",
      "value": mobile2x
    },
    {
      "variable": "mobile2y",
      "value": mobile2y
    }
  ]

  result = my_device.insert(data)  # With response
  if result['status']:
      print(result['result'])
  else:
      print(result['message'])


def mean_rssi(rssi_vals, k):
    rssi_val = sorted(rssi_vals)
    k_highest_val = rssi_val[:k]
    total = 0
    for i in k_highest_val:
        total += i
    return total / k

# estimate the location of the mobile from the received beacons data using multilatiration
def estimate_location(beacons):
    n = len(beacons) - 1  # index of the last entry
    A = numpy.zeros((n, 2))  # matrix A
    b = numpy.arange(n)  # matrix b
    for i in range(n):
        A[i][0] = 2 * (beacons[n][0] - beacons[i][0])
        A[i][1] = 2 * (beacons[n][1] - beacons[i][1])
        b[i] = (beacons[i][2] ** 2) - (beacons[n][2] ** 2) - (beacons[i][0] ** 2) - (beacons[i][1] ** 2) + (
                    beacons[n][0] ** 2) + (beacons[n][1] ** 2)
        # find the location by solving the Ax = b using least square estimation
    return numpy.linalg.lstsq(A, b, rcond=None)[0]


def knn(data, query, k):
    neighbour_distances_and_indices = []
    # For each example [rssi, (x, y)] data in data
    for index, example in enumerate(data):
        # Calculate the distance between the query example and the current example from the data
        # distance = euclidean_distance(example[:-1], query)
        # print(example[:-1][0])
        distance = abs(example[:-1][0] - query)

        # Add the distance and the index of the example to an ordered collection
        neighbour_distances_and_indices.append((distance, index))

    # Sort the ordered collection of distances and indices from smallest to largest (in ascending order) by the
    # distances
    sorted_neighbour_distances_and_indices = sorted(neighbour_distances_and_indices)

    # Pick first K entries
    k_nearest_distances_and_indices = sorted_neighbour_distances_and_indices[:k]

    # Get the labels of the selected K entries
    k_nearest_labels = [data[i][-1] for distance, i in k_nearest_distances_and_indices]

    return mean_coord(k_nearest_labels)
def start_knn(rssi, k, us1, us2):
    step = 1
    # Returns mean coordinates of the 3 nearest nodes
    rssi_coord = knn(SysConf.training_data_rssi, rssi, k)
    loc_rssi = rssi_coord
    # estmate distance using Ultrasonic
    # change beacon number according to coordinates
    ibeacons[0][2] = us1
    ibeacons[1][2] = us2
    loc_unic = estimate_location(ibeacons)
    print(loc_rssi)
    print(loc_unic)

    # for the measured location i.e. the observation for kalman filter
    obs = numpy.append(loc_rssi, loc_unic)
    if (self.step == 1):
        x_init = obs  # the initial position of kalman filter
        cov_init = numpy.eye(len(x_init)) * 1.0  # the initial covariance
        kalman = KalmanFilter(x_init, cov_init, SysConf.A, SysConf.H, SysConf.Q, SysConf.R)
        loc_kalman = kalman.update(obs)
    else:
        # make prediction using kalman filter and update the filter
        loc_kalman = self.kalman.update(obs)
    print(loc_kalman)
    loclog_kalman.append(loc_kalman)
    loclog_rssi.append(loc_rssi)
    loclog_unic.append(loc_unic)
    step += 1
    if (step > 100):  # only store the last 100 measurements
        loclog_kalman.pop(0)
        loclog_rssi.pop(0)
        loclog_unic.pop(0)
    return loc_kalman
# Kalman filter implementation
class KalmanFilter:
    def __init__(self, x_init, cov_init, A, H, Q, R):
        self.ndim = len(x_init)  # dimension of the varible vector
        self.A = A  # state transition model
        self.H = H  # observation model
        self.Q_k = Q  # covariance matrix of process noise
        self.R = R  # covariance matrix of observation noise
        self.cov = cov_init  # covariance matrix
        self.x_hat = x_init  # prediction vector

    # Make prediction and update the filter gain
    def update(self, obs):
        # Make prediction
        self.x_hat_est = numpy.dot(self.A, self.x_hat)
        self.cov_est = numpy.dot(self.A, numpy.dot(self.cov, numpy.transpose(self.A))) + self.Q_k

        # Update estimate
        self.error_x = obs - numpy.dot(self.H, self.x_hat_est)
        self.error_cov = numpy.dot(self.H, numpy.dot(self.cov_est, numpy.transpose(self.H))) + self.R
        self.K = numpy.dot(numpy.dot(self.cov_est, numpy.transpose(self.H)), numpy.linalg.inv(self.error_cov))
        self.x_hat = self.x_hat_est + numpy.dot(self.K, self.error_x)
        if self.ndim > 1:
            self.cov = numpy.dot((numpy.eye(self.ndim) - numpy.dot(self.K, self.H)), self.cov_est)
        else:
            self.cov = (1 - self.K) * self.cov_est

            # return self.x_hat weighted sum
        return [0.5 * self.x_hat[0] + 0.5 * self.x_hat[2], 0.5 * self.x_hat[1] + 0.5 * self.x_hat[3]]


while 1:

    line = ser.readline()
    line = line.decode('utf8')

    j = json.loads(line)

    # WHAT ARE THE COORDINATES OF THESE NODES?
    rssi1 = j[0][0]
    rssi2 = j[1][0]
    rssi3 = j[2][0]
    rssi4 = j[3][0]
    rssi5 = j[4][0]
    rssi6 = j[5][0]
    rssi7 = j[6][0]
    rssi8 = j[7][0]

    # WHICH STATIC NODES HAVE US DISTANCE MEASUREMENTS?
    us1 = j[0][1]
    us2 = j[1][1]
    us3 = j[2][1]
    us4 = j[3][1]

    # DO PROCESSING HERE TO FIND POSITION OF MOBILE NODE 1 AND 2...
    #...
    #...
    #...
    k = 3
    rssi_vals = [rssi1, rssi2, rssi3, rssi4, rssi5, rssi6, rssi7, rssi8]
    rssi = mean_rssi(rssi_vals, k)
    # knn(training_data, rssi, k)
    predicted_coords = start_knn(rssi, k, us1, us2)

    mobile1x = predicted_coords[0]
    mobile1y = predicted_coords[1]
    mobile2x = 0
    mobile2y = 0

    # SEND THE INFORMATION TO THE WEB DASHBOARD
    send_complete_data_to_dashboard(rssi1, rssi2, rssi3, rssi4, rssi5, rssi6, rssi7, rssi8, us1, us2, us3, us4, mobile1x, mobile1y, mobile2x, mobile2y)

    x = mobile1x
    y = mobile1y
    sc.set_offsets(numpy.c_[x, y])
    fig.canvas.draw_idle()
    plt.pause(0.1)





