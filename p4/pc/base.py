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
         [[0, 0], [-79.61333333333333, [0, 1, 2]]], [[4, 0], [-79.52, [1, 2, 5]]], [[4, 3.5], [-78.89333333333333, [2, 5, 6]]], 
         [[4, 4.5], [-83.18, [2, 5, 6]]], [[4, 8], [-78.96, [2, 5, 6]]], [[0, 8], [-75.88000000000001, [1, 2, 5]]], 
         [[0, 4.5], [-72.17999999999999, [0, 2, 3]]], [[0, 3.5], [-72.77333333333333, [2, 3, 6]]], 
         [[4, 1.75], [-77.72666666666667, [1, 2, 6]]], [[2, 0], [-75.61333333333333, [2, 3, 5]]], 
         [[0, 1.75], [-74.45333333333333, [2, 3, 5]]], [[2, 1.75], [-76.73333333333333, [1, 2, 3]]], 
         [[2, 3.5], [-74.56, [0, 2, 3]]], [[4, 6.25], [-77.63333333333334, [2, 3, 5]]], 
         [[2, 4.5], [-73.76666666666667, [1, 2, 5]]], [[0, 6.25], [-74.26666666666667, [0, 2, 3]]], 
         [[2, 8], [-75.36, [2, 3, 6]]], [[2, 6.25], [-77.5, [1, 2, 3]]]
        ]
 #training data collected [rssi, (x, y)]

#[[x, y], [rssi, [ top 3 nodes]]]



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


# Returns [rssi average, top 3 nodes]
def mean_rssi(rssi_vals, k):
    rssi_nodes = sorted(range(len(rssi_vals)), key=lambda i: rssi_vals[i])[-k:]
    rssi_val_avg = sorted(rssi_vals)
    k_highest_val = rssi_val_avg[:k]
    total = 0
    for i in k_highest_val:
        total += i
    avg = [total/k, rssi_nodes]
    return avg

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
    # For each example [[x, y], [rssi, [top 3 nodes]]] data in data
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





