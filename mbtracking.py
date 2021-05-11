import math
import time
import json
import random
import numpy
import pylab
import serial
import tkinter as tk
from threading import Thread
from tkinter import *
from PIL import Image, ImageTk

# The system configuration
class SysConf:
    #GUI parameters
    BOARD_WIDTH = 500
    BOARD_HEIGHT = 900
    DELAY = 100
    GRID_SIZE = 50
    BOARD_MARGIN = 50
    IBEACON_SIZE = 20
    MBNODE_SIZE = 20
    REFRESH_RATE = 200
    
    #beacons and their physical coordinates (x, y, d) in meters
    ibeacons = [[0,0,0],[4,0,0],[4,4,0],[0,4,0]]
    #scalling factor to map the physical board to GUI
    scalling = 100 #1m is equivalent to 100 pixel i.e 4mx4m - 400px-400px
    #mobile device assumptions
    dt = 0.1 #update interval (seconds)
    
    #Kalman filter settings    
    A = numpy.array([(1, 0, dt, 0), (0, 1, 0, dt), (0, 0, 1, 0), (0, 0, 0, 1)]) #[x, y, vx, vy]
    #observation vector [x_rs, y_rs, x_us, y_us] - fusion = (0.5x_rs + 0.5x_us, 0.5y_rs + 0.5y_us)
    H = numpy.array([(0.5, 0, 0, 0), (0, 0.5, 0, 0), (0.5, 0, 0, 0), (0, 0.5, 0, 0)])
    #process error covariance matrix (emprical)    
    Q = numpy.eye(4)*0.1 #Q=eye(n)*process_error % n: number of variables i.e. x, y, vx, vy
    #observation noise covariance matrix (emprical)
    R = numpy.diag([0.5, 0.5, 0.1, 0.1]) #emperical means of measurement errors: RSSI (0.5m) and Ultrasonic (0.1m)

    values = numpy.random.randint(1, 5, 6) # random location values

    training_data_rssi = [
        [-56.8, (0, 0)],
        [-59.8, (0, 0)],
        [-52.18, (0, 0)],
        [-70.1, (0, 0)],
        [-62.0, (0, 0)],
        [-68.68, (0, 0)],
        [-77.74, (0, 0)],
        [-55.54, (4, 0)],
        [-68.08, (4, 0)],
        [-57.4, (4, 0)],
        [-65.02, (4, 0)],
        [-87.46, (4, 0)],
        [-62.0, (4, 0)],
        [-67.64, (4, 0)],
        [-83.02, (4, 0)],
        [-72.04, (4, 3.5)],
        [-66.7, (4, 3.5)],
        [-48.02, (4, 3.5)],
        [-70.78, (4, 3.5)],
        [-84.86, (4, 3.5)],
        [-62, (4, 3.5)],
        [-64.62, (4, 3.5)],
        [-79.78, (4, 3.5)],
        [-80.2, (4, 4.5)],
        [-68.14, (4, 4.5)],
        [-53.44, (4, 4.5)],
        [-63.98, (4, 4.5)],
        [-87.84, (4, 4.5)],
        [-62, (4, 4.5)],
        [-62.42, (4, 4.5)],
        [-81.5, (4, 4.5)],
        [-64.62, (4, 8)],
        [-71.58, (4, 8)],
        [-60.46, (4, 8)],
        [-63.98, (4, 8)],
        [-87.26, (4, 8)],
        [-62.0, (4, 8)],
        [-56.44, (4, 8)],
        [-78.04, (4, 8)],
        [-70.26, (0, 8)],
        [-61.56, (0, 8)],
    ] #training data collected [rssi, (x, y)]

#Kalman filter implementation
class KalmanFilter:
    def __init__(self, x_init, cov_init, A, H, Q, R):
        self.ndim = len(x_init) #dimension of the varible vector 
        self.A = A #state transition model
        self.H = H #observation model
        self.Q_k = Q #covariance matrix of process noise
        self.R = R #covariance matrix of observation noise
        self.cov = cov_init #covariance matrix
        self.x_hat = x_init #prediction vector
        
    #Make prediction and update the filter gain
    def update(self, obs):
        # Make prediction
        self.x_hat_est = numpy.dot(self.A,self.x_hat)
        self.cov_est = numpy.dot(self.A,numpy.dot(self.cov,numpy.transpose(self.A))) + self.Q_k

        # Update estimate
        self.error_x = obs - numpy.dot(self.H, self.x_hat_est)
        self.error_cov = numpy.dot(self.H, numpy.dot(self.cov_est,numpy.transpose(self.H))) + self.R
        self.K = numpy.dot(numpy.dot(self.cov_est, numpy.transpose(self.H)), numpy.linalg.inv(self.error_cov))
        self.x_hat = self.x_hat_est + numpy.dot(self.K, self.error_x)
        if self.ndim > 1:
            self.cov = numpy.dot((numpy.eye(self.ndim) - numpy.dot(self.K,self.H)),self.cov_est)
        else:
            self.cov = (1-self.K)*self.cov_est        
        
        #return self.x_hat weighted sum
        return [0.5*self.x_hat[0] + 0.5*self.x_hat[2], 0.5*self.x_hat[1] + 0.5*self.x_hat[3]]
       
# The system dashboard   
class Dashboard(Canvas):
    def __init__(self, root):
        super().__init__(root, width=SysConf.BOARD_WIDTH, height=SysConf.BOARD_HEIGHT, background="white", highlightthickness=0)
        self.root = root
        #self.grid(sticky=W, row = 0, column = 0)
        self.mbloc = [0, 0] #mobile node location
        self.draw_board()        
        self.pack()

    def draw_board(self):    
        # vertical lines at an interval of "line_distance" pixel
        for x in range(SysConf.BOARD_MARGIN, SysConf.BOARD_WIDTH, SysConf.GRID_SIZE):
            self.create_line(x, SysConf.BOARD_MARGIN, x, SysConf.BOARD_HEIGHT - SysConf.BOARD_MARGIN, fill="#476042")
            self.create_text(x, SysConf.BOARD_MARGIN/2, text="{0}".format(x - SysConf.BOARD_MARGIN), fill="black")
                         
        # horizontal lines at an interval of "line_distance" pixel
        for y in range(SysConf.BOARD_MARGIN, SysConf.BOARD_HEIGHT, SysConf.GRID_SIZE):
            self.create_line(SysConf.BOARD_MARGIN, y, SysConf.BOARD_WIDTH - SysConf.BOARD_MARGIN, y, fill="#476042")
            self.create_text(SysConf.BOARD_MARGIN/2, y, text="{0}".format(y - SysConf.BOARD_MARGIN), fill="black")

        # draw the ibeacons and the mobile node
        try:            
            self.ibeacon = Image.open("ibeacon.png")
            self.ibeacon = self.ibeacon.resize((SysConf.IBEACON_SIZE, SysConf.IBEACON_SIZE), Image.ANTIALIAS)
            self.ibeacon = ImageTk.PhotoImage(self.ibeacon)
            self.create_image(SysConf.BOARD_MARGIN-SysConf.IBEACON_SIZE + 10, SysConf.BOARD_MARGIN-SysConf.IBEACON_SIZE + 10, image=self.ibeacon, anchor=NW,  tag="beacon-1")
            self.create_image(SysConf.BOARD_MARGIN-SysConf.IBEACON_SIZE + 10, SysConf.BOARD_HEIGHT-SysConf.BOARD_MARGIN-SysConf.IBEACON_SIZE + 10, image=self.ibeacon, anchor=NW,  tag="beacon-2")
            self.create_image(SysConf.BOARD_WIDTH-SysConf.BOARD_MARGIN-SysConf.IBEACON_SIZE + 10, SysConf.BOARD_MARGIN-SysConf.IBEACON_SIZE + 10, image=self.ibeacon, anchor=NW,  tag="beacon-3")
            self.create_image(SysConf.BOARD_WIDTH-SysConf.BOARD_MARGIN-SysConf.IBEACON_SIZE + 10, SysConf.BOARD_HEIGHT-SysConf.BOARD_MARGIN-SysConf.IBEACON_SIZE + 10, image=self.ibeacon, anchor=NW,  tag="beacon-4")
            #
            self.imobile = Image.open("bluetooth.png")
            self.imobile = self.imobile.resize((SysConf.MBNODE_SIZE, SysConf.MBNODE_SIZE), Image.ANTIALIAS)
            self.imobile = ImageTk.PhotoImage(self.imobile)
            self.create_image(SysConf.BOARD_MARGIN, SysConf.BOARD_MARGIN, image=self.imobile, anchor=NW,  tag="mobile_node")
            
        except IOError as e:
            print(e)
            sys.exit(1)
    
    #Get the current location of the mobile node on screen
    #def get_location(self):
    #    mobile_node = self.find_withtag("mobile_node")
        
    #Update the location of the mobile node on the dashboard
    def update_node(self, loc):
        mobile_node = self.find_withtag("mobile_node")        
        self.delete(mobile_node)
        self.create_image(loc[0] * SysConf.scalling, loc[1] * SysConf.scalling, image=self.imobile, anchor=NW,  tag="mobile_node")
         
        #dx = (loc[0]- self.mbloc[0]) * SysConf.scalling
        #dy = (loc[1] - self.mbloc[1]) * SysConf.scalling
        #print (dx,dy)
        #self.move(mobile_node, dx, dy)
        #self.mbloc[0] = loc[0]
        #self.mbloc[1] = loc[1]
        
# The system GUI    
class SysGUI(Frame):
    def __init__(self, root):
        super().__init__()
        self.root = root
        root.resizable(False, False)
        self.master.title('MB IoT Localization System (C)')
        self.dashboard = Dashboard(self) 
        #
        self.lbStatus = Label(self, text="Status:")
        self.lbStatus.pack(side=LEFT, padx=5, pady=5)
        #
        self.ic_exit = PhotoImage(file='exit.png')
        self.btnExit = Button(self, text="  Exit", image = self.ic_exit, compound = LEFT, width = 75, command = self.exit)
        self.btnExit.pack(side=RIGHT, padx=5, pady=5)
        #
        self.ic_graph = PhotoImage(file='graph.png')
        self.btnStart = Button(self, text="Graphs", image = self.ic_graph, compound = LEFT, width = 75, command = self.graphs)
        self.btnStart.pack(side=RIGHT, padx=5, pady=5)
        #       
        self.ic_track = PhotoImage(file='track.png')
        self.btnStart = Button(self, text="Tracking", image = self.ic_track, compound = LEFT, width = 75, command = self.tracking)
        self.btnStart.pack(side=RIGHT, padx=5, pady=5)
        #
        self.pack(fill=BOTH, expand=True)               
        
    #start tracking
    def tracking(self):
        #start tracking
        # while True:
        if self.btnStart['text'] == 'Tracking':
            # while True:
                # values = numpy.random.randint(1, 5, 6)
            self.tracker = Tracker(self)
            # self.tracker.start()
            # self.tracker.start_dummy()
            self.tracker.start_knn()
            self.btnStart['text'] = 'Stop'
                # if self.btnStart['text'] == 'Tracking':
                #     break
        else:
            # self.tracker.stop()
            # self.tracker.join()
            self.btnStart['text'] = 'Tracking'
            # break
    
    #plot tracking graphs
    def graphs(self):
        if hasattr(self, 'tracker'):
            self.tracker.graphs()
        
    #exit the program    
    def exit(self):
        if hasattr(self, 'tracker'):
            self.tracker.stop()
            self.tracker.join()        
        self.root.destroy()
    
    #show status on GUI
    def status(self, text):
        tk.messagebox.showinfo(title="system", message=text)


        
    #update the GUI every 1 second
    def update(self):
        #update system clock
        now = time.strftime("%H:%M:%S")
        self.lbStatus.configure(text=now)
        self.root.after(1000, self.update)
        
        #dx, dy = self.get_location_update()   
        #self.dashboard.location_update(dx, dy)
        #if self.btnStart['text'] == 'Stop':    
            
#Tracking the mobile node
class Tracker:
    def __init__(self, gui):
        # Thread.__init__(self)
        self.gui = gui
        self.step = 1 #time step
        self.loclog_kalman = [] #location log by estimated using kalman filter
        self.loclog_rssi = [] #location log by estimated using rssi
        self.loclog_unic = [] #location log by estimated using untrasonic
        #serial communication
        # self.ser = serial.Serial()
        # self.ser.port = "COM10" #"/dev/ttyUSB0"
        # self.ser.baudrate = 9600
        # self.ser.bytesize = serial.EIGHTBITS #number of bits per bytes
        # self.ser.parity = serial.PARITY_NONE #set parity check: no parity
        # self.ser.stopbits = serial.STOPBITS_ONE #number of stop bits
        # self.ser.timeout = None      #block read
        #self.ser.timeout = 1        #non-block read
        #self.ser.timeout = 2        #timeout block read
        #self.ser.xonxoff = False    #disable software flow control
        #self.ser.rtscts = False     #disable hardware (RTS/CTS) flow control
        #self.ser.dsrdtr = False     #disable hardware (DSR/DTR) flow control
        #self.ser.writeTimeout = 2   #timeout for write

    def start_knn(self, rssi, unic):
        # Returns mean coordinates of the 3 nearest nodes
        rssi_coord = knn(SysConf.training_data_rssi, rssi)
        # unic_coord = knn(SysConf.training_data_unic, unic)
        loc_rssi = rssi_coord
        # estmate distance using Ultrasonic
        SysConf.ibeacons[2][2] = SysConf.values[4]
        SysConf.ibeacons[3][2] = SysConf.values[5]
        loc_unic = self.estimate_location(SysConf.ibeacons)
        # loc_unic = unic_coord
        print(loc_rssi)
        print(loc_unic)

        # for the measured location i.e. the observation for kalman filter
        obs = numpy.append(loc_rssi, loc_unic)
        if (self.step == 1):
            x_init = obs  # the initial position of kalman filter
            cov_init = numpy.eye(len(x_init)) * 1.0  # the initial covariance
            self.kalman = KalmanFilter(x_init, cov_init, SysConf.A, SysConf.H, SysConf.Q, SysConf.R)
            loc_kalman = self.kalman.update(obs)
        else:
            # make prediction using kalman filter and update the filter
            loc_kalman = self.kalman.update(obs)
        print(loc_kalman)
        self.loclog_kalman.append(loc_kalman)
        self.loclog_rssi.append(loc_rssi)
        self.loclog_unic.append(loc_unic)
        self.step += 1
        if (self.step > 100):  # only store the last 100 measurements
            self.loclog_kalman.pop(0)
            self.loclog_rssi.pop(0)
            self.loclog_unic.pop(0)
        self.gui.dashboard.update_node(loc_kalman)

    def start_dummy(self):

        val = numpy.random.randint(1, 5, 6)
        new_val = numpy.random.randint(1, 5, 6)
        for i in range(len(val) - 1):
            val[i] = new_val[i] + val[i]
            while val[i] > 4:
                val[i] = val[i] * (1 - numpy.random.uniform(0, 1))
        SysConf.values = val
        print(SysConf.values)
        SysConf.ibeacons[0][2] = SysConf.values[0]
        SysConf.ibeacons[1][2] = SysConf.values[1]
        SysConf.ibeacons[2][2] = SysConf.values[2]
        SysConf.ibeacons[3][2] = SysConf.values[3]
        loc_rssi = self.estimate_location(SysConf.ibeacons)
        # loc_rssi = self.estimate_location()
        print(loc_rssi)
        # estmate distance using Ultrasonic
        SysConf.ibeacons[2][2] = SysConf.values[4]
        SysConf.ibeacons[3][2] = SysConf.values[5]
        loc_unic = self.estimate_location(SysConf.ibeacons)
        # loc_unic = self.estimate_location()
        print(loc_unic)
        # for the measured location i.e. the observation for kalman filter
        obs = numpy.append(loc_rssi, loc_unic)
        if (self.step == 1):
            x_init = obs  # the initial position of kalman filter
            cov_init = numpy.eye(len(x_init)) * 1.0  # the initial covariance
            self.kalman = KalmanFilter(x_init, cov_init, SysConf.A, SysConf.H, SysConf.Q, SysConf.R)
            loc_kalman = self.kalman.update(obs)
        else:
            # make prediction using kalman filter and update the filter
            loc_kalman = self.kalman.update(obs)
        print(loc_kalman)
        self.loclog_kalman.append(loc_kalman)
        self.loclog_rssi.append(loc_rssi)
        self.loclog_unic.append(loc_unic)
        self.step += 1
        if (self.step > 100):  # only store the last 100 measurements
            self.loclog_kalman.pop(0)
            self.loclog_rssi.pop(0)
            self.loclog_unic.pop(0)
        self.gui.dashboard.update_node(loc_kalman)
            # time.sleep(1)
              
    #estimate the location of the mobile from the received beacons data using multilatiration
    def estimate_location(self, beacons):
        # beacons = [
        #    # (xi, yi, di)
        #    (0, 0, 281.84),
        #    (0,400, 283.84),
        #    (400,0, 280.84),
        #    (400,400, 282.84)
        # ]
        
        n = len(beacons)-1 #index of the last entry
        A = numpy.zeros((n, 2)) #matrix A
        b = numpy.arange(n) #matrix b
        for i in range(n):
            A[i][0] = 2*(beacons[n][0] - beacons[i][0])
            A[i][1] = 2*(beacons[n][1] - beacons[i][1])
            b[i] = (beacons[i][2]**2) - (beacons[n][2]**2) - (beacons[i][0]**2) - (beacons[i][1]**2) + (beacons[n][0]**2) + (beacons[n][1]**2)        
        #find the location by solving the Ax = b using least square estimation
        return numpy.linalg.lstsq(A, b, rcond=None)[0]
    
    #plot tracking graphs
    def graphs(self):
        dim = 0
        pylab.figure()
        #pylab.scatter(x_true[:,0],x_true[:,1],s=1, c='b',marker='o', edgecolors='b', label="true location")
        if len(self.loclog_rssi) > 0:
            rssi = numpy.array(self.loclog_rssi)
            pylab.scatter(rssi[:,0],   rssi[:,1], s=10, c='c', marker='*', edgecolors='c', label="rssi measured loc")
            dim += 1
        if len(self.loclog_unic) > 0:
            unic = numpy.array(self.loclog_unic)
            pylab.scatter(unic[:,0],   unic[:,1], s=10, c='g', marker='*', edgecolors='g', label="ultra sonic measured loc")
            dim += 1
        if len(self.loclog_kalman) > 0:        
            kalman = numpy.array(self.loclog_rssi)
            pylab.scatter(kalman[:,0], kalman[:,1], s=10, c='r', marker='s', edgecolors='r', label="kalman estimated loc")
            dim += 1            
        pylab.xlabel('x [m]')
        pylab.ylabel('y [m]')
        pylab.legend(loc = dim)
        pylab.show()

def knn(data, query):
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
    k_nearest_distances_and_indices = sorted_neighbour_distances_and_indices[:3]

    # Get the labels of the selected K entries
    k_nearest_labels = [data[i][-1] for distance, i in k_nearest_distances_and_indices]

    return mean_coord(k_nearest_labels)

def euclidean_distance(point1, point2):
    sum_squared_distance = 0
    for i in range(len(point1)):
        sum_squared_distance += math.pow(point1[i] - point2[i], 2)

    return math.sqrt(sum_squared_distance)

def mean_coord(coord):
    total_x = 0
    total_y = 0
    for x, y in coord:
        total_x += x
        total_y += y

    return ((total_x/3, total_y/3))

    
#The main method        
def main():
    rssi_coord = knn(SysConf.training_data_rssi, -58)
    # unic_coord = knn(SysConf.training_data_unic, unic)
    loc_rssi = rssi_coord
    # loc_unic = unic_coord
    print(loc_rssi)
    # tk = Tk()
    # gui = SysGUI(tk)
    # tk.mainloop()


if __name__ == '__main__':
    main()