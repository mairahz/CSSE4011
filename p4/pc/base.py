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
    mobile1x = 0
    mobile1y = 0
    mobile2x = 0
    mobile2y = 0

    # SEND THE INFORMATION TO THE WEB DASHBOARD
    send_complete_data_to_dashboard(rssi1, rssi2, rssi3, rssi4, rssi5, rssi6, rssi7, rssi8, us1, us2, us3, us4, mobile1x, mobile1y, mobile2x, mobile2y)

    x = mobile1x
    y = mobile1y
    sc.set_offsets(numpy.c_[x, y])
    fig.canvas.draw_idle()
    plt.pause(0.1)





