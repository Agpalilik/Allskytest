import json
import pykml
import pyproj
import math
import simplekml
import numpy as np
import xlsxwriter
from cmath import pi
from math import radians, cos, sin, sqrt, degrees, asin, atan2, acos 
import pyproj
geodesic = pyproj.Geod(ellps='WGS84')


# workbook = xlsxwriter.Workbook('Change_name.xlsx')
# worksheet = workbook.add_worksheet() - This is no longer used


kml = simplekml.Kml(open=1) # Open kml file which can be loaded in google Earth to show the bearings and intersects

input = open("input.txt","r")  # Open input file which specifies names of the two .json files with bearings
frames_per_sec = int(input.readline())   # Number of frames per sec.
dummy = int(input.readline())   # Plot intersections of bearings if dummy = 1
if dummy == 1:
	plot_intersect = True
else:
 	plot_intersect = False
print(plot_intersect)
n_files = int(input.readline())  # Number of input files - should be 2
max_bearings = 1000               # Max number of frames per observation - can be increased if need be
R_Earth = 6371.0

lat_stat = np.arange(max_bearings,dtype=float) # Define arrays to store lat, lon, height, and station number of the involved cameras
lon_stat = np.arange(max_bearings,dtype=float)
height_stat = np.arange(max_bearings,dtype=float)
stat_number = np.arange(3,dtype = int)

obs = [[None]*max_bearings for i in range(-3,3)]   # Define arrys to hold all of the observations
Time_id = [[None]*max_bearings for i in range(3)]  # Define arrays to hold timestamps of observations (string with date and time)
azimuth = np.zeros((3,max_bearings),dtype=float)   # Define arrays to hold all observed bearings

# lat, lon and height of stations used in the calculation - add more if you use new ones.

lat_stat[30] = 53.903437
lon_stat[30] = 13.053244
height_stat[30] = 33.0

lat_stat[34] = 54.4044183
lon_stat[34] = 9.982645
height_stat[34] = 28.0

lat_stat[35] = 53.220087
lon_stat[35] = 11.325496
height_stat[35] = 68.0

lat_stat[62] = 53.52233
lon_stat[62] = 8.58053
height_stat[62] = 25.0

lat_stat[89] = 55.4296
lon_stat[89] = 11.5597
height_stat[89] = 43.0

lat_stat[104] = 44.4066
lon_stat[104] = -70.7911
height_stat[104] = 210.0

lat_stat[141] = 46.178077
lon_stat[141] = -74.336672
height_stat[141] = 100.0


distance = 400


# set various counters to zero initially
n_data = [0,0,0]
start = True
delta_time = 1.0/frames_per_sec
dist_ground_old = 0.0
height_round_old = 0.0
dist_track_old = 0.0
dist_ground = 0.0
height_round = 0.0
dist_track = 0.0
first_obs = True


# Define a class that can store coordintes
class coordinates:
	def __init__(self,lat,lon):
		self.lat = lat
		self.lon = lon


# Define a class that can store coordites and distances
class coor_dist:
	def __init__(self,lat,lon,dist):
		self.lat = lat
		self.lon = lon
		self.dist = dist


# This function calculates the coordinates of a point defined by its distance and bearing from another point at (lat,lon)

def add_distance(lat, lon, bearing, distance):
    # convert Latitude and Longitude
    # into radians for calculation

	EARTH_RADIUS = R_Earth
	latitude = math.radians(lat)
	longitute = math.radians(lon)

    # calculate next latitude
	next_latitude = math.asin(math.sin(latitude) *
					math.cos(distance/EARTH_RADIUS) +
					math.cos(latitude) *
					math.sin(distance/EARTH_RADIUS) *
					math.cos(math.radians(bearing)))

    # calculate next longitude
	next_longitude = longitute + (math.atan2(math.sin(math.radians(bearing)) *
											math.sin(distance/EARTH_RADIUS) *
											math.cos(latitude),
											math.cos(distance/EARTH_RADIUS) -
											math.sin(latitude) *
											math.sin(next_latitude)))

    # convert points into decimal degrees
	new_lat = math.degrees(next_latitude)
	new_lon = math.degrees(next_longitude)

	new_coor = coordinates(new_lat,new_lon)	
	return new_coor

# This function calculates the distance between two points

def dist_2(lat1,lon1,lat2,lon2):
# distance between 2 sets of geographical coordinates
	dlon = pi/180*(lon2 - lon1)
	dlat = pi/180*(lat2 - lat1)
	a = (sin(dlat/2))**2 + cos(lat1) * cos(lat2) * (sin(dlon/2))**2
	print(a,lat1,lon1,lat2,lon2)
	c = 2 * atan2(sqrt(a), sqrt(1-a))
	distance = R_Earth * c
	return distance

# Define a class that can hold all the information required to define a particular observation

class bearings:
	def __init__(self, timestamp, date, hour, minute, second, dec_second, ras, decs, azs, els, color):
		self.timestamp = timestamp
		self.date = date
		self.hour = hour
		self.minute = minute
		self.second = second
		self.dec_second = dec_second
		self.ras = ras
		self.decs = decs
		self.azs = azs
		self.els = els
		self.color = color


# Initialize all observations - negative j values required if there is an offset in the first registration of the fireball between the two datasets.
for i in range(-3,3):
	for j in range(-max_bearings,max_bearings):
		obs[i][j] = bearings('None','None','None','None','None','None','None','None','None','None','None')



# when two cameras observe the same meteor the first observation is rarely the same - the more distant camera will often detect the meteor a 
# few frames later then the proximal camera. The delay, measured in number of frames, is the offset 

def find_offset(dummy):
	for i in range(1,max_bearings):
		for j in range(1,max_bearings):
			#print("ggggg",Time_id[1][i],Time_id[2][j])
			if (Time_id[1][i] != None and Time_id[1][i] == Time_id[2][j]):
				return (j-i)

# The following functions extracts, date, hour, minute sec, and dec_second (milliseconds) from the timestamp array

def date_set(x):
	date_set = x[0:10]
	return date_set

def hour_set(x):
	hour_set = x[11:13]
	return hour_set
	
def minute_set(x):
	minute_set = x[14:16]
	return minute_set

def second_set(x):
	second_set = x[17:19]
	return second_set

def dec_second_set(x):
	dec_second_set = x[20:23]
	return dec_second_set

# This functions assign colors of heading lines to be plotted in google Earth. Lines recorded at the same time from different stations get the same color-code

def linecolor(time):
	print("tid",time)
	color = simplekml.Color.white
	if time== 0:
		color = simplekml.Color.white
	if time== 40:
		color = simplekml.Color.aliceblue
	if time== 80:
		color = simplekml.Color.antiquewhite
	if time== 120:
		color = simplekml.Color.aqua
	if time== 160:
		color = simplekml.Color.aquamarine
	if time== 200:
		color = simplekml.Color.azure
	if time== 240:
		color = simplekml.Color.beige
	if time== 280:
		color = simplekml.Color.bisque
	if time== 320:
		color = simplekml.Color.black
	if time== 360:
		color = simplekml.Color.blanchedalmond
	if time== 400:
		color = simplekml.Color.blue
	if time== 440:
		color = simplekml.Color.blueviolet
	if time== 480:
		color = simplekml.Color.white
	if time== 520:
		color = simplekml.Color.aliceblue
	if time== 560:
		color = simplekml.Color.brown
	if time== 600:
		color = simplekml.Color.burlywood
	if time== 640:
		color = simplekml.Color.cadetblue
	if time== 680:
		color = simplekml.Color.chartreuse
	if time== 720:
		color = simplekml.Color.chocolate
	if time== 760:
		color = simplekml.Color.coral
	if time== 800:
		color = simplekml.Color.cornflowerblue
	if time== 840:
		color = simplekml.Color.crimson
	if time== 880:
		color = simplekml.Color.darkblue
	if time== 920:
		color = simplekml.Color.darkcyan
	if time== 960:
		color = simplekml.Color.darkgoldenrod
	return color


def intersection(x1,y1,b1,x2,y2,b2):

	# The following function calculates the intersection between two bearings:

	# Convert to radian
	lon1 = radians(float(x1))
	lat1 = radians(float(y1))
	b1 = radians(float(b1))

	lon2 = radians(float(x2))
	lat2 = radians(float(y2))
	b2 = radians(float(b2))

	dlon = lon2 - lon1 # Distance between longitude points
	dlat = lat2 - lat1 # Distance between latitude points

	# Great-circle distance between point 1 and point 2
	haversine = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2

	# angular distance between point 1 and point 2
	ang_dist_1_2 = 2 * atan2(sqrt(haversine),sqrt(1-haversine))

	# Initial and final bearings between point 1 and point 2
	initial_bearing = acos((sin(lat2) - sin(lat1) * cos(ang_dist_1_2)) / (sin(ang_dist_1_2) * cos(lat1)))
	final_bearing = acos((sin(lat1) - sin(lat2) * cos(ang_dist_1_2)) / (sin(ang_dist_1_2) * cos(lat2)))

	# Adjust the bearings on the trigonometric circle
	if sin(lon2 - lon1) > 0:
	    bearing_1_2 = initial_bearing
	    bearing_2_1 = (2 * pi) - final_bearing

	else:
	    bearing_1_2 = (2 * pi) - initial_bearing
	    bearing_2_1 = final_bearing


	# Angles between different points
	ang_1 = b1 - bearing_1_2    # angle p2<--p1-->p3
	ang_2 = bearing_2_1 - b2    # angle p1<--p2-->p3
	ang_3 = acos(-cos(ang_1) * cos(ang_2) + sin(ang_1) * sin(ang_2) * cos(ang_dist_1_2))    # angle p1<--p3-->p2

	# angular distance between point 1 and intersection point (point 3)
	ang_dist_1_3 = atan2(sin(ang_dist_1_2) * sin(ang_1) * sin(ang_2), cos(ang_2) + cos(ang_1) * cos(ang_3))

	# Latitude of point 3
	lat3 = asin(sin(lat1) * cos(ang_dist_1_3) + cos(lat1) * sin(ang_dist_1_3) * cos(b1))

	# Longitude of point 3
	delta_long_1_3 = atan2(sin(b1) * sin(ang_dist_1_3) * cos(lat1), cos(ang_dist_1_3) - sin(lat1) * sin(lat3))
	lon3 = lon1 + delta_long_1_3

	lat3 = degrees(lat3)
	lon3 = degrees(lon3)

	distance = ang_dist_1_3*R_Earth

	intersect = coor_dist(lat3,lon3,abs(distance))
	if plot_intersect:
#		single_point = kml.newpoint(coords = [(lon3,lat3)])
		pt2 = kml.newpoint(coords=[(lon3,lat3)])
		pt2.style.iconstyle.icon.href = 'http://maps.google.com/mapfiles/kml/shapes/placemark_circle.png'

	return intersect


##########    MAIN PROGRAM

print("MAIN")
col = 0

for i in range(1,3):
	stat_number[i] = int(input.readline())

	file_name = input.readline()
	file_name = file_name.strip() # Remove return line at the end of the string

	with open(file_name, "r") as read_file:
   		data = json.load(read_file)             # Read the json file with a complete data set from one of the cameras
	n_data[i] = len(data['best_meteor']['dt'])  # Number of observations in this data set
	print("n_data[",i,"] =",n_data[i])
	col = 0 #Start at column 0 in the excel file
	row = 1 #Start at line 1 below headers

	print("data",n_data[1],n_data[2])

	for j in range(0,n_data[i]):
		# Read time, azimuth and elevation of bearing number j
		observation = bearings(data['best_meteor']['dt'][j],date_set(data['best_meteor']['dt'][j]),hour_set(data['best_meteor']['dt'][j]),minute_set(data['best_meteor']['dt'][j]),second_set(data['best_meteor']['dt'][j]),dec_second_set(data['best_meteor']['dt'][j]),data['best_meteor']['ras'][j],data['best_meteor']['decs'][j],data['best_meteor']['azs'][j],data['best_meteor']['els'][j],3)
		azimuth[i][j] = observation.azs
		#Time_id[i][j] = data['best_meteor']['dt'][j]
		Time_id[i][j] = observation.timestamp
		print("sss",observation.timestamp)
		obs[i][j] = observation
		# Calculate a set of coordinates at a preset "distance" from the camera in order to plot a bearing line from the camera 
		end_coordinates = add_distance(lat_stat[stat_number[i]],lon_stat[stat_number[i]],observation.azs,distance)
		# write a label for the plotted line
		dummy = ("AMS",str(stat_number[i])," ",observation.timestamp, " ",str(observation.azs))
		label = ''.join(dummy)
		# add the line to the kml file
		linestring = kml.newlinestring(description=label)
		# give the line a time specific color. Lines recorded by different cameras but at the same time get the same color.
		linestring.style.linestyle.Color = linecolor(int(observation.dec_second))
		if (int(observation.dec_second)) == 0:
			linestring.coords = [(lon_stat[stat_number[i]],lat_stat[stat_number[i]]),(end_coordinates.lon,end_coordinates.lat)]
		elif (int(observation.dec_second)%(1000/frames_per_sec) ==0):
			linestring.coords = [(lon_stat[stat_number[i]],lat_stat[stat_number[i]]),(end_coordinates.lon,end_coordinates.lat)]
		print(int(observation.dec_second)%(1000/frames_per_sec),int(observation.dec_second),1000/frames_per_sec)
		linestring.style.linestyle.color = linecolor(int(observation.dec_second))
		# Lines recorded every full second are shown twice as wide
		if (int(observation.dec_second) == 0):
			linestring.style.linestyle.width= 2
		else:
			linestring.style.linestyle.width= 1
			


#	exel_line  = (
#		[observation.date],
#		[observation.hour],
#		[observation.minute],
#		[int(observation.second)+int(observation.dec_second)/1000 ],
#		[observation.azs],
#		[observation.els],
#	)
#	col = 0
#	for item, in (exel_line):
#		worksheet.write(row, col+(i-1)*10,item)
#		col += 1
#		row += 1

#	workbook.close()



offset = find_offset(obs)
print("offset =",offset)

#print(lat_stat[89],lon_stat[89],azimuth[1][32],lat_stat[62],lon_stat[62],azimuth[2][32+offset])
#dummy = intersection(lon_stat[89],lat_stat[89],azimuth[1][32],lon_stat[62],lat_stat[62],azimuth[2][32+offset])


workbook = xlsxwriter.Workbook('AMS_all.xlsx')
worksheet = workbook.add_worksheet()
bold = workbook.add_format({'bold': True})  #Define bold format in excel files


row = 1


# Write excel file with both datasets and calculate heights and velocities

#define excel file headings
headings  = (
	['Camera'],
	['date'],
	['hour'],
	['minute'],
	['second'],
	['dec_second'],
	['azs',],
	['els',],
	['distance'],
	['height (flat)',],
	['height (round)',],
	['ground dist'],
	['along track dist',],
	['velocity',],
	['Camera'],
	['date'],
	['hour'],
	['minute'],
	['second'],
	['dec_second'],
	['azs',],
	['els',],
	['distance'],
	)

col = 1
for item, in (headings):
		worksheet.write(0,col,item)
		col += 1
col = 0

for i in range(min(0,offset),max(n_data[1],n_data[2])+offset):
	if (Time_id[1][i] != None and Time_id[2][i+offset] != None) and (abs(azimuth[1][i]-azimuth[2][i+offset]) > 5):

		# In the following consider a triangle with angle A at the center of the Earth, 
		# angle B at the camera site and angle C at the intersect between two lines of sight 
		# from two stations.

		dummy = intersection(lon_stat[stat_number[1]],lat_stat[stat_number[1]],azimuth[1][i],lon_stat[stat_number[2]],lat_stat[stat_number[2]],azimuth[2][i+offset])
		print('dummy',i,lon_stat[stat_number[1]],lat_stat[stat_number[1]],azimuth[1][i],lon_stat[stat_number[2]],lat_stat[stat_number[2]],azimuth[2][i+offset])
		if start:
			lat_start = dummy.lat
			lon_start = dummy.lon
			start = False

		A = dummy.dist*pi/20000.
		B = pi/2.+obs[1][i].els*pi/180
		C = pi - A - B
		c = R_Earth
		a = sin(A)*c/sin(C)
		b = sqrt(a**2+c**2-2*a*c*cos(B))
		#distance = intersection(lon_stat[89],lat_stat[89],azimuth[1][i],lon_stat[62],lat_stat[62],azimuth[2][i+offset])
		height_flat = dummy.dist*math.tan(obs[1][i].els*pi/180)
		angle = dummy.dist*pi/20000.
		angle_intersect = pi- angle - (pi/2.+obs[1][i].els*pi/180) #angle_intersect is the angle between line of sight at the intersect and vertical
		#height_round = R_Earth* sin(pi/2.+obs[1][i].els*pi/180.)/sin(pi - angle_intersect - angle) - R_Earth
		height_round_old = height_round
		height_round = b-R_Earth
		if first_obs:   #Save first observed height
			height_start = height_round
			first_obs = False
			time_start = 3600*int(obs[1][i].hour)+60*int(obs[1][i].minute)+int(obs[1][i].second)+int(obs[1][i].dec_second)/1000


		#height_round = sqrt(R_Earth**2+(distance)**2)-R_Earth + height_flat
		#intersect = intersection(lat_start,lon_start,dummy.lat,dummy.lon)
		dist_ground_old = dist_ground
		dist_track_old = dist_track
		dist_ground = dist_2(lat_start,lon_start,dummy.lat,dummy.lon)
		dist_track	= sqrt((dist_ground)**2+(height_round-height_start)**2)
		velocity = (dist_track-dist_track_old)/delta_time
		dist = dummy.dist

		# Save the last observation where the height is above 50 km
		if height_round > 50.0:
			lat_50 = dummy.lat
			lon_50 = dummy.lon
			height_50 = height_round
			distance_50 = dist_track
			time_50 = 3600*int(obs[1][i].hour)+60*int(obs[1][i].minute)+int(obs[1][i].second)+int(obs[1][i].dec_second)/1000

	else:
		height_flat = 0.0
		angle = 0.0
		height_round = 0.0
		dist_track = 0.0
		dist_ground = 0.0
		velocity = 0.0
		dist = 0.0

	excel_line = (
			[stat_number[1]],
			[obs[1][i].date],
			[obs[1][i].hour],
			[obs[1][i].minute],
			[obs[1][i].second],
			[obs[1][i].dec_second],
			[obs[1][i].azs],
			[obs[1][i].els],
			[dist],
			[height_flat],
			[height_round],
			[dist_ground],
			[dist_track],
			[velocity],
			[stat_number[2]],
			[obs[2][i+offset].date],
			[obs[2][i+offset].hour],
			[obs[2][i+offset].minute],
			[obs[2][i+offset].second],
			[obs[2][i+offset].dec_second],
			[obs[2][i+offset].azs],
			[obs[2][i+offset].els]
			)
	col = 1
	for item, in (excel_line):
			worksheet.write(row,col,item)
			col += 1
	row += 1



workbook.close()
strings = ['AMS',str(stat_number[1]), '_AMS',str(stat_number[2]),'_',observation.date,'-',observation.hour,'-',observation.minute,'.kml']

outfilename = ''.join(strings)
#.join('AMS',str(stat_numner[1]))
print(outfilename)
kml.save(outfilename)  # Close kml file

import pyproj
geodesic = pyproj.Geod(ellps='WGS84')
fwd_azimuth,back_azimuth,distance = geodesic.inv(lon_start, lat_start, lon_50, lat_50)
if fwd_azimuth < 0.0:
	fwd_azimuth = fwd_azimuth+360

print("Program ended")
print('Entry velocity =',distance_50/(time_50-time_start),'km/s')
print('bearing',fwd_azimuth)
print('slope',180/3.1415*atan2(height_start-height_50,distance_50))

