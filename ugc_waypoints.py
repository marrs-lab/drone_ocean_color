### A quick script to out UGC waypoints when doing ship-based surveys
# Author: Patrick C Gray
# contact: patrick.c.gray@duke.edu

import math 
import argparse

parser = argparse.ArgumentParser(description='command line program for generating waypoints for UAS surveys')
parser.add_argument('-fn','--filename', help='filename that will be used to save the output', required=True)
parser.add_argument('-b','--bearing', type=float, help='ship bearing in degrees', required=True)
parser.add_argument('-us','--uas_speed', type=float, help='approx planned uas speed in meters/sec', required=True)
parser.add_argument('-ss','--ship_speed', type=float, help='approx planned ship speed in meters/sec', required=True)
parser.add_argument('-lat','--latitude', type=float, help='starting latitude', required=True)
parser.add_argument('-lon','--longitude', type=float, help='starting longitude', required=True)
parser.add_argument('-ft','--flight_time', type=float, help='approx planned flight time in minutes', required=True)
parser.add_argument('-alt','--altitude', type=float, help='approx planned flight altitude in meters', default=50)
args = vars(parser.parse_args())

# arguments:
filename = args['filename']
ship_bearing = args['bearing'] # in degrees
uas_speed = args['uas_speed'] # m/s
ship_speed = args['ship_speed'] # m/s
start_lat = args['latitude']
start_lon = args['longitude']
flight_time = args['flight_time'] # in min
altitude = args['altitude'] # in meters
flight_type = "zigzag" # default is "zigzag"


"""
UGC format is:

QGC WPL <VERSION>
<INDEX> <CURRENT WP> <COORD FRAME> <COMMAND> <PARAM1> <PARAM2> <PARAM3> <PARAM4> <PARAM5/X/LATITUDE> <PARAM6/Y/LONGITUDE> <PARAM7/Z/ALTITUDE> <AUTOCONTINUE>
0    	1    		 0    		   16    	 0.14999  0    	   0    	0    	 8.548    			 47.3759    		  550    			  1

General logic of this script is to take the boat speed divided by UAS speed and
that factor gives us how many times further the UAS can travel than the boat.

Based on that I take that distance and decide on the angle offset from the boat track.
I do cos^-1(1/uas_factor) to find the angle. Then simply find that point via the get_new_pt fcn.

From there I just iterate through a single instance of sweeping from one side of the track to the other.

Finally those coords are written out in the proper format.
"""
R = 6378.1 #Radius of the Earth

def get_new_pt(lat, lon, dist, bearing):
	lat1 = math.radians(lat) #Current lat point converted to radians
	lon1 = math.radians(lon) #Current long point converted to radians
	bearing = math.radians(bearing)

	lat2 = math.asin( math.sin(lat1)*math.cos(dist/R) + 
		math.cos(lat1)*math.sin(dist/R)*math.cos(bearing))

	lon2 = lon1 + math.atan2(math.sin(bearing)*math.sin(dist/R)*math.cos(lat1),
	             math.cos(dist/R)-math.sin(lat1)*math.sin(lat2))

	lat2 = math.degrees(lat2)
	lon2 = math.degrees(lon2)
	return(lat2, lon2)

dist_per_flight = flight_time*60*ship_speed / 1000.0
current_lat = start_lat
current_lon = start_lon
uas_dist = 0
uas_factor = uas_speed/ship_speed*1.0 # this is the distance the uas can travel in the time it take the ship to travel 1 km
uas_bearing_offset = math.degrees(math.acos(1/uas_factor))
print('uas bearing offset in deg', uas_bearing_offset)
leg_shortening_factor = 0.5
uas_leg_dist = uas_factor*leg_shortening_factor
ship_dist = 0

coords = []
coords.append([start_lat,start_lon])

legs_traveled = 0

while ship_dist < dist_per_flight:	
	lat2, lon2 = get_new_pt(current_lat, current_lon, uas_leg_dist, ship_bearing+uas_bearing_offset)
	coords.append([lat2,lon2])

	lat3, lon3 = get_new_pt(lat2,lon2, uas_leg_dist, ship_bearing-uas_bearing_offset)
	coords.append([lat3,lon3])

	lat4, lon4 = get_new_pt(lat3,lon3, uas_leg_dist, ship_bearing-uas_bearing_offset)
	coords.append([lat4,lon4])

	lat5, lon5 = get_new_pt(lat4, lon4, uas_leg_dist, ship_bearing+uas_bearing_offset)
	coords.append([lat5,lon5])

	current_lat = lat5
	current_lon = lon5

	ship_dist += 1*leg_shortening_factor*4
	legs_traveled += 4

print('ship distance covered', ship_dist)
print('uas distance covered', legs_traveled*uas_leg_dist)


with open(filename, 'w') as the_file:
	the_file.write('QGC WPL 110\n')
	idx = 0
	current_wp = 1
	for coord in coords:
	    the_file.write(str(idx)+'\t'+str(current_wp)+'\t0\t0\t0\t0\t0\t0\t'+str(coord[0])+'\t'+str(coord[1])+'\t'+str(altitude)+'\t1\n')
	    idx += 1
	    current_wp=0
#<INDEX> <CURRENT WP> <COORD FRAME> <COMMAND> <PARAM1> <PARAM2> <PARAM3> <PARAM4> <PARAM5/X/LATITUDE> <PARAM6/Y/LONGITUDE> <PARAM7/Z/ALTITUDE> <AUTOCONTINUE>
#0    	1    		 0    		   16    	 0.14999  0    	   0    	0    	 8.548    			 47.3759    		  550    			  1








