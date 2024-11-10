import numpy as np

def load(dataset_folder, dataset_name):
	# 
	# This will load lidar and odometry data from a RAWSEEDS dataset
	# We're matching the format of the data the code was originally written for.
	# This makes it easier to switch between datasets.
	#
	# lidar: t[n_samples, 1, 1], pose[n_samples, 1, 2], rpy[n_samples, 3], scan[n_samples,n_scan]
	# joints: ts[n_samples], rpy[3, n_samples], head_angles[2, n_samples]
	#
	# (Yes, for some reason the lidar data is one sample per row and the joints data is one sample per column)
	#
	# RPY: 
	# - Roll/pitch are used from "joint" for coordinate system conversion, yaw is not used
	# - Yaw is used from "lid" for determining robot rotation, roll/pitch are not used 
	#

	datafile_prefix = f'{dataset_folder}/{dataset_name}/{dataset_name}'
	sick_front_file = f'{datafile_prefix}_SICK_FRONT.csv'
	odometry_file = f'{datafile_prefix}_ODOMETRY_XYT.csv'

	scan = np.genfromtxt(sick_front_file, delimiter=',')
	odometry = np.genfromtxt(odometry_file, delimiter=',')

	# TODO: Swizzle data to above mentioned dimensions

	return [ ]