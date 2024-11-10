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
	# Yes, for some reason the lidar data is one sample per row and the joints data is one sample per column
	# Sample count can differ between lidar/joints datasets. They are matched up by timestamps
	#
	# RPY: 
	# - lidar: Yaw is used for determining robot rotation, roll/pitch are not used 
	# - joints: Roll/pitch are used for coordinate system conversion, yaw is not used
	#

	datafile_prefix = f'{dataset_folder}/{dataset_name}/{dataset_name}'
	sick_front_file = f'{datafile_prefix}-SICK_FRONT.csv'
	odometry_file = f'{datafile_prefix}-ODOMETRY_XYT.csv'

	# Limit number of rows for quicker testing
	max_rows = 10000 

	scan = np.genfromtxt(sick_front_file, delimiter=',', max_rows=max_rows)
	odometry = np.genfromtxt(odometry_file, delimiter=',', max_rows=max_rows)

	# LIDAR data is leading in determining sample count
	sample_count = scan.shape[0]
	scan_count = scan.shape[1]

	print(f"Scan: {scan.shape}")
	print(f"Odo: {odometry.shape}")

	# Lidar dataset
	lidar = \
	{
		't': np.zeros((sample_count, 1, 1)),
		'pose': np.zeros((sample_count, 1, 2)),
		'rpy': np.zeros((sample_count, 3)),
		'scan': scan[:,3:]
	}
	
	# TODO: Merge position and yaw data to lidar data

	# Joints dataset
	# NOTE: We leave all of these zero for now since we assume the robot drives on a perfectly flat floor
	# It also has no head movement
	joints = \
	{
		'ts': np.zeros(sample_count),
		'rpy': np.zeros((3, sample_count)),
		'head_angles': np.zeros((2, sample_count))
	}

	return joints, lidar