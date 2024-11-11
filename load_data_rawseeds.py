import numpy as np
from scipy.spatial import KDTree


def lerp(x, y, t):
	return x + (y - x) * t

def load(dataset_folder, dataset_name):
	# 
	# This will load lidar and odometry data from a RAWSEEDS dataset
	# We're matching the format of the data the code was originally written for.
	# This makes it easier to switch between datasets.
	#
	# It returns two values:
	# lidar: t[n_samples, 1, 1], pose[n_samples, 1, 2], rpy[n_samples, 1, 3], scan[n_samples,n_scan]
	# joints: ts[n_samples], rpy[3, n_samples], head_angles[2, n_samples]
	#
	# lidar is an array of dictionaries, joints is a dictionary of arrays
	#
	# Yes, for some reason the lidar data is one sample per row and the joints data is one sample per column
	# Sample count can differ between lidar/joints datasets. They are matched up by timestamps
	#
	# RPY: 
	# - lidar: Yaw is used for determining robot rotation, roll/pitch are not used 
	# - joints: Roll/pitch are used for coordinate system conversion, yaw is not used
	#

	# Limit number of rows for quicker testing
	#max_rows = 100000 
	max_rows = None


	datafile_prefix = f'{dataset_folder}\\{dataset_name}\\{dataset_name}'
	sick_front_file = f'{datafile_prefix}-SICK_FRONT.csv'
	odometry_file = f'{datafile_prefix}-ODOMETRY_XYT.csv'
	groundtruth_file = f'{datafile_prefix}-GROUNDTRUTH.csv'

	scan = np.genfromtxt(sick_front_file, delimiter=',', max_rows=max_rows)
	odometry = np.genfromtxt(odometry_file, delimiter=',')
	groundtruth = np.genfromtxt(groundtruth_file, delimiter=',')

	# Extract timestamps and positions for KDTree matching
	timestamps = groundtruth[:, 0]  # First column: Timestamps
	positions = groundtruth[:, 1:3]  # Columns 1 and 2: X and Y positions

	# Create KDTree for efficient timestamp matching
	timestamp_tree = KDTree(timestamps.reshape(-1, 1))

	# LIDAR data is leading in determining sample count
	sample_count = scan.shape[0]

	print(f"Scan: {scan.shape}")
	print(f"Odo: {odometry.shape}")
	print(f"Groundtruth: {groundtruth.shape}")

	# Setup data arrays based on desired output format
	# Initialize pose and RPY to zero since they will be filled from odometry data
	t = scan[:,0].reshape((sample_count, 1, 1))
	pose = np.zeros((sample_count, 1, 2))
	rpy = np.zeros((sample_count, 1, 3))
	scan = scan[:,3:]

	# Merge position and yaw data to lidar data
	odo_i = 0
	odo_n = odometry.shape[0]
	for i in range(sample_count):
		desired_ts = scan[i, 0]

		# Find between which odo samples this lidar sample falls
		while True:
			prev_ts = odometry[odo_i, 0]
			next_ts = odometry[odo_i + 1, 0]

			if desired_ts <= prev_ts:
				odo_f = 0
				break

			if desired_ts <= next_ts:
				odo_f = (desired_ts - prev_ts) / float(next_ts - prev_ts)
				break

			if odo_i == odo_n - 1:
				odo_f = 1
				break

			odo_i += 1
		
		# Lerp odomery data based on the found sample range
		x = lerp(odometry[odo_i, 4], odometry[odo_i + 1, 4], odo_f)
		y = lerp(odometry[odo_i, 5], odometry[odo_i + 1, 5], odo_f)
		heading = lerp(odometry[odo_i, 6], odometry[odo_i + 1, 6], odo_f)
	
		pose[i, 0, 0] = x
		pose[i, 0, 1] = y
		rpy[i, 0, 2] = heading

	# Lidar dataset
	lidar = \
	[
		{
			't': t[i,:,:],
			'pose': pose[i,:,:],
			'rpy': rpy[i,:,:],
			'scan': scan[i,:]
		} 
		for i in range(sample_count) 
	]
	

	# Joints dataset
	# NOTE: We leave all of these zero for now since we assume the robot drives on a perfectly flat floor
	# It also has no head movement
	joints = \
	{
		'ts': np.zeros(sample_count),
		'rpy': np.zeros((3, sample_count)),
		'head_angles': np.zeros((2, sample_count))
	}

	return joints, lidar, timestamp_tree, positions