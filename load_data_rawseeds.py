import numpy as np
from scipy.spatial import KDTree

def load(dataset_folder, dataset_name, use_rear_lidar):
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
	max_rows = None


	datafile_prefix = f'{dataset_folder}\\{dataset_name}\\{dataset_name}'
	sick_front_file = f'{datafile_prefix}-SICK_FRONT.csv'
	sick_rear_file = f'{datafile_prefix}-SICK_REAR.csv'
	odometry_file = f'{datafile_prefix}-ODOMETRY_XYT.csv'
	groundtruth_file = f'{datafile_prefix}-GROUNDTRUTH.csv'

	scan_front = np.genfromtxt(sick_front_file, delimiter=',', max_rows=max_rows)
	scan_rear = np.genfromtxt(sick_rear_file, delimiter=',', max_rows=max_rows)
	odometry = np.genfromtxt(odometry_file, delimiter=',', max_rows=max_rows)
	groundtruth = np.genfromtxt(groundtruth_file, delimiter=',', max_rows=max_rows)

	# Extract timestamps and positions for KDTree matching
	timestamps = groundtruth[:, 0]  # First column: Timestamps
	positions = groundtruth[:, 1:3]  # Columns 1 and 2: X and Y positions

	# Create KDTree for efficient timestamp matching
	ts_tree_groundtruth = KDTree(timestamps.reshape(-1, 1))
	ts_tree_scan_rear = KDTree(scan_rear[:, 0].reshape(-1, 1))
	ts_tree_odometry = KDTree(odometry[:, 0].reshape(-1, 1))

	# LIDAR data is leading in determining sample count
	sample_count = scan_front.shape[0]

	print(f"Scan front: {scan_front.shape}")
	print(f"Scan rear: {scan_rear.shape}")
	print(f"Odo: {odometry.shape}")
	print(f"Groundtruth: {groundtruth.shape}")

	# Setup data arrays based on desired output format
	# Rear scan data will be resampled and merged to front scan data
	# Initialize pose and RPY to zero since they will be filled from odometry data
	t = scan_front[:,0].reshape((sample_count, 1, 1))
	scan_front = scan_front[:,3:]
	scan_rear = scan_rear[:,3:]
	scan_rear_resampled = np.zeros((sample_count, scan_front.shape[1]))
	pose = np.zeros((sample_count, 1, 2))
	rpy = np.zeros((sample_count, 1, 3))
 
	# Merge position and yaw data to lidar data
	for i in range(sample_count):
		desired_ts = t[i,0,0]

		# Find which rear scan sample to use
		scan_rear_ts_delta, scan_rear_i = ts_tree_scan_rear.query(desired_ts)

		# Find which odometry sample to use
		odometry_ts_delta, odometry_i = ts_tree_odometry.query(desired_ts)
		
		# Place data in correct data lists
		pose[i, 0, 0] = odometry[odometry_i, 4]
		pose[i, 0, 1] = odometry[odometry_i, 5]
		rpy[i, 0, 2] = odometry[odometry_i, 6]
		scan_rear_resampled[i,:] = scan_rear[scan_rear_i,:]

	if use_rear_lidar:
		scan = np.concatenate([scan_front, scan_rear_resampled], axis=1)
	else:
		scan = scan_front

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

	return joints, lidar, ts_tree_groundtruth, positions