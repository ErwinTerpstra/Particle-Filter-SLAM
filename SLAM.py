import load_data as ld
import numpy as np
from SLAM_helper import *
import matplotlib.pyplot as plt
import MapUtils as maput
import cv2
import random

#### Dataset ####

joint = ld.get_joint("data/train_joint2")
lid = ld.get_lidar("data/train_lidar2")

#### Settings ###

N, N_threshold = 100, 35
step = 1
sample_limit = None

### Data setup ##

# TODO: Find out what this is used for?
angles = np.array([np.arange(-135, 135.25, 0.25) * np.pi / 180.0])

# Particle list. Each has X, Y and heading
particles = np.zeros((N, 3))

# Weight per particle, initialized to evenly distributed
weight = np.ones((N, 1)) * (1.0 / N)

# Map drawing parameters
mapfig = {}
mapfig['res'] = 0.05
mapfig['xmin'] = -40
mapfig['ymin'] = -40
mapfig['xmax'] = 40
mapfig['ymax'] = 40
mapfig['sizex'] = int(np.ceil((mapfig['xmax'] - mapfig['xmin']) / mapfig['res'] + 1))
mapfig['sizey'] = int(np.ceil((mapfig['ymax'] - mapfig['ymin']) / mapfig['res'] + 1))

# Actual map data
# log_map = log likeliness of each cell being occupied
# map = grayscale map data
# show_map = BGR map data
mapfig['log_map'] = np.zeros((mapfig['sizex'], mapfig['sizey']))
mapfig['map'] = np.zeros((mapfig['sizex'], mapfig['sizey']), dtype = np.int8)
mapfig['show_map'] = 0.5 * np.ones((mapfig['sizex'], mapfig['sizey'], 3), dtype = np.int8)

pos_phy, posX_map, posY_map = {}, {}, {}

# Factor that is used to add noise to each particle
# Current setting means 10x as much noise on heading as on x/y positions
factor = np.array([1, 1, 10])

# TODO: What is this used for?
x_im = np.arange(mapfig['xmin'], mapfig['xmax'] + mapfig['res'], mapfig['res'])  # x-positions of each pixel of the map
y_im = np.arange(mapfig['ymin'], mapfig['ymax'] + mapfig['res'], mapfig['res'])  # y-positions of each pixel of the map

# TODO: What is this used for?
x_range = np.arange(-0.05, 0.06, 0.05)
y_range = np.arange(-0.05, 0.06, 0.05)

# Get joint datasets
# TODO: What does ts, h_angle and rpy_robot mean?
ts = joint['ts']
h_angle = joint['head_angles']
rpy_robot = joint['rpy']

# Draw initial map for first dataset sample
lid_p = lid[0]
rpy_p = lid_p['rpy']
ind_0 = np.argmin(np.absolute(ts - lid_p['t'][0][0]))
pos_phy, posX_map, posY_map = mapConvert(lid_p['scan'], rpy_robot[:, ind_0], h_angle[:, ind_0], angles, particles, N, pos_phy, posX_map, posY_map, mapfig)
mapfig = drawMap(particles[0, :], posX_map[0], posY_map[0], mapfig)

pose_p, yaw_p = lid_p['pose'], rpy_p[0, 2]

# Loop over all samples in dataset
timeline = min(sample_limit, len(lid)) if sample_limit else len(lid)
for sample in range(1, timeline, step):
	print("{0}/{1}".format(sample, timeline))

	# Get current lidar data and pose measurement
	lid_c = lid[sample]
	pose_c, rpy_c = lid_c['pose'], lid_c['rpy']
	yaw_c = rpy_c[0, 2]

	yaw_est = particles[:, 2]

	# This does some sort of reference frame conversion between previous pose and the particles
	delta_x_gb = pose_c[0][0] - pose_p[0][0]
	delta_y_gb = pose_c[0][1] - pose_p[0][1]
	delta_theta_gb = yaw_c - yaw_p

	delta_x_lc = (np.cos(yaw_p) * delta_x_gb) + (np.sin(yaw_p) * delta_y_gb)
	delta_y_lc = (-np.sin(yaw_p) * delta_x_gb) + (np.cos(yaw_p) * delta_y_gb)
	delta_theta_lc = delta_theta_gb

	delta_x_gb_new = ((np.cos(yaw_est) * delta_x_lc) - (np.sin(yaw_est) * delta_y_lc)).reshape(-1, N)
	delta_y_gb_new = ((np.sin(yaw_est) * delta_x_lc) + (np.cos(yaw_est) * delta_y_lc)).reshape(-1, N)
	delta_theta_gb_new = np.tile(delta_theta_lc, (1, N))

	# I suspect this gives us the position and heading delta for each particle based on robot measurements
	ut = np.concatenate([np.concatenate([delta_x_gb_new, delta_y_gb_new], axis=0), delta_theta_gb_new], axis=0)
	ut = ut.T

	# Add the robot movement to each particle position and heading
	# Also add a little bit of noise 
	noise = np.random.normal(0, 1e-3, (N, 1)) * factor
	particles = particles + ut + noise

	scan_c = lid_c['scan']
	ind_i = np.argmin(np.absolute(ts - lid_c['t'][0][0]))
	pos_phy, posX_map, posY_map = mapConvert(scan_c, rpy_robot[:, ind_i], h_angle[:, ind_i], angles, particles, N, pos_phy, posX_map, posY_map, mapfig)

	# For each particle, calculate the correlation with the current map
	corr = np.zeros((N, 1))
	for i in range(N):
		size = pos_phy[i].shape[1]
		Y = np.concatenate([pos_phy[i], np.zeros((1, size))], axis = 0)
		corr_cur = maput.mapCorrelation(mapfig['map'], x_im, y_im, Y[0 : 3, :], x_range, y_range)
		ind = np.argmax(corr_cur)

		corr[i] = corr_cur[ind / 3, ind % 3]
		particles[i, 0] += x_range[ind / 3]
		particles[i, 1] += y_range[ind % 3]

	# Update weight of each particle according to correlation
	weight = updateWeights(weight, corr)

	# Determine index of best particle
	ind_best = weight.argmax()

	# Convert location of best particle to map coordinates
	x_r = (np.ceil((particles[ind_best, 0] - mapfig['xmin']) / mapfig['res']).astype(np.int16) - 1)
	y_r = (np.ceil((particles[ind_best, 1] - mapfig['xmin']) / mapfig['res']).astype(np.int16) - 1)

	# Mark location with a blue pixel (index 0 in BGR)
	mapfig['show_map'][x_r, y_r, 0] = 255

	# Draw map 
	# TODO: Why is best particle passed here? What is the difference between particles array and posX_map/posY_Map
	mapfig = drawMap(particles[ind_best, :], posX_map[ind_best], posY_map[ind_best], mapfig)

	# Keep track of previous pose and yaw
	pose_p, yaw_p = pose_c, yaw_c

	# Resample particles if sum of squared weights is lower than threshold
	# Since weights always sum to 1, this seems some sort of measure of whether the 
	# weight distribution is broad or narrow
	N_eff = 1 / np.sum(np.square(weight))
	if N_eff < N_threshold:
		print("Resampling ({0:.2f}/{1})".format(N_eff, N_threshold))

		# Prepare a new list of particles
		particle_New = np.zeros((N, 3))

		# TODO: Figure out how this resamples particles?
		# It seems to mostly re-order particles?
		r = random.uniform(0, 1.0 / N)

		c, i = weight[0], 0
		for m in range(N):
			u = r + m * (1.0 / N)
			
			while u > c:
				i = i + 1
				c = c + weight[i]

			# NOTE: this line was unindented, so it only ran at the end of the loop
			# That seems like a bug to me, otherwise particles_New would mostly be empty
			particle_New[m, :] = particles[i, :]

		particles = particle_New
		
		# Reset the weight array to be uniform again
		# (Weights for new particles are calculated next iteration)
		weight = np.ones((N, 1)) * (1.0 / N)

	#im.set_data(mapfig['show_map'])

	#fig.canvas.draw()
	#fig.canvas.flush_events()


fig = plt.figure(1)
ax = fig.add_subplot(1, 1, 1)
ax.set_title("SLAM Map")

im = ax.imshow(mapfig['show_map'], cmap = "hot")
plt.show()

##plt.imshow(mapfig['show_map'], cmap = "hot")
##plt.show()
