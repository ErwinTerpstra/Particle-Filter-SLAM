import load_data_original as ld_original
import load_data_rawseeds as ld_rawseeds

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from SLAM_helper import *

import cv2
import random
import time

#### Dataset ####

dataset = 'bicocca'
#dataset = 'original'

if dataset == 'original':
	joint = ld_original.get_joint("data/Original/train_joint2")
	lid = ld_original.get_lidar("data/Original/train_lidar2")

	config = \
	{
		'scan_min': 0.1,
		'scan_max': 30,
	}

	# Angle for each sample in LIDAR sweep
	angles = np.array([np.arange(-135, 135.25, 0.25) * np.pi / 180.0])
elif dataset == 'bicocca':
	joint, lid = ld_rawseeds.load('data', 'Bicocca_2009-02-25b')

	config = \
	{
		'scan_min': 0.0,
		'scan_max': 500,
	}
	
	# Angle for each sample in LIDAR sweep
	# SICK frontal sensor has 181 samples in the full frontal 180 degree range
	angles = np.array([np.linspace(-90, 90, 181) * np.pi / 180.0])

#### Settings ###

# Number of particles and threshold for resampling
N, N_threshold = 100, 35

# Number of samples that is skipped each iterations
samples_per_iteration = 10

# Number of iterations between rendering the map
# Only used in when render_animated is True (below)
iterations_per_frame = 10

# Number of samples to limit the simulation to (can be None for full dataset)
sample_limit = None

# Factor that is used to add noise to each particle
# Current setting means 10x as much noise on heading as on x/y positions
factor = np.array([1, 1, 10])

# Std. dev of noise that is added
noise_sigma = 1e-3

# These offsets are used to evaluate map correlation at various offsets of the particle's actual position
# Current settings considers a 3x3 grid for each particle
x_range = np.array([ -0.05, 0.00, 0.05 ])
y_range = np.array([ -0.05, 0.00, 0.05 ])

# Whether to render an animated preview while calculating
# This makes the total calculation quite a bit slower
render_animated = True

# Whether to run the QT event loop each SLAM iteration
# This also slows down calculations, but keeps the window more responsive
run_event_loop = True

### Data setup ##

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
# show_map = RGB map data
mapfig['log_map'] = np.zeros((mapfig['sizex'], mapfig['sizey']))
mapfig['map'] = np.zeros((mapfig['sizex'], mapfig['sizey']), dtype = np.int8)
mapfig['show_map'] = np.zeros((mapfig['sizex'], mapfig['sizey'], 3), dtype = np.uint8)
mapfig['show_map'][:,:,:] = 128

pos_phy, posX_map, posY_map = {}, {}, {}

# Lookup table to convert map coordinate to physical positions
x_im = np.arange(mapfig['xmin'], mapfig['xmax'] + mapfig['res'], mapfig['res'])  # x-positions of each pixel of the map
y_im = np.arange(mapfig['ymin'], mapfig['ymax'] + mapfig['res'], mapfig['res'])  # y-positions of each pixel of the map

# Get joint datasets
# ts = timestamp?
# h_angle = head angles (two joints?)
# rpy_robot = roll, pitch, yaw?
ts = joint['ts']
h_angle = joint['head_angles']
rpy_robot = joint['rpy']

# Draw initial map for first dataset sample
lid_p = lid[0]
rpy_p = lid_p['rpy']

ind_0 = np.argmin(np.absolute(ts - lid_p['t'][0][0])) # Index of closest timestamp match between datasets
pos_phy, posX_map, posY_map = mapConvert(lid_p['scan'], rpy_robot[:, ind_0], h_angle[:, ind_0], angles, particles, N, pos_phy, posX_map, posY_map, mapfig, config)
mapfig = drawMap(particles[0, :], posX_map[0], posY_map[0], mapfig)

pose_p, yaw_p = lid_p['pose'], rpy_p[0, 2]

# Time keeping
start_time = time.perf_counter_ns()
last_print = 0
correlation_time = 0
resample_time = 0
map_draw_time = 0
map_convert_time = 0
update_particles_time = 0
update_weights_time = 0

# Loop over all samples in dataset
timeline = min(sample_limit, len(lid)) if sample_limit else len(lid)
sample = 1

# Function that calls the simulation in animate preview
def animate(frame):
	# Perform a number of iterations before we draw the frame
	next_frame = sample + iterations_per_frame * samples_per_iteration
	while sample < next_frame and sample < timeline:
		slam_iteration()
			
		# This makes rendering a bit slower, but keeps the GUI responsive
		if run_event_loop:
			plt.pause(0.001)

	# Update the image drawer
	im.set_data(mapfig['show_map'])
	return im

# Perform a single SLAM iteration. Moves the sample counter forward by a number of samples
def slam_iteration():
	global particles, weight
	global ts, h_angle, rpy_robot
	global lid_p, rpy_p, ind_0
	global pos_phy, posX_map, posY_map
	global mapfig
	global pose_p, yaw_p
	global sample
	global last_print
	global correlation_time, resample_time
	global map_draw_time, map_convert_time
	global update_particles_time, update_weights_time

	# Get current lidar data and pose measurement
	lid_c = lid[sample]
	pose_c, rpy_c = lid_c['pose'], lid_c['rpy']
	scan_c = lid_c['scan']
	yaw_c = rpy_c[0, 2]

	yaw_est = particles[:, 2]

	# This does some sort of reference frame conversion between previous pose and the particles
	update_particles_start = time.perf_counter_ns()
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
	noise = np.random.normal(0, noise_sigma, (N, 1)) * factor
	particles = particles + ut + noise
	update_particles_time += time.perf_counter_ns() - update_particles_start

	map_convert_start = time.perf_counter_ns()
	ind_i = np.argmin(np.absolute(ts - lid_c['t'][0][0])) # Index of closest timestamp match between datasets
	pos_phy, posX_map, posY_map = mapConvert(scan_c, rpy_robot[:, ind_i], h_angle[:, ind_i], angles, particles, N, pos_phy, posX_map, posY_map, mapfig, config)
	map_convert_time += time.perf_counter_ns() - map_convert_start

	# For each particle, calculate the correlation with the current map
	corr_start = time.perf_counter_ns()
	corr = np.zeros((N, 1))
	for i in range(N):
		# Add an extra row to the occopied positions list for this particle
		# This is necessary since the mapCorrelation function expects a 3xN matrix (even though the 3rd row is not used)
		size = pos_phy[i].shape[1]
		Y = np.concatenate([pos_phy[i], np.zeros((1, size))], axis = 0)

		# Calculate correlation for each of the combinations in x_range/y_range
		corr_cur = mapCorrelation(mapfig['map'], x_im, y_im, Y[0 : 3, :], x_range, y_range)

		# Determine which of the offsets performed best
		ind = np.argmax(corr_cur)

		# Store the correlation for that offset
		corr[i] = corr_cur[ind // 3, ind % 3]

		# Update particle position according to chosen offset
		particles[i, 0] += x_range[ind // 3]
		particles[i, 1] += y_range[ind % 3]

	# Keep track of how much time we spend on particle correlation
	correlation_time += time.perf_counter_ns() - corr_start

	# Update weight of each particle according to correlation
	update_weights_start = time.perf_counter_ns()
	weight = updateWeights(weight, corr)
	update_weights_time += time.perf_counter_ns() - update_weights_start

	# Determine index of best particle
	map_draw_start = time.perf_counter_ns()
	ind_best = weight.argmax()

	# Convert location of best particle to map coordinates
	x_r = (np.ceil((particles[ind_best, 0] - mapfig['xmin']) / mapfig['res']).astype(np.int16) - 1)
	y_r = (np.ceil((particles[ind_best, 1] - mapfig['xmin']) / mapfig['res']).astype(np.int16) - 1)

	# Mark location with a red pixel (index 0 in RGB)
	mapfig['show_map'][x_r, y_r,:] = [ 255, 0, 0]

	# Draw map 
	# TODO: Why is best particle passed here? What is the difference between particles array and posX_map/posY_Map
	mapfig = drawMap(particles[ind_best, :], posX_map[ind_best], posY_map[ind_best], mapfig)
	map_draw_time += time.perf_counter_ns() - map_draw_start

	# Keep track of previous pose and yaw
	pose_p, yaw_p = pose_c, yaw_c

	# Resample particles if sum of squared weights is lower than threshold
	# Since weights always sum to 1, this seems some sort of measure of whether the 
	# weight distribution is broad or narrow. A higher value would mean more of the weight 
	# is "concentrated" in fewer particles, thus we have more effective particles
	resample_start = time.perf_counter_ns()
	N_eff = 1 / np.sum(np.square(weight))
	if N_eff < N_threshold:
		#print("Resampling ({0:.2f}/{1})".format(N_eff, N_threshold))

		# Prepare a new list of particles
		particle_New = np.zeros((N, 3))

		# This resamples particles from the distribution based on their weights
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

	resample_time += time.perf_counter_ns() - resample_start

	sample += samples_per_iteration

	# Print timing stats
	if time.time() - last_print >= 0.5:
		elapsed_time = time.perf_counter_ns() - start_time
		elapsed_iterations = sample / float(samples_per_iteration)
		nfactor = 1 / (1000000 * float(elapsed_iterations))

		print("{0}/{1}".format(sample, timeline))
		print("Total time: {:.1f}s; Avg. per frame: {:.1f}ms".format(elapsed_time / 1000000000.0, elapsed_time * nfactor))
		print("Correlation: {:.1f}ms; Resample: {:.1f}ms".format(correlation_time * nfactor, resample_time * nfactor))
		print("Map draw: {:.1f}ms; Map convert: {:.1f}ms;".format(map_draw_time * nfactor, map_convert_time * nfactor))
		print("Update particles: {:.1f}ms; Update weights: {:.1f}ms".format(update_particles_time * nfactor, update_weights_time * nfactor))
		print("")

		last_print = time.time()

# Setup output
fig = plt.figure(1)
ax = fig.add_subplot(1, 1, 1)
ax.set_title("SLAM Map")

# Setup the animation
if render_animated:
	frame_count = int(np.ceil((timeline - 1) / samples_per_iteration / iterations_per_frame))
	anim = animation.FuncAnimation(fig=fig, func=animate, frames=frame_count, interval=0)
else:
	while sample <  timeline:
		slam_iteration()

im = ax.imshow(mapfig['show_map'])

plt.show()
