import load_data_original as ld_original
import load_data_rawseeds as ld_rawseeds

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from SLAM_helper import *
from scipy.spatial import KDTree
from pathlib import Path

import cv2
import random
import time
import sys
import json


# ---------------------------------------------------------------------------------------------------------------------#
## Hyper parameters
# ---------------------------------------------------------------------------------------------------------------------#
# Dataset configuration: Choose between 'original' or 'bicocca'
dataset = 'bicocca'  # Options: 'original', 'bicocca'
use_rear_lidar = False # Whether to load rear LIDAR data in addition to front (only used in bicocca dataset)

# Rendering and event loop configuration
render_animated = True  # Whether to render an animated preview while calculating. Disable to speed up calculation
render_particles = False # Whether to render all particle positions on the map as well (Only used in render_animated)
render_groundtruth = True # Whether to render a line for ground truth
samples_per_iteration = 1  # Number of samples to skip each iteration (for testing)
iterations_per_frame = 50  # Number of iterations between rendering map frames

# Particle filter configuration
N, N_threshold = 10, 3  # Number of particles and resampling threshold
disable_particle_filtering = False # Set to true to disable particle filtering. In this case, localization is purely from odometry data

# Map and particle noise configuration
noise_sigma = 1e-3 # Noise standard deviation for particles
factor = np.array([1, 1, 10])  # Noise factor for heading (yaw) and position (x, y)

# These offsets are used to evaluate map correlation at various offsets of the particle's actual position
# Current settings considers a 3x3 grid for each particle
# Set both offset and resolution to zero to disable
local_search_offset = 0.1	# The step size for local search. Should in practice usually match the map resolution
local_search_resolution = 1 # The amount of grid cells to look for in each direction. A value of "1" results in a  
							# 3x3 grid. A value of "2" in a 5x5 grid, "3" results in a 7x7 grid, etc.

# Override settings if particle filtering should be disabled
if disable_particle_filtering:
	N = 1
	noise_sigma = 0
	local_search_offset = 0
	local_search_resolution = 0

# ---------------------------------------------------------------------------------------------------------------------#
## Experiment loading
# ---------------------------------------------------------------------------------------------------------------------#

if len(sys.argv) > 1:
	experiment_file = Path(sys.argv[1])
	experiment_repetition = int(sys.argv[2]) if len(sys.argv) > 2 else 0

	print(f'Loading settings for experiment {experiment_file}...')

	# Stem twice to remove extension and input signifier
	experiment_input = Path(experiment_file.stem)
	experiment_output = str(experiment_file.parent.joinpath(experiment_input.stem)) 
	experiment_output += f'.{experiment_repetition}'

	stats_file = Path(experiment_output + ".stats.json")

	if stats_file.is_file():
		print('Skipping experiment because output already exists')
		sys.exit()

	with open(experiment_file) as f:
		experiment = json.load(f)

	# Load settings from data file
	dataset = experiment.get('dataset', dataset)
	
	N = experiment.get('particle_count', N)
	N_threshold = N # Resample every iteration

	noise_sigma = experiment.get('noise_sigma', noise_sigma)
	local_search_offset = experiment.get('local_search_offset', local_search_offset)
	local_search_resolution = experiment.get('local_search_resolution', local_search_resolution)
	use_rear_lidar = experiment.get('use_rear_lidar', use_rear_lidar)

	# Disable animation for experiments
	render_animated = False

	# Print settings
	print(f'Writing experiment output to {experiment_output}.*')
	print(f'Particle count: {N}')
	print(f'Noise sigma: {noise_sigma}')
	print(f'Local search offset: {local_search_offset}')
	print(f'Local search resolution: {local_search_resolution}')
	print(f'Use rear lidar: {use_rear_lidar}')
	print('')
else:
	experiment_output = None
	experiment_repetition = 0

# ---------------------------------------------------------------------------------------------------------------------#
## Script
# ---------------------------------------------------------------------------------------------------------------------#
random.seed(experiment_repetition)
np.random.seed(experiment_repetition)

mapfig = {}

error_values = [ ]
rmse_values = []

print(f'Loading dataset "{dataset}"...')
if dataset == 'original':
	joint = ld_original.get_joint("data/Original/train_joint2")
	lid = ld_original.get_lidar("data/Original/train_lidar2")

	config = {'scan_min': 0.1,'scan_max': 30}
	start_sample = 0
	sample_limit = None

	mapfig['res'] = 0.05
	mapfig['xmin'] = -40
	mapfig['ymin'] = -40
	mapfig['xmax'] = 40
	mapfig['ymax'] = 40

	# Angle for each sample in LIDAR sweep
	angles = np.array([np.arange(-135, 135.25, 0.25) * np.pi / 180.0])
elif dataset == 'bicocca':
	joint, lid, timestamp_tree, groundtruth = ld_rawseeds.load('data', 'Bicocca_2009-02-25b', use_rear_lidar)

	config = {'scan_min': 0.1,'scan_max': 80}
	start_sample = 99580 # Gives about ~4700 samples of ground truth data
	sample_limit = 104640
	
	mapfig['res'] = 0.1
	mapfig['xmin'] = -50
	mapfig['ymin'] = -60
	mapfig['xmax'] = 50
	mapfig['ymax'] = 40
	
	# Angle for each sample in LIDAR sweep
	# SICK frontal sensor has 181 samples in the full frontal 180 degree range
	# SICK rear sensor has 181 samples in the full rear 180 degree range
	if use_rear_lidar:
		angles_front = np.linspace(-90, 90, 181) * np.pi / 180.0
		angles_rear = np.linspace(90, 270, 181) * np.pi / 180.0
		angles = np.concatenate([angles_front, angles_rear])
	else:
		angles = np.linspace(-90, 90, 181) * np.pi / 180.0

### Data setup ##

# Particle list. Each has X, Y and heading
particles = np.zeros((N, 3))

# Weight per particle, initialized to evenly distributed
weight = np.ones((N, 1)) * (1.0 / N)

# Calculate positions for the local search grid
local_search_range = local_search_resolution * local_search_offset
local_search_grid_size = 2 * local_search_resolution + 1
x_range = np.linspace(-local_search_range, local_search_range, local_search_grid_size)
y_range = np.linspace(-local_search_range, local_search_range, local_search_grid_size)

# Map drawing parameters
mapfig['sizex'] = int(np.ceil((mapfig['xmax'] - mapfig['xmin']) / mapfig['res'] + 1)) # sizex = x-size of the map
mapfig['sizey'] = int(np.ceil((mapfig['ymax'] - mapfig['ymin']) / mapfig['res'] + 1)) # sizey = y-size of the map

# Actual map data
# log_map = log likeliness of each cell being occupied
# map = grayscale map data
# show_map = RGB map data
mapfig['log_map'] = np.zeros((mapfig['sizex'], mapfig['sizey'])) # np-array with cell for each pixel and size equal to map drawing
mapfig['map'] = np.zeros((mapfig['sizex'], mapfig['sizey']), dtype = np.int8) # np-array with grey value cell for each pixel and size equal to map drawing
mapfig['show_map'] = np.zeros((mapfig['sizex'], mapfig['sizey'], 3), dtype = np.uint8) # np-array with color value for cell for each pixel and size equal to map drawing
mapfig['show_map'][:,:,:] = 128 # all colors set to grey

pos_phy, posX_map, posY_map = {}, {}, {}

# Get joint datasets
# ts = timestamp?
# h_angle = head angles (two joints?)
# rpy_robot = roll, pitch, yaw?
ts = joint['ts']
h_angle = joint['head_angles']
rpy_robot = joint['rpy']

# Draw initial map for first dataset sample
lid_p = lid[start_sample]
rpy_p = lid_p['rpy']

ts_start = lid_p['t'][0][0]
ind_0 = np.argmin(np.absolute(ts - ts_start)) # Index of closest timestamp match between datasets
pos_phy, posX_map, posY_map = mapConvert(lid_p['scan'], rpy_robot[:, ind_0], h_angle[:, ind_0], angles, particles, N, pos_phy, posX_map, posY_map, mapfig, config)
mapfig = drawMap(particles[0, :], posX_map[0], posY_map[0], mapfig)

pose_p, yaw_p = lid_p['pose'], rpy_p[0, 2]

# Get ground truth for first sample
distance, index = timestamp_tree.query(ts_start)
ground_truth_offset = groundtruth[index, 0:2]
ground_truth_heading = groundtruth[index, 2]
ground_truth_rotation = -np.array([[np.cos(ground_truth_heading), -np.sin(ground_truth_heading)], [np.sin(ground_truth_heading), np.cos(ground_truth_heading)]])

# Time keeping
start_time = time.time()
last_print = 0
correlation_time = 0
resample_time = 0
map_draw_time = 0
map_convert_time = 0
update_particles_time = 0
update_weights_time = 0

# Loop over all samples in dataset
timeline = min(sample_limit, len(lid)) if sample_limit else len(lid)
sample = start_sample + 1

# Function that calls the simulation in animate preview
def animate(frame):
	# Perform a number of iterations before we draw the frame
	iterations_this_frame = 0
	while iterations_this_frame < iterations_per_frame and sample < timeline:
		slam_iteration()

		iterations_this_frame += 1

	# Update the image drawer
	updateDisplayMap(mapfig)

	if render_particles:
		drawParticles(mapfig, scatter, particles)

	update_plots()

	return [ im, rmse_plot ]

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
	global error_values, rmse_values
	global timestamp_tree
	
	# Get current lidar data and pose measurement
	lid_c = lid[sample]
	pose_c, rpy_c = lid_c['pose'], lid_c['rpy']
	scan_c = lid_c['scan']
	yaw_c = rpy_c[0, 2]

	yaw_est = particles[:, 2]

	estimated_timestamp = lid_c['t'][0][0]

    # Find the closest ground truth position using KDTree
	distance, index = timestamp_tree.query(estimated_timestamp)
	true_position = groundtruth[index, 0:2] - ground_truth_offset  # Ground truth x, y coordinates
	true_heading = groundtruth[index, 2] - ground_truth_heading # Ground truth heading

	true_position = np.matmul(ground_truth_rotation, true_position)

	# This does some sort of reference frame conversion between previous pose and the particles
	update_particles_start = time.perf_counter_ns() # Start timer
	delta_x_gb = pose_c[0][0] - pose_p[0][0]
	delta_y_gb = pose_c[0][1] - pose_p[0][1]
	delta_theta_gb = yaw_c - yaw_p
	
	while delta_theta_gb > np.pi:
		delta_theta_gb -= np.pi * 2
	
	while delta_theta_gb < -np.pi:
		delta_theta_gb += np.pi * 2

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
	noise = np.random.normal(0, noise_sigma, (N, 1)) * factor # Calculate noise
	particles = particles + ut + noise # New position + noise 
	update_particles_time += time.perf_counter_ns() - update_particles_start # particle runtime to particle time

	map_convert_start = time.perf_counter_ns() # start timer
	ind_i = np.argmin(np.absolute(ts - lid_c['t'][0][0])) # Index of closest timestamp match between datasets
	pos_phy, posX_map, posY_map = mapConvert(scan_c, rpy_robot[:, ind_i], h_angle[:, ind_i], angles, particles, N, pos_phy, posX_map, posY_map, mapfig, config)
	map_convert_time += time.perf_counter_ns() - map_convert_start # map convert runtime to map convert time

	# For each particle, calculate the correlation with the current map
	corr_start = time.perf_counter_ns() #start timer
	corr = np.zeros((N, 1))
	for i in range(N):
		# Calculate correlation for each of the combinations in x_range/y_range
		corr_cur = mapCorrelation(mapfig, pos_phy[i], x_range, y_range)

		# Determine which of the offsets performed best
		ind = np.argmax(corr_cur)

		# Store the correlation for that offset
		corr[i] = corr_cur[ind // local_search_grid_size, ind % local_search_grid_size]

		# Update particle position according to chosen offset
		particles[i, 0] += x_range[ind // local_search_grid_size]
		particles[i, 1] += y_range[ind % local_search_grid_size]

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
	y_r = (np.ceil((particles[ind_best, 1] - mapfig['ymin']) / mapfig['res']).astype(np.int16) - 1)

	# Extract SLAM estimate (timestamp, x, y)
	est_x, est_y = particles[ind_best, 0], particles[ind_best, 1]
	
    # Calculate RMSE for the current sample
	error = (true_position[0] - est_x) ** 2 + (true_position[1] - est_y) ** 2
	error_values.append(error)

	rmse = np.sqrt(np.mean(error_values))
	rmse_values.append(rmse)

	# Mark location with a red pixel (index 0 in RGB)
	mapfig['show_map'][x_r, y_r,:] = [ 255, 0, 0]
	
	if render_groundtruth:
		# Convert ground truth location to map coordinates
		x_t = (np.ceil((true_position[0] - mapfig['xmin']) / mapfig['res']).astype(np.int16) - 1)
		y_t = (np.ceil((true_position[1] - mapfig['ymin']) / mapfig['res']).astype(np.int16) - 1)

		# Mark location with a blue pixel (index 2 in RGB)
		mapfig['show_map'][x_t, y_t,:] = [ 0, 0, 255]

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
	if time.time() - last_print >= 1.0:
		elapsed_time = time.time() - start_time
		elapsed_iterations = (sample - start_sample) / float(samples_per_iteration)
		nfactor = 1 / (1000000 * float(elapsed_iterations))

		print("{0}/{1}".format(sample, timeline))
		print(f"Current average RMSE: {rmse:.4f} meters")
		print("Total time: {:.1f}s; Avg. per frame: {:.1f}ms".format(elapsed_time, elapsed_time * 1000 / float(elapsed_iterations)))
		print("Correlation: {:.1f}ms; Resample: {:.1f}ms".format(correlation_time * nfactor, resample_time * nfactor))
		print("Map draw: {:.1f}ms; Map convert: {:.1f}ms;".format(map_draw_time * nfactor, map_convert_time * nfactor))
		print("Update particles: {:.1f}ms; Update weights: {:.1f}ms".format(update_particles_time * nfactor, update_weights_time * nfactor))

		print("")

		last_print = time.time()

def update_plots():
	sample_count = len(rmse_values)

	im.set_data(mapfig['show_map'])
	rmse_plot.set_data(list(range(sample_count)), rmse_values)
	
	ax2.set_xlim(xmax=sample_count)
	ax2.set_ylim(ymax=max(rmse_values) * 1.1)

# Setup output
fig = plt.figure(1, figsize=(20, 10))

ax1 = fig.add_subplot(1, 2, 1)
ax1.set_title("SLAM Map")

ax2 = fig.add_subplot(1, 2, 2)
ax2.set_title("RMSE")
ax2.set_xlabel("Sample")
ax2.set_ylabel("RMSE (m)")
ax2.set_xlim(xmin=0)
ax2.set_ylim(ymin=0)

# Prepare plots
im = ax1.imshow(mapfig['show_map'])
rmse_plot, = ax2.plot([], [])

if render_particles:
	scatter = ax1.scatter([], [], s=0.1)

# Either have an animation call the update function or manually run all iterations
if render_animated:
	# Setup the animation
	frame_count = int(np.ceil((timeline - 1) / samples_per_iteration / iterations_per_frame))
	anim = animation.FuncAnimation(fig=fig, func=animate, frames=frame_count, interval=0, repeat=False, blit=False, cache_frame_data=False)
else:
	# Run all iterations
	while sample <  timeline:
		slam_iteration()

	updateDisplayMap(mapfig)
	update_plots()

# Write output of experiment
if experiment_output is not None:
	# Tally final time
	total_time = time.time() - start_time

	# Save image
	plt.savefig(experiment_output + '.map.png')

	# Save stats
	output = { }
	output['dataset'] = dataset
	output['particle_count'] = N
	output['noise_sigma'] = noise_sigma
	output['local_search_offset'] = local_search_offset
	output['local_search_resolution'] = local_search_resolution
	output['use_rear_lidar'] = use_rear_lidar
	output['runtime'] = total_time
	output['rmse'] = rmse_values[-1]

	with open(experiment_output + '.stats.json', 'w') as f:
		json.dump(output, f, indent=1)

	with open('experiments/results.csv', 'a') as f:
		fields = \
		[
			experiment_output,
			experiment_repetition,
			dataset, 
			use_rear_lidar,
			N,
			noise_sigma,
			local_search_resolution,
			local_search_offset,
			total_time,
			rmse_values[-1] 
		]

		fields = [ str(field) for field in fields ]

		f.write(', '.join(fields))
		f.write('\n')

	print('Experiment finished!')
else:
	# Show the window
	plt.show()