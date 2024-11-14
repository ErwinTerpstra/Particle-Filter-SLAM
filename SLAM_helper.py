import numpy as np
import cv2
import random

def updateWeights(weights, corr):
	# Convert the weights to "log space". Correlation is added in log space 
	wtmp = np.log(weights) + corr
	
	# Substract maximum weight from all entries
	# This makes sure weights are all negative with the highest becoming 0
	wtmp_max = wtmp[np.argmax(wtmp)]
	wtmp = wtmp - wtmp_max
	
	# Calculates the log-sum-exponent of the weights\
	# Subtracting that from wtmp ensures wtmp sums to 1
	lse = np.log(np.sum(np.exp(wtmp)))
	wtmp = wtmp - lse

	# Convert the weights back from log space to "normal" space
	return np.exp(wtmp)

def getW2B(particles, ori_robot):
	"""Calculates W2B transformation for all particles at once"""
	r, p, y = ori_robot[0], ori_robot[1], particles[:,2]
	n = particles.shape[0]

	cosY = np.cos(y)
	sinY = np.sin(y)

	cosP = np.cos(p)
	sinP = np.sin(p)

	cosR = np.cos(r)
	sinR = np.sin(r)

	r11 = cosY * cosP
	r12 = cosY * sinP * sinR - sinY * cosR
	r13 = cosY * sinP * cosR + sinY * sinR

	r21 = sinY * cosP
	r22 = sinY * sinP * sinR + cosY * cosR
	r23 = sinY * sinP * cosR - cosY * sinR

	r31 = np.full(n, -sinP)
	r32 = np.full(n, cosP * sinR)
	r33 = np.full(n, cosP * cosR)

	t_w2b = np.zeros((n, 4, 4))

	t_w2b[:,0,0] = r11
	t_w2b[:,0,1] = r12
	t_w2b[:,0,2] = r13
	t_w2b[:,1,0] = r21
	t_w2b[:,1,1] = r22
	t_w2b[:,1,2] = r23
	t_w2b[:,2,0] = r31
	t_w2b[:,2,1] = r32
	t_w2b[:,2,2] = r33

	t_w2b[:,0,3] = particles[:,0]
	t_w2b[:,1,3] = particles[:,1]
	t_w2b[:,2,3] = 0.93

	t_w2b[:,3,3] = 1

	return t_w2b

def getB2L(head_a):
	"""Calculates B2L matrix for the given head angles"""
	cosHA0 = np.cos(head_a[0])
	sinHA0 = np.sin(head_a[0])

	cosHA1 = np.cos(head_a[1])
	sinHA1 = np.sin(head_a[1])

	t_b2h = np.array([[cosHA0, -sinHA0, 0, 0],
					[sinHA0, cosHA0, 0, 0],
					[0, 0, 1, 0.33],
					[0, 0, 0, 1]])

	t_h2l = np.array([[cosHA1, 0, sinHA1, 0],
						[0, 1, 0, 0],
						[-sinHA1, 0, cosHA1, 0.15],
						[0, 0, 0, 1]])

	t_b2l = np.matmul(t_b2h, t_h2l)
	return t_b2l

def mapConvert(scan, ori_robot, head_a, angles, particles, N, pos_phy, posX_map, posY_map, m, config):
	indValid = np.logical_and((scan < config['scan_max']), (scan > config['scan_min']))
	scan_valid = scan[indValid]
	angles_valid = angles[indValid]

	# Determines the grid locations that are hit by the sensor scan
	# This essentially converts polar to cartesian coordinates
	xs0 = np.array([scan_valid * np.cos(angles_valid)])
	ys0 = np.array([scan_valid * np.sin(angles_valid)])

	Y = np.concatenate([np.concatenate([np.concatenate([xs0, ys0], axis=0), np.zeros(xs0.shape)], axis=0), np.ones(xs0.shape)], axis=0)

	# Create transformation matrices
	# w2b converts from world orientation to robot local orientation
	# b2l converts from robot base orientation to robot head local orientation
	t_w2b = getW2B(particles, ori_robot)
	t_b2l = getB2L(head_a)
	
	for i in range(N):
		trans_cur = np.matmul(t_w2b[i,:,:], t_b2l)

		res = np.matmul(trans_cur, Y)
		ind_notG = res[2, :] > 0.1

		pos_phy[i] = res[0 : 2, ind_notG]
		posX_map[i] = (np.ceil((res[0, ind_notG] - m['xmin']) / m['res']).astype(np.int16) - 1)
		posY_map[i] = (np.ceil((res[1, ind_notG] - m['ymin']) / m['res']).astype(np.int16) - 1)

	return pos_phy, posX_map, posY_map


# INPUT 
# im              the map 
# vp(0:2,:)       occupied x,y positions from range sensor (in physical unit)  
# xs,ys           physical x,y,positions you want to evaluate "correlation" 
#
# OUTPUT 
# c               sum of the cell values of all the positions hit by range sensor
def mapCorrelation(m, vp, xs, ys):
	im = m['map']

	nx = im.shape[0]
	ny = im.shape[1]

	nxs = xs.size
	nys = ys.size
	cpr = np.zeros((nxs, nys))
	for jy in range(0,nys):
		y1 = vp[1,:] + ys[jy] # 1 x 1076
		iy = (np.ceil((y1 - m['ymin']) / m['res']).astype(np.int16) - 1)
		for jx in range(0,nxs):
			x1 = vp[0,:] + xs[jx] # 1 x 1076
			ix = (np.ceil((x1 - m['xmin']) / m['res']).astype(np.int16) - 1)

			# Create a mask that indicates which positions are valid (within the mask range)
			valid = np.logical_and(np.logical_and((iy >=0), (iy < ny)), \
									np.logical_and((ix >=0), (ix < nx)))
			
			# Count how many positions are actually occupied that the sensor expects to be occupied
			cpr[jx,jy] = np.sum(im[ix[valid],iy[valid]])

	return cpr

def drawMap(particle_cur, xis, yis, m):
	x_sensor = (np.ceil((particle_cur[0] - m['xmin']) / m['res']).astype(np.int16) - 1)
	y_sensor = (np.ceil((particle_cur[1] - m['ymin']) / m['res']).astype(np.int16) - 1)

	x_occupied = np.concatenate([xis, [x_sensor]])
	y_occupied = np.concatenate([yis, [y_sensor]])

	m['log_map'][xis, yis] += 2 * np.log(9)
	polygon = np.zeros((m['sizey'], m['sizex']))

	occupied_ind = np.vstack((y_occupied, x_occupied)).T.astype(np.int32)

	cv2.drawContours(image = polygon, contours = [occupied_ind], contourIdx = 0, color = np.log(1.0 / 9), thickness = -1)
	m['log_map'] += polygon

	occupied = m['log_map'] > 0
	m['map'][occupied] = 1

	return m

def updateDisplayMap(m):
	occupied = m['log_map'] > 0
	empty = m['log_map'] < 0
	route = (m['show_map'][:, :, 0] == 255)
	groundtruth = (m['show_map'][:, :, 2] == 255)
	path = np.logical_or(route, groundtruth)

	m['show_map'][np.logical_and(occupied, ~path), :] = 0
	m['show_map'][np.logical_and(empty, ~path), :] = 254 # Set empty space to 254 so 255 keeps being reserved for the route

def drawParticles(m, scatter, particles):
	# Convert particle positions to map coordinates
	particles_x = (np.ceil((particles[:,0] - m['xmin']) / m['res']).astype(np.int16) - 1)
	particles_y = (np.ceil((particles[:,1] - m['ymin']) / m['res']).astype(np.int16) - 1)

	# Map is rendered with X coordinates as first dimension instead of Y
	# So we need to swap X and Y here
	scatter.set_offsets(np.array([particles_y, particles_x]).T)