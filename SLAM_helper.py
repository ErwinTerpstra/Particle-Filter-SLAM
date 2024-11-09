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

def getW2B(part_cur, ori_robot):
	r, p, y = ori_robot[0], ori_robot[1], part_cur[2]

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

	r31 = -sinP
	r32 = cosP * sinR
	r33 = cosP * cosR
	
	t_w2b = np.array(
			[
				[r11, r12, r13, part_cur[0]],
				[r21, r22, r23, part_cur[1]],
				[r31, r32, r33, 0.93],
				[0, 0, 0, 1]
			])
	
	return t_w2b


def convertFrame(part_cur, ori_robot, head_angles):
	cosHA0 = np.cos(head_angles[0])
	sinHA0 = np.sin(head_angles[0])

	cosHA1 = np.cos(head_angles[1])
	sinHA1 = np.sin(head_angles[1])
	
	t_w2b = getW2B(part_cur, ori_robot)

	t_b2h = np.array([[cosHA0, -sinHA0, 0, 0],
						[sinHA0, cosHA0, 0, 0],
						[0, 0, 1, 0.33],
						[0, 0, 0, 1]])

	t_h2l = np.array([[cosHA1, 0, sinHA1, 0],
						[0, 1, 0, 0],
						[-sinHA1, 0, cosHA1, 0.15],
						[0, 0, 0, 1]])
	
	return np.matmul(t_w2b, t_b2h, t_h2l)


def mapConvert(scan, ori_robot, head_a, angles, particles, N, pos_phy, posX_map, posY_map, m):
	indValid = np.logical_and((scan < 30), (scan > 0.1))
	scan_valid = scan[indValid]
	angles_valid = angles[indValid]

	xs0 = np.array([scan_valid * np.cos(angles_valid)])
	ys0 = np.array([scan_valid * np.sin(angles_valid)])

	Y = np.concatenate([np.concatenate([np.concatenate([xs0, ys0], axis=0), np.zeros(xs0.shape)], axis=0), np.ones(xs0.shape)], axis=0)

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

	for i in range(N):
		trans_cur = np.matmul(getW2B(particles[i,:], ori_robot), t_b2l)

		res = np.matmul(trans_cur, Y)
		ind_notG = res[2, :] > 0.1

		pos_phy[i] = res[0 : 2, ind_notG]
		posX_map[i] = (np.ceil((res[0, ind_notG] - m['xmin']) / m['res']).astype(np.int16) - 1)
		posY_map[i] = (np.ceil((res[1, ind_notG] - m['ymin']) / m['res']).astype(np.int16) - 1)

	return pos_phy, posX_map, posY_map


def drawMap(particle_cur, xis, yis, m):
	x_sensor = (np.ceil((particle_cur[0] - m['xmin']) / m['res']).astype(np.int16) - 1)
	y_sensor = (np.ceil((particle_cur[1] - m['ymin']) / m['res']).astype(np.int16) - 1)

	x_occupied = np.concatenate([xis, [x_sensor]])
	y_occupied = np.concatenate([yis, [y_sensor]])

	m['log_map'][xis, yis] += 2 * np.log(9)
	polygon = np.zeros((m['sizey'], m['sizex']))

	occupied_ind = np.vstack((y_occupied, x_occupied)).T
	cv2.drawContours(image = polygon, contours = [occupied_ind], contourIdx = 0, color = np.log(1.0 / 9), thickness = -1)
	m['log_map'] += polygon

	occupied = m['log_map'] > 0
	empty = m['log_map'] < 0
	route = (m['show_map'][:, :, 0] == 255)

	m['map'][occupied] = 1
	m['show_map'][occupied, :] = 0
	m['show_map'][np.logical_and(empty, ~route), :] = 1

	return m

