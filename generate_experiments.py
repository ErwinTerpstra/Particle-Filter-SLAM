import json
import itertools

# Helper script to generate experiment config files from a template
with open('experiments/template.json') as f:
	config = json.load(f)

	fields = ['local_search_resolution', 'noise_sigma', 'particle_count', 'use_rear_lidar' ]
	other_fields = [ key for key in config.keys() if key not in fields ]

	for i, permutation in enumerate(itertools.product(*[ config[key] for key in fields])):
		experiment = { key: config[key] for key in other_fields }

		for key_idx, key in enumerate(fields):
			experiment[key] = permutation[key_idx]

			file_name = f'experiments/experiment{i + 1}.json'
			with open(file_name, 'w') as f:
				json.dump(experiment, f)

			print(f'{file_name} done!')
