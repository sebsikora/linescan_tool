from linescan import LineScan, WaveFromFile
import csv
from os.path import join
from os import walk

folder_path = '/home/sebsikora/Documents/linescan_tool/test/'
output_filename = 'overall_analysis_results.csv'
output_directory = folder_path
lines_per_second = 188.0
line_width = 0.0000002

intensity_thresholds = [10.0, 12.5, 15.0, 17.5, 20.0]

def getFilenames(image_parent_directory):
	list_of_files = []
	for root, directories, filenames in walk(image_parent_directory):
		for filename in filenames:
			if filename.endswith('.png'):
				list_of_files.append(join(root, filename))
	return list_of_files

def CreateTableRow(image_name, intensity_threshold, status, top_results, bottom_results, mean_result):
	if status == 'empty':
		row_to_return = [image_name, 'Cannot find event leading edge above ' + str(intensity_threshold) + ' standard deviations above the background intensity']
	else:
		if top_results and bottom_results:
			top_gradient = abs(top_results[0][0][0])
			top_r_squared = abs(top_results[1])
			bottom_gradient = abs(bottom_results[0][0][0])
			bottom_r_squared = abs(bottom_results[1])
			top_velocity = abs(top_results[3])
			bottom_velocity = abs(bottom_results[3])
		elif top_results and (not bottom_results):
			top_gradient = abs(top_results[0][0][0])
			top_r_squared = abs(top_results[1])
			bottom_gradient = 'NA'
			bottom_r_squared = 'NA'
			top_velocity = abs(top_results[3])
			bottom_velocity = 'NA'
		elif (not top_results) and bottom_results:
			top_gradient = 'NA'
			top_r_squared = 'NA'
			bottom_gradient = abs(bottom_results[0][0][0])
			bottom_r_squared = abs(bottom_results[1])
			top_velocity = 'NA'
			bottom_velocity = abs(bottom_results[3])
		row_to_return = [image_name, str(intensity_threshold), status, str(top_r_squared), str(top_gradient), str(top_velocity), str(bottom_r_squared), str(bottom_gradient), str(bottom_velocity), str(mean_result[0]), str(mean_result[1])]
	return row_to_return
			
def WriteOutputTable(mean_result, output_directory, output_filename, rows):
	output_file = open(output_directory + output_filename + '_wave_analysis_results.csv', 'ab')
	csv_writer = csv.writer(output_file, delimiter = ',')
	csv_writer.writerow(['Image name', 'Intensity threshold', 'Status', 'Top fit r^2', 'Top Gradient', 'Top Velocity', 'Bottom fit r^2', 'Bottom Gradient', 'Bottom Velocity', 'Mean Gradient', 'Mean Velocity'])
	for row in rows:
		csv_writer.writerow(row)
	csv_writer.writerow(['', '', '', '', '', '', '', '', '', '', 'Mean of mean gradients', mean_result[0]])
	csv_writer.writerow(['', '', '', '', '', '', '', '', '', '', 'Mean of mean velocities', mean_result[1]])
	output_file.close()

file_paths = getFilenames(folder_path)

for current_input_file in file_paths:
	image_name = current_input_file.split('/')[-1].split('.')[0]
	rows = []
	mean_of_mean_gradients = 0.0
	mean_of_mean_velocities = 0.0
	number_of_good_analyses = 0
	for current_intensity_threshold in intensity_thresholds:
		wave = WaveFromFile(file_path = current_input_file, lines_per_second = lines_per_second, line_width = line_width)
		status, top_results, bottom_results, mean_result = wave.Analyse(display_plots = False, output_directory = '', background_span = 25, distance_threshold = 10, fit_window_span = 10, peak_threshold_multiplier = current_intensity_threshold)
		if status == 'good':
			sum_of_mean_gradients = mean_of_mean_gradients + mean_result[0]
			sum_of_mean_velocities = mean_of_mean_velocities + mean_result[1]
			number_of_good_analyses += 1
		next_row = CreateTableRow(image_name, current_intensity_threshold, status, top_results, bottom_results, mean_result)
		print next_row
		rows.append(next_row)
	good_mean_gradient = sum_of_mean_gradients / float(number_of_good_analyses)
	good_mean_velocity = sum_of_mean_velocities / float(number_of_good_analyses)
	WriteOutputTable([good_mean_gradient, good_mean_velocity], output_directory, output_filename, rows)

	
	
