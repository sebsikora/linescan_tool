from linescan import LineScan, Wave
import csv
from os.path import join
from os import walk

# ---------------------------------------------------------------------------------------------------------------------------------

# The root directory for your folder structure containing the image directories.
parent_directory = '/home/sebsikora/Documents/linescan_tool/test_images/good/'

# These account for the fact that the image will be rotated by 90 degrees to run left-right,
# IE, the short dimension will be vertical and the long dimension will be horizontal.
image_height = 512
image_width = 30000

minimum_feature_width = 50							# Pixels width beneath which to class as 'too short to measure'.
background_sampling_window_width = 25				# Width of image on the left-hand-side over which the background is characterised.

splitting_brightness_sd_multiplier = 3.0			# Number of standard deviations above the background mean for splitting threshold.

feature_detection_brightness_sd_multipliers = [10.0, 12.5, 15.0, 17.5, 20.0]

show_split_images = False							# Show the split raw wave images as you go (True/False).
show_event_fitting = False							# Show the edge detection and linear fit images as you go (True/False).

# These are currently non-functional.

scan_lines_per_second = 188
scan_row_thickness = 1.0

# ---------------------------------------------------------------------------------------------------------------------------------

def getFilenames(image_parent_directory):
	list_of_files = []
	for root, directories, filenames in walk(image_parent_directory):
		for filename in filenames:
			if filename.endswith('.pic'):
				list_of_files.append(join(root, filename))
	return list_of_files

file_paths = getFilenames(parent_directory)

for file_path in file_paths:
	
	print "------------------------------------------------------------------------------------------------------------"
	print file_path
	
	input_filename = file_path.split('/')[-1][:-4]
	output_directory = ''.join([chunk + '/' for chunk in file_path.split('/')[:-1]])
	
	output_file = open(output_directory + '/' + input_filename + '_wave_analysis_results.csv', 'wb')
	csv_writer = csv.writer(output_file, delimiter = ',')
	
	linescan = LineScan(file_path, image_height, image_width)
	events, event_arrays = linescan.Split(background_window = background_sampling_window_width, intensity_threshold_multiplier = splitting_brightness_sd_multiplier, minimum_feature_width = minimum_feature_width, save_events_images = True, save_events_table = True, show_events_images = show_split_images)
	
	if events != None:
		csv_writer.writerow(['Image filename', input_filename, str(len(events)) + ' detected'])
		
		for index, current_event in enumerate(events):
			event_status = current_event[2]
			
			if event_status == 'good':
				csv_writer.writerow(['Event Index', 'Feature detection brightness sd multiplier', 'Upper Gradient', 'Upper r^2', 'Lower Gradient', 'Lower r^2', 'Mean Gradient', 'Mean r^2', 'Comments'])	
				event_total_mean_gradient = 0.0
				event_total_mean_r_squared = 0.0
				event_intensities_attempted = 0
				
				for current_multiplier in feature_detection_brightness_sd_multipliers:
					event_name = 'Name = ' + input_filename + ', event = ' + str(index) + ', threshold = ' + str(current_multiplier) + ','
					wave = Wave(name = event_name, image_array = event_arrays[index], lines_per_second = scan_lines_per_second, line_width = scan_row_thickness)
					status, notes, top_results, bottom_results = wave.Analyse(display = show_event_fitting, output_directory = output_directory, background_span = background_sampling_window_width, peak_threshold_multiplier = current_multiplier)
					print event_name + notes
					row_to_write = [str(index), str(current_multiplier)]
					
					if status != 'empty':
						if top_results and bottom_results:
							top_gradient = abs(top_results[0][0][0])
							top_r_sqaured = top_results[1]
							bottom_gradient = abs(bottom_results[0][0][0])
							bottom_r_squared = bottom_results[1]
							row_to_write.append(str(top_gradient))
							row_to_write.append(str(top_r_sqaured))
							row_to_write.append(str(bottom_gradient))
							row_to_write.append(str(bottom_r_squared))
							mean_gradient = (top_gradient + bottom_gradient) / 2.0
							mean_r_squared = (top_r_sqaured + bottom_r_squared) / 2.0
							row_to_write.append(str(mean_gradient))
							row_to_write.append(str(mean_r_squared))
						elif top_results:
							top_gradient = abs(top_results[0][0][0])
							top_r_sqaured = top_results[1]
							row_to_write.append(str(top_gradient))
							row_to_write.append(str(top_r_sqaured))
							row_to_write.append(' ')
							row_to_write.append(' ')
							mean_gradient = top_gradient
							mean_r_squared = top_r_sqaured
							row_to_write.append(str(mean_gradient))
							row_to_write.append(str(mean_r_squared))
						elif bottom_results:
							bottom_gradient = abs(bottom_results[0][0][0])
							bottom_r_squared = bottom_results[1]
							row_to_write.append(' ')
							row_to_write.append(' ')
							row_to_write.append(str(bottom_gradient))
							row_to_write.append(str(bottom_r_squared))
							mean_gradient = bottom_gradient
							mean_r_squared = bottom_r_squared
							row_to_write.append(str(mean_gradient))
							row_to_write.append(str(mean_r_squared))
						if status == 'short':
							row_to_write.append('Warning - fitted region less than 1/4 of frame height')
						else:
							event_total_mean_gradient = event_total_mean_gradient + mean_gradient
							event_total_mean_r_squared = event_total_mean_r_squared + mean_r_squared
							event_intensities_attempted += 1
					else:
						row_to_write.append('Cannot find event leading edge above ' + str(current_multiplier) + ' standard deviations above the background intensity')
					csv_writer.writerow(row_to_write)
				
				if event_intensities_attempted > 0:
					divisor = float(event_intensities_attempted) 
				else:
					divisor = 1
				
				csv_writer.writerow([' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', 'Mean of mean gradients (exclusive of poor fits)', event_total_mean_gradient / divisor, 'Mean of mean r^2s (exclusive of poor fits)', event_total_mean_r_squared / divisor])
				#csv_writer.writerow([' ', ' ', ' ', ' ', ' ', ' ', ' ', 'Mean of mean r^2s (exclusive of poor fits)', event_total_mean_r_squared / divisor])
				
	else:
		csv_writer.writerow(['Image filename', input_filename, 'No events detected beyond ' + str(splitting_brightness_sd_multiplier) + ' standard deviations of the mean background intensity'])	
	
	output_file.close()
						
