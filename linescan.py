from PIL import Image, ImageFilter
import numpy as np		
from scipy.optimize import curve_fit
from scipy.stats import linregress
import matplotlib.pyplot as plt
import csv
from os import path, makedirs

class LineScan():
	def __init__(self, source_filepath, image_height, image_width):
		self.image_height = image_height
		self.image_width = image_width
		self.source_filepath = source_filepath
		self.input_filename = source_filepath[:-4].split('/')[-1]
		
		# For opening .tif files.
		#~self.original_image = Image.open(self.source_filepath)
		
		# For opening .pic raw files.
		image_file_handle = open(self.source_filepath,'rb')
		image_data = np.fromfile(image_file_handle, dtype=np.uint8,count=self.image_width * self.image_height)
		image_data = image_data.reshape((self.image_width, self.image_height))
		image_file_handle.close()
		
		# For reasons that I have not yet understood, unlike ImageJ (which loads the .pic file juuuust fine),
		# we need to take the first 76 bytes from the beginning of the raw, one-dimensional byte stream and 
		# append them to the end. Otherwise the whole image appears rolled in the direction parallel to the
		# short axis, by 76 pixels. Weird ???.
		self.original_image = Image.fromarray(self.__ReAlign__(image_data, self.image_height, self.image_width, 76))
		
		self.original_image_array = np.array(self.original_image)
		self.original_image_array = np.transpose(self.original_image_array)
		blurred_image = self.original_image.filter(ImageFilter.BLUR)
		self.filtered_image = blurred_image.filter(ImageFilter.SMOOTH)
		self.filtered_image_array = np.array(self.filtered_image)
		self.filtered_image_array = np.transpose(self.filtered_image_array)
		self.column_average_intensities = self.__CalcColumnIntensities__(self.filtered_image_array)
	
	def __ReAlign__(self, image_array, image_height, image_width, roll_amount):
		array = image_array.reshape(((image_height * image_width), 1))
		rolled_array = np.concatenate((array[roll_amount:], array[0:roll_amount]))
		output_image_array = rolled_array.reshape((image_width, image_height))
		return output_image_array
	
	def __CreateFolders__(self, directories):
		for current_directory in directories:
			if current_directory != '':
				if not path.exists(current_directory):
					makedirs(current_directory)
	
	def __CalcColumnIntensities__(self, image_array):
		column_average_intensities = []
		for i in range(image_array.shape[1]):
			column_average = np.mean(image_array[:,i])
			column_average_intensities.append(column_average)
		return column_average_intensities
	
	def __CharacteriseBackgroundColumnwise__(self, sub_image_array):
		sub_column_average_intensities = self.__CalcColumnIntensities__(sub_image_array)
		background_sd = np.std(sub_column_average_intensities)
		background_mean = np.mean(sub_column_average_intensities)
		return background_mean, background_sd
	
	def __FindEvents__(self, image_array, column_average_intensities, background_mean, background_sd, intensity_threshold_multiplier):
		events = []
		event_started = False
		for j, current_column_intensity in enumerate(column_average_intensities[1:]):
			if event_started == False:
				if (current_column_intensity > (background_mean + (background_sd * intensity_threshold_multiplier))) and (column_average_intensities[j] < (background_mean + (background_sd * intensity_threshold_multiplier))):
					event_started = True
					current_event = [j, 0]
			elif event_started == True:
				if (current_column_intensity < (background_mean + (background_sd * intensity_threshold_multiplier))) and (column_average_intensities[j] > (background_mean + (background_sd * intensity_threshold_multiplier))):
					event_started = False
					current_event[1] = j
					events.append(current_event)
		return events
	
	def __ScreenEvents__(self, image_array, events, background_padding, gap_ignore_threshold, minimum_feature_width):
		screened_events = []
		current_event = events[0]
		added_events = 0
		
		for i in range(1, len(events)):
			if (events[i][0] - background_padding) < current_event[1]:
				current_event[1] = events[i][1]
				if (events[i][0] - gap_ignore_threshold) > current_event[1]:
					added_events = added_events + 1
			else:
				if current_event[0] - background_padding > 0:
					event_start_pos = current_event[0] - background_padding
				else:
					event_start_pos = 0
				if current_event[1] + background_padding > image_array.shape[1]:
					event_end_pos = image_array.shape[1]
				else:
					event_end_pos = current_event[1] + background_padding
				if (current_event[1] - current_event[0]) > minimum_feature_width:
					if added_events == 0:
						event_status = 'good'
					else:
						event_status = 'multiple'
				else:
					event_status = 'short'	
				screened_events.append((event_start_pos, event_end_pos, event_status))
				added_events = 0
				current_event = events[i]
		if (current_event[1] - current_event[0]) > minimum_feature_width:
			screened_events.append((current_event[0] - background_padding, current_event[1] + background_padding, 'good' if added_events == 0 else 'multiple'))
		else:
			screened_events.append((current_event[0] - background_padding, current_event[1] + background_padding, 'short'))
		return screened_events
	
	def __ShowEventImages__(self, image_array, events): 
		for current_event in events:
			temp_event_array = image_array[:,current_event[0]:current_event[1]]
			temp_event_image = Image.fromarray(temp_event_array)
			temp_event_image.show()
	
	def __SaveEventImages__(self, source_filepath, events, image_array):
		output_directory = ''.join([chunk + '/' for chunk in source_filepath.split('/')[:-1]])
		source_filename = source_filepath.split('/')[-1][:-4]
		self.__CreateFolders__([output_directory + 'good', output_directory + 'suspect'])
		for index, current_event in enumerate(events):
			event_array = image_array[:,current_event[0]:current_event[1]]
			event_image = Image.fromarray(event_array)
			if current_event[2] == 'good':
				output_filename = source_filename + '_G_event_' + str(index) + '.png'
				event_image.save(output_directory + 'good/' + output_filename)
			elif current_event[2] == 'multiple':
				output_filename = source_filename + '_M_event_' + str(index) + '.png'
				event_image.save(output_directory + 'suspect/' + output_filename)
			elif current_event[2] == 'short':
				output_filename = source_filename + '_S_event_' + str(index) + '.png'
				event_image.save(output_directory + 'suspect/' + output_filename)
	
	def __SplitArray__(self, image_array, events):
		event_arrays = []
		for index, current_event in enumerate(events):
			event_arrays.append(image_array[:, current_event[0]:current_event[1]])
		return event_arrays
	
	def __SaveEventsTable__(self, source_filepath, screened_events, minimum_feature_width, intensity_threshold_multiplier, gap_ignore_threshold, background_padding):
		output_directory = ''.join([chunk + '/' for chunk in source_filepath.split('/')[:-1]])
		source_filename = source_filepath.split('/')[-1][:-4]
		output_filename = source_filename + '_split_results_summary.csv'
		self.__CreateFolders__([output_directory[:-1]])
		if output_directory != '':
			output_path = output_directory + '/' + output_filename
		else:
			output_path = output_filename
		with open(output_path, 'wb') as csv_file:
			csv_writer = csv.writer(csv_file, delimiter = ',')
			for index in range(len(screened_events) + 3):
				if index == 0:
					row = ['Minimum feature width', 'Intensity threshold multiplier', 'Gap ignore threshold', 'Background padding']
				elif index == 1:
					row = [str(i) for i in [minimum_feature_width, intensity_threshold_multiplier, gap_ignore_threshold, background_padding]]
				elif index == 2:
					row = ['Start position', 'End position', 'Comments']
				else:
					row = [str(i) for i in screened_events[index - 3]]
				csv_writer.writerow(row)
	
	def ShowImage(self):
		self.image.show()
	
	def ShowIntensities(self):
		fig = plt.figure()
		plt.plot(self.column_average_intensities, 'red')
		plt.show()
	
	def Split(self, background_window = 50, intensity_threshold_multiplier = 3.0, background_padding = 100, minimum_feature_width = 100, gap_ignore_threshold = 10, show_events_images = True, show_events_table = False, save_events_images = False, save_events_table = False):
		self.background_window = background_window
		self.intensity_threshold_multiplier = intensity_threshold_multiplier
		self.minimum_feature_width = minimum_feature_width
		self.gap_ignore_threshold = gap_ignore_threshold
		self.background_padding = background_padding
		
		self.sub_image_array = self.filtered_image_array[:, 0:background_window]
		self.background_mean, self.background_sd = self.__CharacteriseBackgroundColumnwise__(self.sub_image_array)
		
		events = self.__FindEvents__(self.filtered_image_array, self.column_average_intensities, self.background_mean, self.background_sd, self.intensity_threshold_multiplier)
		
		if len(events) > 0:
			self.screened_events = self.__ScreenEvents__(self.filtered_image_array, events, self.background_padding, self.gap_ignore_threshold, self.minimum_feature_width)
			
			if show_events_images == True:
				self.__ShowEventImages__(self.original_image_array, self.screened_events)
			if save_events_images == True:
				self.__SaveEventImages__(self.source_filepath, self.screened_events, self.original_image_array)
			if show_events_table == True:
				for index, current_event in enumerate(self.screened_events):
					print current_event
			if save_events_table == True:
				self.__SaveEventsTable__(self.source_filepath, self.screened_events, self.minimum_feature_width, self.intensity_threshold_multiplier, self.gap_ignore_threshold, self.background_padding)
			
			return self.screened_events, self.__SplitArray__(self.original_image_array, self.screened_events)
		else:
			print "This linescan contains no events beyond " + str(self.intensity_threshold_multiplier) + " standard deviations of the mean brightness" 
			return None, None
	
class Wave():
	def __init__(self, name, image_array, lines_per_second, line_width):
		self.load_type = 'array'
		self.source_filepath = name
		self.original_image_array = image_array
		self.lines_per_second = lines_per_second
		self.line_width = line_width
		self.original_image = Image.fromarray(image_array)
		blurred_image = self.original_image.filter(ImageFilter.BLUR)
		self.filtered_image = blurred_image.filter(ImageFilter.SMOOTH)
		#self.filtered_image.show()
		self.filtered_image_array = np.array(self.filtered_image)
		
	def __f__(self, x, A, B): # this is your 'straight line' y=f(x)
		return A*x + B
	
	def __CreateFolders__(self, directories):
		for current_directory in directories:
			if current_directory != '':
				if not path.exists(current_directory):
					makedirs(current_directory)
	
	def __CharacteriseBackgroundRowWise__(self, image_array, background_span):
		sub_image_start = image_array[:, 0:background_span]
		background_row_sds = []
		background_row_means = []
		for current_row_index in range(sub_image_start.shape[0]):
			current_row_sd = np.std(sub_image_start[current_row_index, :])
			current_row_mean = np.mean(sub_image_start[current_row_index, :])
			background_row_sds.append(current_row_sd)
			background_row_means.append(current_row_mean)
		return background_row_means, background_row_sds
	
	def __FindPeaks__(self, image_array, background_row_means, background_row_sds, peak_threshold_multiplier):
		row_peaks = []
		for current_row_index in range(image_array.shape[0]):
			for current_column_index in range(image_array.shape[1]):
				if image_array[current_row_index, current_column_index] > ((background_row_sds[current_row_index] * peak_threshold_multiplier) + background_row_means[current_row_index]):
					row_peaks.append((current_column_index, current_row_index))
					break
		return row_peaks
	
	def __FilterPeaks__(self, peaks, distance_threshold, fit_window_span):
		final_peaks = []
		peaks = np.array(peaks)
		for i, current_peak in enumerate(peaks):
			if (i <= fit_window_span):
				peaks_subset = peaks[0: fit_window_span]
			elif (i <= len(peaks) - (fit_window_span + 1)) and (i > fit_window_span):
				peaks_subset = peaks[i - fit_window_span: i + fit_window_span]
			elif (i > len(peaks) - (fit_window_span + 1)):
				peaks_subset = peaks[len(peaks) - (fit_window_span + 1): -1]
			result = curve_fit(self.__f__, peaks_subset[:, 1], peaks_subset[:, 0])
			fit_point = self.__f__(current_peak[1], result[0][0], result[0][1])
			if (current_peak[0] > (fit_point + distance_threshold)) or (current_peak[0] < (fit_point - distance_threshold)):
				pass
			else:
				final_peaks.append(current_peak)	
		return np.array(final_peaks)
		
	def __FindSplit__(self, image_array, final_peaks):
		min_x_position = image_array.shape[1]
		min_x_index = 0
		for i, current_peak in enumerate(final_peaks):
			if current_peak[0] <= min_x_position:
				min_x_position = current_peak[0]
				min_x_index = i
		top_peaks = final_peaks[0:min_x_index]
		bottom_peaks = final_peaks[min_x_index:]
		#print top_peaks[0:20]
		return top_peaks, bottom_peaks
	
	def __FitRegion__(self, region_peaks, filtered_peaks):
		lowest_sse = 9.9e25
		fit_window_span = int(region_peaks.shape[0]*(3.0/4.0))
		if fit_window_span > (filtered_peaks.shape[0] / 4) :
			for i in range(region_peaks.shape[0] - fit_window_span):
				# Get a subset of the data, to which we will try and fit a straight line.
				peaks_fitting_window = region_peaks[i:i + fit_window_span,:]
				result = curve_fit(self.__f__, peaks_fitting_window[:, 0], peaks_fitting_window[:, 1])
				# Calculate the fit data and calculate the sum of the squared error between it and the actual wavefront.
				fit_y = self.__f__(peaks_fitting_window[:,0], result[0][0], result[0][1])
				sse = np.sum((peaks_fitting_window[:,1] - fit_y) ** 2.0)
				# If it's the lowest sse yet, store this fit.
				#print sse
				if sse <= lowest_sse:
					lowest_sse = sse
					best_result = result
					plotting_window = peaks_fitting_window
					r_squared = abs(linregress(peaks_fitting_window[:,0], peaks_fitting_window[:,1])[2])
			return_result = [best_result, r_squared, plotting_window]
		else:
			return_result = []
		return return_result
		
	def __CalculateGradients__(self, image_array, settings):
		background_span, peak_threshold_multiplier, distance_threshold, fit_window_span = settings
		background_row_means, background_row_sds = self.__CharacteriseBackgroundRowWise__(image_array, background_span)
		row_peaks = self.__FindPeaks__(image_array, background_row_means, background_row_sds, peak_threshold_multiplier)
		filtered_peaks = self.__FilterPeaks__(row_peaks, distance_threshold, fit_window_span)
		top_peaks, bottom_peaks = self.__FindSplit__(image_array, filtered_peaks)
		top_results = self.__FitRegion__(top_peaks, filtered_peaks)
		bottom_results = self.__FitRegion__(bottom_peaks, filtered_peaks)
		return filtered_peaks, top_results, bottom_results
	
	def __GradientToVelocity__(self, gradient):
		velocity = gradient * self.line_width * self.lines_per_second
		return velocity
	
	def __PlotResults__(self, display, output_directory, top_results, bottom_results, filtered_peaks):
		if self.load_type == 'file':
			if not output_directory:
				print self.source_filepath
				output_directory = ''.join([chunk + '/' for chunk in self.source_filepath.split('/')[:-1]])
			source_filename = self.source_filepath.split('/')[-1][:-4]	
		else:
			source_filename = self.source_filepath
		self.__CreateFolders__([output_directory + 'good', output_directory + 'suspect'])
		
		label_lines = []
		if top_results and bottom_results:
			fit_gradient = (abs(bottom_results[0][0][0]) + abs(top_results[0][0][0])) / 2.0
			fit_velocity = (abs(bottom_results[3] + abs(top_results[3]))) / 2.0
			label_lines.append('Gradient: Upper = ' + str(abs(top_results[0][0][0])))
			label_lines.append('          Lower = ' + str(abs(bottom_results[0][0][0])))
			label_lines.append('          Mean = ' + str(fit_gradient))
			label_lines.append('Velocity: Upper = ' + str(abs(top_results[3])))
			label_lines.append('          Lower = ' + str(abs(bottom_results[3])))
			label_lines.append('          Mean = ' + str(fit_velocity))
			label_lines.append('r^2     : Upper = ' + str(top_results[1]))
			label_lines.append('          Lower = ' + str(bottom_results[1]))
		elif bottom_results:
			fit_gradient = abs(bottom_results[0][0][0])
			label_lines.append('Gradient: Lower = ' + str(abs(bottom_results[0][0][0])))
			label_lines.append('r^2       Lower = ' + str(bottom_results[1]))
		elif top_results:
			fit_gradient = abs(top_results[0][0][0])
			label_lines.append('Gradient: Upper = ' + str(abs(top_results[0][0][0])))
			label_lines.append('r^2     : Upper = ' + str(top_results[1]))
		
		self.fig = plt.figure()
		plt.imshow(self.original_image)
		plt.title(source_filename)
		
		for i, current_label_line in enumerate(label_lines):
			plt.text(10, 20 + (20 * i), current_label_line, fontsize=6, color = 'white')
		
		if bottom_results:
			#plt.plot(bottom_results[2][:,0], bottom_results[2][:,1], 'red')
			self.__PlotSeries__(bottom_results[2][:,0], bottom_results[2][:,1], 'red')
			#plt.plot(self.__f__(bottom_results[2][:,1], bottom_results[0][0][0], bottom_results[0][0][1]), bottom_results[2][:,1], 'black')
			self.__PlotSeries__(bottom_results[2][:,0], self.__f__(bottom_results[2][:,0], bottom_results[0][0][0], bottom_results[0][0][1]),'black')
		if top_results:
			#plt.plot(top_results[2][:,0], top_results[2][:,1], 'red')
			self.__PlotSeries__(top_results[2][:,0], top_results[2][:,1], 'red')
			#plt.plot(self.__f__(top_results[2][:,1], top_results[0][0][0], top_results[0][0][1]), top_results[2][:,1], 'black')
			self.__PlotSeries__(top_results[2][:,0], self.__f__(top_results[2][:,0], top_results[0][0][0], top_results[0][0][1]), 'black')
		
		plt.axis((0, self.original_image_array.shape[1], self.original_image_array.shape[0], 0))
		plt.axis('off')
		self.fig.tight_layout()
		
		if filtered_peaks.shape[0] < (self.original_image_array.shape[0] / 4.0):
			output_direction_choice = 'suspect/'
		else:
			output_direction_choice = 'good/'
		plt.savefig(output_directory + output_direction_choice + source_filename + '.png', bbox_inches='tight')
		
		if display == True:
			plt.show()
		else:
			plt.close()
	
	def __PlotSeries__(self, x_values, y_values, colour):
		plt.plot(x_values, y_values, color = colour)
	
	def __CreateTableEntry__(self, top_results, bottom_results, filtered_peaks):
		if filtered_peaks.shape[0] < int(self.original_image_array.shape[0] / 4.0):
			short = True
		else:
			short = False
		notes = ''
		if bottom_results:
			notes = notes + ' Lower gradient = ' + str(abs(bottom_results[0][0][0])) + ', Lower r^2 = ' + str(bottom_results[1]) + ','
			status = 'good'
		if top_results:
			notes = notes + ' Upper gradient = ' + str(abs(top_results[0][0][0])) + ', Upper r^2 = ' + str(top_results[1]) + ','
			status = 'good'
		if short:
			notes = notes + ' Warning - fitted region less than 1/4 of frame height'
			status = 'short'
		return status, notes
	
	def Analyse(self, display = False, output_directory = '', background_span = 50, peak_threshold_multiplier = 30.0, distance_threshold = 30, fit_window_span = 20):
		settings = [background_span, peak_threshold_multiplier, distance_threshold, fit_window_span]
		filtered_peaks, top_results, bottom_results = self.__CalculateGradients__(self.filtered_image_array, settings)
		if top_results:
			top_results.append(self.__GradientToVelocity__(top_results[0][0][0]))
		if bottom_results:
			bottom_results.append(self.__GradientToVelocity__(bottom_results[0][0][0]))
		if top_results or bottom_results:
			self.__PlotResults__(display, output_directory, top_results, bottom_results, filtered_peaks)
			status, notes = self.__CreateTableEntry__(top_results, bottom_results, filtered_peaks)
		else:
			status, notes = 'empty', 'Cannot find event leading edge above ' + str(peak_threshold_multiplier) + ' standard deviations above the background intensity'
		return status, notes, top_results, bottom_results
		
class WaveFromFile(Wave):
	def __init__(self, file_path, lines_per_second, line_width):
		self.load_type = 'file'
		self.source_filepath = file_path
		self.original_image = Image.open(file_path)
		self.lines_per_second = lines_per_second
		self.line_width = line_width
		self.original_image_array = np.array(self.original_image)
		blurred_image = self.original_image.convert('L').filter(ImageFilter.BLUR)
		self.filtered_image = blurred_image.filter(ImageFilter.SMOOTH)
		self.filtered_image_array = np.array(self.filtered_image)
