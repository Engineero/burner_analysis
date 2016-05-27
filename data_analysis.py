#****************************************************************************
#
# Filename        : data_analysis.py
# Author          : Nathan L. Toner
# Created         : 2016-05-27
# Modified        : 2016-05-27
# Modified By     : Nathan L. Toner
#
# Description:
# Utilities for analyzing experimental data.
#
# Copyright (C) 2016 Nathan L. Toner
#
#***************************************************************************/

import timeit
import numpy as np

from database import *
from scipy import signal
from scipy.linalg import hankel
from sklearn.metrics import mutual_info_score, mean_squared_error

def do_stft(data, fft_size, fs, overlap_fac=0.5):
  """
  Generates a short-time fourier transform waterfall matrix by performing FFT
  of the input data over windows of fft_size length, overlapping by a factor
  of overlap_fac until the end of the data set. The resulting matrix contains
  the decibel power indexed by frequency and time.

  Args:
    data (float array): long time series data to be analyzed
    fft_size (int): number of samples used for each FFT window
    fs (float): sample rate, Hz
    overlap_fac (float, default=0.5): amount by which to overlap adjacent
      windows

  Returns:
    array: Decibel power indexed by frequency and time
  """

  hop_size = np.int32(np.floor(fft_size * (1-overlap_fac)))
  pad_end_size = fft_size  # the last segment can overlap the end of the data array by no more than one window size
  total_segments = np.int32(np.ceil(len(data) / np.float32(hop_size)))
  t_max = len(data) / np.float32(fs)

  window = np.hanning(fft_size)  # our half cosine window
  inner_pad = np.zeros(fft_size)  # the zeros which will be used to double each segment size

  proc = np.concatenate((data, np.zeros(pad_end_size)))  # the data to process
  result = np.empty((total_segments, fft_size), dtype=np.float32)    # space to hold the result

  for i in range(total_segments):  # for each segment
    current_hop = hop_size * i  # figure out the current segment offset
    segment = proc[current_hop:current_hop+fft_size]  # get the current segment
    windowed = segment * window  # multiply by the half cosine function
    padded = np.append(windowed, inner_pad)  # add 0s to double the length of the data
    spectrum = np.fft.rfft(padded, norm="ortho")  # take the Fourier Transform and scale by the number of samples
    autopower = np.abs(spectrum)**2  # find the autopower spectrum
    result[i, :] = autopower[:fft_size]  # append to the results array

  result = 20*np.log10(result)  # scale to db
  return (np.clip(result, -40, 200), t_max)  # clip values

def CC_waterfall(x, y, num_samp=1000, overlap_fac=0.5):
  """
  Generate a 2D data set of the cross correlation between two different
  microphone samples where each row represents cross correlation between a
  single sample from the first microphone and a delayed sample with variable
  delay from the second microphone. Each column represents a new sample from
  the first microphone. Overlap between first microphone samples are given by
  overlap_fac.

  Args:
    x (float array): first (upstream) microphone
    y (float array): second (downstream) microphone over which we slide
      first mic sample
    num_samp (int, default=1000): number of samples to compare, i.e. window
      size
    overlap_fac (float, default=0.5): amount by which to overlap sample
      windows on first data set. overlap_fac in [0, 1)

  Returns:
    array: 2D array of cross correlation indexed by location (time) through
      the first data set, and by time delay.
  """

  hop_size = np.int(np.floor(num_samp * (1-overlap_fac)))  # size of hops through first distribution, x
  total_segments = np.int(np.floor((len(x) - 2*num_samp)/np.float(hop_size)))  # number of hops to end of x
  result = np.empty((total_segments, num_samp+1))
  start = 0  # moving start location for x
  try:
    for seg in range(total_segments):
      stationary = y[start:start+2*num_samp]
      cc = np.correlate(stationary, x[start:start+num_samp], mode="valid")
      result[seg,:] = cc
      start += np.int(num_samp*(1-overlap_fac))  # move window across x
  except:
    print("Error at segment {}/{}, jump {}/{}".format(seg, total_segments, num, num_jumps))
    traceback.print_exc()
  result = 20*np.log10(result**2)  # scale to db
  return np.clip(result, -40, 200)  # clip values

def calc_MI(x, y, bins):
  """
  Calculate the mutual information between two distributions.

  Args:
    x (float array): true distribution
    y (float array): predicted distribution
    bins (int): number of bins to use in building a histogram of x and y

  Returns:
    float: mutual information between x and y
  """

  c_xy = np.histogram2d(x, y, bins)[0]
  mi = mutual_info_score(None, None, contingency=c_xy)
  return mi

def MI_waterfall(x, y, bins, num_samp=1000, jump=1, num_jumps=1000, overlap_fac=0.5):
  """
  Generate a 2D data set of mutual information between two different microphone
  samples where each row represents mutual information between a single sample
  from the first microphone and a delayed sample with variable delay from the
  second microphone. Each column represents a new sample from the first
  microphone. Overlap between first microphone samples are given by
  overlap_fac.

  Args:
    x (float array): true distribution
    y (float array): predicted distribution
    bins (int): number of bins to use in building a histogram of x and y
    num_samp (int, default=1000): number of samples to compare, i.e. window
      size
    jump (int, default=1): number of points to shift second window for each
      calculation
    num_jumps (int, default=1000): number of times to shift second window
    overlap_fac (float, default=0.5): amount by which to overlap sample windows
      on first data set. overlap_fac in [0, 1)

  Returns:
    array: 2D array of mutual information indexed by location (time) through
      the first data set, and time delay.
  """

  hop_size = np.int(np.floor(num_samp * (1-overlap_fac)))  # size of hops through first distribution, x
  total_segments = np.int(np.floor((len(x) - (jump*num_jumps+num_samp))/np.float(hop_size)))  # number of hops to end of x
  result = np.empty((total_segments, num_jumps))
  start = 0  # moving start location for x
  progress = np.floor(total_segments/50)
  try:
    for seg in range(total_segments):
      if seg % progress == 0:
        print(".", end="", sep="", flush=True)
      stationary = x[start:start+num_samp]
      pos = start  # moving start location for y
      for num in range(num_jumps):
        result[seg, num] = calc_MI(stationary, y[pos:pos+num_samp], bins)
        pos += jump
      start += np.int(num_samp*(1-overlap_fac))  # move window across x
  except:
    print("Error at segment {}/{}, jump {}/{}".format(seg, total_segments, num, num_jumps))
    traceback.print_exc()
  return result

if __name__=="__main__":
  # Initialize constants
  step = 1/10e3  # microphone sample period, sec
  mic_list = ("Ambient", "Mic 0", "Mic 1", "Mic 2", "Mic 3")  # for setting the legend
  fs = 1/step  # sample rate, Hz
  end_t = 0.0  # end time for measurements, sec

  # Load data from the database
  host = "mysql.ecn.purdue.edu"  # 128.46.154.164
  user = "op_point_test"
  database = "op_point_test"
  table_name = "100_op_point_test"
  with open("password.txt", "r") as f:
    password = f.read().rstrip()
    print(password)
  eng = connect_to_db(host, user, password, database)
  tic = timeit.default_timer()
  data = import_data(eng, table_name)
  toc = timeit.default_timer()
  if eng.open:
      eng.close()
  print("Elapsed time: {} sec".format(toc-tic))

  # Do FFT waterfall analysis and generate data files
  fft_size = np.int(fs/2)
  overlap_fac = 0.5  # amount by which to overlap windows. Chosen so that amplitude doesn't get all wonky
  for index in range(5):
    print("Doing FFT waterfall of mic {} ... ".format(mic_list[index]), end="")
    tic = timeit.default_timer()
    stft_res, end_t = do_stft(data["dynamicP"][index], fft_size, fs,
        overlap_fac)
    toc = timeit.default_timer()
    print("elapsed time: {} sec".format(toc-tic), flush=True)
    fname = "Processed/fft_waterfall_{}.txt".format(mic_list[index])
    with open(fname, "w") as f:
      f.write("{}\n".format(mic_list[index]))
      f.write("End time: {}\n".format(end_t))
      for line in stft_res:
        line.tofile(f, sep=",")
        f.write("\n")

  # Do auto-correlation analysis
  cc_samp = 1000  # samples taken for auto-correlation
  for index in range(5):
    print("Doing auto-correlation of {} ... ".format(mic_list[index]), end="")
    tic = timeit.default_timer()
    cc_res = CC_waterfall(data["dynamicP"][index], data["dynamicP"][index],
        cc_samp, overlap_fac=0)
    toc = timeit.default_timer()
    print("elapsed time: {} sec".format(toc-tic), flush=True)
    fname = "Processed/fft_autocorr_{}.txt".format(mic_list[index])
    with open(fname, "w") as f:
      f.write("{}\n".format(mic_list[index]))
      f.write("End time: {}\n".format(end_t))
      for line in cc_res:
        line.tofile(f, sep=",")
        f.write("\n")

  # Do auto-mutual information analysis
  bins = 6  # number of histogram bins to represent data's pdf
  # mi_samp = np.int(fs/2)
  mi_samp = 2000
  jump = 1
  num_jumps = 100
  delay = np.arange(0, step*num_jumps*jump, step*jump)  # array of time delays
  for index in range(5):
    print("Doing auto-MI of {} ".format(mic_list[index]), end="")
    tic = timeit.default_timer()
    mi_res = MI_waterfall(data["dynamicP"][index], data["dynamicP"][index],
        bins, num_samp, jump, num_jumps, overlap_fac=0.5)
    toc = timeit.default_timer()
    print(" elapsed time: {} sec".format(toc-tic), flush=True)
    fname = "Processed/fft_autoMI_{}.txt".format(mic_list[index])
    with open(fname, "w") as f:
      f.write("{}\nDelay: ".format(mic_list[index]))
      delay.tofile(f, sep=",")
      f.write("\n")
      for line in mi_res:
        line.tofile(f, sep=",")
        f.write("\n")
