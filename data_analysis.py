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
        overlap_fac (float, default=0.5): amount by which to overlap adjacent windows

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

if __name__=="__main__":
  # Initialize constants
  step = 1/10e3  # microphone sample period, sec
  mic_list = ("Ambient", "Mic 0", "Mic 1", "Mic 2", "Mic 3")  # for setting the legend
  fs = 1/step  # sample rate, Hz
  fft_size = np.int(fs/2)
  overlap_fac = 0.5  # amount by which to overlap windows. Chosen so that amplitude doesn't get all wonky
  result = []
  end_time = []

  # Load data from the database
  host = "mysql.ecn.purdue.edu"  # 128.46.154.164
  user = "op_point_test"
  database = "op_point_test"
  table_name = "10_op_point_test"
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

  # Do analysis and generate data files
  for index in range(5):
    print("Working on mic {}...".format(index), end="")
    tic = timeit.default_timer()
    res, t = do_stft(data["dynamicP"][index], fft_size, fs, overlap_fac)
    toc = timeit.default_timer()
    print("elapsed time: {} sec".format(toc-tic), flush=True)
    result.append(res)
    end_time.append(t)
    fname = "Processed/fft_waterfall_{}.txt".format(mic_list[index])
    with open(fname, "w") as f:
      f.write("{}\n".format(mic_list[index]))
      f.write("End time: {}\n".format(t))
      for line in res:
        f.write(line)
