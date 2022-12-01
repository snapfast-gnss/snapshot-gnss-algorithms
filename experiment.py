# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 21:49:53 2020

@author: Jonas Beuchert
"""

from encodings import utf_8
import numpy as np
import glob
import eph_util as ep
import coarse_time_navigation as ctn
import xml.etree.ElementTree as et
from concurrent import futures
import time as tm
import matplotlib.pyplot as plt

from rinex_preprocessor import preprocess_rinex
import pymap3d as pm
import shapely.geometry as sg
import os

def worker(data, mode, source):
    """Process one snapperGPS dataset with GPS L1.

    Inputs:
        data - Uppercase character indicating the SnapperGPS dataset ("A"-"K")
        mode - CTN Algorithm to use for positioning
        source - "snapper" or list of directory names containing code_phasess to use for positioning, for the same dataset
                "snapper" runs the eph.acquisition_simplified to process the snapshot raw data

    Output:
        all_errors - dictionnary of errors for "snapper" acquisition and one key for each source

    Author: Jonas Beuchert, adapted by Guillaume Thivolet
    """

    print("Start processing dataset {}.".format(data))

    # List of ground truth positions (static) / initial positions (dynamic)
    init_positions = {
        "A": np.array([50.870492, -1.562298, 100.0]),
        "B": np.array([51.763991, -1.259858, 100.0]),
        "C": np.array([51.751285, -1.246198, 100.0]),
        "D": np.array([51.760732, -1.257458, 100.0]),
        "E": np.array([51.735383, -1.211070, 100.0]),
        "F": np.array([51.735383, -1.211070, 100.0]),
        "G": np.array([51.735383, -1.211070, 100.0]),
        "H": np.array([51.735383, -1.211070, 100.0]),
        "I": np.array([51.755258204127756, -1.2591261135480434, 100.0]),
        "J": np.array([51.755258204127756, -1.2591261135480434, 100.0]),
        "K": np.array([51.755258204127756, -1.2591261135480434, 100.0])
        }
        
    pos_ref_geo = init_positions[data]

     # RINEX navigation data files for different navigation satellite systems
    # You do not need all of them, just use None for those that you do not want
    # Broadcasted ephemeris can be found on
    # https://cddis.nasa.gov/archive/gps/data/daily/2021/brdc/
    rinex_file = glob.glob(os.path.join("data", data,
                                        "BRDC00IGS_R_*_01D_MN.rnx"))[0]
    eph_dict = {}
    eph_dict['G'], eph_dict['E'], eph_dict['C'] = preprocess_rinex(
        rinex_file
        )

    # Ground truth track for dynamic
    gt_files = glob.glob(os.path.join("data", data, "ground_truth*"))
    if len(gt_files) > 0:
        gt_enu = []
        for gt_file in gt_files:

            # Load ground truth
            root = et.parse(gt_file).getroot()
            file_ending = gt_file[-3:]
            print("Open ground truth file of type " + file_ending + ".")
            if file_ending == "gpx":
                # Ground truth position
                gt_geo = [(
                    float(child.attrib['lat']),
                    float(child.attrib['lon'])
                    ) for child in root[-1][-1]]
            elif file_ending == "kml":
                # Get coordinates of path
                try:
                    gt_string = root[-1][-1][-1][-1][-1][-1].text
                except(IndexError):
                    gt_string = root[-1][-1][-1][-1].text
                gt_geo = np.fromstring(
                    gt_string.replace('\n', '').replace('\t', '').replace(
                        ' ', ','), sep=',')
                gt_geo = [(lat, lon)
                        for lat, lon in zip(gt_geo[1::3], gt_geo[::3])]
            else:
                raise ValueError(
                    "Ground truth file format {} not recognized.".format(
                        file_ending))

            # Transform to ENU coordinates with same reference
            gt_enu.append(np.array(pm.geodetic2enu(
                [g[0] for g in gt_geo], [g[1] for g in gt_geo], 0,
                pos_ref_geo[0], pos_ref_geo[1], pos_ref_geo[2]
                )).T)

        # Concatenate both parts, if there are two
        gt_enu = np.vstack(gt_enu)

        # Convert to line
        gt_enu_line = sg.LineString([(p[0], p[1]) for p in gt_enu])

    else:
        print("No ground truth file.")
        gt_file = None

    # Get all names of data files
    filenames = glob.glob(os.path.join("data", data, "*.bin"))

    modes = ["ls-single", "ls-linear", "ls-combo", "ls-sac", "mle",
             "ls-sac/mle", "dpe"]
    ls_modes = {key: val for key, val in zip(
        modes, ["single", "snr", "combinatorial", "ransac", None, "ransac",
                None]
        )}
 
    mle_modes = {key: val for key, val in zip(
        modes, [False, False, False, False, True, True, False]
        )}
    # Maximum number of satellites for CTN
    max_sat_count = {key: val for key, val in zip(
        modes, [5, 15, 10, 15, None, 15]
        )}


    gnss_list = ['G']

    # Frequency offsets of GNSS front-ends
    frequency_offsets = {
        "A": -864.0,
        "B": -384.0,
        "C": -384.0,
        "D": -768.0 + 900.0,
        "E": -768.0 - 300.0,
        "F": -768.0 - 300.0,
        "G": -768.0 - 300.0,
        "H": -768.0 - 300.0,
        "I": -768.0,
        "J": -768.0,
        "K": -768.0
        }

    # Intermediate frequency [Hz]
    intermediate_frequency = 4092000.0
    # Correct intermediate frequency
    intermediate_frequency = intermediate_frequency + frequency_offsets[data]

    # Diameter of temporal search space [s] (MLE and DPE only)
    search_space_time = {
        "A": 2.0,
        "B": 10.0,
        "C": 2.0,
        "D": 2.0,
        "E": 2.0,
        "F": 2.0,
        "G": 10.0,
        "H": 2.0,
        "I": 2.0,
        "J": 2.0,
        "K": 2.0
        }


    # Get all names of data files
    filenames = glob.glob(os.path.join("data", data, "*.bin"))

    all_errors = dict()


    # Iterate over all files
    for idx, filename in enumerate(filenames):

        #filename = 'data/A/20201206_153820.bin'

        print('Snapshot {} of {}'.format(idx+1, len(filenames)))

         # Random error in box
        if gt_file is None:
            init_err_east = np.random.uniform(low=-10.0e3, high=10.0e3)
            init_err_north = np.random.uniform(low=-10.0e3, high=10.0e3)
            init_err_height = np.random.uniform(low=-100.0, high=100.0)
        else:
            init_err_east = np.random.uniform(low=-1.0e3, high=1.0e3)
            init_err_north = np.random.uniform(low=-1.0e3, high=1.0e3)
            init_err_height = np.random.uniform(low=-10.0, high=10.0)
        pos_geo = np.empty(3)
        pos_geo[0], pos_geo[1], pos_geo[2] = pm.enu2geodetic(
            init_err_east, init_err_north, init_err_height,
            pos_ref_geo[0], pos_ref_geo[1], pos_ref_geo[2])

        # Ground truth time from filename
        YYYY = filename[-19:-15]
        MM = filename[-15:-13]
        DD = filename[-13:-11]
        hh = filename[-10:-8]
        mm = filename[-8:-6]
        ss = filename[-6:-4]
        utc = np.datetime64(YYYY
                            + "-" + MM
                            + "-" + DD
                            + "T" + hh
                            + ":" + mm
                            + ":" + ss)

        # Read signals from files
        # How many bytes to read
        bytes_per_snapshot = int(4092000.0 * 12e-3 / 8)
        # Read binary raw data from file
        signal_bytes = np.fromfile(filename, dtype='>u1',
                                   count=bytes_per_snapshot)
        # Get bits from bytes
        # Start here if data is passed as byte array
        signal = np.unpackbits(signal_bytes, axis=-1, count=None,
                               bitorder='little')
        # Convert snapshots from {0,1} to {-1,+1}
        signal = -2 * signal + 1

        ###################################################################
        # Acquisition
        ###################################################################

        # Store acquisition results in dictionaries with one element per GNSS
        snapshot_idx_dict = {}
        prn_dict = {}
        code_phase_dict = {}
        eph_dict_curr = {}
        snr_dict = {}

        # Loop over all GNSS
        for gnss in gnss_list:
            ###################################################################
            # Acquisition
            ###################################################################
            snapshot_idx_dict[gnss], prn_dict[gnss], code_phase_dict[gnss], \
                snr_dict[gnss], eph_idx, _, _ = ep.acquisition_simplified(
                    np.array([signal]),
                    np.array([utc]),
                    pos_geo,
                    eph=eph_dict[gnss],
                    system_identifier=gnss,
                    intermediate_frequency=intermediate_frequency,
                    frequency_bins=np.linspace(-0, 0, 1),
                    )

            eph_dict_curr[gnss] = eph_dict[gnss][:, eph_idx]

        #start_time = tm.time()

        ###################################################################
        # Positioning
        ###################################################################

        # Estimate all positions with a single function call
        # Correct timestamps, too
        # Finally, estimate the horizontal one-sigma uncertainty
        latitude_estimates, longitude_estimates, time_utc_estimates, \
            uncertainty_estimates \
            = ctn.positioning_simplified(
                    snapshot_idx_dict,
                    prn_dict,
                    code_phase_dict,
                    snr_dict,
                    eph_dict_curr,
                    np.array([utc]),
                    # Initial position goes here or
                    # if data is processed in mini-batches, last plausible position
                    pos_geo[0], pos_geo[1], pos_geo[2],
                    # If we could measure the height, it would go here (WGS84)
                    observed_heights=None,
                    # If we measure pressure & temperature, we can estimate the height
                    pressures=None, temperatures=None,
                    # There are 5 different modes, 'snr' is fast, but inaccurate
                    # In the future, 'ransac' might be the preferred option
                    ls_mode=ls_modes[mode],
                    # Turn mle on to get a 2nd run if least-squares fails (recommended)
                    mle=mle_modes[mode],
                    # This parameter is crucial for speed vs. accuracy/robustness
                    # 10-15 is good for 'snr', 10 for 'combinatorial', 15 for 'ransac'
                    max_sat_count=max_sat_count[mode],
                    # These parameters determine the max. spatial & temporal distance
                    # between consecutive snapshots to be plausible
                    # Shall depend on the application scenario
                    max_dist=15.0e3, max_time=30.0,
                    # If we would know an initial offset of the timestamps
                    # If data is processed in mini-batches, the error from previous one
                    time_error=0.0,
                    search_space_time=search_space_time[data])

        # Measure time spent on positioning
        #all_time.append(tm.time() - start_time)

        # Calculate positioning error in ENU coordinates [m,m,m]
        err_east, err_north, err_height \
            = pm.geodetic2enu(latitude_estimates[0], longitude_estimates[0], pos_ref_geo[2],
                              pos_ref_geo[0], pos_ref_geo[1], pos_ref_geo[2])

        if gt_file is not None:
            pos_enu = np.array([err_east, err_north, err_height])
            # Get nearest point on line for all estimated points
            nearest_point = gt_enu_line.interpolate(gt_enu_line.project(
                sg.Point((pos_enu[0], pos_enu[1]))
                ))

            # Calculate horizontal error
            err = np.linalg.norm(nearest_point.coords[0] - pos_enu[:2])
        else:
            err = np.linalg.norm(np.array([err_east, err_north]))

        if np.isnan(err):
            err = np.inf

        print('(Snapper) Resulting horizontal error: {:.0f} m'.format(err))

        if not 'snapper' in all_errors:
            all_errors['snapper'] = [err]
        else:
            all_errors['snapper'].append(err)

        ###################################################################
        # Positioning from matlab generated code phase
        ###################################################################

        basename = os.path.basename(filename)
        for key in source:
            filename = os.path.join("data", key, basename + "_result.txt")

            print(source)

            # Read binary raw data from file
            input = np.fromfile(filename, dtype='f4', count=64)
            code_phases_matlab = input[:32]
            snrs_matlab = input[32:]

            # Filter out the non visible satellites
            code_phases_matlab = {'G': np.array([(4092 - code_phases_matlab[x-1]) / 4.092e3 for x in prn_dict['G']])}
            snrs_matlab = {'G': np.array([snrs_matlab[x-1] for x in prn_dict['G']])}

            print(np.round(code_phases_matlab['G'] * 4092))
            print(np.round(code_phase_dict['G'] * 4092))

            print(np.round(snr_dict['G'], 2))
            print(np.round(snrs_matlab['G'], 2))

            error_cp_dict = code_phase_dict['G'] - code_phases_matlab['G']
            error_snr_dict = snr_dict['G'] - snrs_matlab['G']

            error_cp_dict = [error_cp_dict[id] if error_cp_dict[id] > 1e-3 else 0 for id, val in enumerate(error_cp_dict)] 
            error_snr_dict = [error_snr_dict[id] if error_cp_dict[id] > 0 else 0 for id, val in enumerate(error_snr_dict)] 

        #   error_snr_dict = error_snr_dict if error_cp_dict is not 0 else 0 

            print(error_cp_dict)
            print(error_snr_dict)

            # Estimate all positions with a single function call
            # Correct timestamps, too
            # Finally, estimate the horizontal one-sigma uncertainty
            latitude_estimates_matlab, longitude_estimates_matlab, time_utc_estimates_matlab, \
                uncertainty_estimates_matlab \
                = ctn.positioning_simplified(
                        snapshot_idx_dict,
                        prn_dict,
                        code_phases_matlab,
                        snrs_matlab,
                        eph_dict_curr,
                        np.array([utc]),
                        # Initial position goes here or
                        # if data is processed in mini-batches, last plausible position
                        pos_geo[0], pos_geo[1], pos_geo[2],
                        # If we could measure the height, it would go here (WGS84)
                        observed_heights=None,
                        # If we measure pressure & temperature, we can estimate the height
                        pressures=None, temperatures=None,
                        # There are 5 different modes, 'snr' is fast, but inaccurate
                        # In the future, 'ransac' might be the preferred option
                        ls_mode=ls_modes[mode],
                        # Turn mle on to get a 2nd run if least-squares fails (recommended)
                        mle=mle_modes[mode],
                        # This parameter is crucial for speed vs. accuracy/robustness
                        # 10-15 is good for 'snr', 10 for 'combinatorial', 15 for 'ransac'
                        max_sat_count=max_sat_count[mode],
                        # These parameters determine the max. spatial & temporal distance
                        # between consecutive snapshots to be plausible
                        # Shall depend on the application scenario
                        max_dist=15.0e3, max_time=30.0,
                        # If we would know an initial offset of the timestamps
                        # If data is processed in mini-batches, the error from previous one
                        time_error=0.0,
                        search_space_time=search_space_time[data])

            # Calculate positioning error in ENU coordinates [m,m,m]
            err_east, err_north, err_height \
                = pm.geodetic2enu(latitude_estimates_matlab[0], longitude_estimates_matlab[0], pos_ref_geo[2],
                                pos_ref_geo[0], pos_ref_geo[1], pos_ref_geo[2])

            if gt_file is not None:
                pos_enu = np.array([err_east, err_north, err_height])
                # Get nearest point on line for all estimated points
                nearest_point = gt_enu_line.interpolate(gt_enu_line.project(
                    sg.Point((pos_enu[0], pos_enu[1]))
                    ))

                # Calculate horizontal error
                err = np.linalg.norm(nearest_point.coords[0] - pos_enu[:2])
            else:
                err = np.linalg.norm(np.array([err_east, err_north]))

            if np.isnan(err):
                err = np.inf

            print('(Matlab - {}) Resulting horizontal error: {:.0f} m'.format(key, err))

            if not key in all_errors:
                all_errors[key] = [err]
            else:                
                all_errors[key].append(err)

      # break

    return all_errors #{'snapper': all_error_snapper, 'matlab': all_error_matlab}, all_time

if __name__ == '__main__':

    np.random.seed(0)

    # List of folders
    #data = list(map(chr, range(ord('A'), ord('K')+1)))
    data = ['A']
    source = [['A_20bins_fft4096', 'A_20bins_fft4092', 'A_40bins_fft4092', 'A_80bins_fft4092']]

    def write_results(all_satellites, all_snrs, all_filenames, folder):
        path = os.path.join("data", folder, 'results.txt')
        
        with open(path, mode="w", encoding="utf8") as file_results:
            for id, filename in enumerate(all_filenames):
                satellites = ','.join(map(str, all_satellites[id]))
                snrs = ','.join(map(str, all_snrs[id]))

                file_results.write(filename + '\n')
                file_results.write(satellites + '\n')
                file_results.write(snrs + '\n')
                        
    with futures.ProcessPoolExecutor() as pool:
        results = pool.map(worker, data, ["ls-linear"], source)

    for id, result in enumerate(results):
        errors = result
        #write_results(all_satellites, all_snrs, all_filenames, data[id])

    def cdf(x, plot=True, *args, **kwargs):
        """Plot cumulative error."""
        x, y = sorted(x), np.arange(len(x)) / len(x)
        return plt.plot(x, y, *args, **kwargs) if plot else (x, y)

    def reliable(errors):
        """Portion of horizontal errors below 200 m."""
        return (np.array(errors) < 200).sum(axis=0) / len(errors)

    print()
    print("(Snapper) Median horizontal error: {:.1f} m".format(np.median(errors['snapper'])))
    print("(Snapper) Error < 200 m: {:.0%}".format(reliable(errors['snapper'])))

    for key in source[0]:
        print("(Matlab - {}) Median horizontal error: {:.1f} m".format(key, np.median(errors[key])))
        print("(Matlab - {}) Error < 200 m: {:.0%}".format(key, reliable(errors[key])))
        print()

    # Plot CDF
    cdf(errors['snapper'])

    for key in source[0]:
        cdf(errors[key])

    plt.xlim(0, 200)
    plt.ylim(0, 1)
    plt.grid()
    plt.yticks(np.linspace(0, 1, 11))
    plt.xlabel("horizontal error [m]")
    plt.legend(['SnapperGPS eph_acquisition', 
    'SnapFast acquisition - 20 doppler bins - 4096 points DFT',
    'SnapFast acquisition - 20 doppler bins - 4092 points DFT',
    'SnapFast acquisition - 40 doppler bins - 4092 points DFT',
    'SnapFast acquisition - 80 doppler bins - 4092 points DFT'])
    plt.title(f"CDF of error against reference, dataset A, mode ls-linear")
    plt.grid(True)

    plt.show()
