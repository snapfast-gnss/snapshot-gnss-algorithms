# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 21:49:53 2020

@author: Jonas Beuchert
"""

from encodings import utf_8
import numpy as np
import glob
import eph_util as ep
import xml.etree.ElementTree as et
from concurrent import futures
from rinex_preprocessor import preprocess_rinex
import pymap3d as pm
import shapely.geometry as sg
import os

def worker(data):
    """Process one snapperGPS dataset with GPS L1.

    Inputs:
        data - Uppercase character indicating the SnapperGPS dataset ("A"-"K")

    Output:
        all_satellites - list of observable satellites, one array for each data
        all_snrs - list of acquisition SNRs for each observable satellite, one array for each data [dB]
        all_filenames - list of filenames from the dataset, to match satellites and snrs later on
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

    # Get all names of data files
    filenames = glob.glob(os.path.join("data", data, "*.bin"))
    all_filenames = []
    all_satellites = []
    all_snrs = []

    # Iterate over all files
    for idx, filename in enumerate(filenames):

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
        snr_dict = {}

        # Loop over all GNSS
        for gnss in gnss_list:
            # Acquisitiong
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

            all_satellites.append(prn_dict['G'])
            all_snrs.append(snr_dict['G'])
            all_filenames.append(filename)
        break

    return all_satellites, all_snrs, all_filenames

if __name__ == '__main__':

    np.random.seed(0)

    # List of folders
    data = list(map(chr, range(ord('A'), ord('K')+1)))
    
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
        results = pool.map(worker, data)

        results = []
        
        for d in data:
            all_satellites, all_snrs, all_filenames = worker(d)
            write_results(all_satellites, all_snrs, all_filenames, d)