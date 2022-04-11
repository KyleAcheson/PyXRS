import numpy as np
import os
#import natsort
from io import StringIO
import scipy.io as sio




def read_trajs_from_xyz(fpaths_list, code_type, natom):
    # main function for reading trajectories specified in every subdir listed
    # in the parent dir fpaths
    # filters all possible subdirs down to suitable trajectory ones on the
    # basis they contain the file traj_file and it is readable
    # by default it currently reads only SHARC files but can be easily extended
    # returns a np array of shape [natom, 3, ntraj, nts] where nts = maximum
    # run time over all trajs read in
    # trajs that crash before nts have subsequent elements set to NaN
    # also returns a list of each trajs run time and the total num. of trajs
    if code_type == 'sharc':
        traj_file = 'output.xyz'
    else:
        print("Only SHARC trajectories currently available.")

    dirs = []
    for fpaths in fpaths_list:
        dirs.extend([(fpaths + '/' + d) for d in os.listdir(fpaths) if os.path.isdir(fpaths + '/' + d)])
        #dirs = natsort.natsorted(dirs)
    dirs = [d for d in dirs if os.path.isfile(d + '/' + traj_file)]
    if len(dirs) < 1:
        raise Exception("No directories containing trajectories found. Please check cwd.")
    fpaths = [(f + '/' + traj_file) for f in dirs]
    print("Attempting to read in %s trajectories from xyz files" % len(fpaths))
    all_trajs, run_times, ntraj = load_trajs_xyz(fpaths, natom, 'sharc') # currently default is SHARC only
    trajs = format_traj_array(all_trajs, run_times, natom, ntraj)
    return trajs, run_times


def format_traj_array(traj_list, run_times, natom, ntraj):
    # formats nested list of trajectories into a np array of dimensions (natom,
    # 3, ntraj, nts)
    # trajs might not all run over time frame, therefore nts is taken to be the
    # longest
    # run time of all trajectories together. For trajectories that crash before
    # the int nts,
    # the subsequent array indexes are filled with NaN

    max_run_time = max(run_times)
    trajs = np.full((natom, 3, ntraj, max_run_time), np.nan)
    for i in range(ntraj):
        run_time = run_times[i]
        trajs[:, :, i, 0:run_time] = np.swapaxes(np.swapaxes(np.array(traj_list[i]), 0, 2), 0, 1) # must do some reshaping
    return trajs


def load_trajs_xyz(fpaths, natom, traj_type='sharc'):
    # iterates over file paths for all trajectories in subdirs and returns
    # all suitable trajs in a nested list along with the run time for each traj
    # simulation and the total traj number
    # currently only works for SHARC formatted trajectory files (see call to
    # read_sharc() func)
    # can further extend to other quantum codes by adding a suitable function
    # that returns a
    # list of trajs and their corrosponding run times
    run_times = []
    all_trajs = []
    ntraj = 0
    for idx, fpath in enumerate(fpaths):
        with open(fpath, 'r') as trj_file:

            if traj_type.lower() == 'sharc':
                traj_coords, traj_nts = read_sharc(trj_file, natom)
            else:
                pass # extend to other quantum md code readers if needed - write another function like read_sharc to call

            if traj_nts > 1:
                ntraj += 1
                run_times.append(traj_nts)
                all_trajs.append(traj_coords)
            else:
                print("%s contains less than one time step - excluding from analysis." % fpath)
    return all_trajs, run_times, ntraj


def read_sharc(trj_file, natom):
    # reads SHARC style trajectory files in xyz files
    # returns a list of a single trajectories coordinates and its run time
    count = 0
    na = 0
    nts = 0
    geom_all = []
    geom_temp = []
    for idx, line in enumerate(trj_file):
        idx += 1
        count += 1
        if count > 2:
            atom_coord = np.genfromtxt(StringIO(line))[1:4].tolist()
            geom_temp.append(atom_coord)
            na += 1
        if na == natom:
            na = 0
            count = 0
            nts += 1
            geom_all.append(geom_temp)
            geom_temp = []

    return geom_all, nts


if __name__== "__main__":
    natom = 3
    parent_dir = '/Users/kyleacheson/PycharmProjects/SHARC_NMA/'
    trajs_dirs = ['Singlet_1', 'Singlet_2']
    fpaths = [parent_dir + tdir for i, tdir in enumerate(trajs_dirs)]
    trajs = read_trajs_from_xyz(fpaths, 'sharc', natom)
    #sio.savemat('trajectories_test.mat', {'Q': trajs})
