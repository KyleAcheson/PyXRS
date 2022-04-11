import numpy as np
import readers
import f_functions as ffuncs
import matplotlib.pyplot as plt

au2ang = 0.52917721092
ang2au = 1/au2ang

def setup_qvec(kin, theta_lim, Nq):
    qmin = 2 * kin * np.sin(0.5*theta_lim[0])
    qmax = 2 * kin * np.sin(0.5*theta_lim[1])
    q = np.linspace(qmin, qmax, Nq)
    qAng = q * ang2au
    qAng = qAng[np.where(qAng < 8 * np.pi)] # make sure never above 8*pi
    return qAng


def calculate_scattering(trajs, traj_count, qAng, FLAGelec):

    Nq = len(qAng)
    [Nat, _, Ntraj, Nts] = np.shape(trajs)
    [FF, fq] = ffuncs.scattering_factors(qAng, atmnum, FLAGelec)
    Iat = sum(fq**2) # atomic scattering background term (when i == j) - coherent addition of scattering from each isolated atom

    Wiam_avg = np.zeros((Nq, Nts))
    for ts in range(Nts):
        Wiam = np.zeros((Ntraj, Nq))
        Q = trajs[:, :, :, ts]
        for traj in range(Ntraj):
            Imol = np.zeros(Nq)
            sinQ = np.zeros(Nq)
            for i in range(Nat):
                for j in range(i+1, Nat): # distance mat symmetric - take only upper triangle for speed and multiply by factor of 2 after
                    Dq = qAng * np.linalg.norm(Q[i, 0:3, traj] - Q[j, 0:3, traj])
                    sinQ = np.sin(Dq)/Dq
                    inds = np.where(abs(Dq) < 1.E-9)
                    sinQ[inds] = 1.E0
                    Imol = Imol + (2 * FF[i, j, :] * sinQ) # molecular scattering (i != j) - incoherent summation of amplitudes

            if FLAGelec:
                Wiam[traj, :] = (qAng * Imol)/Iat
            else:
                Wiam[traj, :] = Imol + Iat # total scattering for each traj is the sum of molecular and atomic terms

        Wiam_avg[:, ts] = np.nansum(Wiam, 0)/traj_count[ts]

    return Wiam_avg


def calc_traj_time(run_times, Ntraj):
    traj_count = np.zeros(Nts)
    for i in range(Ntraj):
        traj_count[0:run_times[i]] = traj_count[0:run_times[i]] + 1
    return traj_count


def percentage_diff_signal(Wiam, exfrac, ref_signal, Nts, Nq, FLAGelec):
    pdw = np.zeros((Nq, Nts))
    if FLAGelec:
       for ts in range(Nts):
           pdw[:, ts] = 100 * exfrac * (Wiam[:, ts] - ref_signal)
    else:
        for ts in range(Nts):
            pdw[:, ts] = 100 * exfrac * ((Wiam[:, ts] - ref_signal) / ref_signal)

    return pdw


def plot_signal(pdw, qAng, time):
    [Q, T] = np.meshgrid(qAng, time)
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot_surface(Q, T, np.transpose(pdw), edgecolor='none', cmap='RdBu')
    plt.xlim([0, 12])


if __name__ == "__main__":

    ### INPUT PARAMS  ###
    parent_dir = '/Users/kyleacheson/PycharmProjects/SHARC_NMA/' # parent dir of every traj
    trajs_dirs = ['Singlet_1', 'Singlet_2'] # each subdir that contains TRAJ_XXXXX subdirs
    FLAGelec = True # True = electron scattering (to be finished)
    FLAGsave = False # save percentage difference, qvec and time to file
    atmnum = [6, 16, 16] # atomic numbers - must be same ordering as trajs
    natom = 3 # number atoms
    dt = 0.5 # time step in traj simulations
    exfrac = 0.03 # excitation fraction (1 = 100 %)
    Nq = 481 # number of points in q space
    qmax = 24 # max q value (in inv angstrom)

    # if you wish to set up qvec based on angles using the setup_qvec function, uncomment below
    #kin = 12 * au2ang # incident beam energy
    #theta_lim = [0, np.pi/2] # min and max scattering angles

    ### END OF INPUT ###

    #qAng = setup_qvec(kin, theta_lim, Nq) # determine qvec from angles and beam energy
    qAng = np.linspace(0, qmax, Nq)

    fpaths = [parent_dir + tdir for i, tdir in enumerate(trajs_dirs)]
    # note trajs have np.nan at timesteps where that traj does not exist
    trajs, run_times = readers.read_trajs_from_xyz(fpaths, 'sharc', natom)
    [Nat, _, Ntraj, Nts] = np.shape(trajs)
    traj_count = calc_traj_time(run_times, Ntraj)

    Wiam = calculate_scattering(trajs, traj_count, qAng, FLAGelec)
    ref_signal = Wiam[:, 0]
    pdw = percentage_diff_signal(Wiam, exfrac, ref_signal, Nts, Nq, FLAGelec)

    time = np.linspace(0, (Nts-1)*dt, Nts)
    #plot_signal(pdw, qAng, time)

    if FLAGsave:
        pass # add some functionality to save to mat, npy or text files for experimentalists
        # save pdw - dims Nq Nts
        # save qAng
        # save time
