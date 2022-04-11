import numpy as np


def form_factors_xrs(atmnum, qAng):

    DEBUG = False

    Nat = len(atmnum)
    Nq = len(qAng)

    # dict parameterised using: International Tables for Crystallography, Volume C, section 6.1 (p. 476)
    # add atoms that are needed here - key corresponds to atomic number of that atom
    atoms = {1: {'a': [0.493002, 0.322912, 0.140191,  0.040810], 'b': [10.5109,  26.1257,   3.14236,  57.7997], 'c': 0.003038},
             6: {'a': [2.26069, 1.56165,  1.05075, 0.839259], 'b': [22.6907, 0.656665, 9.75618, 55.5949], 'c': 0.286977},
             16: {'a': [6.90530, 5.20340, 1.43790, 1.58630], 'b': [1.46790, 22.2151, 0.253600, 56.1720], 'c': 0.866900}

             }

    fq = np.zeros((Nat, Nq))

    for i, num in enumerate(atmnum):
        try:
            tmp = np.zeros(Nq)
            for j in range(4):
                tmp = tmp + atoms[num]['a'][j] * np.exp(-atoms[num]['b'][j] * (qAng / (4 * np.pi))**2)
            fq[i, :] = atoms[num]['c'] + tmp

        except KeyError:
            raise Exception('Atomic Number %s not parameterised - edit atoms dict using ITC data' % num)

    if DEBUG:
        plot_form_factor(qAng, fq)

    return fq



def form_factors_elec(atmnum, qAng):
    pass


def scattering_factors(qAng, atmnum, FLAGelec):

    Nat = len(atmnum)
    Nq = len(qAng)

    if FLAGelec:
        #fq = form_factors_elec(atmnum, qAng) # to be parameterised in a function using Mott-Bethe
        raise Exception("Electron Scattering Not Possible")
    else:
        fq = form_factors_xrs(atmnum, qAng) # get form factors

    FF = np.zeros((Nat, Nat, Nq)) # calculate form factor product
    for i in range(Nat):
        for j in range(Nat):
            FF[i, j, :] = fq[i, :] * fq[j, :]

    return FF, fq


def plot_form_factor(qAng, fq):
    import matplotlib.pyplot as plt
    plt.plot(qAng, np.transpose(fq))
    plt.show()