y"""
Code to take non-trivial p(z|d) data and infer dN/dz..
"""

import numpy as np

trueamps = np.array([0.20, 0.35, 0.55])
trueamps /= np.sum(trueamps)
cumamps = np.cumsum(trueamps)
truemeans = np.array([0.5, 0.2, 0.75])
truesigmas = np.array([0.4, 0.2, 0.1])

def make_one_ztrue():
    ztrue = -1.
    while ztrue < 0. or ztrue > 1.:
        r = np.random.uniform(0., cumamps[-1])
        c = 0
        for k in range(1, len(cumamps)):
            if r > cumamps[k-1]:
                c = k
        ztrue = truemeans[c] + truesigmas[c] * np.random.normal()
        print(c, ztrue)
    return ztrue

def sample_true_prior(N):
    print("Making ztrue values...")
    ztrues = np.array([make_one_ztrue() for n in range(N)])
    print(ztrues)
    return ztrues

def evaluate_true_prior(zs):
    """
    Doesn't return properly normalized fuunction.
    """
    ps = np.zeros_like(zs)
    for c in range(len(trueamps)):
        ps += trueamps[c] / (np.sqrt(2. * np.pi) * truesigmas[c]) * \
            np.exp(-0.5 * (zs - truemeans[c]) ** 2 / truesigmas[c] ** 2)
    ps[zs < 0.] = 0.
    ps[zs > 1.] = 0.
    return ps

true_lf_sigma = 0.05
true_lf_outlier_fraction = 0.2
true_lf_outlier_mean = 0.4
true_lf_outlier_sigma = 0.1

def sample_likelihood(ztrue):
    if np.random.uniform() < true_lf_outlier_fraction:
        return true_lf_outlier_mean + true_lf_outlier_sigma * np.random.normal()
    return ztrue + true_lf_sigma * np.random.normal()

def evaluate_likelihood(zobs, ztrues):
    """
    assumes zobs is a scalar and ztrues is a vector.
    """
    ps = np.zeros_like(ztrues)
    ps += true_lf_outlier_fraction / (np.sqrt(2. * np.pi) * true_lf_outlier_sigma) * \
        np.exp(-0.5 * (zobs - true_lf_outlier_mean) ** 2 / true_lf_outlier_sigma ** 2)
    ps += (1. - true_lf_outlier_fraction) / (np.sqrt(2. * np.pi) * true_lf_sigma) * \
        np.exp(-0.5 * (zobs - ztrues) ** 2 / true_lf_sigma ** 2)
    return ps

def evaluate_interim_prior(ztrues):
    ps = np.ones_like(ztrues)
    ps[ztrues < 0.] = 0.
    ps[ztrues > 1.] = 0.
    return ps

K = 16
binfactor = 16
KK = K * binfactor # number of fine bins
dzfine = 1. / KK
zfine = np.arange(0.5 * dzfine, 1.0, dzfine)
interimfine = evaluate_interim_prior(zfine)
dzcoarse = 1. / K
zcoarse = np.arange(0.5 * dzcoarse, 1.0, dzcoarse)
interim = evaluate_interim_prior(zcoarse)

def get_binned_interim_posterior(zobs):
    """
    stupidly slow, but only called once.
    """
    ps = interimfine * evaluate_likelihood(zobs, zfine)
    ps /= np.sum(ps) * dzfine
    coarseps = np.array([np.sum(ps[k * binfactor : (k + 1) * binfactor]) * dzfine for k in range(K)])
    return coarseps / dzcoarse

def get_rectangular_data(zobss):
    ps = np.zeros((len(zobss), K))
    for n,zobs in enumerate(zobss):
        ps[n] = get_binned_interim_posterior(zobs)
    return ps

def hyper_lfs(data, lnamps):
    amps = np.exp(lnamps - np.max(lnamps))
    amps /= np.sum(amps)
    print(data.shape)
    print(lnamps.shape)
    print(amps.shape)
    print(interim.shape)
    return np.sum(amps[None,:] * data / interim[None,:], axis=1)

def hyper_lnlf(data, lnamps):
    return np.sum(np.log(hyper_lfs(data, lnamps)))

def hyper_lnprior(lnamps):
    """
    Stupid Gaussian prior.
    """
    return -0.5 * np.sum((lnamps - 0.) ** 2 / 1.)

def hyper_lnposterior(lnamps, data):
    return hyper_lnprior(lnamps) + hyper_lnlf(data, lnamps)

if __name__ == "__main__":
    import matplotlib.pylab as plt

    N = 10000
    ztrues = sample_true_prior(N)
    nplot = 1000
    zplot = np.arange(0.5 / nplot, 1.0, 1.0 / nplot)
    plt.clf()
    plt.hist(ztrues, bins=100, normed=True, color="k")
    plt.plot(zplot, evaluate_true_prior(zplot), "r-")
    plt.xlabel("ztrue")
    plt.savefig("ztrues.png")

    zobss = np.array([sample_likelihood(ztrue) for ztrue in ztrues])
    data = get_rectangular_data(zobss)
    plt.clf()
    plt.plot(ztrues, zobss, "k.", alpha=0.5)
    plt.xlabel("ztrue")
    plt.ylabel("zobs")
    plt.savefig("scatter.png")

    plt.clf()
    plt.hist(zobss, bins=100, normed=True, color="k")
    plt.plot(zplot, evaluate_true_prior(zplot), "r-")
    plt.plot(zcoarse, np.sum(data, axis=0) / N, "go")
    plt.xlabel("z")
    plt.savefig("zobs.png")

    for n,zobs in enumerate(zobss[:10]):
        plt.clf()
        plt.plot(zplot, evaluate_likelihood(zobs, zplot), "k-")
        plt.plot(zcoarse, get_binned_interim_posterior(zobs), "ko")
        plt.axvline(zobs, color="k")
        plt.xlabel("ztrue")
        plt.xlim(0., 1.)
        plt.title("likelihood function for object {}".format(n))
        plt.savefig("lf_{:02d}.png".format(n))

    guess = np.log(evaluate_true_prior(zcoarse))
    print("hyperposterior at truth:", hyper_lnposterior(guess, data))
    guess = np.log(interim)
    print("hyperposterior at interim prior:", hyper_lnposterior(guess, data))
