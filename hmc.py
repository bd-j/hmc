import numpy as np
import matplotlib.pyplot as pl
from numpy import polyval


class BasicHMC(object):

    def __init__(self, verbose=True):
        self.verbose = verbose

    def sample(self, initial, model, iterations=1, length=2, epsilon=None,
               nadapt=0, store_trajectories=False):
        """Sample for `iterations` trajectories (i.e., compute that many
        trajectories, resampling the momenta at the end of each trajectory.
        """
        # set some initial values
        self.nadapt = nadapt
        if epsilon is None:
            epsilon = self.find_reasonable_epsilon(initial.copy(), model)
            print('using epsilon = {0}'.format(epsilon))
        self.reset()
        effective_length = epsilon * length
        self.mu = np.log(10 * epsilon)

        # set up the output
        self.chain = np.zeros([iterations, len(initial)])
        self.lnprob = np.zeros([iterations])
        self.accepted = np.zeros([iterations])
        if store_trajectories:
            self.trajectories = np.zeros([iterations, length, len(initial)])
        theta = initial.copy()
        self.traj_num = 0
        # loop over trajectories
        lnp, grad = None, None  # initial P and lnP are unknown
        for i in xrange(int(iterations)):
            #self.epsilon  = np.random.normal(1.0,1) * self.epsilon
            #epsilon = self.find_reasonable_epsilon(initial.copy(), model)

            if self.verbose:
                print('eps, L={0}, {1}'.format(epsilon, length))
            info = self.trajectory(theta, model, epsilon, length,
                                   lnP0=lnp, grad0=grad)
            theta, lnp, grad, epsilon = info
            self.lnprob[i] = info[1]
            self.chain[i, :] = info[0]
            self.traj_num += 1
        return theta, lnp, epsilon

    def trajectory(self, theta0, model, epsilon, length, lnP0=None, grad0=None):
        """Compute one trajectory for a given starting location,
        epsilon, and length.  The momenta in each direction are
        drawn from a gaussian before performing 'length' leapfrog
        steps.  If the trajectories attribute exists, store the
        path of the trajectory."""

        # Set up for the run
        # save initial position
        theta = theta0.copy()
        # random initial momenta
        p0 = np.random.normal(0, 1, len(theta0))
        if grad0 is None:
            # gradient in U at initial position, negative of gradient lnP
            grad0 = -model.lnprob_grad(theta0)
        if lnP0 is None:
            lnP0 = model.lnprob(theta0)
        # use initial gradient
        grad = grad0.copy()
        # use initial momenta
        p = p0.copy()

        # do 'length' leapfrog steps along the trajectory (and store?)
        for step in xrange(int(length)):
            theta, p, grad = self.leapfrog(theta, p, epsilon, grad, model,
                                           check_oob=hasattr(model, 'check_constrained'))
            try:
                self.trajectories[self.traj_num, step, :] = theta
            except:
                pass

        # Odds ratio of the proposed move
        lnP = model.lnprob(theta)
        dU = lnP0 - lnP  # change in potential = negative change in lnP
        dK = 0.5 * (np.dot(p, p.T) - np.dot(p0, p0.T))  # change in kinetic
        alpha = np.exp(-dU - dK)  # acceptance criterion
        if self.verbose:
            print('H={0}, dU={1}, dK={2}'.format(alpha, dU, dK))

        # Adapt epsilon?
        #if self.traj_num <= self.nadapt and length > 1:
        #    epsilon = self.adjust_epsilon(alpha)
        #    print(epsilon)
        #elif self.nadapt > 0:
        #    epsilon = np.exp(self.logepsbar)
        #    self.nadapt = 0
        # Accept or reject
        if np.random.uniform(0, 1) < alpha:
            self.accepted[self.traj_num] = 1
            return theta, lnP, grad, epsilon
        else:
            return theta0, lnP0, grad0, epsilon

    def leapfrog(self, theta, p, epsilon, grad, model, check_oob=False):
        """Perfrom one leapfrog step, updating the momentum and
        position vectors. This uses one call to the model.lnprob_grad()
        function, which must be defined. It also performs an optional
        check on the value of the new position to make sure it satistfies
        any parameter constraints, for which the check_constrained
        method of model is called.
        """

        # half step in p
        p -= 0.5 * epsilon * grad
        # full step in theta
        theta += epsilon * p
        # check for constraints on theta
        while check_oob:
            theta, sign, check_oob = model.check_constrained(theta)
            p *= sign  # flip the momentum if necessary
        # compute new gradient in U, which is negative of gradient in lnP
        grad = -model.lnprob_grad(theta)
        # another half step in p
        p -= 0.5 * epsilon * grad
        return theta, p, grad

    def langevin(self):
        """Special case of length = 1 trajectories"""
        pass

    def find_reasonable_epsilon(self, theta0, model, epsilon_guess=1):
        epsilon = epsilon_guess
        lnP0, grad0 = model.lnprob(theta0), model.lnprob_grad(theta0)
        p0 = np.random.normal(0, 1, len(theta0))
        condition, a = True, 0
        i = 0
        while condition:
            p = p0.copy()
            epsilon = 2.**a * epsilon
            thetaprime, pprime, gradprime = self.leapfrog(theta0.copy(), p, epsilon, grad0,
                                                          model, check_oob=True)
            lnP = model.lnprob(thetaprime)
            # change in potential = negative change in lnP
            dU = lnP0 - lnP
            # change in kinetic
            dK = 0.5 * (np.dot(pprime, pprime.T) - np.dot(p0, p0.T))
            alpha = np.exp(-dU - dK)
            if a is 0:  # this is the first try
                a = 2 * (alpha > 0.5) - 1.0  # direction to change epsilon in the future, + or -
            condition = (alpha**a) > (2**(-a))
            i += 1
            print(i, epsilon, alpha)
            if alpha is 0.0:
                raise ValueError('alpha is 0')
        return epsilon

    def reset(self):
        # use this to keep track of the trajectory number within the trajectory
        # (for storage)
        self.traj_num = 0
        self.H_t = 0
        self.logepsbar = 0
        self.delta = 0.65

    def adjust_epsilon(self, alpha, gamma=1.0, t0=10, kappa=0.75):
        t = self.traj_num + 1
        eta = 1/float(t + t0)
        self.H_t = (1 - eta) * self.H_t + eta * (self.delta - alpha)
        logeps = self.mu - np.sqrt(t)/gamma * self.H_t
        xi = t**(-kappa)
        self.logepsbar = xi * logeps + (1 - xi) * self.logepsbar
        return np.exp(logeps)


class TestModel(object):
    """A simple correlated normal distribution to sample.
    """

    def __init__(self):
        self.A = np.asarray([[50.251256, -24.874372],
                            [-24.874372, 12.562814]])
        self.has_constraints = False

    def lnprob_grad(self, theta):
        return -np.dot(theta, self.A)

    def lnprob(self, theta):
        return 0.5 * np.dot(self.lnprob_grad(theta), theta.T)


class MixModel(object):
    """A simple line in 2-d space (but constrained) to sample.
    """

    def __init__(self):
        self.A = np.array([10., 20.])
        # constraints
        self.lower = 0.
        self.upper = 10.

    def model(self, theta):
        # super simple model
        return (self.A * theta).sum()

    def lnprob(self, theta):
        # probability of that simple model given observations (which must be defined)
        return -0.5 * ((self.model(theta) - self.obs)**2 /
                       self.obs_unc**2).sum()

    def lnprob_grad(self, theta):
        # with simple gradients of the probbility
        grad = -(self.model(theta)-self.obs)/self.obs_unc**2 * self.A
        return grad

    def check_constrained(self, theta):
        """Method that checks the value of theta against constraints.
        If theta is above or below the boundaries, the sign of the momentum
        is flipped and theta is adjusted as if the trajectory had
        bounced off the constraint. Returns the new theta vector, a
        vector of multiplicative signs for the momenta, and a flag for
        if the values are still out of bounds.
        """

        # initially no flips
        sign = np.ones_like(theta)
        oob = True  # pretend we started out-of-bounds to force at least one check
        #print('theta_in ={0}'.format(theta))
        while oob:
            above = theta > self.upper
            theta[above] = 2*self.upper - theta[above]
            sign[above] *= -1
            below = theta < self.lower
            theta[below] = 2*self.lower - theta[below]
            sign[below] *= -1
            oob = np.any(below | above)
            #print('theta_out ={0}'.format(theta))
        return theta, sign, oob


def test_mix_hmc(epsilon=0.2, length=10, iterations=100, snr=10):
    """Sample the mixing model using hmc, and plot the results.
    """
    model = MixModel()
    D = 2

    #generate the mock
    mock_theta = np.random.uniform(1, 5, D)
    #print('mock_theta={0}'.format(mock_theta))
    mock = model.model(mock_theta)
    noised_mock = mock * (1 + np.random.normal(0, 1, 1) / snr)
    noise = mock/snr

    #add the mock to the model
    model.obs = noised_mock
    model.obs_unc = noise
    theta0 = np.random.uniform(0, 10, D)

    #initialize sampler and sample
    sampler = BasicHMC(verbose=False)
    pos, prob, eps = sampler.sample(theta0, model, iterations=iterations,
                                    epsilon=epsilon, length=length,
                                    store_trajectories=True)
    print mock_theta/(np.mean(pos, axis=0))
    print('mock_theta = {0}'.format(mock_theta))

    #plot trajectories
    pl.figure(1)
    pl.clf()
    color = ['red', 'blue']
    pl.plot(sampler.chain[::10, 0], sampler.chain[::10, 1], '.', label='Thinned samples')
    for it in np.arange(20) + int(iterations/3):
        pl.plot(sampler.trajectories[it, :, 0],
                sampler.trajectories[it, :, 1],
                color=color[int(sampler.accepted[it])])
    pl.plot(mock_theta[0], mock_theta[1], 'g.', markersize=20, label='Truth (noiseless)')
    pl.plot(theta0[0], theta0[1], 'c.', markersize=15, label='Initial')
    pl.legend(loc='upper right')
    pl.title(r'$Z = \theta^T x$, $\theta >0$, $\epsilon = ${0}, Length = {1}, $f_{{accept}} =$ {2}'.format(epsilon, length, sampler.accepted.sum()/iterations))
    pl.xlabel(r'$\theta_1$')
    pl.ylabel(r'$\theta_2$')

    pl.show()
    return sampler


def test_hmc(epsilon=0.1, length=10, iterations=100):
    """sample the correlated normal using hmc"""
    model = TestModel()
    D = 2
    theta0 = np.random.normal(0, 1, D)

    sampler = BasicHMC(model)
    pos, prob, eps = sampler.sample(theta0.copy(), model, iterations=iterations,
                                    epsilon=epsilon, length=length, nadapt=0)
    print (np.std(pos, axis=0))
    print (theta0)
    return sampler
