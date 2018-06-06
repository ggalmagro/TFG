def gen_rand_const(X, y, mat_const, nb_const, noise = 0, 
	prop = False)
def LCVQE(X, K, constraints, max_iter = 300, 
	centroids = None)
def TVClust(X, K, constraints, max_iter = 300, 
	is_keep_l = 1, alpha0 = 1.2, stop_thr = 0.0005)

def CEKM(X, K, constraints, max_iter = 300, alpha = 1, 
	rho = 100, xi = 0.5, stop_thr = 1e-3, init = 'rand')

def COPKM(X, K, constraints, max_iter = 300,  tol = 1e-4,
	init = 'rand')

def RDPM(X, lamb, constraints, max_iter = 300, xi_0 = 0.1, 
	xi_rate = 1)