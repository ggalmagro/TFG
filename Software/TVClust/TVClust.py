import numpy as np
import math as m
import scipy.special as sps


def logB(W, v):
    p = np.shape(W)[0]
    result = (-v/2) * m.log(np.linalg.det(W))
    tmp = (v * p / 2) * m.log(2) + (p * (p - 1) / 4) * m.log(m.pi) + np.sum(sps.gammaln((v + 1 - np.array(range(1,p+1)) ) / 2 ) )
    return result - tmp


def H(W, v):
    p = np.shape(W)[0]
    E = np.sum(sps.digamma( (v+1 - np.array(range(1, p+1))) / 2 ) ) + p * m.log(2) + m.log( np.linalg.det(W) )
    return -logB(W,v) - (v-p-1)/2 * E + v * p / 2


def TVClust( x, SM, C, K, iterN = 100, isKeepL = 1, alpha0 = 1.2, stopThreshold=0.0005 ):

    INFINITY = 10 ** 300

    n, p = np.shape(x)

    meanX = np.mean(x, 0)
    x = x - meanX
    sdX = np.std(x, 0, ddof=1)
    sdX[sdX == 0.0] = 0.000000000001
    x = x / sdX

    # hyper parameters
    mu0 = np.zeros(p)
    beta0 = 1
    W0 = 1 * np.identity(p)
    nu0 = p
    alpha_p_0 = 1
    beta_p_0 = 10
    alpha_q_0 = 10
    beta_q_0 = 1

    rQ = np.ones((n, K)) / K
    gammaQ = np.ones((K - 1, 2))
    betaQ = np.repeat(beta0, K)
    muQ = np.random.randn(p, 1, K)
    muQ = np.reshape(muQ, (K, p, 1))
    WQ = np.zeros((K, p, p))

    for k in range(K):
        WQ[k,:,:] = W0
    nuQ = np.repeat(nu0, K)
    alpha_p = 1
    beta_p = 10
    alpha_q = 10
    beta_q = 1
    pImprove = 1

    L = np.zeros(iterN)
    nRun = iterN
    iter = 0

    while iter < iterN:
        for i in range(n):
            r_i = np.zeros(K)
            for k in range(K):
                E_ln_lambda_k = np.sum(sps.digamma((nuQ[k] + 1 - np.array(range(1, p+1), dtype=np.float) ) / 2 ) ) + m.log(np.linalg.det(WQ[k, :, :] ) )
                E_sqr = p / betaQ[k] + nuQ[k] * np.dot(np.dot((x[i, :] - muQ[k, :, :].T), WQ[k, :, :]), (x[i,] - muQ[k, :, :].T).T)

                if k < K-1:
                    tmp = E_ln_lambda_k / 2 - E_sqr / 2 + sps.digamma(gammaQ[k, 0]) - sps.digamma(np.sum(gammaQ[k, :]))
                if k == K-1:
                    tmp = E_ln_lambda_k / 2 - E_sqr / 2
                if k > 0:
                    for l in range(k):
                        tmp = tmp + sps.digamma(gammaQ[l, 1]) - sps.digamma(np.sum(gammaQ[l, ]))

                for j in range(n):
                    if C[i, j] == 1:
                        tmpj = 0
                        tmpj = tmpj + SM[i, j] * (sps.digamma(alpha_p) - sps.digamma(alpha_p + beta_p)\
                            - sps.digamma(beta_q) + sps.digamma(alpha_q + beta_q)) - (1 - SM[i, j]) * \
                            (sps.digamma(alpha_q) - sps.digamma(alpha_q + beta_q) - sps.digamma(beta_p)\
                             + sps.digamma(alpha_p + beta_p))

                        tmp = tmp + rQ[j, k] * tmpj

                r_i[k] = np.exp(tmp)
                if m.isinf(r_i[k]):
                    r_i[k] = INFINITY

            r_i += + 10 ** (-100)
            rQ[i, ] = r_i / np.sum(r_i)

        N = np.sum(np.array(rQ), 0)
        for k in range(K-1):
            gammaQ[k, 0] = 1 + N[k]
            gammaQ[k, 1] = alpha0 + sum(N[(k + 1):K])

        rQ = np.matrix(rQ)
        for k in range(K):
            x_bar_k = np.sum(np.multiply(x, rQ[: ,k]), 0) / N[k]
            S_k = np.dot((x - x_bar_k).T, np.multiply(x - x_bar_k, rQ[:, k])) / N[k]
            betaQ[k] = beta0 + N[k]
            muQ[k, :, :] = ((beta0 * mu0 + N[k] * x_bar_k) / betaQ[k]).T
            tmp = np.dot(((x_bar_k - mu0).T), (x_bar_k - mu0)) * beta0 * N[k] / (beta0 + N[k])
            WQ[k, :, :] = np.linalg.inv(tmp + N[k] * S_k + np.linalg.inv(W0))
            nuQ[k] = nu0 + N[k]

        tmpRQ = np.dot(rQ, rQ.T)
        alpha_p = alpha_p_0 + np.sum(np.multiply(np.multiply(SM, C), tmpRQ))
        beta_p = beta_p_0 + np.sum(np.multiply(np.multiply(1 - SM, C), tmpRQ))
        alpha_q = alpha_q_0 + np.sum(np.multiply(np.multiply(1 - SM, C), 1 - tmpRQ))
        beta_q = beta_q_0 + np.sum(np.multiply(np.multiply(SM, C), 1 - tmpRQ))

        rQ = rQ + 10**(-40)  # prevent log(0)
        rQ = rQ / np.matrix(np.sum(np.array(rQ), 1)).T
        term75 = np.sum(np.multiply(rQ, np.log(rQ)))

        if term75 > -0.1:
            nRun = iter
            iter = iterN + 1

        if (isKeepL == 1) & (iter <= iterN):
            term71 = 0
            term72 = 0
            term73 = 0
            term74 = 0
            term76 = 0
            term77 = 0

            for k in range(K):
                if N[k] > 10**(-10):
                    E_ln_lambda_k = np.sum(sps.digamma((nuQ[k] + 1 - np.array(range(1, p + 1), dtype=np.float)) / 2)) + m.log(np.linalg.det(WQ[k, :, :]))
                    x_bar_k = np.sum(np.multiply(x, rQ[:, k]), 0) / N[k]
                    S_k = np.dot((x - x_bar_k).T, np.multiply(x - x_bar_k, rQ[:, k])) / N[k]
                    term71 = term71 + 0.5 * N[k] * (E_ln_lambda_k - p / betaQ[k] - nuQ[k] * np.sum(np.diag(np.dot(S_k, WQ[k, :, :]) ) ) - nuQ[k] * np.dot(np.dot((x_bar_k - muQ[k, :, :].T), WQ[k, :, :]), ((x_bar_k - muQ[k, :, :].T)).T))

                if k < K - 1:
                    tmp = N[k] * (sps.digamma(gammaQ[k, 0]) - sps.digamma(np.sum(gammaQ[k, :])))
                    for j in range(k+1, K):
                        tmp = tmp + N[j] * (sps.digamma(gammaQ[k, 1]) - sps.digamma(np.sum(gammaQ[k, :])))

                    term72 = term72 + tmp
                    term73 = term73 + (alpha0 - 1) * (sps.digamma(gammaQ[k, 1]) - sps.digamma(np.sum(gammaQ[k, :])))


                tmp2 = (-p * beta0 / betaQ[k] - beta0 * nuQ[k] * np.dot(np.dot((muQ[k, :, :].T - mu0), WQ[k, :, :]), (((muQ[k, :,:].T - mu0).T) ))) / 2
                tmp3 = nuQ[k] * sum(np.diag(np.dot(np.linalg.inv(W0), WQ[k, :, :]))) / 2
                term74 = term74 + E_ln_lambda_k * (nu0 - p) / 2 + tmp2 - tmp3

                if k < K - 1:
                    term76 = term76 + (gammaQ[k, 0] - 1) * (sps.digamma(gammaQ[k, 0]) - sps.digamma(np.sum(gammaQ[k, :]))) + (gammaQ[k, 1] - 1) * (sps.digamma(gammaQ[k, 1]) - sps.digamma(np.sum(gammaQ[k, :]))) - sps.betaln(gammaQ[k, 0], gammaQ[k, 1])

                term77 = term77 + E_ln_lambda_k / 2 + p / 2 * m.log(betaQ[k]) - H(WQ[k, :, :], nuQ[k])

            #Los valores de las cuatro siguientes variables no coinciden con R
            ElnP = sps.digamma(alpha_p) - sps.digamma(alpha_p + beta_p)
            ElnMinusP = sps.digamma(beta_p) - sps.digamma(alpha_p + beta_p)
            ElnQ = sps.digamma(alpha_q) - sps.digamma(alpha_q + beta_q)
            ElnMinusQ = sps.digamma(beta_q) - sps.digamma(alpha_q + beta_q)
            tmpRQ = np.dot(rQ, rQ.T)
            termE_lnPez = np.sum(np.multiply(np.multiply(SM, C), tmpRQ)) * ElnP + np.sum(np.multiply(np.multiply(1 - SM, C), tmpRQ)) * ElnMinusP + np.sum(np.multiply(np.multiply(SM, C), 1 - tmpRQ)) * ElnMinusQ + np.sum(np.multiply(np.multiply(1 - SM, C), 1 - tmpRQ)) * ElnQ
            termE_ln_pq = (alpha_p_0 - alpha_p) * ElnP + (beta_p_0 - beta_p) * ElnMinusP + (alpha_q_0 - alpha_q) * ElnQ + (beta_q_0 - beta_q) * ElnMinusQ + sps.betaln(alpha_p, beta_p) + sps.betaln(alpha_q, beta_q)

            L[iter] = term71 + term72 + term73 + term74 - term75 - term76 - term77 + termE_lnPez + termE_ln_pq

            pImprove = 1
            if iter > 0:
                pImprove = (L[iter] - L[iter - 1]) / np.absolute(L[iter - 1])
        print("Iter: " + str(iter) + " L: " + str(L[iter]) + " improvement: " + str(pImprove*100))
        if pImprove < stopThreshold:
            nRun = iter
            iter = iterN + 1

        iter += 1

    membership_vector = np.array([np.argmax(rQ[i, :]) for i in range(np.shape(rQ)[0])], dtype=np.uint8)
    return membership_vector
