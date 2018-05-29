import numpy as np
import scipy.sparse
import math
import gc


def solqp(Q, A, b, c, x, toler = 1.e-5, beta = 0.95, alpha = 0.95):

    m, n = np.shape(A)
    if np.shape(x)[0] == 0:
        a = b - np.dot(A, np.ones((n,1)))
        x = np.ones((n+1,1))
        z = 0
        ob = x[n]
        obhis = [ob]
        gap = ob-z

        while gap >= toler:
            ################### SSPHASE1 ###################
            dx = np.ones((n,1))/x[:n]
            DD = scipy.sparse.spdiags(np.multiply(dx,dx).T, 0, n, n)
            DD = DD.todense()
            aux_row1 = np.concatenate((DD, A.conj().T), axis=1)
            aux_row2 = np.concatenate((A, np.zeros((m,m))), axis=1)
            aux_matrix1 = np.concatenate((aux_row1, aux_row2))
            aux_row1 = np.concatenate((dx, np.zeros((n,1))), axis=1)
            aux_row2 = np.concatenate((np.zeros((m,1)), a), axis=1)
            aux_matrix2 = np.concatenate((aux_row1, aux_row2))
            aux_matrix3 = np.linalg.solve(aux_matrix1, aux_matrix2)

            y1 = np.matrix(aux_matrix3[n:n+m, 0]).reshape(30,1)
            y2 = np.matrix(aux_matrix3[n:n+m, 1]).reshape(30,1)
            w1 = ((1/ob - np.dot(a.conj().T, y1))/(1/ob**2 - np.dot(a.conj().T, y2)))[0,0]
            w2 = 1/(1/ob**2 - np.dot(a.conj().T, y2))[0,0]

            y1 = y1 - w1 * y2
            y2 = -w2 * y2

            w1 = np.dot(b.conj().T, y1)[0,0]
            w2 = np.dot(b.conj().T, y2)[0,0]
            y1 = y1/(1+w1)
            y2 = y2 - w2*y1

            u_row1 = np.multiply(x[:n], np.dot(-y2.conj().T, A).conj().T)
            u_row2 = x[n] * 1 - np.dot(y2.conj().T, a)
            u_row3 = np.matrix([w2/(1+w1)])

            u = np.concatenate((u_row1, u_row2))
            u = np.concatenate((u, u_row3))

            v_row1 = np.multiply(x[:n], np.dot(y1.conj().T, A).conj().T)
            v_row2 = x[n] * np.dot(y1.conj().T, a)
            v_row3 = np.matrix([1 / (1 + w1)])

            v = np.concatenate((v_row1, v_row2))
            v = np.concatenate((v, v_row3))

            if np.min(u - z*v) >= 0:
                y = y2 + z*y1
                z = int(np.dot(b.conj().T, y))

            u = u - z * v - int((ob-z)/(n+2))*np.ones((n+2,1))
            nora = np.max(u)

            if nora == u[n+1]:
                alpha = 1

            v = np.ones((n+2, 1)) - (alpha/nora)*u
            x = np.multiply(x, v[:n+1])/v[n+1]
            ################## ################### ###################

            ob = x[n]
            obhis = np.concatenate((obhis, ob))
            gap = ob - z

            if z>0:
                print("The sistem has no feasible solution")
                return x, y, obhis

    else:
        ob = 0.5 * (np.dot(np.dot(x.conj().T, Q), x)) + np.dot(c.conj().T, x)

    alpha = 0.9
    x = x[0:n]
    comp = np.random.rand(n, 1)
    aux_row1 = np.concatenate((np.identity(n), A.conj().T), 1)
    aux_row2 = np.concatenate((A, np.zeros((m,m))), 1)
    aux_matrix1 = np.concatenate((aux_row1, aux_row2))
    aux_matrix2 = np.concatenate((comp, np.zeros((m,1))))
    solve = np.linalg.solve(aux_matrix1, aux_matrix2)
    comp = solve[0:n]

    aux_row1 = []
    aux_row2 = []
    aux_matrix1 = []
    aux_matrix2 = []
    gc.collect(2)

    nora = np.min(comp / x)

    if nora<0:
        nora = -0.01/nora
    else:
        nora = np.max(comp / x)

        if nora == 0:
            print("The problem has a unique feasible point")
            return x, y, obhis

        nora = 0.01/nora

    x = x + nora*comp

    obvalue = np.dot(x.conj().T, np.dot(Q,x))/2 + np.dot(c.conj().T, x)
    obhis = [obvalue]
    lower = float("-inf")
    zhis = [lower]
    gap = 1
    lamda = np.max(np.concatenate((np.matrix([1]), np.abs(obvalue)/np.sqrt(np.sqrt(n))), 1))
    iter = 0

    while gap >= toler:

        ################### SSPHASE2 ###################
        lamda = (1-beta)*lamda
        go = 0
        gg = np.dot(Q,x)+c
        XX = scipy.sparse.spdiags(x.T, 0, n, n)
        XX = XX.todense()
        AA = np.dot(A, XX)
        XX = np.dot(np.dot(XX, Q), XX)


        while go <=0:
            aux_row1 = XX + lamda * scipy.sparse.csr_matrix(np.identity(n))
            aux_row1 = np.concatenate((aux_row1, AA.conj().T), 1)
            aux_row2 = np.concatenate((AA, np.zeros((m, m))), 1)
            aux_matrix1 = np.concatenate((aux_row1, aux_row2))
            aux_matrix2 = np.concatenate(( np.array(-x) * np.array(gg), np.zeros((m, 1))))
            u = np.linalg.solve(aux_matrix1, aux_matrix2)
            xx = x + np.array(x) * np.array(u[:n])
            go = np.min(xx)

            if go > 0: ##########
                ob = np.dot(np.dot(xx.conj().T, Q), xx)/2 + np.dot(c.conj().T, xx)
                go = np.min([go, obvalue-ob+2.2204e-16])

            lamda = 2*lamda

            if lamda >= (1+np.absolute(obvalue))/toler:
                y = -u[n+1:n+m]
                return x, y, obhis

        y = -u[n:n + m]
        u = u[:n]
        nora = np.min(u)
        if nora < 0:
            nora = -alpha/nora

        elif nora == 0:
            nora = alpha
        else:
            nora = float("-inf")

        u = np.array(x) * (u)
        w1 = np.dot(np.dot(u.conj().T, Q), u)
        w2 = np.dot((-u).conj().T, gg)

        if w1 > 0:
            nora = min(w1[0,0]/w2[0,0], nora)
        else:
            x = x + nora*u
            ob = np.dot(np.dot(x.conj().T, Q), x) / 2 + np.dot(c.conj().T, x)

        ################## ################### ###################

        if(math.isinf(ob) and ob < 0):
            gap = 0
            print("The problem is unbounded")
            return x, y, obhis
        else:
            obhis = np.append(obhis, ob)
            comp = np.dot(Q, x) + c - np.dot(A.conj().T, y)
            if np.min(comp) >= 0:
                zhis = np.append(zhis, ob - np.dot(x.conj().T, comp))
                lower = zhis[iter+1]
                gap = (ob- lower)/(1+abs(ob))
                obvalue = ob
            else:
                zhis = np.append(zhis, zhis[iter])
                lower = zhis[iter+1]
                gap = (obvalue-ob)/(1+abs(ob))
                obvalue = ob
        if iter > 200:
            print ("gap = %f, toler = %f", gap, toler)

        iter = iter + 1

    return x, y, obhis


