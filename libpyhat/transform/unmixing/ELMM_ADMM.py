from numpy import *
from scipy.optimize import nnls
import math

'''

Corresponds with ELMM_ADMM.m from the toolbox at following link:
https://openremotesensing.net/knowledgebase/spectral-variability-and-extended-linear-mixing-model/

   The algorithm is presented in detail in:

   L. Drumetz, M. A. Veganzones, S. Henrot, R. Phlypo, J. Chanussot and 
   C. Jutten, "Blind Hyperspectral Unmixing Using an Extended Linear
   Mixing Model to Address Spectral Variability," in IEEE Transactions on 
   Image Processing, vol. 25, no. 8, pp. 3890-3905, Aug. 2016.

'''
    
def elmm_admm( data, A_init, psis_init, S0, lambda_s, lambda_a, lambda_psi, **kwargs):
    '''
    Unmix hyperspectral data using the Extended Linear Mixing Model    

     Mandatory inputs:
    -data: m*n*L image cube, where m is the number of rows, n the number of
    columns, and L the number of spectral bands.
    -A_init: P*N initial abundance matrix, with P the number of endmembers
    to consider, and N the number of pixels (N=m*n)
    -psis_init: P*N initial scaling factor matrix
    -S0: L*P reference endmember matrix
    -lambda_s: regularization parameter on the ELMM tightness
    -lambda_a: regularization parameter for the spatial regularization on
    the abundances.
    -lambda_psi: regularization parameter for the spatial regularization on
    the scaling factors
    The spatial regularization parameters can be scalars, in which case 
    they will apply in the same way for all the terms of the concerned
    regularizations. If they are vectors, then each term of the sum
    (corresponding to each material) will be differently weighted by the
    different entries of the vector.

   Optional inputs (arguments are to be provided in the same order as in 
    the following list):
    -norm_sr: choose norm to use for the spatial regularization on the
    abundances. Can be '2,1' (Tikhonov like penalty on the gradient) or
    '1,1' (Total Variation) (default: '1,1')
    -verbose: flag for display in console. Display if true, no display
    otherwise (default: true)
    -maxiter_anls: maximum number of iterations for the ANLS loop (default:
    100)
    -maxiter_admm: maximum number of iterations for the ADMM loop (default:
    100)
    -epsilon_s: tolerance on the relative variation of S between two
    consecutive iterations (default: 10^(-3))
    -epsilon_a: tolerance on the relative variation of A between two
    consecutive iterations (default: 10^(-3))
    -epsilon_psi: tolerance on the relative variation of psi between two
    consecutive iterations (default: 10^(-3))
    -epsilon_admm_abs: tolerance on the absolute part of the primal and
    dual residuals (default: 10^(-2))
    -epsilon_admm_rel: tolerance on the relative part of the primal and
    dual residuals (default: 10^(-2))

   Outputs:
    -A: P*N abundance matrix
    -psi_maps: P*N scaling factor matrix
    -S: L*P*N tensor constaining all the endmember matrices for each pixel
    -optim_struct: structure containing the values of the objective
    function and its different terms at each iteration
    '''
 
    # set default values for optional parameters
    norm_sr = kwargs.get('norm_sr', '1,1')
    verbose = kwargs.get('verbose', True)
    maxiter_anls = kwargs.get('maxiter_anls', 100)
    maxiter_admm = kwargs.get('maxiter_admm', 100)
    epsilon_s = kwargs.get('epsilon_s', 10**(-3))
    epsilon_a = kwargs.get('epsilon_a', 10**(-3))
    epsilon_psi = kwargs.get('epsilon_psi', 10**(-3))
    epsilon_admm_abs = kwargs.get('epsilon_admm_abs', 10**(-2))
    epsilon_admm_rel = kwargs.get('epsilon_admm_rel', 10**(-2))


    P = A_init.shape[0]  # number of endmembers

    scalar_lambda_a = False
    scalar_lambda_psi = False
    
    if lambda_a.shape[1] == 1:
        scalar_lambda_a = True
    elif lambda_a.shape[1] == P:
        
        if lambda_a.shape[0] == 1:
            lambda_a = lamdba_a.transpose()
    else:
        raise ValueError('lambda_a must be a scalar or a P-dimensional vector')

    if lambda_psi.shape[1] == 1:
        scalar_lambda_psi = True
    elif lambda_psi.shape[1] == P:
        if lambda_psi.shape[0] == 1:
            lambda_psi = lamdba_psi.transpose()
    else:
        raise ValueError('lambda_psi must be a scalar or a P-dimensional vector')


    m, n, L = data.shape
    N = m*n

    # data_r = data.reshape((N, L)).transpose()
    data_r = data.copy().reshape((N, L) ,order='F').conj().T
   
    rs = zeros((maxiter_anls,1))
    ra = zeros((maxiter_anls,1))
    rpsi = zeros((maxiter_anls,1))

    A = A_init
    # MATLAB: S = repmat(S0,[1,1,N]);
    S = array([tile(S0,(1,1)) for i in range(N)]).T
    
    psi_maps = psis_init

    S0ptS0 = diag(S0.T@S0)
    S0ptS0 = S0ptS0[None,].T

    objective = zeros((maxiter_anls,1))
    norm_fitting = zeros((maxiter_anls,1))
    source_model = zeros((maxiter_anls,1))
    
    if scalar_lambda_a:
        TV_a = zeros((maxiter_anls,1))
    else:
        TV_a = zeros((maxiter_anls,P))

    if scalar_lambda_psi:
        smooth_psi = zeros((maxiter_anls,1))
    else:
        smooth_psi = zeros((maxiter_anls,P))

    # forward first order horizontal difference operator
    FDh = zeros((m,n))
    FDh[0, n-1] = -1
    FDh[m-1,n-1] = 1
    FDh = fft.fft2(FDh)
    FDhC = conj(FDh)

    # forward first order vertical  difference operator
    FDv = zeros((m,n))
    FDv[0, n-1] = -1
    FDv[m-1,n-1] = 1;
    FDv = fft.fft2(FDh)
    FDvC = conj(FDh)

    # barrier parameter of ADMM and related
    rho = zeros((maxiter_admm,1))
    rho[0] = 10
    tau_incr = 2
    tau_decr = 2
    nu = 10


    # EXPECTED BUG: matrix vs element multiplication
    for i in range(maxiter_anls):
        S_old = S.copy()
        psi_maps_old = psi_maps.copy()
        A_old_anls = A.copy()
        
        
        #S_update
        for k in range(N):
            first_op = data_r[:,k]@A[:,k].conj().T+(lambda_s*S0)@diag(psi_maps[:,k])
            second_op = A[:,k]@A[:,k].conj().T+lambda_s*eye(P)
            S[:,:,k] = dot(first_op, linalg.pinv(second_op))  # first_op / second_op
            S[:,:,k] = maximum(pow(10,-6), S[:,:,k])


        # A_update

        if any( lambda_a ):
            # initialize split variables
            v1 = A
            v1_im = conv2im(v1,m,n,P)
            v2 = ConvC(A,FDh,m,n,P)
            v3 = ConvC(A,FDv,m,n,P)
            v4 = A

            # initialize Lagrange multipliers
            d1 = zeros((P,N))
            d2 = zeros((v2.shape))
            d3 = zeros((v3.shape))
            d4 = zeros((psi_maps.shape))

            mu = zeros((1,N))

            # initialize primal and dual variables
            primal = zeros((maxiter_admm,1))
            dual = zeros((maxiter_admm,1))

            # precomputing
            Hvv1 = ConvC(v1,FDv,m,n,P)
            Hhv1 = ConvC(v1,FDh,m,n,P)

            for j in range(maxiter_admm):
                A_old = A
                v1_old = v1
                p_res2_old = v2 - Hhv1;
                p_res3_old = v3 - Hvv1
                v4_old = v4
                d1_old = d1
                d4_old = d4

                for k in range(N):
                    ALPHA = S[:,:,k].T@S[:,:,k]+2*rho[j]*eye(P)
                    ALPHA_INVERTED = linalg.inv(ALPHA)
                    BETA = ones((P,1))
                    s = ALPHA_INVERTED.sum(axis=0)
                    SEC_MEMBER = concatenate(( S[:,:,k].T@data_r[:,k] + rho[j]*(v1[:,k] + d1[:,k] + v4[:,k] + d4[:,k]), array([1])), axis=0)
                    OMEGA_a = concatenate((ALPHA_INVERTED@(eye(P)-1/s*ones((P,P))@ALPHA_INVERTED), 1/s * ALPHA_INVERTED * BETA), axis=None)
                    OMEGA_b = concatenate( (1/s*BETA.T*ALPHA_INVERTED, -1/s), axis=None)
                    print(OMEGA_a.shape)
                    print(OMEGA_b.shape)
                    OMEGA_INV = concatenate(( OMEGA_a, OMEGA_b), axis=0)
                    print(OMEGA_INV.shape)
                    print(SEC_MEMBER.shape)
                    X = OMEGA_INV @ SEC_MEMBER

                    A[:,k] = X[0:-2]
                    mu[k] = X[-1]

                A_im = conv2im(A,m,n,P)
                d1_im = conv2im(d1,m,n,P)
                d2_im = conv2im(d2,m,n,P)
                d3_im = conv2im(d3,m,n,P)
                v2_im = conv2im(v2,m,n,P)
                v3_im = conv2im(v3,m,n,P)

                # update in the Fourier domain

                for p in range(P):
                    sec_spectral_term = fft.fft2(squeeze(A_im[:,:,p]) - squeeze(d1_im[:,:,p])) + fft.fft2(squeeze((v2_im[:,:,p]+squeeze(d2_im[:,:,p])))*FDhC + fft.fft2(squeeze(v3_im[:,:,p]+squeeze(d3_im[:,:,p]))))*FDvC
                    v1_im[:,:,p] = dot(real(ftt.ifft2((sec_spectral_term), linalg.pinv(ones((m,n)) + abs(FDh)**2 + abs(FDv)**2))))


                # convert back necessary variables into matrices

                v1 = conv2mat(v1_im)
                Hvv1 = ConvC(v1,FDv)
                Hhv1 = ConvC(v1, FDh)


                # min w.r.t. v2 and v3

                if scalar_lambda_a:
                    if norm_sr == '2,1':
                        v2 = vector_soft_col( -(d2-Hhv1), lambda_a/rho[j])
                        v3 = vector_soft_col( -(d3-Hvv1), lambda_a/rho[j])
                    elif norm_sr == '1,1':
                        v2 = soft( -(d2-Hhv1), lambda_a/rho[j])
                        v3 = soft( -(d3-Hvv1), lambda_a/rho[j])
                else:
                    if norm_sr == '2,1':
                        for p in range(P):
                            v2[p,:] = vector_soft_col(-(d2[p,:] - Hhv1[p,:]), lambda_a[p]/rho[j])
                            v3[p,:] = vector_soft_col(-(d3[p,:] - Hvv1[p,:]), lambda_a[p]/rho[j])
                    elif norm_sr == '1,1':
                            v2[p,:] = soft(-(d2[p,:] - Hhv1[p,:]), lambda_a[p]/rho[j])
                            v3[p,:] = soft(-(d3[p,:] - Hvv1[p,:]), lambda_a[p]/rho[j])

                            
                # min w.r.t. v4

                v4 = max(A-d4, zeros(A.shape))


                # dual update
                # compute necessary variables for the residuals and update lagrange multipliers
                p_res1 = v1 - A;
                p_res2 = v2 - Hhv1;
                p_res3 = v3 - Hvv1;
                p_res4 = v4 - A;
                
                d1 = d1 + p_res1;
                d2 = d2 + p_res2;
                d3 = d3 + p_res3;
                d4 = d4 + p_res4;


                # primal and dual residuals

                primal[j] = math.sqrt( linalg.norm(p_res1, 'fro')**2 + linalg.norm(p_res2, 'fro')**2 + linalg.norm(p_res3, 'fro')**2 + linalg.norm(p_res4, 'fro')**2 )
                dual[j] = rho[j] * math.sqrt( linalg.norm(v1_old-v1,'fro')**2 + linalg.norm(v4_old-v4,'fro')**2 )

                # compute termination values

                epsilon_primal = math.sqrt(4*P*N) * epsilon_admm_abs + epsilon_admm_rel*max(math.sqrt(2*linalg.norm(A,'fro')**2), math.sqrt(linalg.norm(v1_old,'fro')**2 + linalg.norm(p_res2_old,'fro')**2 + linalg.norm(p_res3_old,'fro')**2 + linalg.norm(v4_old,'fro')**2))
                epsilon_dual = math.sqrt(P*N)*epsilon_admm_abs + rho[j] * epsilon_admm_rel * math.sqrt(linalg.norm(d1_old,'fro')+linalg.norm(d4_old,'fro')**2)

                rel_A = dot(abs(linalg.norm(A,'fro')-linalg.norm(A_old,'fro')), linalg.pinv(linalg.norm(A_old,'fro')))


                # display of admm results

                if verbose:
                    print(f'iter {j}, rel_A = {rel_A}, primal = {primal[j]}, eps_p = {epsilon_primal}, dual = {dual[j]}, eps_d = {epsilon_dual}, rho = {rho[j]}')

                if j > 1 and ((primal[j] < epsilon_primal and dual[j] < epsilon_dual)):
                    break


                # rho update

                if j < maxiter_admm:
                    if norm(primal[j]) > nu*norm(dual[j]):
                        rho[j+1] = tau_incr*rho[j]
                        A = A/tau_incr
                    elif norm(dual[j]) < nu*norm(primal[j]):
                        rho[j+1] = rho[j]/tau_decr
                        A = tau_decr * A
                    else:
                        rho[j+1] = rho[j]

            # end for loop

        else:
            # without spatial regularization
            for k in range(N):
                A[:,k] = FCLSU(data_r[:,k],S[:,:,k])

            if verbose:
                print("Done")
                print("updating psi..")

            # psi_update

            if any(lambda_psi):
                # with spatial regularization
                if scalar_lambda_psi:
                    for p in range(P):
                        numerator = 0 # TODO
                        psi_maps_im = real(fft.ifft2(fft.fft2(numerator)/((lambda_psi*(abs(FDh)**2+abs(FDv)**2)+lambda_s*S0ptS0[p]))))
                        psi_maps[p,:] = psi_maps_im[:]

                else:
                    for p in range(P):
                        numerator = 0 # TODO (translate from matlab)
                        psi_maps_im = real(fft.ifft2(fft.fft2(numerator)/((lambda_psi[p]*(abs(FDh)**2+abs(FDv)**2)+lambda_s*S0ptS0[p]))))
                        psi_maps[p,:] = psi_maps_im[:]
            else:
                for p in range(P):
                    psi_maps_temp = zeros((N,1))
                    for k in range(N):
                        psi_maps_temp[k] = (S0[:,p].T@S[:,p,k])/S0ptS0[p]
                        
                    psi_maps[p,:] = psi_maps_temp.flatten()

            if verbose:
                print("Done")

                
            # residuals of the ANLS loops
            rs_vect = zeros((N,1))

            for k in range(N):
                rs_vect[k] = linalg.norm(squeeze(S[:,:,k])-squeeze(S_old[:,:,k]),'fro')/linalg.norm(squeeze(S_old[:,:,k]),'fro')

            rs[i] = rs_vect.mean(axis=0)
            ra[i] = linalg.norm(A[:]-A_old_anls[:],2)/linalg.norm(A_old_anls[:],2)
            rpsi[i] = linalg.norm(psi_maps-psi_maps_old,'fro')/(linalg.norm(psi_maps_old,'fro'))

            # compute objective function value

            SkAk = zeros((L,N))
            S0_psi = ndarray((L,P,N))  # S0_psi initializes automatically in matlab in forloop, manually here
            for k in range(N):
                SkAk[:,k] = squeeze(S[:,:,k]@A[:,k])
                S0_psi[:,:,k] = S0*diag(psi_maps[:,k])

            norm_fitting[i] = 1/2*linalg.norm(data_r[:]-SkAk[:])**2

            source_model[i] = 1/2*linalg.norm(S[:]-S0_psi[:])**2

            if any(lambda_psi) and any(lambda_a):  # different objective functions depending on the chosen regularizations
                if scalar_lambda_psi:
                    smooth_psi[i] = 1/2*(sum(sum((ConvC(psi_maps,FDh,m,n,P)**2))) + sum(sum((ConvC(psi_maps,FDv,m,n,P)**2))))
                else:
                    CvCpsih = ConvC(psi_maps,FDh,m,n,P)
                    CvCpsiv = ConvC(psi_maps,FDv,m,n,P)
                    for p in range(P):
                        smooth_psi[i,p] = 1/2*(sum(sum((CvCpsih[p,:h]**2))) + sum(sum((CVCpsiv[p,:]**2))))


                if scalar_lambda_a:
                    if norm_sr == '2,1':
                        TV_a[i] = sum(sum(math.sqrt(ConvC(A,FDh,m,n,P)**2 + ConvC(A,FDv,m,n,P)**2)))
                    elif norm_sr == '1,1':
                        TV_a[i] = sum(sum(abs(ConvC(A,FDh,m,n,P)) + abs(ConvC(A,FDv,m,n,P))))
                else:
                    CvCAh = ConvC(A,FDh,m,n,P)
                    CvCAv = ConvC(A,FDv,m,n,P)

                    if norm_sr == '2,1':
                        for p in range(P):
                            TV_a[i,p] = sum(sum(math.sqrt(CvCAh[p,:]**2 + CvCAv[p,:]**2)))
                    elif norm_sr == '1,1':
                        for p in range(P):
                            TV_a[i,p] = sum(sum(abs(CvCAh[p,:])+abs(CvCAv[p,h])))

                objective[i] = norm_fitting[i] + lambda_s * source_model[i] + lambda_a.transpose() * TV_a[i,:].transpose() + lambda_psi.transpose * smooth_psi[i,:].transpose()

            elif not(any(lambda_psi)) and any(lambda_a):

                if scalar_lambda_a:
                    if norm_sr == '2,1':
                        TV_a[i] = sum(sum(math.sqrt(ConvC(A,FDh,m,n,P)**2 + ConvC(A,FDv,m,n,P)**2)))
                    elif norm_sr == '1,1':
                        TV_a[i] = sum(sum(abs(ConvC(A,FDh,m,n,P)) + abs(ConvC(A,FDv,m,n,P))))
                else:
                    CvCAh = ConvC(A,FDh,m,n,P)
                    CvCAv = ConvC(A,FDv,m,n,P)

                    if norm_sr == '2,1':
                        for p in range(P):
                            TV_a[i,p] = sum(sum(math.sqrt(CvCAh[p,:]**2 + CvCAv[p,:]**2)))
                    elif norm_sr == '1,1':
                        for p in range(P):
                            TV_a[i,p] = sum(sum(abs(CvCAh[p,:])+abs(CvCAv[p,h])))


                objective[i] = norm_fitting[i] + lambda_s * source_model[i] + lambda_a.transpose() * TV_a[i,:].transpose()


            elif any(lambda_psi) and not(any(lambda_a)):
                if scalar_lambda_psi:
                    smooth_psi[i] = 1/2*(sum(sum((ConvC(psi_maps,FDh,m,n,P)**2))) + sum(sum((ConvC(psi_maps,FDv,m,n,P)**2))))
                else:
                    CvCpsih = ConvC(psi_maps,FDh,m,n,P)
                    CvCpsiv = ConvC(psi_maps,FDv,m,n,P)
                    for p in range(P):
                        smooth_psi[i,p] = 1/2*(sum(sum((CvCpsih[p,:h]**2))) + sum(sum((CVCpsiv[p,:]**2))))

                objective[i] = norm_fitting[i] + lambda_s * source_model[i] + lambda_psi.transpose() * smooth_psi[i,:].transpose()

            else:
                objective[i] = norm_fitting[i] + lambda_s * source_model[i]

            # termination test
            print(f'iteration: {i}');
            if (rs[i] < epsilon_s) and (ra[i] < espilon_a) and (rpsi[i] < epsilon_psi):
                break
                            


    # gather processed output

    '''
    Outputs:
    -A: P*N abundance matrix
    -psi_maps: P*N scaling factor matrix
    -S: L*P*N tensor constaining all the endmember matrices for each pixel
    -optim_struct: structure containing the values of the objective
    function and its different terms at each iteration
    '''

    outputs = []
    outputs.append(A)
    outputs.append(psi_maps)
    outputs.append(S)
    

    return outputs



# # define some auxiliary functions:

# Fully Constrained Linear Spectral Unmixing
# has it's own .m file in source code
# may be useful to implement in it's own file
def FCLSU(HIM,M):
    # depends on scipy nnls import
    # this was a hacky implementation
    # should be tested

    if len(HIM.shape) == 1:
        HIM = HIM[:,None]
       
    ns = HIM.shape[1]
    l = M.shape[0]
    p = M.shape[1]
    Delta = 1/1000
    N = zeros((l+1,p))
    N[0:l,0:p] = Delta*M
    N[l,:] = ones((1,p))
    s = zeros((l+1,1))
    
    out = zeros((ns,p))

    for i in range(ns):
        s[0:l] = Delta*HIM[:,i,None] 
        s[l] = 1
        Abundances = nnls(N,s.flatten())[0].T
        out[i,:] = Abundances
    return out

# circular convolution
def ConvC(X, FK, m, n, P):
    # matlab:
    # reshape(real(ifft2(fft2(reshape(X', m,n,P)).*repmat(FK,[1,1,P])) ), m*n,P)';

    # AN ESPECIALLY WEIRD m->py DISCREPANCY
    # not exactly sure how this works, but it's what I've been using.
    # MATLAB: S = repmat(S0,[1,1,N]);
    # python: S = array([tile(S0,(1,1)) for i in range(N)]).T
    
    first_op = fft.fft2(X.T.reshape(m,n,P, order='F'))
    second_op = real( fft.ifft2( first_op * array([tile(S0,(1,1)) for i in range(P)]).T))
    third_op = second_op.reshape( m*n, P ).T

    return third_op

# convert matrix to image
def conv2im(A, m, n, P):
    return A.T.reshape((m,n,P))

# convert image to matrix
def conv2mat(A, m, n, P):
    return A.reshape((m*n,P)).T

# # soft(x,T) and vector_soft_col(X, tau) are in a separate .m file in source code

# soft-thresholding function
def soft(x, T):
    if sum(abs(T.flatten(1))) == 0:
        y = x
    else:
        y = max(abs(x)-T, 0)
        y = y/(y+T) * x
    return y

# computes the vector soft columnwise
def vector_soft_col(X, tau):
    NU = math.sqrt(sum(X**2))
    A = max(0,NU-tau)
    Y = kron(ones((size(X, axis=1),1)), (A/(A+tau))) * X
    return Y

'''
# code block used for testing by running this .py file

# this test case is NOT sufficient
# only used for getting the basic matrix shapes correct
# It's recommended that a real test case is set up before much further work on the algorithm itself
if __name__ == '__main__':

    # print statements are littered around the algorithm
    # this is how it was done in MATLAB
    # should be removed before final PyHAT implementation

    m = 5
    n = 5
    L = 5
    P = 5
    N = m * n

    arb = 5  # arbitrary integer
    
    data = ones((n,n,L))
    A_init = ones((P, N))
    psis_init = ones((P, N))
    S0 = ones((L, P))
    lambda_s = arb
    lambda_a = ones((arb,arb))
    lambda_psi = ones((arb,arb))
    
    output = elmm_admm( data, A_init, psis_init, S0, lambda_s, lambda_a, lambda_psi )

    print( output )
'''
