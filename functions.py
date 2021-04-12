import numpy as np
import scipy
import scipy.sparse
import scipy.sparse.linalg
import pyamg
import time
from scipy.optimize import fsolve
from singleton_classes import RK4, RK3, RK2,DIRK2
from tensorflow.keras.models import model_from_json
# from tf_siren import SIRENModel, SinusodialRepresentationDense, SIRENInitializer
from tf_siren.siren import *
from tensorflow.keras.initializers import he_uniform
tf.keras.utils.get_custom_objects().update({
    'Sine': Sine,
    'SIRENInitializer': SIRENInitializer,
    'SIRENFirstLayerInitializer': SIRENFirstLayerInitializer,
    'SinusodialRepresentationDense': SinusodialRepresentationDense,
    'ScaledSinusodialRepresentationDense': ScaledSinusodialRepresentationDense,
})

class func:
    def __init__(self, probDescription,bcs_type=None):
        self.probDescription = probDescription
        self.bcs_type = bcs_type

    def periodic_scalar(self, f):
        # set periodicity on the scalar
        f[-1, :] = f[1, :]
        f[0, :] = f[-2, :]
        f[:, -1] = f[:, 1]
        f[:, 0] = f[:, -2]

    def periodic_u(self, u):
        # set periodicity in x
        u[:, -1] = u[:, 1]
        u[:, 0] = u[:, -2]
        # set periodicity in y
        u[-1, :] = u[1, :]
        u[0, :] = u[-2, :]

    def periodic_v(self, v):
        # set periodicity in y
        v[-1, :] = v[1, :]
        v[0, :] = v[-2, :]
        # set periodicity in x
        v[:, -1] = v[:, 1]
        v[:, 0] = v[:, -2]

    def top_wall(self,u,v,u_func,v_func):
        xu, yu = self.probDescription.get_XVol()
        xv, yv = self.probDescription.get_YVol()
        u[-1, 1:] = 2*u_func(xu) - u[-2, 1:]
        v[-1, 1:-1] = v_func(xv)

    def bottom_wall(self,u,v,u_func,v_func):
        xu, yu = self.probDescription.get_XVol()
        xv, yv = self.probDescription.get_YVol()
        u[0, 1:] =  2 * u_func(xu) - u[1, 1:]
        v[1, 1:-1] = v_func(xv)

    def left_wall(self,u,v,u_func,v_func):
        xu, yu = self.probDescription.get_XVol()
        xv, yv = self.probDescription.get_YVol()
        u[1:-1, 1] = u_func(yu)
        v[1:, 0] = 2*v_func(yv) -v[1:, 1]

    def right_wall(self,u,v,u_func,v_func):
        xu, yu = self.probDescription.get_XVol()
        xv, yv = self.probDescription.get_YVol()
        u[1:-1, -1] = u_func(yu)
        v[1:,-1] = 2.0*v_func(yv) - v[1:,-2]

    def Newton_solver(self,guesses:list,old_vals:list,residual_funcs:list,bcs_func:list,Jacobian_builder,Tol,verbose=False,ghost_cells=2):
        info = {"non-linear-iterations_res":{}}
        num = len(guesses) # number of residual functions
        size = 0
        num_eq = 0
        num_eq_per_guess = [0]
        num_eq_per_guess_wgc = [0] # including ghost cells
        size_guess = []
        size_guess_ngc = [] # size without ghost cells
        for i in range(num):
            size_y, size_x = guesses[0].shape
            size_guess.append((size_y,size_x))
            num_eq_per_guess_wgc.append(num_eq_per_guess_wgc[-1]+size_y*size_x)
            nx = size_x - ghost_cells
            ny = size_y - ghost_cells
            num_eq_per_guess.append(num_eq_per_guess[-1] + nx*ny)
            size_guess_ngc.append((ny,nx))
            num_eq+= nx*ny
            size += size_x*size_y

        info['num_equations'] = num_eq
        num_eq_per_guess.append(-1)
        num_eq_per_guess_wgc.append(-1)

        delta = np.zeros(num_eq)

        initial_guess = None
        if num == 1:
            initial_guess=guesses[0].ravel()
        else:
            initial_guess = np.append(guesses[0].ravel(),guesses[1].ravel())

            if num >=2:
                for i in range(2,num):
                    initial_guess = np.append(initial_guess,guesses[i].ravel())

        lastX = initial_guess

        nextX = lastX + 10 *Tol # "different than lastX so loop starts OK

        res = np.linalg.norm((lastX - nextX),np.inf)

        deltas = []
        for i in range(num):
            deltas.append(np.zeros_like(guesses[i]))

        non_linear_iterations = 0
        info_cg=None
        info['condition_num'] = []
        while (res > Tol):  # this is how you terminate the loop
            lastX = nextX

            # guesses_k = [lastX[:num_eq_per_guess_wgc[1]].reshape(size_guess[0][0],size_guess[0][1])]
            guesses_k = []
            # for i in range(1,num-2):
            for i in range(num):
                size_y,size_x = size_guess[i]
                guesses_k.append(lastX[num_eq_per_guess_wgc[i]:num_eq_per_guess_wgc[i+1]].reshape(size_y, size_x))
            # guesses_k.append(lastX[num_eq_per_guess_wgc[num-1]:].reshape(size_guess[num-1][0], size_guess[num-1][1]))

            # apply boundary conditions
            for i in range(num):
                bcs_func[i](guesses_k[i])

            # residual functions
            F = None
            if num == 1:
                F = residual_funcs[0](*old_vals)(*guesses_k)[1:-1,1:-1].ravel()
            else:
                F = np.append(residual_funcs[0](*old_vals)(*guesses_k)[1:-1,1:-1].ravel(), residual_funcs[1](*old_vals)(*guesses_k)[1:-1,1:-1].ravel())
                if num >= 2:
                    for i in range(2, num):
                        F = np.append(F, residual_funcs[i](*old_vals)(*guesses_k)[1:-1,1:-1].ravel())

            # check if the residual functions has infinity values
            inf = np.isinf(F)
            nan = np.isnan(F)
            print("     F has inf: {}\n     F has nan: {}".format(inf.any(),nan.any()))

            # Jacobian
            # J = Jacobian_builder.Sparse_Jacobian_deprecated(*guesses_k)
            J = Jacobian_builder.Sparse_Jacobian(*guesses_k)
            cond_num = pyamg.util.linalg.condest(J)
            print('     condition number=',cond_num )
            info['condition_num'].append(cond_num)

            # conjugate Gradient Pyamg
            # delta,info_cg = pyamg.krylov.cg(J,-F, maxiter=10000, tol=Tol)

            ml = pyamg.ruge_stuben_solver(J)
            residuals = []
            delta = ml.solve(-F,x0=np.zeros_like(F), maxiter=1000, tol= Tol, residuals=residuals)

            # collect data
            info["non-linear-iterations_res"][non_linear_iterations] = residuals
            # delta,info =scipy.sparse.linalg.cg(J, -F, tol=1e-1*Tol, maxiter=50000)
            if verbose:
                print(ml)
                print('residuals=', residuals)
                # print('convergence info:',info_cg)
            if info_cg!=None:
                info["conjugate-gradient-cgne "] = info_cg
            # deltas with ghost cells
            # deltas[0][1:-1, 1:-1] = [delta[:num_eq_per_guess[1]].reshape(size_guess_ngc[0][0], size_guess_ngc[0][1])]
            # deltas[0][1:-1, 1:-1] = [delta[:num_eq_per_guess[1]].reshape(size_guess_ngc[0][0], size_guess_ngc[0][1])]
            for i in range(num):
                size_y, size_x = size_guess_ngc[i]
                deltas[i][1:-1, 1:-1] = delta[num_eq_per_guess[i]:num_eq_per_guess[i + 1]].reshape(size_y, size_x)
            # deltas[num - 1][1:-1, 1:-1] = [delta[num_eq_per_guess[num - 1]:].reshape(size_guess_ngc[num - 1][0], size_guess_ngc[num - 1][1])]

            # compute the new delta as a vector
            new_delta = None
            if num == 1:
                new_delta = deltas[0].ravel()
            else:
                new_delta = np.append(deltas[0].ravel(), deltas[1].ravel())
                if num >= 2:
                    for i in range(2, num):
                        new_delta = np.append(new_delta, deltas[i].ravel())

            # update the next X
            nextX = lastX + new_delta  # update estimate using N-R

            # infinity norm
            # infty_norm = np.linalg.norm(new_delta, np.inf)
            res = residuals[-1]
            non_linear_iterations += 1

        # return the sol a list
        # sol = [nextX[:num_eq_per_guess_wgc[1]].reshape(size_guess[0][0], size_guess[0][1])]
        sol = []
        for i in range(num):
            size_y, size_x = size_guess[i]
            sol.append(nextX[num_eq_per_guess_wgc[i]:num_eq_per_guess_wgc[i + 1]].reshape(size_y, size_x))
        # sol.append(nextX[num_eq_per_guess_wgc[num - 1]:].reshape(size_guess[num - 1][0], size_guess[num - 1][1]))
        info['non-linear-iterations'] = non_linear_iterations
        info['infty-norm'] = res

        return sol, non_linear_iterations, res, info


    def urhs(self, u, v):
        μ = self.probDescription.μ
        dx = self.probDescription.dx
        dy = self.probDescription.dy
        nx = self.probDescription.nx
        ny = self.probDescription.ny

        urhs = np.zeros_like(u)
        # do x-momentum first - u is of size (nx + 2) x (ny + 2)
        ue = 0.5 * (u[1:ny + 1, 2:nx + 2] + u[1:ny + 1, 1:nx + 1])
        uw = 0.5 * (u[1:ny + 1, 1:nx + 1] + u[1:ny + 1, :nx])

        un = 0.5 * (u[2:ny + 2, 1:nx + 1] + u[1:ny + 1, 1:nx + 1])
        us = 0.5 * (u[1:ny + 1, 1:nx + 1] + u[:ny, 1:nx + 1])

        vn = 0.5 * (v[2:ny + 2, 1:nx + 1] + v[2:ny + 2, :nx])
        vs = 0.5 * (v[1:ny + 1, 1:nx + 1] + v[1:ny + 1, :nx])

        # convection = - d(uu)/dx - d(vu)/dy
        convection = - (ue * ue - uw * uw) / dx - (un * vn - us * vs) / dy

        # diffusion = d2u/dx2 + d2u/dy2
        diffusion = μ * ((u[1:ny + 1, 2:nx + 2] - 2.0 * u[1:ny + 1, 1:nx + 1] + u[1:ny + 1, :nx]) / dx / dx + (
                    u[2:ny + 2, 1:nx + 1] - 2.0 * u[1:ny + 1, 1:nx + 1] + u[:ny, 1:nx + 1]) / dy / dy)

        urhs[1:ny + 1, 1:nx + 1] = convection + diffusion

        # set periodicity in x and y
        self.periodic_u(urhs)
        return urhs

    def urhs_bcs(self,u,v):
        μ = self.probDescription.μ
        dx = self.probDescription.dx
        dy = self.probDescription.dy
        nx = self.probDescription.nx
        ny = self.probDescription.ny

        urhs = np.zeros_like(u)

        ue = 0.5 * (u[1:ny + 1, 3:nx + 2] + u[1:ny + 1, 2:nx + 1])
        uw = 0.5 * (u[1:ny + 1, 2:nx + 1] + u[1:ny + 1, 1:nx])

        un = 0.5 * (u[2:ny + 2, 2:nx + 1] + u[1:ny + 1, 2:nx + 1])
        us = 0.5 * (u[1:ny + 1, 2:nx + 1] + u[0:ny, 2:nx + 1])

        vn = 0.5 * (v[2:ny + 2, 2:nx + 1] + v[2:ny + 2, 1:nx])
        vs = 0.5 * (v[1:ny + 1, 2:nx + 1] + v[1:ny + 1, 1:nx])

        # convection = - d(uu)/dx - d(vu)/dy
        convection = - (ue * ue - uw * uw) / dx - (un * vn - us * vs) / dy

        # diffusion = d2u/dx2 + d2u/dy2
        diffusion = μ * ((u[1:ny + 1, 3:nx + 2] - 2.0 * u[1:ny + 1, 2:nx + 1] + u[1:ny + 1, 1:nx]) / dx / dx + (
                    u[2:ny + 2, 2:nx + 1] - 2.0 * u[1:ny + 1, 2:nx + 1] + u[0:ny, 2:nx + 1]) / dy / dy)

        urhs[1:ny + 1, 2:nx + 1] = convection + diffusion
        return urhs

    def vrhs(self, u, v):
        μ = self.probDescription.μ
        dx = self.probDescription.dx
        dy = self.probDescription.dy
        nx = self.probDescription.nx
        ny = self.probDescription.ny
        vrhs = np.zeros_like(u)
        # do y-momentum
        ve = 0.5 * (v[1:ny + 1, 2:nx + 2] + v[1:ny + 1, 1:nx + 1])
        vw = 0.5 * (v[1:ny + 1, 1:nx + 1] + v[1:ny + 1, :nx])

        ue = 0.5 * (u[1:ny + 1, 2:nx + 2] + u[:ny, 2:nx + 2])
        uw = 0.5 * (u[1:ny + 1, 1:nx + 1] + u[:ny, 1:nx + 1])

        vn = 0.5 * (v[2:ny + 2, 1:nx + 1] + v[1:ny + 1, 1:nx + 1])
        vs = 0.5 * (v[1:ny + 1, 1:nx + 1] + v[:ny, 1:nx + 1])

        # convection = d(uv)/dx + d(vv)/dy
        convection = - (ue * ve - uw * vw) / dx - (vn * vn - vs * vs) / dy
        # diffusion = d2u/dx2 + d2u/dy2
        diffusion = μ * ((v[1:ny + 1, 2:nx + 2] - 2.0 * v[1:ny + 1, 1:nx + 1] + v[1:ny + 1, :nx]) / dx / dx + (
                    v[2:ny + 2, 1:nx + 1] - 2.0 * v[1:ny + 1, 1:nx + 1] + v[:ny, 1:nx + 1]) / dy / dy)

        vrhs[1:ny + 1, 1:nx + 1] = convection + diffusion
        # set periodicity in x and y
        self.periodic_v(vrhs)
        return vrhs

    def vrhs_bcs(self,u,v):
        μ = self.probDescription.μ
        dx = self.probDescription.dx
        dy = self.probDescription.dy
        nx = self.probDescription.nx
        ny = self.probDescription.ny

        vrhs = np.zeros_like(u)
        # do y-momentum - only need to do interior points
        ve = 0.5 * (v[2:ny + 1, 2:nx + 2] + v[2:ny + 1, 1:nx + 1])
        vw = 0.5 * (v[2:ny + 1, 1:nx + 1] + v[2:ny + 1, 0:nx])

        ue = 0.5 * (u[2:ny + 1, 2:nx + 2] + u[1:ny, 2:nx + 2])
        uw = 0.5 * (u[2:ny + 1, 1:nx + 1] + u[1:ny, 1:nx + 1])

        vn = 0.5 * (v[3:ny + 2, 1:nx + 1] + v[2:ny + 1, 1:nx + 1])
        vs = 0.5 * (v[2:ny + 1, 1:nx + 1] + v[1:ny, 1:nx + 1])

        # convection = d(uv)/dx + d(vv)/dy
        convection = - (ue * ve - uw * vw) / dx - (vn * vn - vs * vs) / dy

        # diffusion = d2u/dx2 + d2u/dy2
        diffusion = μ * ((v[2:ny + 1, 2:nx + 2] - 2.0 * v[2:ny + 1, 1:nx + 1] + v[2:ny + 1, 0:nx]) / dx / dx + (
                    v[3:ny + 2, 1:nx + 1] - 2.0 * v[2:ny + 1, 1:nx + 1] + v[1:ny, 1:nx + 1]) / dy / dy)

        vrhs[2:ny + 1, 1:nx + 1] = convection + diffusion

        return vrhs

    def div(self, fx, fy):
        dx = self.probDescription.dx
        dy = self.probDescription.dy
        nx = self.probDescription.nx
        ny = self.probDescription.ny

        divf = np.zeros([ny + 2, nx + 2])
        for i in range(1, nx + 1):
            for j in range(1, ny + 1):
                divf[j, i] = (fx[j, i + 1] - fx[j, i]) / dx + (fy[j + 1, i] - fy[j, i]) / dy
        return divf

    def Gpx(self, p):
        dx = self.probDescription.dx
        nx = self.probDescription.nx
        ny = self.probDescription.ny

        gpx = np.zeros([ny + 2, nx + 2])
        gpx[1:-1, 1:] = ((p[1:-1, 1:] - p[1:-1, :-1]) / dx)
        if self.bcs_type=="periodic":
            self.periodic_u(gpx)
        return gpx

    def Gpy(self, p):
        dy = self.probDescription.dy
        nx = self.probDescription.nx
        ny = self.probDescription.ny
        gpy = np.zeros([ny + 2, nx + 2])
        gpy[1:, 1:-1] = ((p[1:, 1:-1] - p[:-1, 1:-1]) / dy)
        if self.bcs_type == "periodic":
            self.periodic_v(gpy)
        return gpy

    def BackwardEuler(self,u_n,v_n,p_n):
        dt = self.probDescription.dt
        size_x, size_y = u_n.shape
        flat_size = size_x * size_y

        def residual_function(x, data_n):
            oldVals = data_n[0]
            uold = oldVals[:flat_size]
            vold = oldVals[flat_size:]
            uold = uold.reshape(size_y, size_x)
            vold = vold.reshape(size_y, size_x)
            u_r = x[:flat_size]
            v_r = x[flat_size:2 * flat_size]
            p_r = x[2 * flat_size:]
            u_it = u_r.reshape(size_y, size_x)
            v_it = v_r.reshape(size_y, size_x)
            p_it = p_r.reshape(size_y, size_x)
            self.periodic_u(u_it)
            self.periodic_v(v_it)
            self.periodic_scalar(p_it)
            f1 = (u_it - uold)/dt -  (self.urhs(u_it, v_it) - self.Gpx(p_it))
            f2 = (v_it - vold)/dt -  (self.vrhs(u_it, v_it) - self.Gpy(p_it))
            f3 = (self.div(self.Gpx(p_it), self.Gpy(p_it)) - self.div(self.urhs(u_it, v_it), self.vrhs(u_it, v_it)) ) - self.div(uold,vold)/dt
            F = np.append(f1.ravel(), f2.ravel())
            F = np.append(F, f3.ravel())
            # print('it Norm = ',np.linalg.norm(F,np.inf))

            return F

        guesses = np.append(u_n.ravel(), v_n.ravel())
        guesses = np.append(guesses, p_n.ravel())

        data_n = np.append(u_n.ravel(), v_n.ravel())
        sol, info, _, _ = fsolve(residual_function, guesses, args=[data_n],xtol=1e-11, full_output=True)
        u_f = sol[:flat_size]
        v_f = sol[flat_size:2 * flat_size]
        p_f = sol[2 * flat_size:]
        u_sol = u_f.reshape(size_y, size_x)
        v_sol = v_f.reshape(size_y, size_x)
        p_sol = p_f.reshape(size_y, size_x)
        self.periodic_u(u_sol)
        self.periodic_v(v_sol)
        self.periodic_scalar(p_sol)
        return u_sol, v_sol, p_sol, info

    def DIRK_S1(self,u_n,v_n,p_n,integ):
        dt = self.probDescription.dt
        size_x, size_y = u_n.shape
        flat_size = size_x * size_y
        a11 = integ.a11
        def residual_function(x, data_n):
            oldVals = data_n[0]
            uold = oldVals[:flat_size]
            vold = oldVals[flat_size:]
            uold = uold.reshape(size_y, size_x)
            vold = vold.reshape(size_y, size_x)
            u_r = x[:flat_size]
            v_r = x[flat_size:2 * flat_size]
            p_r = x[2 * flat_size:]
            u_it = u_r.reshape(size_y, size_x)
            v_it = v_r.reshape(size_y, size_x)
            p_it = p_r.reshape(size_y, size_x)
            self.periodic_u(u_it)
            self.periodic_v(v_it)
            self.periodic_scalar(p_it)
            F1 = (u_it - uold) / dt - a11*(self.urhs(u_it, v_it) - self.Gpx(p_it))
            F2 = (v_it - vold) / dt - a11*(self.vrhs(u_it, v_it) - self.Gpy(p_it))
            F3 = (self.div(self.Gpx(p_it), self.Gpy(p_it)) - self.div(self.urhs(u_it, v_it),
                                                                  self.vrhs(u_it, v_it))) - self.div(uold, vold) / dt/a11
            F = np.append(F1.ravel(), F2.ravel())
            F = np.append(F, F3.ravel())
            # print('it Norm = ',np.linalg.norm(F,np.inf))

            return F

        guesses = np.append(u_n.ravel(), v_n.ravel())
        guesses = np.append(guesses, p_n.ravel())

        data_n = np.append(u_n.ravel(), v_n.ravel())
        sol, info, _, _ = fsolve(residual_function, guesses, args=[data_n],xtol=1e-11, full_output=True)
        u_f = sol[:flat_size]
        v_f = sol[flat_size:2 * flat_size]
        p_f = sol[2 * flat_size:]
        u_sol = u_f.reshape(size_y, size_x)
        v_sol = v_f.reshape(size_y, size_x)
        p_sol = p_f.reshape(size_y, size_x)
        self.periodic_u(u_sol)
        self.periodic_v(v_sol)
        self.periodic_scalar(p_sol)
        return u_sol, v_sol, p_sol, info

    def DIRK_S2(self,u_n,v_n,p_n,rhs_u1,rhs_v1,integ):
        dt = self.probDescription.dt
        size_x, size_y = u_n.shape
        flat_size = size_x * size_y
        a11 = integ.a11
        a21 = integ.a21
        a22 = integ.a22
        def residual_function(x, data_n):
            oldVals = data_n[0]
            rhsVals = data_n[1]
            rhs_u1_r = rhsVals[:flat_size]
            rhs_v1_r = rhsVals[flat_size:]
            rhs_u1 = rhs_u1_r.reshape(size_y, size_x)
            rhs_v1 = rhs_v1_r.reshape(size_y, size_x)
            uold = oldVals[:flat_size]
            vold = oldVals[flat_size:]
            uold = uold.reshape(size_y, size_x)
            vold = vold.reshape(size_y, size_x)
            u_r = x[:flat_size]
            v_r = x[flat_size:2 * flat_size]
            p_r = x[2 * flat_size:]
            u_it = u_r.reshape(size_y, size_x)
            v_it = v_r.reshape(size_y, size_x)
            p_it = p_r.reshape(size_y, size_x)
            self.periodic_u(u_it)
            self.periodic_v(v_it)
            self.periodic_scalar(p_it)
            F1 = (u_it - uold) / dt - a21 * rhs_u1 - a22 * (self.urhs(u_it, v_it) - self.Gpx(p_it))
            F2 = (v_it - vold) / dt - a21 * rhs_v1 - a22 * (self.vrhs(u_it, v_it) - self.Gpy(p_it))
            F3 = (self.div(self.Gpx(p_it), self.Gpy(p_it)) - self.div(self.urhs(u_it, v_it),self.vrhs(u_it, v_it))) -a21*self.div(rhs_u1,rhs_v1)/a22- self.div(uold, vold) / dt/a22
            F = np.append(F1.ravel(), F2.ravel())
            F = np.append(F, F3.ravel())
            # print('it Norm = ',np.linalg.norm(F,np.inf))

            return F

        guesses = np.append(u_n.ravel(), v_n.ravel())
        guesses = np.append(guesses, p_n.ravel())

        data_n = np.append(u_n.ravel(), v_n.ravel())
        rhs_1 = np.append(rhs_u1.ravel(),rhs_v1.ravel())
        sol, info, _, _ = fsolve(residual_function, guesses, args=[data_n,rhs_1],xtol=1e-11, full_output=True)
        u_f = sol[:flat_size]
        v_f = sol[flat_size:2 * flat_size]
        p_f = sol[2 * flat_size:]
        u_sol = u_f.reshape(size_y, size_x)
        v_sol = v_f.reshape(size_y, size_x)
        p_sol = p_f.reshape(size_y, size_x)
        self.periodic_u(u_sol)
        self.periodic_v(v_sol)
        self.periodic_scalar(p_sol)
        return u_sol, v_sol, p_sol, info

    # defining pressure matrix

    def A(self):
        dx = self.probDescription.dx
        dy = self.probDescription.dy
        nx = self.probDescription.nx
        ny = self.probDescription.ny
        # build pressure coefficient matrix
        Ap = np.zeros([ny, nx])
        Ae = 1.0 / dx / dx * np.ones([ny, nx])
        As = 1.0 / dy / dy * np.ones([ny, nx])
        An = 1.0 / dy / dy * np.ones([ny, nx])
        Aw = 1.0 / dx / dx * np.ones([ny, nx])
        # # set left wall coefs
        Aw[:, 0] = 0.0
        # # set right wall coefs
        Ae[:, -1] = 0.0

        Awb = 1.0 / dx / dx * np.ones([ny, nx])
        Awb[:, 1:] = 0

        Asb = 1.0 / dx / dx * np.ones([ny, nx])
        Asb[1:, :] = 0

        Aeb = 1.0 / dx / dx * np.ones([ny, nx])
        Aeb[:, :-1] = 0

        Anb = 1.0 / dx / dx * np.ones([ny, nx])
        Anb[:-1, :] = 0

        Ap = -(Aw + Ae + An + As + Awb + Aeb)

        n = nx * ny
        d0 = Ap.reshape(n)
        # print(d0)
        de = Ae.reshape(n)[:-1]
        # print(de)
        dw = Aw.reshape(n)[1:]
        # print(dw)
        ds = As.reshape(n)[nx:]
        # print(ds)
        dn = An.reshape(n)[:-nx]
        # print(dn)
        dwb = Awb.reshape(n)[:-nx + 1]
        # print(dwb)
        dsb = Asb.reshape(n)[:nx]
        # print(dsb)
        deb = Aeb.reshape(n)[nx - 1:]
        # print(deb)
        dnb = Anb.reshape(n)[-nx:]
        # print(dnb)
        A1 = scipy.sparse.diags([d0, de, dw, dn, ds, dwb, dsb, deb, dnb],
                                [0, 1, -1, nx, -nx, nx - 1, nx * (ny - 1), -nx + 1, -nx * (ny - 1)], format='csr')
        return A1

    def A_channel_flow(self):
        dx = self.probDescription.dx
        dy = self.probDescription.dy
        nx = self.probDescription.nx
        ny = self.probDescription.ny
        # build pressure coefficient matrix
        Ap = np.zeros([ny, nx])
        Ae = 1.0 / dx / dx * np.ones([ny, nx])
        As = 1.0 / dy / dy * np.ones([ny, nx])
        An = 1.0 / dy / dy * np.ones([ny, nx])
        Aw = 1.0 / dx / dx * np.ones([ny, nx])
        # set left wall coefs
        Aw[:, 0] = 0.0
        # set right wall coefs
        Ae[:, -1] = 0.0
        # set top wall coefs
        An[-1, :] = 0.0
        # set bottom wall coefs
        As[0, :] = 0.0
        Ap = -(Aw + Ae + An + As)
        # for the outflow boundary condition to get -3/dx/dx
        Ap[:, -1] -= 2 / dx / dx

        n = nx * ny
        d0 = Ap.reshape(n)
        # print(d0)
        de = Ae.reshape(n)[:-1]
        # print(de)
        dw = Aw.reshape(n)[1:]
        # print(dw)
        ds = As.reshape(n)[nx:]
        # print(ds)
        dn = An.reshape(n)[:-nx]
        # print(dn)
        A1 = scipy.sparse.diags([d0, de, dw, dn, ds], [0, 1, -1, nx, -nx], format='csr')

        return A1

    def A_Lid_driven_cavity(self):
        dx = self.probDescription.dx
        dy = self.probDescription.dy
        nx = self.probDescription.nx
        ny = self.probDescription.ny
        # build pressure coefficient matrix
        Ap = np.zeros([ny, nx])
        Ae = 1.0 / dx / dx * np.ones([ny, nx])
        As = 1.0 / dy / dy * np.ones([ny, nx])
        An = 1.0 / dy / dy * np.ones([ny, nx])
        Aw = 1.0 / dx / dx * np.ones([ny, nx])
        # set left wall coefs
        Aw[:, 0] = 0.0
        # set right wall coefs
        Ae[:, -1] = 0.0
        # set top wall coefs
        An[-1, :] = 0.0
        # set bottom wall coefs
        As[0, :] = 0.0
        Ap = -(Aw + Ae + An + As)

        n = nx * ny
        d0 = Ap.reshape(n)
        # print(d0)
        de = Ae.reshape(n)[:-1]
        # print(de)
        dw = Aw.reshape(n)[1:]
        # print(dw)
        ds = As.reshape(n)[nx:]
        # print(ds)
        dn = An.reshape(n)[:-nx]
        # print(dn)
        A1 = scipy.sparse.diags([d0, de, dw, dn, ds], [0, 1, -1, nx, -nx], format='csr')

        return A1

    def ImQ(self, uh, vh, Coef, p0, tol=None, atol=None):
        dx = self.probDescription.dx
        dy = self.probDescription.dy
        dt = self.probDescription.dt
        nx = self.probDescription.nx
        ny = self.probDescription.ny
        unp1 = np.zeros_like(uh)
        vnp1 = np.zeros_like(vh)
        divuhat = self.div(uh, vh)

        prhs = 1.0 / dt * divuhat[1:-1, 1:-1]
        # plt.imshow(prhs,origin='bottom',cmap='jet',vmax=80, vmin=-80)
        # # plt.contourf((psol[-1][1:-1,1:] - psol[-1][1:-1,:-1])/dx)
        # v = np.linspace(-80, 80, 4, endpoint=True)
        # plt.colorbar(ticks=v)
        # plt.title('Prhs')
        # plt.show()

        rhs = prhs.ravel()

        def solver(A, b):
            num_iters = 0

            def callback(xk):
                nonlocal num_iters
                num_iters += 1

            ml = pyamg.ruge_stuben_solver(A)
            ptmp = ml.solve(b, tol=1e-12, callback=callback)
            # ptmp = scipy.sparse.linalg.spsolve(A,b,callback=callback)
            # atol = 1e-5
            # tol = 1e-10
            # if atol!=None or tol!=None:
            #     ptmp,_ = scipy.sparse.linalg.cg(A, b,p0[1:-1, 1:-1].ravel(),callback=callback,tol=tol)
            # else:
            #     ptmp, _ = scipy.sparse.linalg.cg(A, b, p0[1:-1, 1:-1].ravel(), callback=callback, tol=1e-12)

            # if max(tol*np.linalg.norm(b,2),atol) == atol:
            #     print('atol')
            # else:
            #     print('tol')
            return ptmp, num_iters

        ptmp, num_iters = solver(Coef, rhs)
        p = np.zeros([ny + 2, nx + 2])
        p[1:-1, 1:-1] = ptmp.reshape([ny, nx])

        # set periodicity on the pressure
        self.periodic_scalar(p)

        # time advance
        unp1[1:-1, 1:] = uh[1:-1, 1:] - dt * (p[1:-1, 1:] - p[1:-1, :-1]) / dx
        vnp1[1:, 1:-1] = vh[1:, 1:-1] - dt * (p[1:, 1:-1] - p[:-1, 1:-1]) / dy

        self.periodic_u(unp1)
        self.periodic_v(vnp1)

        return unp1, vnp1, p, num_iters

    def ImQ_bcs(self, uh, vh, Coef, p0,bcs_pressure,is_post_processing=False,m_t=None, tol=None, atol=None):
        dx = self.probDescription.dx
        dy = self.probDescription.dy
        if is_post_processing:
            dt = self.probDescription.dt_post_processing
        else:
            dt = self.probDescription.dt
        nx = self.probDescription.nx
        ny = self.probDescription.ny
        unp1 = np.zeros_like(uh)
        vnp1 = np.zeros_like(vh)
        divuhat = self.div(uh, vh)

        if type(m_t) == type(np.ndarray):
            m_t[2:-2, 2:-2] = 0.0
            prhs = 1.0 / dt * divuhat[1:-1, 1:-1] - m_t[1:-1, 1:-1]
        else:
            prhs = 1.0 / dt * divuhat[1:-1, 1:-1]
        # plt.imshow(prhs,origin='bottom',cmap='jet',vmax=80, vmin=-80)
        # # plt.contourf((psol[-1][1:-1,1:] - psol[-1][1:-1,:-1])/dx)
        # v = np.linspace(-80, 80, 4, endpoint=True)
        # plt.colorbar(ticks=v)
        # plt.title('Prhs')
        # plt.show()

        rhs = prhs.ravel()

        def solver(A, b):
            num_iters = 0

            def callback(xk):
                nonlocal num_iters
                num_iters += 1

            ml = pyamg.ruge_stuben_solver(A)
            ptmp = ml.solve(b, tol=1e-16, callback=callback)
            # ptmp = scipy.sparse.linalg.spsolve(A,b,callback=callback)
            atol = 1e-5
            tol = 1e-12
            # ptmp,_ = scipy.sparse.linalg.cg(A, b,p0[1:-1, 1:-1].ravel(),callback=callback,tol=tol)
            # if max(tol*np.linalg.norm(b,2),atol) == atol:
            #     print('atol')
            # else:
            #     print('tol')
            return ptmp, num_iters

        ptmp, num_iters = solver(Coef, rhs)
        p = np.zeros([ny + 2, nx + 2])
        p[1:-1, 1:-1] = ptmp.reshape([ny, nx])

        # set bcs on the pressure
        bcs_pressure(p)

        # time advance
        unp1[1:-1, 1:] = uh[1:-1, 1:] - dt * (p[1:-1, 1:] - p[1:-1, :-1]) / dx
        vnp1[1:, 1:-1] = vh[1:, 1:-1] - dt * (p[1:, 1:-1] - p[:-1, 1:-1]) / dy

        return unp1, vnp1, p, num_iters

    def pressure_solver(self, prhs, Coef):
        nx = self.probDescription.nx
        ny = self.probDescription.ny
        rhs = prhs.ravel()

        def solver(A, b):
            num_iters = 0

            def callback(xk):
                nonlocal num_iters
                num_iters += 1

            ml = pyamg.ruge_stuben_solver(A)
            ptmp = ml.solve(b, tol=1e-12, callback=callback)
            # ptmp = scipy.sparse.linalg.spsolve(A,b,callback=callback)
            # ptmp,_ = scipy.sparse.linalg.cg(A, b, callback=callback)
            return ptmp, num_iters

        ptmp, num_iters = solver(Coef, rhs)
        p = np.zeros([ny + 2, nx + 2])
        p[1:-1, 1:-1] = ptmp.reshape([ny, nx])

        # set periodicity on the pressure
        self.periodic_scalar(p)
        return p

    def ImQ_post_processing(self, uh, vh, Coef, p0, a=1):
        dx = self.probDescription.dx
        dy = self.probDescription.dy
        dt = self.probDescription.dt
        nx = self.probDescription.nx
        ny = self.probDescription.ny
        unp1 = np.zeros_like(uh)
        vnp1 = np.zeros_like(vh)
        divuhat = self.div(uh, vh)

        prhs =1/dt/a *  divuhat[1:-1, 1:-1]
        # plt.imshow(prhs,origin='bottom',cmap='jet',vmax=80, vmin=-80)
        # # plt.contourf((psol[-1][1:-1,1:] - psol[-1][1:-1,:-1])/dx)
        # v = np.linspace(-80, 80, 4, endpoint=True)
        # plt.colorbar(ticks=v)
        # plt.title('Prhs')
        # plt.show()

        rhs = prhs.ravel()

        def solver(A, b):
            num_iters = 0

            def callback(xk):
                nonlocal num_iters
                num_iters += 1

            # ml = pyamg.ruge_stuben_solver(A)
            # ptmp = ml.solve(b, tol=1e-12, callback=callback)
            # ptmp = scipy.sparse.linalg.spsolve(A,b,callback=callback)
            tol = 1e-12
            atol = 1e-5
            ptmp, _ = scipy.sparse.linalg.cg(A, b, p0[1:-1, 1:-1].ravel(), callback=callback, tol=tol)
            if max(tol * np.linalg.norm(b, 2), atol) == atol:
                print('atol')
            else:
                print('tol')
            return ptmp, num_iters

        ptmp, num_iters = solver(Coef, rhs)
        p = np.zeros([ny + 2, nx + 2])
        p[1:-1, 1:-1] = ptmp.reshape([ny, nx])

        # set periodicity on the pressure
        self.periodic_scalar(p)

        # time advance
        unp1[1:-1, 1:] = uh[1:-1, 1:] - a*dt* (p[1:-1, 1:] - p[1:-1, :-1]) / dx
        vnp1[1:, 1:-1] = vh[1:, 1:-1] - a*dt* (p[1:, 1:-1] - p[:-1, 1:-1]) / dy

        self.periodic_u(unp1)
        self.periodic_v(vnp1)

        return unp1, vnp1, p, num_iters

    def ImQ_wasatch(self, uh, vh, Coef, prhs, b, p0=0):
        dx = self.probDescription.dx
        dy = self.probDescription.dy
        dt = self.probDescription.dt
        nx = self.probDescription.nx
        ny = self.probDescription.ny

        unp1 = np.zeros_like(uh)
        vnp1 = np.zeros_like(vh)
        # divuhat = div(uh,vh,dx,dy,nx,ny)
        #
        # prhs = 1.0/dt * divuhat[1:-1,1:-1]
        # plt.imshow(prhs,origin='bottom',cmap='jet',vmax=80, vmin=-80)
        # # plt.contourf((psol[-1][1:-1,1:] - psol[-1][1:-1,:-1])/dx)
        # v = np.linspace(-80, 80, 4, endpoint=True)
        # plt.colorbar(ticks=v)
        # plt.title('Prhs')
        # plt.show()

        rhs = prhs.ravel()

        def solver(A, b):
            num_iters = 0

            def callback(xk):
                nonlocal num_iters
                num_iters += 1

            ml = pyamg.ruge_stuben_solver(A)
            ptmp = ml.solve(b, tol=1e-12, callback=callback)
            # ptmp = scipy.sparse.linalg.spsolve(A,b,callback=callback)
            # ptmp,_ = scipy.sparse.linalg.cg(A, b, callback=callback)
            return ptmp, num_iters

        ptmp, num_iters = solver(Coef, rhs)
        p = np.zeros([ny + 2, nx + 2])
        p[1:-1, 1:-1] = ptmp.reshape([ny, nx])

        # set periodicity on the pressure
        self.periodic_scalar(p)

        # time advance
        unp1[1:-1, 1:] = uh[1:-1, 1:] - b * dt * (p[1:-1, 1:] - p[1:-1, :-1]) / dx
        vnp1[1:, 1:-1] = vh[1:, 1:-1] - b * dt * (p[1:, 1:-1] - p[:-1, 1:-1]) / dy

        self.periodic_u(unp1)
        self.periodic_v(vnp1)

        return unp1, vnp1, p, num_iters


    def Guess(self, pold, order=None, integ='Rk3', type='regular',theta=None,ml_model='',ml_weights=''):
        pn = pold[0]
        pnm1 = pold[1]

        Gpnx = self.Gpx(pn)
        Gpny = self.Gpy(pn)
        f1x = np.zeros_like(Gpnx)
        f1y = np.zeros_like(f1x)

        if integ == 'RK4':
            integ = RK4(type)
            a21 = integ.a21
            a31 = integ.a31
            a32 = integ.a32

            pnm2 = pold[2]
            Gpnm1x = self.Gpx(pnm1)
            Gpnm1y = self.Gpy(pnm1)

            Gpnm2x = self.Gpx(pnm2)
            Gpnm2y = self.Gpy(pnm2)

            pnm3 = pold[3]
            Gpnm3x = self.Gpx(pnm3)
            Gpnm3y = self.Gpy(pnm3)

            f2x = np.zeros_like(Gpnx)
            f2y = np.zeros_like(Gpny)

            f3x = np.zeros_like(Gpnx)
            f3y = np.zeros_like(Gpny)

            if order == 'post-processing-approx':
                dt = self.probDescription.get_dt()
                Pnx = Gpnx
                Pny = Gpny

                Pnx_p = (3*Gpnx - 4*Gpnm1x +Gpnm2x)/2/dt # Pnx'
                Pny_p =(3*Gpny - 4*Gpnm1y +Gpnm2y)/2/dt  # Pny'

                Pnx_pp = (Gpnx - 2*Gpnm1x + Gpnm2x) / dt /dt # Pnx'
                Pny_pp = (Gpny - 2*Gpnm1y + Gpnm2y) / dt /dt # Pny'

                f1x = Pnx
                f1y = Pny

                f2x = Pnx + a21 *dt* Pnx_p
                f2y = Pny + a21 *dt* Pny_p

                f3x = Pnx + (a31 + a32) * dt * Pnx_p + a32 * a21 * dt*dt* Pnx_pp
                f3y = Pny + (a31 + a32) * dt * Pny_p + a32 * a21 * dt*dt* Pny_pp


            elif order =='second':
                dt = self.probDescription.get_dt()
                # Pnx = (15 * Gpnx - 10 * Gpnm1x + 3 * Gpnm2x) / 8
                Pnx = Gpnx +  (Gpnx - Gpnm1x)/2
                # Pnx = Gpnx
                # Pny = (15 * Gpny - 10 * Gpnm1y + 3 * Gpnm2y) / 8
                Pny = Gpny +  (Gpny - Gpnm1y)/2
                # Pny = Gpny
                Pnx_p = (2 * Gpnx - 3 * Gpnm1x + Gpnm2x) / dt  # Pnx'  # from lagrange polynomial jupyter notebook
                Pny_p = (2 * Gpny - 3 * Gpnm1y + Gpnm2y) / dt  # Pny'
                Pnx_pp = (Gpnx - 2 * Gpnm1x + Gpnm2x) / 2 / dt / dt  # Pnx''
                Pny_pp = (Gpny - 2 * Gpnm1y + Gpnm2y) / 2 / dt / dt  # Pny''

                f1x = Pnx
                f1y = Pny

                f2x = Pnx + a21 * dt * Pnx_p
                f2y = Pny + a21 * dt * Pny_p

                f3x = Pnx + (a31 + a32) * dt * Pnx_p + a32 * a21 * dt * dt * Pnx_pp
                f3y = Pny + (a31 + a32) * dt * Pny_p + a32 * a21 * dt * dt * Pny_pp

            elif order =='third':
                dt = self.probDescription.get_dt()
                Pnx = 25*Gpnx/12 -23*Gpnm1x/12 + 13*Gpnm2x/12 - Gpnm3x/4
                Pny = 25*Gpny/12 -23*Gpnm1y/12 + 13*Gpnm2y/12 - Gpnm3y/4

                Pnx_p = (35*Gpnx/12 - 23 * Gpnm1x/4 + 15*Gpnm2x/4 - 11*Gpnm3x/12) / dt  # Pnx'
                Pny_p = (35*Gpny/12 - 23 * Gpnm1y/4 + 15*Gpnm2y/4 - 11*Gpnm3y/12) / dt  # Pny'

                Pnx_pp = (5*Gpnx/2 - 13 * Gpnm1x/2 + 11*Gpnm2x/2 - 3*Gpnm3x/2) / dt / dt  # Pnx''
                Pny_pp = (5*Gpny/2 - 13 * Gpnm1y/2 + 11*Gpnm2y/2 - 3*Gpnm3y/2) / dt / dt  # Pny''

                f1x = Pnx
                f1y = Pny

                f2x = Pnx + a21 * dt * Pnx_p
                f2y = Pny + a21 * dt * Pny_p

                f3x = Pnx + (a31 + a32) * dt * Pnx_p + a32 * a21 * dt * dt * Pnx_pp
                f3y = Pny + (a31 + a32) * dt * Pny_p + a32 * a21 * dt * dt * Pny_pp

            elif order == None:
                f1x = np.zeros_like(pn)
                f1y = np.zeros_like(pn)
                f2x = np.zeros_like(pn)
                f2y = np.zeros_like(pn)
                f3x = np.zeros_like(pn)
                f3y = np.zeros_like(pn)

            return f1x, f1y, f2x, f2y, f3x, f3y

        if integ == 'RK3':
            integ = RK3(type)
            a21 = integ.a21
            a31 = integ.a31
            a32 = integ.a32
            b3 = integ.b3
            Gpnm1x = self.Gpx(pnm1)
            Gpnm1y = self.Gpy(pnm1)
            f2x = np.zeros_like(Gpnx)
            f2y = np.zeros_like(f1x)

            if order == 'first':
                ## first order f1
                f1x = self.Gpx(pn)
                f1y = self.Gpy(pn)
                ## first order f2
                f2x = self.Gpx(pn) + a21 * (Gpnx - Gpnm1x)
                f2y = self.Gpy(pn) + a21 * (Gpny - Gpnm1y)

            elif order == 'second':
                f1x = Gpnx + (Gpnx - Gpnm1x) / 2
                f1y = Gpny + (Gpny - Gpnm1y) / 2
                f2x = f1x + (a21) * (Gpnx - Gpnm1x)
                f2y = f1y + (a21) * (Gpny - Gpnm1y)

            elif order == 'third':
                pnm2 = pold[2]
                Gpnm2x = self.Gpx(pnm2)
                Gpnm2y = self.Gpy(pnm2)

                # f1x = 5.0*Gpnx/3 - 5.0*Gpnm1x / 6 + 1.0*Gpnm2x/6
                # f1y = 5.0*Gpny/3 - 5.0*Gpnm1y / 6 + 1.0*Gpnm2y/6
                f1x = 11 * Gpnx / 6 - 7.0 * Gpnm1x / 6 + 1.0 * Gpnm2x / 3
                f1y = 11 * Gpny / 6 - 7.0 * Gpnm1y / 6 + 1.0 * Gpnm2y / 3
                f2x = f1x + (a21) * (Gpnx - Gpnm1x)
                f2y = f1y + (a21) * (Gpny - Gpnm1y)

            elif order == 'post_projection':
                pnm2 = pold[2]
                Gpnm2x = self.Gpx(pnm2)
                Gpnm2y = self.Gpy(pnm2)

                f1x = Gpnx
                f1y = Gpny

                Pnx_p = (3 * Gpnx - 4 * Gpnm1x + Gpnm2x) / 2   # Pnx'
                Pny_p = (3 * Gpny - 4 * Gpnm1y + Gpnm2y) / 2   # Pny'

                f2x = f1x + (a21) * Pnx_p
                f2y = f1y + (a21) * Pny_p

            elif order == 'capuano_ci_00':
                ##
                f1x = Gpnx + (Gpnx - Gpnm1x) / 2 + a21 * (Gpnx - Gpnm1x) / 2
                f1y = Gpny + (Gpny - Gpnm1y) / 2 + a21 * (Gpny - Gpnm1y) / 2
                f2x = Gpnx + (Gpnx - Gpnm1x) / 2 + (a31 + a32) * (Gpnx - Gpnm1x) / 2
                f2y = Gpny + (Gpny - Gpnm1y) / 2 + (a31 + a32) * (Gpny - Gpnm1y) / 2

            elif order == 'capuano_ci_01':
                ##
                f1x = Gpnx + (Gpnx - Gpnm1x) / 2 + a21 * (Gpnx - Gpnm1x) / 2
                f1y = Gpny + (Gpny - Gpnm1y) / 2 + a21 * (Gpny - Gpnm1y) / 2
                f2x = 0
                f2y = 0

            elif order == 'capuano_ci_10':
                ##
                f1x = 0
                f1y = 0
                f2x = Gpnx + (Gpnx - Gpnm1x) / 2 + (a31 + a32) * (Gpnx - Gpnm1x) / 2
                f2y = Gpny + (Gpny - Gpnm1y) / 2 + (a31 + a32) * (Gpny - Gpnm1y) / 2

            elif order == 'capuano_00':
                # in this form no need for c_i
                f1x = Gpnx + (Gpnx - Gpnm1x) / 2 +  (Gpnx - Gpnm1x) / 3
                f1y = Gpny + (Gpny - Gpnm1y) / 2 +  (Gpny - Gpnm1y) / 3
                f2x = Gpnx + (Gpnx - Gpnm1x) / 2 +  (Gpnx - Gpnm1x) / 3
                f2y = Gpny + (Gpny - Gpnm1y) / 2 +  (Gpny - Gpnm1y) / 3

            elif order == 'capuano_10':
                # in this form no need for c_i
                f1x = 0
                f1y = 0
                f2x = Gpnx + (Gpnx - Gpnm1x) / 2 +  (Gpnx - Gpnm1x) / 6 / (a31 + a32) / b3
                f2y = Gpny + (Gpny - Gpnm1y) / 2 +  (Gpny - Gpnm1y) / 6 / (a31 + a32) / b3

            elif order == 'capuano_01':
                # in this form no need for c_i
                f1x = Gpnx + (Gpnx - Gpnm1x) / 2
                f1y = Gpny + (Gpny - Gpnm1y) / 2
                f2x = 0
                f2y = 0

            elif order == None:
                ## f1 and f2 are zeros
                f1x = np.zeros_like(pn)
                f1y = np.zeros_like(pn)
                f2x = np.zeros_like(pn)
                f2y = np.zeros_like(pn)

            return f1x, f1y, f2x, f2y

        elif integ == 'RK2':
            integ = RK2(type,theta)
            a21 = integ.a21
            if order == 'second':
                Gpnm1x = self.Gpx(pnm1)
                Gpnm1y = self.Gpy(pnm1)
                ## first order f1
                f1x = Gpnx + (Gpnx - Gpnm1x) / 2
                f1y = Gpny + (Gpny - Gpnm1y) / 2
            elif order == 'first':
                ## first order f1
                f1x = self.Gpx(pn)
                f1y = self.Gpy(pn)
            elif order == 'capuano':
                ##
                Gpnm1x = self.Gpx(pnm1)
                Gpnm1y = self.Gpy(pnm1)
                f1x = Gpnx + (Gpnx - Gpnm1x) / 2 + a21 * (Gpnx - Gpnm1x) / 2
                f1y = Gpny + (Gpny - Gpnm1y) / 2 + a21 * (Gpny - Gpnm1y) / 2

            elif order == 'ml':
                # load json and create model
                json_file = open(ml_model, 'r')
                loaded_model_json = json_file.read()
                json_file.close()
                from tf_siren.siren import Sine
                loaded_model = model_from_json(loaded_model_json)
                # load weights into new model
                loaded_model.load_weights(ml_weights)
                loaded_model.compile(loss='mean_squared_error', metrics=['mean_absolute_error'], optimizer='Adam')

                Pn = pn.ravel()
                Pnm1 = pnm1.ravel()
                len = Pn.shape[0]
                input = np.ndarray(shape=(len, 2))
                input[:, 1] = Pn
                input[:, 0] = Pnm1

                p_predict= loaded_model.predict(input)
                p_predict= p_predict.reshape(pn.shape)

                Pnx = self.Gpx(p_predict)
                Pny = self.Gpy(p_predict)

                f1x = Pnx
                f1y = Pny

            elif order == 'ml_dt':
                # load json and create model
                json_file = open(ml_model, 'r')
                loaded_model_json = json_file.read()
                json_file.close()
                from tf_siren.siren import Sine
                loaded_model = model_from_json(loaded_model_json)
                # load weights into new model
                loaded_model.load_weights(ml_weights)
                loaded_model.compile(loss='mean_squared_error', metrics=['mean_absolute_error'], optimizer='Adam')

                Pn = pn.ravel()
                Pnm1 = pnm1.ravel()
                len = Pn.shape[0]
                input = np.ndarray(shape=(len, 3))
                input[:, 2] = Pn
                input[:, 1] = self.probDescription.dt
                input[:, 0] = Pnm1

                p_predict = loaded_model.predict(input)
                p_predict = p_predict.reshape(pn.shape)

                Pnx = self.Gpx(p_predict)
                Pny = self.Gpy(p_predict)

                f1x = Pnx
                f1y = Pny

            elif order == None:
                f1x = np.zeros_like(pn)
                f1y = np.zeros_like(pn)

            return f1x, f1y

    def observed_order(self, E2, E1, r):
        return -np.log(E2 / E1) / np.log(r)

    def order(self, E3, E2, E1, r):
        return -np.log((E3 - E2) / (E2 - E1)) / np.log(r)

    def gx(self, E2, E1, rx, h, p_hat):
        Norm = np.abs(E2 - E1)
        return Norm / ((h ** p_hat) * (rx ** p_hat - 1))

    def gt(self, E2, E1, rt, dx, p_hat):
        Norm = np.abs(E2 - E1)
        return Norm / ((dx ** p_hat) * (rt ** p_hat - 1))
