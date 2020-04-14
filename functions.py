import numpy as np
import scipy
import scipy.sparse
import scipy.sparse.linalg
import pyamg
import time
from singleton_classes import RK4, RK3, RK2


class func:
    def __init__(self, probDescription):
        self.probDescription = probDescription

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

    def urhs(self, u, v, splited=False):
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

        if splited:
            convection_x = np.zeros_like(urhs)
            diffusion_x = np.zeros_like(urhs)
            self.periodic_u(convection)
            self.periodic_u(diffusion)
            convection_x[1:ny + 1, 1:nx + 1] = convection
            diffusion_x[1:ny + 1, 1:nx + 1] = diffusion
            return convection_x, diffusion_x
        else:
            return urhs

    def nonlinear_conv_diff_x(self, u, v, splited=False):
        μ = self.probDescription.μ
        dx = self.probDescription.dx
        dy = self.probDescription.dy
        nx = self.probDescription.nx
        ny = self.probDescription.ny

        urhs = np.zeros_like(u)
        U = u.copy() - 1
        V = v.copy() - 1
        # do x-momentum first - u is of size (nx + 2) x (ny + 2)
        ue = 0.5 * (u[1:ny + 1, 2:nx + 2] + u[1:ny + 1, 1:nx + 1])
        uw = 0.5 * (u[1:ny + 1, 1:nx + 1] + u[1:ny + 1, :nx])

        un = 0.5 * (u[2:ny + 2, 1:nx + 1] + u[1:ny + 1, 1:nx + 1])
        us = 0.5 * (u[1:ny + 1, 1:nx + 1] + u[:ny, 1:nx + 1])

        vn = 0.5 * (v[2:ny + 2, 1:nx + 1] + v[2:ny + 2, :nx])
        vs = 0.5 * (v[1:ny + 1, 1:nx + 1] + v[1:ny + 1, :nx])

        # convection = - d(uu)/dx - d(vu)/dy
        convection = - U[1:ny + 1, 1:nx + 1] * (ue - uw) / dx - V[1:ny + 1, 1:nx + 1] * (vn - vs) / dy

        # diffusion = d2u/dx2 + d2u/dy2
        diffusion = μ * ((u[1:ny + 1, 2:nx + 2] - 2.0 * u[1:ny + 1, 1:nx + 1] + u[1:ny + 1, :nx]) / dx / dx + (
                    u[2:ny + 2, 1:nx + 1] - 2.0 * u[1:ny + 1, 1:nx + 1] + u[:ny, 1:nx + 1]) / dy / dy)

        convection_x = np.zeros_like(urhs)
        diffusion_x = np.zeros_like(urhs)

        self.periodic_u(convection)
        self.periodic_u(diffusion)

        convection_x[1:ny + 1, 1:nx + 1] = -convection
        diffusion_x[1:ny + 1, 1:nx + 1] = -diffusion
        return convection_x, diffusion_x

    def u_hat(self, u, v):
        μ = self.probDescription.μ
        dx = self.probDescription.dx
        dy = self.probDescription.dy
        nx = self.probDescription.nx
        ny = self.probDescription.ny
        dt = self.probDescription.dt

        uh = np.zeros_like(u)
        # uh[1:ny+1,1:nx+1] = u[1:ny+1,1:nx+1] + dt *urhs(u,v,μ,dx,dy,nx,ny)
        uh[1:ny + 1, 1:nx + 1] = u[1:ny + 1, 1:nx + 1] + dt * self.urhs(u, v, μ, dx, dy, nx, ny)
        # set periodicity in x and y
        self.periodic_u(uh)

        return uh

    def vrhs(self, u, v, splited=False):
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

        if splited:
            convection_y = np.zeros_like(vrhs)
            diffusion_y = np.zeros_like(vrhs)
            self.periodic_v(convection)
            self.periodic_v(diffusion)
            convection_y[1:ny + 1, 1:nx + 1] = convection
            diffusion_y[1:ny + 1, 1:nx + 1] = diffusion
            return convection_y, diffusion_y
        else:
            return vrhs

    def nonlinear_conv_diff_y(self, u, v):
        μ = self.probDescription.μ
        dx = self.probDescription.dx
        dy = self.probDescription.dy
        nx = self.probDescription.nx
        ny = self.probDescription.ny
        vrhs = np.zeros_like(u)
        U = u.copy() - 1
        V = v.copy() - 1
        # do y-momentum
        ve = 0.5 * (v[1:ny + 1, 2:nx + 2] + v[1:ny + 1, 1:nx + 1])
        vw = 0.5 * (v[1:ny + 1, 1:nx + 1] + v[1:ny + 1, :nx])

        ue = 0.5 * (u[1:ny + 1, 2:nx + 2] + u[:ny, 2:nx + 2])
        uw = 0.5 * (u[1:ny + 1, 1:nx + 1] + u[:ny, 1:nx + 1])

        vn = 0.5 * (v[2:ny + 2, 1:nx + 1] + v[1:ny + 1, 1:nx + 1])
        vs = 0.5 * (v[1:ny + 1, 1:nx + 1] + v[:ny, 1:nx + 1])

        # convection = d(uv)/dx + d(vv)/dy
        convection = - U[1:ny + 1, 1:nx + 1] * (ve - vw) / dx - V[1:ny + 1, 1:nx + 1] * (vn - vs) / dy
        # diffusion = d2u/dx2 + d2u/dy2
        diffusion = μ * ((v[1:ny + 1, 2:nx + 2] - 2.0 * v[1:ny + 1, 1:nx + 1] + v[1:ny + 1, :nx]) / dx / dx + (
                    v[2:ny + 2, 1:nx + 1] - 2.0 * v[1:ny + 1, 1:nx + 1] + v[:ny, 1:nx + 1]) / dy / dy)

        convection_y = np.zeros_like(vrhs)
        diffusion_y = np.zeros_like(vrhs)
        self.periodic_v(convection)
        self.periodic_v(diffusion)
        convection_y[1:ny + 1, 1:nx + 1] = -convection
        diffusion_y[1:ny + 1, 1:nx + 1] = -diffusion
        return convection_y, diffusion_y

    def v_hat(self, u, v):
        μ = self.probDescription.μ
        dx = self.probDescription.dx
        dy = self.probDescription.dy
        nx = self.probDescription.nx
        ny = self.probDescription.ny
        dt = self.probDescription.dt

        vh = np.zeros_like(u)
        # do y-momentum
        vh[1:ny + 1, 1:nx + 1] = v[1:ny + 1, 1:nx + 1] + dt * self.vrhs(u, v, μ, dx, dy, nx, ny)
        # set periodicity in x and y
        self.periodic_v(vh)
        return vh

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
        self.periodic_u(gpx)
        return gpx

    def Gpy(self, p):
        dy = self.probDescription.dy
        nx = self.probDescription.nx
        ny = self.probDescription.ny
        gpy = np.zeros([ny + 2, nx + 2])
        gpy[1:, 1:-1] = ((p[1:, 1:-1] - p[:-1, 1:-1]) / dy)
        self.periodic_v(gpy)
        return gpy

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

    def ImQ(self, uh, vh, Coef, p0=0):
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
            # ptmp,_ = scipy.sparse.linalg.cg(A, b, callback=callback)
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

    def ImQ_timed(self, uh, vh, Coef, p0=0):
        dx = self.probDescription.dx
        dy = self.probDescription.dy
        dt = self.probDescription.dt
        nx = self.probDescription.nx
        ny = self.probDescription.ny

        unp1 = np.zeros_like(uh)
        vnp1 = np.zeros_like(vh)
        divuhat = self.div(uh, vh, dx, dy, nx, ny)

        prhs = 1.0 / dt * divuhat[1:-1, 1:-1]

        rhs = prhs.ravel()

        def solver(A, b):
            num_iters = 0

            def callback(xk):
                nonlocal num_iters
                num_iters += 1

            ml = pyamg.ruge_stuben_solver(A)
            ptmp = ml.solve(b, tol=1e-12, callback=callback)
            # ptmp = scipy.sparse.linalg.spsolve(A,b,callback=callback)
            return ptmp, num_iters

        solver_time_start = time.clock()
        ptmp, num_iters = solver(Coef, rhs)
        solver_time = time.clock() - solver_time_start
        p = np.zeros([ny + 2, nx + 2])
        p[1:-1, 1:-1] = ptmp.reshape([ny, nx])

        # set periodicity on the pressure
        self.periodic_scalar(p)

        # time advance
        unp1[1:-1, 1:] = uh[1:-1, 1:] - dt * (p[1:-1, 1:] - p[1:-1, :-1]) / dx
        vnp1[1:, 1:-1] = vh[1:, 1:-1] - dt * (p[1:, 1:-1] - p[:-1, 1:-1]) / dy

        self.periodic_u(unp1)
        self.periodic_v(vnp1)

        return unp1, vnp1, p, num_iters, solver_time

    def Guess(self, pold, order=None, integ='Rk3', type='regular'):
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
            f2x = np.zeros_like(Gpnx)
            f2y = np.zeros_like(Gpny)

            f3x = np.zeros_like(Gpnx)
            f3y = np.zeros_like(Gpny)

            if order == 'third':
                dt = self.probDescription.get_dt()
                Pnx = (15 * Gpnx - 10 * Gpnm1x + 3 * Gpnm2x) / 8
                # Pnx = Gpnx +  (Gpnx - Gpnm1x)/2
                # Pnx = Gpnx
                Pny = (15 * Gpny - 10 * Gpnm1y + 3 * Gpnm2y) / 8
                # Pny = Gpny +  (Gpny - Gpnm1y)/2
                # Pny = Gpny
                Pnx_p = (2 * Gpnx - 3 * Gpnm1x + Gpnm2x)/dt  # Pnx'
                Pny_p = (2 * Gpny - 3 * Gpnm1y + Gpnm2y)/dt  # Pny'
                Pnx_pp = (Gpnx - 2 * Gpnm1x + Gpnm2x) / 2 /dt/dt  # Pnx''
                Pny_pp = (Gpny - 2 * Gpnm1y + Gpnm2y) / 2 /dt/dt # Pny''

                f1x = Pnx
                f1y = Pny

                f2x = Pnx + a21 *dt* Pnx_p
                f2y = Pny + a21 *dt* Pny_p

                f3x = Pnx + (a31 + a32) * dt * Pnx_p + a32 * a21 * dt*dt* Pnx_pp
                f3y = Pny + (a31 + a32) * dt * Pny_p + a32 * a21 * dt*dt* Pny_pp


            elif order =='fourth':
                dt = self.probDescription.get_dt()
                pnm3 = pold[3]
                Gpnm3x = self.Gpx(pnm3)
                Gpnm3y = self.Gpy(pnm3)

                Pnx = (35 * Gpnx - 35 * Gpnm1x + 21 * Gpnm2x - 5*Gpnm3x) / 16
                # Pnx = Gpnx +  (Gpnx - Gpnm1x)/2
                # Pnx = Gpnx
                Pny = (35 * Gpny - 35 * Gpnm1y + 21 * Gpnm2y- 5*Gpnm3y) / 16
                # Pny = Gpny +  (Gpny - Gpnm1y)/2
                # Pny = Gpny
                Pnx_p = (71 * Gpnx - 141 * Gpnm1x + 93*Gpnm2x -23*Gpnm3x) / dt /24  # Pnx'
                Pny_p = (71 * Gpny - 141 * Gpnm1y + 93*Gpnm2y -23*Gpnm3y) / dt /24 # Pny'
                Pnx_pp = (5*Gpnx - 13 * Gpnm1x + 11*Gpnm2x -3*Gpnm3x) / 4 / dt / dt  # Pnx''
                Pny_pp = (5*Gpny - 13 * Gpnm1y + 11*Gpnm2y -3*Gpnm3y) / 4 / dt / dt  # Pny''

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

            elif order == 'capuano':
                ##
                f1x = Gpnx + (Gpnx - Gpnm1x) / 2 + a21 * (Gpnx - Gpnm1x) / 2
                f1y = Gpny + (Gpny - Gpnm1y) / 2 + a21 * (Gpny - Gpnm1y) / 2
                f2x = Gpnx + (Gpnx - Gpnm1x) / 2 + (a31 + a32) * (Gpnx - Gpnm1x) / 2
                f2y = Gpny + (Gpny - Gpnm1y) / 2 + (a31 + a32) * (Gpny - Gpnm1y) / 2

                # in this form no need for c_i
                # f1x = Gpnx + (Gpnx - Gpnm1x) / 2 +  (Gpnx - Gpnm1x) / 3
                # f1y = Gpny + (Gpny - Gpnm1y) / 2 +  (Gpny - Gpnm1y) / 3
                # f2x = Gpnx + (Gpnx - Gpnm1x) / 2 +  (Gpnx - Gpnm1x) / 3
                # f2y = Gpny + (Gpny - Gpnm1y) / 2 +  (Gpny - Gpnm1y) / 3

            elif order == None:
                ## f1 and f2 are zeros
                f1x = np.zeros_like(pn)
                f1y = np.zeros_like(pn)
                f2x = np.zeros_like(pn)
                f2y = np.zeros_like(pn)

            return f1x, f1y, f2x, f2y

        elif integ == 'RK2':
            integ = RK2(type)
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
