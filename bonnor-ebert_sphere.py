import numpy as np
import pylab as pl
from scipy.integrate import odeint, quad

#x = r/r_c
#y = ln(rho/rho_c)
#z1 = y
#z2 = dz1/dx
#dz2/dx = -2/x*z2-np.exp(z1)

#define Bonnor-Ebert equilibria
def BE(boundaries, x):
	z1 = boundaries[0]
	z2 = boundaries[1]
	dz1dx = z2
	dz2dx = -2.0/x*z2 - np.exp(z1)
	return [dz1dx, dz2dx]

#boundary conditions and x-range
BCs = [0,0]
x = np.logspace(-3,7,1000)

#solution to ODE
solution = odeint(BE, BCs, x)
z1 = solution[:, 0]
z2 = solution[:, 1]

#set in terms of what we want
y = z1
rho_rhoc = np.exp(y)
r_rc = x


'''MAKE SURE UNITS WORK OUT'''
########
###a)###
########
pl.rc('font', size=15)
pl.rc('lines', linewidth=2)

#figure rho/rho_c vs. r/r_c
pl.loglog(x,rho_rhoc,'k-')
pl.loglog(x,1.7*x**(-2),'b--', label=r'$\propto x^{-2}$')

pl.axhline(y = 1, c='r', ls='--', label=r'$\propto x^0$')
#pl.axvline(x = 30, c='r', ls='--')

pl.legend(loc=1,fontsize=18)
pl.ylim(1E-4,10)
pl.xlim(1E-1,50)
pl.xlabel(r'$r/r_c$', size=18)
pl.ylabel(r'$\rho / \rho_c$', size=18)
pl.savefig('density.pdf', bbox_inches='tight')
#pl.show()
pl.close()

########
###b)###
########

##define mass of sphere
def M(rho,r):
#    return 4*np.pi*rho*r**3/3.0
    return rho*r**3/3.0

mass = M(rho_rhoc,x)

#figure M(r) vs. x
pl.loglog(x,mass, 'k-')
pl.loglog(x,4./(4*np.pi)*x**(3),'b--', label=r'$\propto x^{3}$')
pl.loglog(x,7./(4*np.pi)*x**(1),'r--', label=r'$\propto x^{1}$')

pl.legend(loc=2,fontsize=18)
pl.xlim(1E-1,50)
pl.ylim(1E-3,1E3)
pl.ylabel(r'$m_r$ $[c_s^3 \rho_c^{-1/2} G^{3/2}] $', size=18)
pl.xlabel(r'$r/r_c$', size=18)
pl.savefig('mass.pdf', bbox_inches='tight')
#pl.show()
pl.close()

########
###c)###
########

dimless_mass = mass*np.sqrt(rho_rhoc)/np.sqrt(2.9246796896)#/(np.pi**2*2)/1.0887942136 ####GET RID OF THESE NORMALIZATION FACTORS####

print "max m(r_0) resides at x = ", round(x[np.where(dimless_mass == max(dimless_mass))[0][0]],3)

#figure of dimensionless mass
pl.semilogx(x,dimless_mass, 'k-')
pl.xlim(1E-1,50)
pl.ylabel(r'$m(r_0)$ $[M(r) / (c_s^3 \rho_0^{-1/2} G^{3/2})] $', size=18)
pl.xlabel(r'$r_0/r_c$', size=18)
pl.rc('font', size=15)
pl.savefig('dimless_mass.pdf', bbox_inches='tight')
#pl.show()
pl.close()

########
###d)###
########

pressure = dimless_mass**2

print max(pressure)
print "max p(r_0) resides at x = ", round(x[np.where(pressure == max(pressure))[0][0]],3)

#figure of pressure
pl.semilogx(x,pressure, 'k-')
#pl.semilogx(x,0.03*x**(4),'b--', label=r'$\propto x^{4}$')
#pl.loglog(x,475*x**(-3)+0.16,'r--', label=r'$\propto x^{-3}$')

pl.xlim(1E-1,50)
pl.ylim(0,1.4)
pl.ylabel(r'$P_0(x)$ $[c_s^8 G^3 M^2] $', size=18)
pl.xlabel(r'$r_0/r_c$', size=18)
#pl.legend(loc=2,fontsize=18)
pl.savefig('pressure.pdf', bbox_inches='tight')
#pl.show()
pl.close()


#max mass
print "the maximun mass of a BE sphere is max(m(r)) * c_s^4/P_0^{1/2} G^{3/2}"
print "max mass = ", round(max(dimless_mass),3)," c_s^4/P_0^{1/2} G^{3/2}"

rho_rhoc/=2.7777

#figure of pressure vs. y
pl.semilogx(rho_rhoc,pressure, 'k-')
#pl.semilogx(x,0.03*x**(4),'b--', label=r'$\propto x^{4}$')
#pl.loglog(x,475*x**(-3)+0.16,'r--', label=r'$\propto x^{-3}$')
pl.semilogx(rho_rhoc[np.where(pressure == max(pressure))[0][0]], max(pressure), 'bo')
pl.axvline(x=rho_rhoc[np.where(pressure == max(pressure))[0][0]],c='b',ls='--')
pl.text(0.1,1.3, r'$(\rho / \rho_c)_{max} \sim 1/14.1$', size=18)
pl.xlim(1E-3,2)
pl.ylim(0,1.4)
pl.xlabel(r'$\rho / \rho_c$', size=18)
pl.ylabel(r'$P_0(x)$ $[c_s^8 G^3 M^2] $', size=18)
#pl.legend(loc=2,fontsize=18)
pl.savefig('pressure_density.pdf', bbox_inches='tight')
#pl.show()
pl.close()

print round(rho_rhoc[np.where(pressure == max(pressure))[0][0]],3)

