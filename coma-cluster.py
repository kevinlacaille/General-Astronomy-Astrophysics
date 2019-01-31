import numpy as np
import matplotlib.pyplot as pl
from scipy.optimize import curve_fit
import coords as co
import xlrd
import matplotlib.patches as patches
from PIL import Image

# Define a Gaussian function
def Gaussian(x, A, mu, sigma):
    return A*np.exp(-0.5*((x-mu)/sigma)**2)

# Physical properties from Ned Wright cosmo calc.
DL = 101.6*1e6 #pc
scale = 0.471 #kpc/"
# Coma Cluster properties from Zwicky+1942
z_coma = 0.023100
v_coma = 6925.0 #km/s
sigma_coma = 1000.0 #km/s
RA_coma=co.convHMS('12:59:48.7') #deg
DEC_coma=co.convDMS('+27:58:50') #deg


################
# Importing Data
################
loc = ('coma.xlsx')
wb=xlrd.open_workbook(loc)
sheet=wb.sheet_by_index(0)

# Sorting data and converting to useful units
ID = []
RA = []
DEC = []
type = []
vel = []
z = []
mag = []
d_coma = []
for i in range(sheet.nrows)[1:]:
    ID.append(str(sheet.cell_value(i,0)) + str(sheet.cell_value(i,1)))
    RA_tmp = str(sheet.cell_value(i,2))
    RA.append(co.convHMS(RA_tmp.split('m')[0][:2] +':'+ RA_tmp.split('m')[0][3:]+':'+RA_tmp.split('m')[1][:4]))
    DEC_tmp = str(sheet.cell_value(i,3))
    DEC.append(co.convDMS(DEC_tmp.split('m')[0][1:3]+':'+DEC_tmp.split('m')[0][-2:]+':'+DEC_tmp.split('m')[1][0:2]))
    type.append(str(sheet.cell_value(i,4)))
    vel.append(float(sheet.cell_value(i,5)))
    z.append(float(sheet.cell_value(i,6)))
    mag.append(str(sheet.cell_value(i,7)))
    d_coma.append(float(sheet.cell_value(i,8)))
ID = np.array(ID)
RA = np.array(RA)
DEC = np.array(DEC)
type = np.array(type)
vel = np.array(vel) #km/s
z = np.array(z)
mag = np.array(mag)
d_coma = np.array(d_coma)*60*scale #kpc

# Selcting all r- or R-band data
r_ID = []
r_RA = []
r_DEC = []
r_z = []
r_d_coma = []
r_mag = []
for i in range(len(mag)):
    if mag[i][-1]=='r' or mag[i][-1]=='R':
        r_ID.append(ID[i])
        r_RA.append(RA[i])
        r_DEC.append(DEC[i])
        r_z.append(z[i])
        r_d_coma.append(d_coma[i])
        r_mag.append(float(mag[i][:-1]))
r_ID = np.array(r_ID)
r_RA = np.array(r_RA)
r_DEC = np.array(r_DEC)
r_z = np.array(r_z)
r_d_coma = np.array(r_d_coma)
r_mag = np.array(r_mag)
r = r_mag-5*(np.log10(DL)-1)
L_r = 10**(0.4*(4.60-r)) #Lsun -- Sun_R (Coursins-R) = 4.60 #http://mips.as.arizona.edu/~cnaw/sun.html

# Selcting all g- or G-band data
g_ID = []
g_RA = []
g_DEC = []
g_z = []
g_d_coma = []
g_mag = []
for i in range(len(mag)):
    if mag[i][-1]=='g' or mag[i][-1]=='G':
        g_ID.append(ID[i])
        g_RA.append(RA[i])
        g_DEC.append(DEC[i])
        g_z.append(z[i])
        g_d_coma.append(d_coma[i])
        g_mag.append(float(mag[i][:-1]))
g_ID = np.array(g_ID)
g_RA = np.array(g_RA)
g_DEC = np.array(g_DEC)
g_z = np.array(g_z)
g_d_coma = np.array(g_d_coma)
g_mag = np.array(g_mag)
g = g_mag-5*(np.log10(DL)-1)
L_g = 10**(0.4*(5.11-g)) #Lsun -- Sun_g (SDSS-g) = 5.11 #http://mips.as.arizona.edu/~cnaw/sun.html

# Selcting all b- or B-band data
b_ID = []
b_RA = []
b_DEC = []
b_z = []
b_d_coma = []
b_mag = []
for i in range(len(mag)):
    if mag[i][-1]=='b' or mag[i][-1]=='B':
        b_ID.append(ID[i])
        b_RA.append(RA[i])
        b_DEC.append(DEC[i])
        b_z.append(z[i])
        b_d_coma.append(d_coma[i])
        b_mag.append(float(mag[i][:-1]))
b_ID = np.array(b_ID)
b_RA = np.array(b_RA)
b_DEC = np.array(b_DEC)
b_z = np.array(b_z)
b_d_coma = np.array(b_d_coma)
b_mag = np.array(b_mag)
b = b_mag-5*(np.log10(DL)-1)
L_b = 10**(0.4*(5.31-b)) #Lsun -- Sun_g (Johnson-B) = 5.31 #http://mips.as.arizona.edu/~cnaw/sun.html

# Measure median and luminosity-weighted mean coordinates
median_RA = np.median(RA_coma-RA)
median_DEC = np.median(DEC_coma-DEC)
mean_RA_r = RA_coma-np.average(r_RA,weights=1/L_r)
mean_DEC_r = DEC_coma-np.average(r_DEC,weights=1/L_r)
mean_RA_b = RA_coma-np.average(b_RA,weights=1/L_b)
mean_DEC_b = DEC_coma-np.average(b_DEC,weights=1/L_b)
mean_RA_g = RA_coma-np.average(g_RA,weights=1/L_g)
mean_DEC_g = DEC_coma-np.average(g_DEC,weights=1/L_g)

print 'median: ' + co.deg2HMS(median_RA), co.deg2DMS(median_DEC)
print 'r-band mean: ' + co.deg2HMS(mean_RA_r), co.deg2DMS(mean_DEC_r) + ', ' + str(round(mean_RA_r*3600,1)),str(round(mean_DEC_r*3600,1)) + 'arcsec = ' + str(round(np.sqrt(((mean_RA_r*3600)*np.cos(DEC_coma*np.pi/180))**2 + (mean_DEC_r*3600)**2),1)) + 'arcsec, ' + str(round(mean_RA_r*3600*scale,1)),str(round(mean_DEC_r*3600*scale,1)) + 'kpc = ' + str(round(np.sqrt(((mean_RA_r*3600)*np.cos(DEC_coma*np.pi/180))**2 + (mean_DEC_r*3600)**2)*scale,1)) + 'kpc'
print 'b-band mean: ' + co.deg2HMS(mean_RA_b), co.deg2DMS(mean_DEC_b) + ', ' + str(round(mean_RA_b*3600,1)),str(round(mean_DEC_b*3600,1)) + 'arcsec = ' + str(round(np.sqrt(((mean_RA_b*3600)*np.cos(DEC_coma*np.pi/180))**2 + (mean_DEC_b*3600)**2),1)) + 'arcsec, ' + str(round(mean_RA_b*3600*scale,1)),str(round(mean_DEC_b*3600*scale,1)) + 'kpc = ' + str(round(np.sqrt(((mean_RA_b*3600)*np.cos(DEC_coma*np.pi/180))**2 + (mean_DEC_b*3600)**2)*scale,1)) + 'kpc'
print 'g-band mean: ' + co.deg2HMS(mean_RA_g), co.deg2DMS(mean_DEC_g) + ', ' + str(round(mean_RA_g*3600,1)),str(round(mean_DEC_g*3600,1)) + 'arcsec = ' + str(round(np.sqrt(((mean_RA_g*3600)*np.cos(DEC_coma*np.pi/180))**2 + (mean_DEC_g*3600)**2),1)) + 'arcsec, ' + str(round(mean_RA_g*3600*scale,1)),str(round(mean_DEC_g*3600*scale,1)) + 'kpc = ' + str(round(np.sqrt(((mean_RA_g*3600)*np.cos(DEC_coma*np.pi/180))**2 + (mean_DEC_g*3600)**2)*scale,1)) + 'kpc'

###############
# Plot galaxies
###############
fig, axs = pl.subplots(2, 1,figsize=(5,8))
ax1=axs[0]
ax2=axs[1]
font = {'family' : 'sans-serif',
        'weight' : 'normal',
        'size'   : 14}
pl.rc('font', **font)
ax1.set_xscale('linear')
ax1.set_yscale('linear')

# Whole field
ax1.plot(RA_coma-g_RA,DEC_coma-g_DEC,marker='o',linestyle='none',c='g',alpha=0.2,zorder=9,label='g-band: '+str(len(g_ID)))
ax1.plot(RA_coma-r_RA,DEC_coma-r_DEC,marker='o',linestyle='none',c='r',alpha=0.2,zorder=10,label='r-band: '+str(len(r_ID)))
ax1.plot(RA_coma-b_RA,DEC_coma-b_DEC,marker='o',linestyle='none',c='b',alpha=0.2,zorder=10,label='b-band: '+str(len(b_ID)))
ax1.plot(RA_coma-RA,DEC_coma-DEC,marker='o',linestyle='none',c='k',alpha=0.2,zorder=0,label='no mag: '+str(len(ID)-len(g_ID)+len(r_ID)+len(b_ID)))

# Create a Rectangle patch for core
rect = patches.Rectangle((-0.3,0.3),0.6,-0.6,linewidth=2,edgecolor='k',facecolor='none')
ax1.add_patch(rect)

ax1.grid()
ax1.set_ylim(-2,2)
ax1.set_xlim(-2,2)
ax1.legend(fontsize=12,loc=1,ncol=1,frameon=True,numpoints=1,scatterpoints=1,bbox_to_anchor=(1.47, 1.01))
ax1.set_xlabel(r'$\Delta$ R.A. (deg)')
ax1.set_ylabel(r'$\Delta$ Decl. (deg)')

# Core
ax2.plot(RA_coma-g_RA,DEC_coma-g_DEC,marker='o',linestyle='none',c='g',alpha=0.2,zorder=9)
ax2.plot(RA_coma-r_RA,DEC_coma-r_DEC,marker='o',linestyle='none',c='r',alpha=0.2,zorder=10)
ax2.plot(RA_coma-b_RA,DEC_coma-b_DEC,marker='o',linestyle='none',c='b',alpha=0.2,zorder=10)
ax2.plot(RA_coma-RA,DEC_coma-DEC,marker='o',linestyle='none',c='k',alpha=0.2,zorder=0)

ax2.plot(mean_RA_g,mean_DEC_g,marker='X',ms=20,linestyle='none',c='g',alpha=0.5, label='g-band centre',zorder=10)
ax2.plot(mean_RA_r,mean_DEC_r,marker='X',ms=20,linestyle='none',c='r',alpha=0.5, label='r-band centre',zorder=10)
ax2.plot(mean_RA_b,mean_DEC_b,marker='X',ms=20,linestyle='none',c='b',alpha=0.5, label='b-band centre',zorder=10)
ax2.plot(median_RA,median_DEC,marker='X',ms=20,linestyle='none',c='k',alpha=0.5, label='median centre',zorder=10)

ax2.grid()
ax2.set_ylim(-0.3,0.3)
ax2.set_xlim(-0.3,0.3)
ax2.legend(fontsize=12,loc=1,ncol=1,frameon=True,numpoints=1,scatterpoints=1,bbox_to_anchor=(1.5, 1.01))
ax2.set_xlabel(r'$\Delta$ R.A. (deg)')
ax2.set_ylabel(r'$\Delta$ Decl. (deg)')

pl.savefig('coma_cluster.pdf',bbox_inches='tight')
pl.close()





############
# Histograms
############
fig, axs = pl.subplots(2, 1,figsize=(5,8))
ax1=axs[0]
ax2=axs[1]
#fig, (ax1, ax2) = pl.subplots(nrows=2, sharex=True,figsize=(6,10))
font = {'family' : 'sans-serif',
        'weight' : 'normal',
        'size'   : 14}
pl.rc('font', **font)
ax1.set_xscale('linear')
ax1.set_yscale('linear')

# Compute redshift distrubution and fit Gaussian
zspace=np.linspace(0.02,0.026,150)
z_data, bins = np.histogram(z,bins=20)
popt, pcov = curve_fit(Gaussian, xdata=bins[:-1], ydata=z_data, p0=[40,np.mean(z), np.std(z)])
print popt,np.sqrt(np.diag(pcov))
A,mu,sigma=popt
e_A,e_mu,e_sigma=np.sqrt(np.diag(pcov))

print '<z> = ' + str(round(mu,4)) + '+/-' + str(round(e_sigma,4))
print '<z> - z_coma = ' + str(round((mu-z_coma),6)) + ': <z>/z_coma = ' + str(round(mu/z_coma,5)) + ': e_<z>/delta z = ' + str(round(e_mu/(mu-z_coma),5))
print 'sigma(z) = ' + str(round(sigma,4)) + '+/-' + str(round(e_sigma,4))

ax1.hist(z,20)
ax1.plot(zspace,Gaussian(zspace,A,mu,sigma),c='r',ls='-')
ax1.plot(zspace,Gaussian(zspace,A,mu,sigma+e_sigma),c='r',ls='--')
ax1.plot(zspace,Gaussian(zspace,A,mu,sigma-e_sigma),c='r',ls='--')

ax1.set_xlabel(r'$z$')
ax1.set_ylabel('N')

# Compute velocity distrubution and fit Gaussian
vspace=np.linspace(6000,8000,150)
v_data, bins = np.histogram(vel,bins=20)
popt, pcov = curve_fit(Gaussian, xdata=bins[:-1], ydata=v_data, p0=[40,np.mean(vel), np.std(vel)])
print popt,np.sqrt(np.diag(pcov))
A,mu,sigma=popt
e_A,e_mu,e_sigma=np.sqrt(np.diag(pcov))

print '<v> = ' + str(int(round(mu,0))) + '+/-' + str(int(round(e_sigma,0))) + ' km/s'
print '<v> - v_coma = ' + str(round((mu-v_coma),6)) + ': <v>/v_coma = ' + str(round(mu/v_coma,5)) + ': e_<v>/delta v = ' + str(round(e_mu/(mu-v_coma),5))
print 'sigma(v) = ' + str(int(round(sigma,0))) + '+/-' + str(int(round(e_sigma,0))) + ' km/s'
print '<sigma> - sigma_coma = ' + str(round((sigma-sigma_coma),6)) + ': <sigma>/sigma_coma = ' + str(round(sigma/sigma_coma,5)) + ': e_<sigma>/delta sigma = ' + str(round(e_sigma/(sigma-sigma_coma),5))

ax2.hist(vel,20)
ax2.plot(vspace,Gaussian(vspace,A,mu,sigma),c='r',ls='-')
ax2.plot(vspace,Gaussian(vspace,A,mu,sigma+e_sigma),c='r',ls='--')
ax2.plot(vspace,Gaussian(vspace,A,mu,sigma-e_sigma),c='r',ls='--')
ax2.set_xlabel(r'$v$ (km s$^{-1}$)')
ax2.set_ylabel('N')

pl.savefig('z_v.pdf')
pl.close()





# Compute counts for all data
r = np.sqrt(((RA_coma-RA)*np.cos(RA_coma*np.pi/180))**2 + (DEC_coma-DEC)**2)
N, radius = np.histogram(r,15)
# Compute counts for g-band data - they resemble the field, mostly
g_r = np.sqrt(((RA_coma-g_RA)*np.cos(RA_coma*np.pi/180))**2 + (DEC_coma-g_DEC)**2)
g_N, g_radius = np.histogram(g_r,15)

# Manually sum up the luminosities for each bin
L_g_binned = np.array((sum(L_g[:g_N[0]])
,sum(L_g[g_N[0]:g_N[0]+g_N[1]])
,sum(L_g[g_N[0]+g_N[1]:g_N[0]+g_N[1]+g_N[2]])
,sum(L_g[g_N[0]+g_N[1]+g_N[2]:g_N[0]+g_N[1]+g_N[2]+g_N[3]])
,sum(L_g[g_N[0]+g_N[1]+g_N[2]+g_N[3]:g_N[0]+g_N[1]+g_N[2]+g_N[3]+g_N[4]])
,sum(L_g[g_N[0]+g_N[1]+g_N[2]+g_N[3]+g_N[4]:g_N[0]+g_N[1]+g_N[2]+g_N[3]+g_N[4]+g_N[5]])
,sum(L_g[g_N[0]+g_N[1]+g_N[2]+g_N[3]+g_N[4]+g_N[5]:g_N[0]+g_N[1]+g_N[2]+g_N[3]+g_N[4]+g_N[5]+g_N[6]])
,sum(L_g[g_N[0]+g_N[1]+g_N[2]+g_N[3]+g_N[4]+g_N[5]+g_N[6]:g_N[0]+g_N[1]+g_N[2]+g_N[3]+g_N[4]+g_N[5]+g_N[6]+g_N[7]])
,sum(L_g[g_N[0]+g_N[1]+g_N[2]+g_N[3]+g_N[4]+g_N[5]+g_N[6]+g_N[7]:g_N[0]+g_N[1]+g_N[2]+g_N[3]+g_N[4]+g_N[5]+g_N[6]+g_N[7]+g_N[8]])
,sum(L_g[g_N[0]+g_N[1]+g_N[2]+g_N[3]+g_N[4]+g_N[5]+g_N[6]+g_N[7]+g_N[8]:g_N[0]+g_N[1]+g_N[2]+g_N[3]+g_N[4]+g_N[5]+g_N[6]+g_N[7]+g_N[8]+g_N[9]])
,sum(L_g[g_N[0]+g_N[1]+g_N[2]+g_N[3]+g_N[4]+g_N[5]+g_N[6]+g_N[7]+g_N[8]+g_N[9]:g_N[0]+g_N[1]+g_N[2]+g_N[3]+g_N[4]+g_N[5]+g_N[6]+g_N[7]+g_N[8]+g_N[9]+g_N[10]])
,sum(L_g[g_N[0]+g_N[1]+g_N[2]+g_N[3]+g_N[4]+g_N[5]+g_N[6]+g_N[7]+g_N[8]+g_N[9]+g_N[10]:g_N[0]+g_N[1]+g_N[2]+g_N[3]+g_N[4]+g_N[5]+g_N[6]+g_N[7]+g_N[8]+g_N[9]+g_N[10]+g_N[11]])
,sum(L_g[g_N[0]+g_N[1]+g_N[2]+g_N[3]+g_N[4]+g_N[5]+g_N[6]+g_N[7]+g_N[8]+g_N[9]+g_N[10]+g_N[11]:g_N[0]+g_N[1]+g_N[2]+g_N[3]+g_N[4]+g_N[5]+g_N[6]+g_N[7]+g_N[8]+g_N[9]+g_N[10]+g_N[11]+g_N[12]])
,sum(L_g[g_N[0]+g_N[1]+g_N[2]+g_N[3]+g_N[4]+g_N[5]+g_N[6]+g_N[7]+g_N[8]+g_N[9]+g_N[10]+g_N[11]+g_N[12]:g_N[0]+g_N[1]+g_N[2]+g_N[3]+g_N[4]+g_N[5]+g_N[6]+g_N[7]+g_N[8]+g_N[9]+g_N[10]+g_N[11]+g_N[12]+g_N[13]])
,sum(L_g[g_N[0]+g_N[1]+g_N[2]+g_N[3]+g_N[4]+g_N[5]+g_N[6]+g_N[7]+g_N[8]+g_N[9]+g_N[10]+g_N[11]+g_N[12]+g_N[13]:g_N[0]+g_N[1]+g_N[2]+g_N[3]+g_N[4]+g_N[5]+g_N[6]+g_N[7]+g_N[8]+g_N[9]+g_N[10]+g_N[11]+g_N[12]+g_N[13]+g_N[14]])))

######################
# Number density plots
######################
fig, axs = pl.subplots(2, 1,figsize=(5,8))
ax1=axs[0]
ax2=axs[1]
font = {'family' : 'sans-serif',
        'weight' : 'normal',
        'size'   : 14}
pl.rc('font', **font)

# Counts density
ax1.set_xscale('linear')
ax1.set_yscale('linear')

ax1.plot(radius[1:],N,drawstyle='steps',c='k',label='all')
ax1.plot(g_radius[1:],g_N,drawstyle='steps',c='g',label='g-band')

ax1.set_xlim(0,2)
ax1.legend(fontsize=12,loc=1,ncol=1,frameon=True,numpoints=1,scatterpoints=1)#,bbox_to_anchor=(1.35, 1.01))
ax1.set_xlabel(r'r$_{proj.}$ (deg)')
ax1.set_ylabel(r'N (deg$^{-1}$)')

# Luminosity density
ax2.set_xscale('linear')
ax2.set_yscale('log')
ax2.plot(radius[1:],L_g_binned/(np.pi*radius[1:]**2),drawstyle='steps',c='g',label='g-band')
#Poisson error statistics - error \propto 1/\sqrt(N)
ax2.errorbar(radius[1:]-(radius[1]-radius[0])/2,L_g_binned/(np.pi*radius[1:]**2),yerr=L_g_binned/(np.pi*radius[1:]**2)*np.sqrt(1/N.astype(np.float)), linestyle='none', c='g')

ax2.set_xlabel(r'r$_{proj.}$ (deg)')
ax2.set_ylabel(r'L$_g$ (L$_\odot$ deg$^{-1}$)')
ax2.set_xlim(0,2)


pl.savefig('number_luminosity.pdf',bbox_inches='tight')
pl.close()
