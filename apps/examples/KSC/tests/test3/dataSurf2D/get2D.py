import numpy as np

lim = 2.6  # x,y min,max
num = 250
delt = 2*lim/(num-1)

xdat = np.arange(-lim,lim+delt,delt)
ydat = xdat.copy()
xydat = np.zeros((num*num,2))

for ix in range(num):
  for iy in range(num):
    i = ix*num + iy
    xydat[i,0] = -lim+ix*delt
    xydat[i,1] = -lim+iy*delt


np.savetxt("xydata.dat",xydat, fmt="%8.4e") 
np.savetxt("xdata.dat", xdat, fmt="%8.4e")
np.savetxt("ydata.dat", ydat, fmt="%8.4e")

