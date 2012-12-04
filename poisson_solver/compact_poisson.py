import numpy as np

Nx = 3
Ny = 3


def d(i,j):
    if i == j:
        return 1
    if i != j:
        return 0

def m(i):
    return (i%Ny, i/Ny)


D = np.zeros((Nx*Ny,Nx*Ny))

bn = np.ones((Nx,Ny))
bs = np.ones((Nx,Ny))
be = np.ones((Nx,Ny))
bw = np.ones((Nx,Ny))

cn = -1
cs = 1
ce = 1
cw = 1

for i in range(Nx*Ny):
    for j in range(Nx*Ny):
        k, l = m(i)
#+ cn*d(k,Ny)*bn[k,l] + cs*d(k,0)*bs[k,l] + ce*d(Nx,l)*be[k,l] + cw*d(0,l)*bw[k,l]
        D[i,j] = -((cn*d(Nx-1,l)+1)*bn[k,l]+bs[k,l]+bw[k,l]+be[k,l])*d(i,j) + bn[k,l]*d(i,j-Ny) + bs[k,l]*d(i,j+Ny) + be[k,l]*d(i-1,j) + bw[k,l]*d(i+1,j) 

print D
