import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy
from tqdm import tqdm
import time
from mpl_toolkits.axes_grid1 import make_axes_locatable
from numpy import random
from matplotlib import cm
from scipy import optimize
from scipy.optimize import curve_fit
from scipy import stats
import matplotlib.style
import scipy.special
from scipy.special import binom, comb
from scipy.stats import chisquare
from scipy.special import expit
import matplotlib.patches as mpl_patches
from matplotlib.patches import Polygon
from scipy.special import gamma
import sys, os, os.path

""" (1)

#parametri di prova, si possono usare nel caso in cui non si voglia lavorare con gli input, scegliendo
#dunque di variare i parametri direttamente nel codice. 
#In tal caso commentare i commentare gli input e scommentare in corrispondenza dei numeri 1 e 2.

N=np.array([1000])
passi=np.array([100])
step=1
#step=0.015
stepx=stepy=step
px=np.array([0.9])
py=np.array([0.6])
p=np.array([0.44])
n00=2

(2) """ 

#---
px=py=p=np.empty(0)
passi=N=np.empty(0, dtype=int)
n00=int(input('Si vuole effettuare una simulazione 2D (digita 1) o una 3D (digita 2)?		'))
print('	')
n0=int(input('Inserire il numero di simulazioni da effettuare		'))
print('	')

if(n0==1):
	n2=int(input('inserisci il numero delle palline		'))
	print('	')
	N=np.append(N,n2)	
	n3=int(input('inserisci il numero di scelte (passi)			'))
	print('	')
	passi=np.append(passi,n3)
	if(n00==1):
		n4=float(input('inserisci la probabilità			'))
		print('	')
		p=np.append(p,n4)
	if(n00==2):
		n5=float(input('inserisci la probabilità su X 		'))
		print('	')
		px=np.append(px,n5)
		n6=float(input('inserisci la probabilità su Y 		'))
		print('	')
		py=np.append(py,n6)

if(n0>1):
	print('Di quale variabile si vogliono effettuare più simulazioni?		')
	print('	')
	n1=int(input('Digita 1 per il numero delle palline, 2 per il numero delle scelte e 3 per la probabilità		'))
	print('	')
	if(n1==1):
		for i in range(0,n0):
			n2=int(input('inserisci il numero di palline contenute nel set numero 'f'{i+1}		'))
			print('	')
			N=np.append(N,n2)

		n3=int(input('inserisci il numero di scelte (passi)			'))
		print('	')
		passi=np.append(passi,n3)

		if(n00==1):
			n4=float(input('inserisci la probabilità			'))
			print('	')
			p=np.append(p,n4)
		if(n00==2):
			n5=float(input('inserisci la probabilità su X 		'))
			print('	')
			px=np.append(px,n5)
			n6=float(input('inserisci la probabilità su Y 		'))
			print('	')
			py=np.append(py,n6)

	if(n1==2):
		n2=int(input('inserisci il numero delle palline		'))
		print('	')
		N=np.append(N,n2)

		for i in range(0,n0):
			n3=int(input('inserisci il numero di scelte (passi) contenute nel set numero 'f'{i+1}		'))
			print('	')
			passi=np.append(passi,n3)

		if(n00==1):
			n4=float(input('inserisci la probabilità		'))
			print('	')
			p=np.append(p,n4)
		if(n00==2):
			n5=float(input('inserisci la probabilità su X 		'))
			print('	')
			px=np.append(px,n5)
			n6=float(input('inserisci la probabilità su Y 		'))
			print('	')
			py=np.append(py,n6)


	if(n1==3):
		n2=float(input('inserisci il numero delle palline		'))
		print('	')
		N=np.append(N,int(n2))

		n3=float(input('inserisci il numero di scelte (passi)		'))
		print('	')
		passi=np.append(passi,int(n3))


		if(n00==1):
			for i in range(0,n0):
				n4=float(input('inserisci la probabilità per il set numero 'f'{i+1}		'))
				print('	')
				p=np.append(p,n4)

		if(n00==2):
			for i in range (0,n0):
				n5=float(input('inserisci la probabilità su X per il set numero 'f'{i+1}		'))
				print('	')
				px=np.append(px,n5)
				n6=float(input('inserisci la probabilità su Y per il set numero 'f'{i+1}		'))
				print('	')
				py=np.append(py,n6)	
#---


def distribution_2D(p,n):
	"""
	funzione che simula la discesa di una pallina in 2D

	n= numero di passi (scelte)
	p= probabilità di andare a destra (es. 0.6 rappresenta una probabilità del 60%
	di andare a destra)

	"""
	position=0
	step=1

	check=np.random.random(n)
	for c in check:
		if c<=p:
			position=position+1

	return position

def distribution_3D(px,py,n):
	"""
	funzione che simula la discesa di una pallina in 3D

	n=numero di passi
	px=probabilità di andare a destra sull'asse X
	py=probabilità di andare a destra sull'asse Y
	"""
	positionx=positiony=0
	checkx, checky=np.random.rand(2, n)

	for cx in checkx:
		if cx<=px:
			positionx=positionx+1

	for cy in checky:
		if cy<=py:
			positiony=positiony+1

	return positionx, positiony

def gaussian_1D(x,mu, sigma,A):
	"""
	funzione gaussiana g(x)= ((sigma*sqrt(2*pi))^-1)*e^-(x-mu)^2/2*sigma^2
	mu= centro della distribuzione
	sigma= larghezza della distribuzione
	A=numero di palline considerate
	"""
	return A*np.exp(-((x-mu)**2)/(2*sigma**2))

def binomial(k,p,n,A):	
	"""
	PDF per la distribuzione binomiale.

	k=numero di successi (valori dell'istogramma da fittare)
	n=numero di prove
	p=probabilità di successo per una prova
	A=numero di prove (palline)
	"""
	return A*binom(n, k) * p**k * (1-p)**(n-k)


def chi2(n,y):
	"""
	Chi quadro di Pearson con 2 cifre significative.
	"""
	f=0
	for i in range (0, len(n)):
		if(y[i]!=0):
			f+=((n[i]-y[i])**2)/(y[i])

	return int(100* f/ (len(n)-1) )/100


#-------------------- Macchina di Galton 2D -----------------


mu=sigma=np.empty(0)
alpha=np.empty(0)

if(n00==1):
	for l in range (0, len(passi)):
		for k in range (0, len(p)):
			for j in range(0,len(N)):
				array_plot=np.empty(shape=N[j])
				array_plot_fit=np.empty(shape=N[j])
				plt.style.use('bmh')


				for i in range(0, N[j]):
					array_plot[i]=distribution_2D(p[k], passi[l])
				mu=np.append(mu,np.mean(array_plot))
				sigma=np.append(sigma,np.std(array_plot))
				

				n,bis,pp=plt.hist(array_plot, bins=passi[l]+1, label=r'Dati simulati, $\mu='f'{int(mu[j+k+l])}$,'' $\sigma='f'{int(10*sigma[j+k+l])/10}$', range=(0,passi[l]+1), color='royalblue')
				plt.legend(loc='best', frameon=12, fontsize=12)
				#calcolo della media e della deviazione standard
				mask=np.nonzero(n)
				plt.hist(array_plot, bins=passi[l]+1, range=(0,passi[l]+1), color='black', histtype='step')
				plt.ylabel('Numero di palline', fontsize=12)
				plt.xlabel('# passi a dx', fontsize=12)
				plt.title(r'Macchina di Galton 2D con $p='f'{p[k]}'' ,n='f'{passi[l]}'' ,N_p='f'{N[j]}$', fontsize=14)

				plt.show()

				bincenters=(bis[:-1]+bis[1:])/2

				par,pcov=curve_fit(gaussian_1D,xdata=bincenters[mask], ydata=n[mask],sigma=np.sqrt(n[mask]),p0=[mu[j+k+l],sigma[j+k+l], N[j]], absolute_sigma=True, maxfev=100000)
				par_binom, pcov_binom=curve_fit(binomial, xdata=bincenters[mask], ydata=n[mask],sigma=np.sqrt(n[mask]*p[k]*(1-p[k])),p0=[p[k],passi[l],N[j]], absolute_sigma=True, maxfev=10000000)

				x_fit=np.linspace(0,passi[l]+1,num=2*passi[l])
				x_hist=np.linspace(0, passi[l]+1, num=2*passi[l])

				alpha=np.append(alpha, par_binom)
				chi2_binom=1
				chi2_gauss=1
				chi2_binom=chi2(n[mask],binomial(bincenters[mask],par_binom[0],par_binom[1],par_binom[2]))
				chi2_gauss=chi2(n[mask],gaussian_1D(bincenters[mask],par[0],par[1],par[2]))

				fig, ax = plt.subplots(2, 1,gridspec_kw={'height_ratios': [3, 1]})
				fig.tight_layout()
				ax[0].plot(x_fit, binomial(x_fit,par_binom[0],par_binom[1],par_binom[2]),c='darkgreen', label=r'Binomiale, $\~{\chi}^2='f'{chi2_binom}$')
				ax[0].plot(x_fit, gaussian_1D(x_fit,par[0],par[1],par[2]),c='darkblue',label=r'Gaussiana, $\~{\chi}^2='f'{chi2_gauss}$')
				ax[0].hist(array_plot, bins=passi[l]+1,label='Dati simulati', range=(0,passi[l]+1),color='royalblue')
				ax[0].hist(array_plot, bins=passi[l]+1, range=(0,passi[l]+1),color='black',histtype='step' )

				ax[0].set_xlabel('# passi a destra', fontsize=12)
				ax[0].set_ylabel('Numero di palline', fontsize=12)
				ax[0].errorbar(bincenters[mask],n[mask],yerr=np.sqrt(n[mask]*p[k]*(1-p[k])),fmt='o', color='darkred', markersize='4')
				ax[0].legend(loc='best', frameon=False, fontsize=12)
				ax[0].set_title(r'Fit dei dati simulati con  $p='f'{p[k]}'' ,n='f'{passi[l]}'' ,N_p='f'{N[j]}$', fontsize=14)


				ax[1].axhline(1, color='darkorange')
				ax[1].errorbar(bincenters[mask],n[mask]/binomial(bincenters[mask],par_binom[0],par_binom[1],par_binom[2]), fmt='o', label='dati/binomiale', yerr=np.sqrt(n[mask]*p[k]*(1-p[k]))/
				binomial(bincenters[mask],par_binom[0],par_binom[1],par_binom[2]), markersize=4, color='darkgreen')
				ax[1].errorbar(bincenters[mask],n[mask]/gaussian_1D(bincenters[mask],par[0],par[1],par[2]), fmt='o', label='dati/gaussiana',
				 yerr=sigma[j]/gaussian_1D(bincenters[mask],par[0],par[1],par[2]), markersize=4, c='darkblue')
				ax[1].legend(loc='best', frameon=False)
				ax[1].set_ylabel('Dati/Fit (logscale)', fontsize=12)
				ax[1].set_xlabel('# passi a destra', fontsize=12)
				ax[1].grid(True)
				plt.show()

	if(len(passi)!=1 or len(p)!=1):
		fig,ax=plt.subplots(1,1)
		fig.tight_layout()

		for j in range(0,len(N)):
			for l in range(0, len(passi)):
				for i in range (0,len(p)):
					x_fit=np.linspace(0,passi[l]+1,num=5*passi[l])
					x_text=1.1*mu[j+i+l]
					ax.plot(x_fit, binomial(x_fit,alpha[(i+j+l)*3],alpha[(i+j+l)*3 +1],alpha[(i+j+l)*3 +2]), label=r'$\mu=$'f'{int(mu[i+j+l])}'', $\sigma=$'f'{int(10*sigma[i+j+l])/10}')
					ax.legend(loc='best', frameon=False)
					if(len(p)>1):
						ax.text(x_text,max(binomial(x_fit,alpha[(i+j+l)*3],alpha[(i+j+l)*3 +1],alpha[(i+j+l)*3 +2])), r'$p='f'{p[j+i+l]}$', fontsize=14)

					if(len(N)>1):
						ax.text(x_text,max(binomial(x_fit,alpha[(i+j+l)*3],alpha[(i+j+l)*3 +1],alpha[(i+j+l)*3 +2])), r'$N_p='f'{N[j+i+l]}$', fontsize=14)
			
					if(len(passi)>1):
						ax.text(x_text,max(binomial(x_fit,alpha[(i+j+l)*3],alpha[(i+j+l)*3 +1],alpha[(i+j+l)*3 +2])), r'$n='f'{passi[j+i+l]}$', fontsize=14)

					ax.set_xlabel('# passi a dx', fontsize=12)
					ax.set_ylabel('Numero palline', fontsize=12)
					ax.set_title('Funzione binomiale di fit per i diversi parametri inseriti', fontsize=14)
	plt.show()

#-------------------- Macchina di Galton 3D -----------------

alphax=alphay=Chi2binom_x=Chi2binom_y=Chi2gauss_x=Chi2gauss_y=mux=muy=sigmax=sigmay=np.empty(0)

if(n00==2):
	for l in range (0, len(passi)):
		for k in range (0,len(px)):
			for j in range(0, len(N)):
				array_plotx=np.empty(shape=N[j])
				array_ploty=np.empty(shape=N[j])
				plt.style.use('bmh')
				for i in range(0,N[j]):
					array_plotx[i],array_ploty[i]=distribution_3D(px[k],py[k],passi[l])

				ax1 = plt.subplot(221)

				nx,bisx,pxx=ax1.hist(array_plotx, bins=passi[l]+1,label='Asse X', range=(0,passi[l]+1), color='lightblue')
				ax1.hist(array_plotx, bins=passi[l]+1,label='Asse X', range=(0,passi[l]+1),histtype='step', color='black')
				ax1.set_title(r'Passi asse X	$p_x='f'{px[k]}$')
				ax1.set_ylabel('# di palline')
				ax1.set_xlabel('# di passi a dx')

				ax2=plt.subplot(222)
				ny,bisy,pyy=ax2.hist(array_ploty, bins=passi[l]+1,label='Asse Y', range=(0,passi[l]+1),color='orange')
				ax2.hist(array_ploty, bins=passi[l]+1,label='Asse Y', range=(0,passi[l]+1),histtype='step',color='black')
				ax2.set_title(r'Passi asse Y	$p_y='f'{py[k]}$')
				ax2.set_ylabel('# di palline')
				ax2.set_xlabel('# di passi[l] a dx')

				ax3=plt.subplot(212)
				ax3.set_title(r'Confronto XY con $N_p='f'{N[j]}$'', $n='f'{passi[l]}$')
				ax3.set_xlabel('# di passi a destra')
				ax3.hist(array_ploty, bins=passi[l]+1,label='Asse Y', range=(0,passi[l]+1),color='orange', alpha=0.85)
				ax3.hist(array_ploty, bins=passi[l]+1, range=(0,passi[l]+1),histtype='step',color='black',alpha=0.75)
				ax3.hist(array_plotx, bins=passi[l]+1,label='Asse X', range=(0,passi[l]+1), color='lightblue',alpha=0.75)
				ax3.hist(array_plotx, bins=passi[l]+1, range=(0,passi[l]+1),histtype='step', color='black',alpha=0.75)
				ax3.legend(loc='best',frameon=False)
				plt.show()

				fig, ax4 = plt.subplots(figsize=(6, 6))

				h=ax4.hist2d(array_plotx, array_ploty,cmap=plt.cm.BuPu,bins=min(passi[l]+1,100), range=((0,passi[l]+1),(0,passi[l]+1)))
				ax4.set_xlabel('# passi a destra in direzione X')
				ax4.set_ylabel('# passi a destra in direzione Y')
				ax4.set_aspect(1.)

				divider = make_axes_locatable(ax4)
				ax_histx = divider.append_axes("top", 1.2, pad=0.15, sharex=ax4)
				ax_histy = divider.append_axes("right", 1.2, pad=0.15, sharey=ax4)

				fig.colorbar(h[3],cax=divider.append_axes("bottom",0.1,pad=0.5), orientation='horizontal')
				ax_histx.xaxis.set_tick_params(labelbottom=False)
				ax_histy.yaxis.set_tick_params(labelleft=False)

				ax_histx.hist(array_plotx, bins=passi[l]+1,label='Dati X', range=(0,passi[l]+1),color='lightblue')
				ax_histx.hist(array_plotx, bins=passi[l]+1, range=(0,passi[l]+1),histtype='step',color='black')
				ax_histx.legend(loc='best', frameon=False)

				ax_histx.set_ylabel('numero di palline')
				ax_histy.set_xlabel('numero di palline')
				ax_histx.set_title(r'Asse X	$p_y='f'{px[k]}$')
				ax_histy.set_title(r' Asse Y	$p_y='f'{py[k]}$')

				ax_histy.hist(array_ploty, bins=passi[l]+1,label=r'Dati Y', range=(0,passi[l]+1),color='orange', orientation='horizontal')
				ax_histy.hist(array_ploty, bins=passi[l]+1, range=(0,passi[l]+1),histtype='step',color='black', orientation='horizontal')
				ax_histy.legend(loc='best', frameon=False)

				plt.show()
			#------------------

				mux=np.append(mux,np.mean(array_plotx))
				muy=np.append(muy,np.mean(array_ploty))
				sigmax=np.append(sigmax,np.std(array_plotx))
				sigmay=np.append(sigmay,np.std(array_ploty))

				maskx=np.nonzero(nx)
				masky=np.nonzero(ny)

				bincentersx = (bisx[:-1] + bisx[1:])/2
				bincentersy = (bisy[:-1] + bisy[1:])/2

				xdata_hist=np.linspace(0,passi[l]+1, num=1000)
				ydata_hist=np.linspace(0,passi[l]+1, num=1000)

				parx, pcovx = curve_fit(gaussian_1D, xdata=bincentersx[maskx], ydata=nx[maskx],sigma=np.sqrt(nx[maskx]),p0=[mux[j+k+l],sigmax[j+k+l], N[j]/ sigmax[j+k]*np.sqrt(2*np.pi)], absolute_sigma=True, maxfev=10**7 )
				pary, pcovy = curve_fit(gaussian_1D, xdata=bincentersy[masky], ydata=ny[masky],sigma=np.sqrt(ny[masky]),p0=[muy[j+k+l],sigmay[j+k+l], N[j]/ sigmay[j+k]* np.sqrt(2*np.pi)], absolute_sigma=True, maxfev=10**7)

				errx=np.sqrt(nx[maskx]*px[k]*(1-px[k]))
				erry=np.sqrt(ny[masky]*py[k]*(1-py[k]))

				parx_binom, pcovx_binom=curve_fit(binomial, xdata=bincentersx[maskx], ydata=nx[maskx],sigma=np.sqrt(nx[maskx]*px[k]*(1-px[k])),p0=[px[k],passi[l], N[j]], absolute_sigma=True, maxfev=10**7)
				pary_binom, pcovy_binom=curve_fit(binomial, xdata=bincentersy[masky], ydata=ny[masky],sigma=np.sqrt(ny[masky]*py[k]*(1-py[k])),p0=[py[k],passi[l], N[j]], absolute_sigma=True, maxfev=10**7)

				alphax=np.append(alphax, parx_binom)
				alphay=np.append(alphay, pary_binom)

				Chi2binom_x=np.append(Chi2binom_x,chi2(nx[maskx],binomial(bincentersx[maskx],parx_binom[0],parx_binom[1], parx_binom[2]))) 
				Chi2gauss_x=np.append(Chi2gauss_x,chi2(nx[maskx],gaussian_1D(bincentersx[maskx], parx[0],parx[1], parx[2]))) 

				Chi2binom_y=np.append(Chi2binom_y, chi2(ny[masky],binomial(bincentersy[masky],pary_binom[0],pary_binom[1], pary_binom[2])))
				Chi2gauss_y=np.append(Chi2gauss_y, chi2(ny[masky],gaussian_1D(bincentersy[masky], pary[0],pary[1], pary[2])))


				fig, ax = plt.subplots(2, 2,gridspec_kw={'height_ratios': [3, 1]})
				fig.subplots_adjust(hspace=0)
				fig.tight_layout()

				ax[0,0].hist(array_plotx, bins=passi[l]+1,label='Dati simulati X', range=(0,passi[l]+1), color='lightblue')
				ax[0,0].hist(array_plotx, bins=passi[l]+1, range=(0,passi[l]+1),histtype='step', color='black')

				ax[0,0].plot(xdata_hist, binomial(xdata_hist,parx_binom[0], parx_binom[1], parx_binom[2]), c='royalblue',label=r'Binomiale, $ \~{\chi} ^2$='f'{Chi2binom_x[j]} ')
				ax[0,0].plot(xdata_hist, gaussian_1D(xdata_hist, parx[0],parx[1],parx[2]),label=r'Gaussiana, $ \~{\chi} ^2$='f'{Chi2gauss_x[j]}', c='darkgreen')
				ax[0,0].errorbar(bincentersx[maskx],nx[maskx],yerr=errx, markersize=4,fmt='o',color='darkred')
				ax[0,0].set_ylabel('# di palline', fontsize=12)
				ax[0,0].set_xlabel('# di passi[l] a dx')
				ax[0,0].set_title(r'Fit asse X	$p_x='f'{px[k]}$')
				ax[0,0].legend(loc='best', frameon=False)

				ax[1,0].set_yscale('log')
				ax[1,0].errorbar(bincentersx[maskx], nx[maskx]/binomial(bincentersx[maskx],parx_binom[0], parx_binom[1], parx_binom[2]), fmt='o', color='darkgreen',yerr=errx/binomial(bincentersx[maskx],parx_binom[0], parx_binom[1], parx_binom[2]), label='Binoamiale',markersize=4 )
				ax[1,0].axhline(1, color='darkorange')
				ax[1,0].set_ylabel('Dati/Fit (logscale)') 
				ax[1,0].set_xlabel('# di passi a dx')
				ax[1,0].errorbar(bincentersx[maskx], nx[maskx]/gaussian_1D(bincentersx[maskx], parx[0],parx[1],parx[2]), yerr=errx/gaussian_1D(bincentersx[maskx],parx[0], parx[1], parx[2]), fmt='o', color='royalblue', label='Gaussiana',markersize=4)
				ax[1,0].legend(loc='best', frameon=False)
				ax[1,0].grid(True)

				ax[0,1].set_xlabel('# di passi a sx')
				ax[0,1].hist(array_ploty, bins=passi[l]+1,label='Dati simulati Y', range=(0,passi[l]+1),color='orange')
				ax[0,1].hist(array_ploty, bins=passi[l]+1, range=(0,passi[l]+1),histtype='step', color='black')

				ax[0,1].plot(ydata_hist, binomial(ydata_hist,pary_binom[0],pary_binom[1], pary_binom[2]), c='royalblue', label=r'Binomiale, $\~{\chi} ^2$='f'{Chi2binom_y[j]}')
				ax[0,1].plot(ydata_hist, gaussian_1D(ydata_hist, pary[0],pary[1],pary[2]),label=r'Gaussiana, $ \~{\chi} ^2$='f'{Chi2gauss_y[j]}', c='darkgreen')
				ax[0,1].set_ylabel('# di palline', fontsize=12)
				ax[0,1].errorbar(bincentersy[masky],ny[masky],yerr=erry,fmt='o',markersize=4, color='darkred')
				ax[0,1].set_title(r'Fit asse Y	$p_y='f'{py[k]}$')
				ax[0,1].legend(loc='best', frameon=False)

				ax[1,1].set_yscale('log')
				ax[1,1].errorbar(bincentersy[masky], ny[masky]/binomial(bincentersy[masky],pary_binom[0], pary_binom[1], pary_binom[2]), yerr=erry/binomial(bincentersy[masky],pary_binom[0], pary_binom[1], pary_binom[2]), fmt='o', color='darkgreen',label='Binomiale', markersize=4)
				ax[1,1].axhline(1, color='darkorange')
				ax[1,1].set_ylabel('Dati/Fit (logscale)')
				ax[1,1].set_xlabel('# di passi a dx', fontsize=12) 
				ax[1,1].errorbar(bincentersy[masky], ny[masky]/gaussian_1D(bincentersy[masky], pary[0],pary[1],pary[2]), yerr=erry/ gaussian_1D(bincentersy[masky],pary[0], pary[1], pary[2]), fmt='o', color='royalblue', label='Gaussiana',markersize=4)
				ax[1,1].legend(loc='best', frameon=False)
				ax[1,1].grid(True)
				plt.show()

	if(len(passi)!=1 or len(px)!=1):
		fig,ax=plt.subplots(2,1)
		fig.tight_layout()

		for j in range(0,len(N)):
			for l in range(0, len(passi)):
				for i in range (0,len(px)):
					x_fit=np.linspace(0,passi[l]+1,num=5*passi[l])
					x_text=(1.05)*mux[j+i+l]
					y_text=(1.05)*muy[j+i+l]

					ax[0].plot(x_fit, binomial(x_fit,alphax[(i+j+l)*3],alphax[(i+j+l)*3 +1],alphax[(i+j+l)*3 +2]), label=r'$\mu_x=$'f'{int(mux[i+j+l])}'', $\sigma_x=$'f'{int(10*sigmax[i+j+l])/10}')
					ax[0].legend(loc='best', frameon=False, fontsize=12)
					ax[1].plot(x_fit, binomial(x_fit,alphay[(i+j+l)*3],alphay[(i+j+l)*3 +1],alphay[(i+j+l)*3 +2]), label=r'$\mu_y=$'f'{int(muy[i+j+l])}'', $\sigma_y=$'f'{int(10*sigmay[i+j+l])/10}')
					ax[1].legend(loc='best', frameon=False, fontsize=12)
					if(len(px)>1):
						ax[0].text(x_text,max(binomial(x_fit,alphax[(i+j+l)*3],alphax[(i+j+l)*3 +1],alphax[(i+j+l)*3 +2])), r'$p_x='f'{px[j+i+l]}$', fontsize=14)
						ax[1].text(y_text,max(binomial(x_fit,alphay[(i+j+l)*3],alphay[(i+j+l)*3 +1],alphay[(i+j+l)*3 +2])), r'$p_y='f'{py[j+i+l]}$', fontsize=14)

					if(len(N)>1):
						ax[0].text(x_text,max(binomial(x_fit,alphax[(i+j+l)*3],alphax[(i+j+l)*3 +1],alphax[(i+j+l)*3 +2])), r'$N_p='f'{N[j+i+l]}$', fontsize=14)
						ax[1].text(y_text,max(binomial(x_fit,alphay[(i+j+l)*3],alphay[(i+j+l)*3 +1],alphay[(i+j+l)*3 +2])), r'$N_p='f'{N[j+i+l]}$', fontsize=14)

					if(len(passi)>1):
						ax[0].text(x_text,max(binomial(x_fit,alphax[(i+j+l)*3],alphax[(i+j+l)*3 +1],alphax[(i+j+l)*3 +2])), r'$n='f'{passi[j+i+l]}$', fontsize=14)
						ax[1].text(y_text,max(binomial(x_fit,alphay[(i+j+l)*3],alphay[(i+j+l)*3 +1],alphay[(i+j+l)*3 +2])), r'$n='f'{passi[j+i+l]}$', fontsize=14)

						ax[0].set_xlabel('# passi a dx', fontsize=12)
						ax[0].set_ylabel('Numero palline', fontsize=12)
						ax[0].set_title('Asse X: binomiale per i diversi parametri inseriti', fontsize=14)
						ax[1].set_xlabel('# passi a dx', fontsize=12)
						ax[1].set_ylabel('Numero palline', fontsize=12)
						ax[1].set_title('Asse Y:binomiale per i diversi parametri inseriti', fontsize=14)
	plt.show()
