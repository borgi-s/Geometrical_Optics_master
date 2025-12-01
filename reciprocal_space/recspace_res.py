# Model for rec. space resolution function for DFXM paper.
# The objective is modelled as an isotropic Gaussian with an NA and in addition
# a square phyical aperture of d´side length D.

# H.F. Poulsen, June 16, 2020, version 1.0
# H.F. Poulsen, Jan 22, 2021, version 1.1

# input parameters:
#   Nrays: numer of rays to be used
#   qi1_range, qi2_range, qi3_range: ranges for Resq_i in crystal system, in inverse AA.
#   npoints1,  npoints2, npoints3:  nr of points within each range
#   plot_figs: a flag; if 1 then plots will be generated, otherwise not
# output parameters:
#   Resq_i: voxelized rec. space resolution function in the IMAGING system.
#                                       Normalised to a max value of 1
#   ratio_outside: The fraction of rays not within the range defined by
#                                       qi1_range, qi2_range, qi3_range.

# Libraries
import numpy as np
import scipy.stats
import sys, os, gc
import pickle
import matplotlib.pyplot as plt
import xraylib as xrl


def check_folder(path, folder_name):
    folder_path = os.path.join(path, folder_name)
    # print(folder_path,'folder_path')
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder {folder_path} created.")
    else:
        print(f"Folder {folder_path} already exists.")

check_folder('','pkl_files')


def reciprocal_res_func(Nrays, npoints1, npoints2, npoints3, qi1_range, qi2_range, qi3_range, plot_figs, save_resqi, zeta_v_fwhm, zeta_h_fwhm, NA_rms, eps_rms, theta, phys_aper, date, mem_save = True, beamstop = False, bs_height = None, return_qs = False, aperture = False, knife_edge = False, dphi_range = 0.0):


    print('Defining properties of rays')
    # Define the properties of one ray


    zeta_v_sigma = zeta_v_fwhm/2.355
    # Cut off at 140 micro radians
    lower = -1.4e-4
    upper = 1.4e-4
    mu = 0 # that should be the offset of the distribution
    # zeta_v = scipy.stats.truncnorm.rvs((lower - mu), (upper - mu), loc = mu, scale = zeta_v_fwhm, size = Nrays)
    zeta_v = scipy.stats.truncnorm.rvs((lower - mu) / zeta_v_sigma, (upper - mu) / zeta_v_sigma, loc=mu, scale=zeta_v_sigma, size=Nrays)
    if plot_figs == 1:
        # Plot the histogram of zeta_v
        n, bins, patches = plt.hist(zeta_v, bins=50, density=True, alpha=0.7, edgecolor='black', label='Histogram') 

        # Normalize the histogram
        bin_width = bins[1] - bins[0]
        n_normalized = n / (np.sum(n) * bin_width)

        # Create a range of x values for the PDF
        x = np.linspace(lower*2, upper*2, 200)

        # Calculate the PDF using the standard normal distribution parameters
        pdf = scipy.stats.truncnorm.pdf(x, (lower - mu) / zeta_v_sigma, (upper - mu) / zeta_v_sigma, loc=mu, scale=zeta_v_sigma)

        # Plot the PDF curve of zeta_v
        plt.plot(x, pdf, 'r-', lw=2, label='PDF')

        plt.xlabel('zeta_v')
        plt.ylabel('Probability Density')
        plt.title('Distribution of zeta_v')
        plt.grid(True)
        plt.legend()
        plt.show()
    #zeta_v = (np.random.normal(size=Nrays)*zeta_v_fwhm/2.35)
    zeta_h = (np.random.normal(size=Nrays)*zeta_h_fwhm/2.35)
    eps = (np.random.normal(size=Nrays)*eps_rms) 
    if dphi_range > 0.0:
        dphi = np.random.uniform(-dphi_range/2, dphi_range/2, Nrays)
    else:
        dphi = 0.0
    print('Properties of rays defined')

    x1 = (np.random.normal(size=int(1.01*Nrays))*NA_rms)
    x2 = (np.random.normal(size=int(1.01*Nrays))*NA_rms)
    delta_2theta = x1[np.abs(x1)<phys_aper/2][:Nrays]
    xi = x2[np.abs(x2)<phys_aper/2][:Nrays]
    if len(xi)<Nrays:
        exit('Not enough values for xi')
    if len(delta_2theta)<Nrays:
        exit('Not enough values for delta_2theta')
    if len(delta_2theta) == Nrays and len(xi) == Nrays:
        print('Found trial delta_2theta and xi')

    #all this eq. 43,44,45 in Poulsen 2021, or from S
    qrock = ((-zeta_v/2) - (delta_2theta/2)) + dphi
    qroll = (-zeta_h/(2*np.sin(theta)) -  xi/(2*np.sin(theta))) 
    qpar = (eps + (1/np.tan(theta))*(-zeta_v/2 + delta_2theta/2))
    print('Converted to crystal system coordinates')
    # % Convert from crystal to imaging system  (Eq. 65 and 66 in Poulsen 2017)
    qrock_prime = np.cos(theta)*qrock + np.sin(theta)*qpar
    q2th =  - np.sin(theta)*qrock + np.cos(theta)*qpar
    print('Converted to image system system ')
    
    
    #new beamstop
    qroll_plot = qroll.copy() #hopefully keep the full shape for plotting later
    if beamstop == True:
        
        def bfp_x_to_alpha(x): #in mm 
            phi = 0.008684440640353642 # unitless, see simons 2017 eq. 22
            f = 21214.67 #mm single lenslet focal distance
            N = 88
            return x*np.sin(N*phi)/(f*phi) #return angle
        
        def bfp_alpha_to_x(alpha): #in mm 
            phi = 0.008684440640353642 # unitless, see simons 2017 eq. 22
            f = 21214.67 #mm single lenslet focal distance
            N = 88
            return alpha/np.sin(N*phi)*(f*phi) #return x
            
        def absorption_wire(alpha, thick_tot):
            '''
            Vectorized version of adding the beam stop
            '''
            # Convert angle to x
            x = bfp_alpha_to_x(alpha)

            # Only compute thickness for rays where x < thick_tot
            inside_mask = x < thick_tot

            # Compute absorption probability
            thick = np.zeros_like(alpha)
            thick[inside_mask] = np.sqrt(thick_tot**2 - x[inside_mask]**2)

            mu = xrl.CS_Total(74, 17)  # Tungsten, 17 keV
            chance = np.exp(-mu * thick / 10 * 19.254)

            # Draw uniform random numbers and accept/reject based on chance
            rand = np.random.rand(alpha.size)
            absorption = np.logical_or(x >= thick_tot, rand < chance)

            return absorption, x, chance
        def apply_aperture(alpha_x, alpha_y, square_length):
            """
            Square aperture version: absorb rays outside square_length
            in either x or y direction.
            """
            x = bfp_alpha_to_x(alpha_x)
            y = bfp_alpha_to_x(alpha_y)

            # absorb everything outside the square
            absorption = (np.abs(x) > square_length) | (np.abs(y) > square_length)
            return ~absorption, x, y
        # bs_height = 0.06 # in mm, height of beamstop (diameter)
        delta2ThetaMin = bfp_x_to_alpha(bs_height) # in rad
        print("shapes before: "+str(qroll.shape)+str(qrock_prime.shape)+str(q2th.shape))
        def sim_knife_edge(alpha, edge_pos, dx = 0.0):
            '''
            Vectorized version of adding the knife-edge, including it's movement along x.
            '''
            # Convert angle to x
            x = bfp_alpha_to_x(alpha)
            x += dx  # Apply the knife-edge shift

            # Knife-edge absorbs everything beneath it
            absorption = x < edge_pos

            return absorption
        
        #version with absorption
        # mu = xrl.CS_Total(74, 17)  # Tungsten, 17 keV
        if aperture == True and knife_edge == False:
            absorption, x_res, y_res = apply_aperture(np.abs(delta_2theta/2), np.abs(xi/2), bs_height/2)
        elif knife_edge == True and aperture == False:
            absorption = sim_knife_edge(delta_2theta/2, bs_height/2)
        elif aperture == False and knife_edge == False:
            absorption, x_res, chance_res = absorption_wire(np.abs(delta_2theta/2),bs_height/2)
        # plt.figure(0) # just to check
        # plt.clf()
        # plt.plot(x_res,chance_res,'x')
        qrock = qrock[absorption == 1][:Nrays]
        qroll = qroll[absorption == 1][:Nrays]
        qpar = qpar[absorption == 1][:Nrays]
        qrock_prime = qrock_prime[absorption == 1][:Nrays]
        q2th = q2th[absorption == 1][:Nrays]    
        
        #version without
#        qroll = qroll[np.abs(delta_2theta/2)>=delta2ThetaMin/2][:Nrays]
#        qrock_prime = qrock_prime[np.abs(delta_2theta/2)>=delta2ThetaMin/2][:Nrays]
#        q2th = q2th[np.abs(delta_2theta/2)>=delta2ThetaMin/2][:Nrays]
        print("shapes after:  "+str(qroll.shape)+str(qrock_prime.shape)+str(q2th.shape))
    cos_t, sin_t = np.cos(theta), np.sin(theta)

    def bin_once(qrock_prime_k, q2th_k):
        index1 = np.floor((qrock_prime_k + (qi1_range/2)) / qi1_range * npoints1).astype(np.int16)
        index2 = np.floor((qroll         + (qi2_range/2)) / qi2_range * npoints2).astype(np.int16)
        index3 = np.floor((q2th_k        + (qi3_range/2)) / qi3_range * npoints3).astype(np.int16)
        Res = np.zeros((npoints1, npoints2, npoints3), dtype=np.uint32)
        idx = (index1>=0)&(index2>=0)&(index3>=0)&(index1<npoints1)&(index2<npoints2)&(index3<npoints3)
        np.add.at(Res, (index1[idx], index2[idx], index3[idx]), 1)
        return Res

        # Single “position”
    if beamstop == False:
        qrock_prime = cos_t*qrock + sin_t*qpar
        q2th        = -sin_t*qrock + cos_t*qpar
    if beamstop == True:
        qrock_prime = qrock_prime
        q2th        = q2th

    if save_resqi == True: 
        Res = bin_once(qrock_prime, q2th)
        normResq_i = Res/Res.max()
        output = open('pkl_files/Resq_i_{0}.pkl'.format(date), 'wb')
        pickle.dump(normResq_i, output)
        output.close()
        # np.savetxt('reciprocal space/Resq_i_{0}.csv'.format(date), normResq_i, delimiter=',')
        print('Resq_i saved as Resq_i_{0}.pkl'.format(date))
    if return_qs == True:
        return qrock, qroll, qpar, qrock_prime, q2th, delta_2theta
        


    # if plot_figs == 1:
    #     plt.plot(np.linspace(-qi1_range,qi1_range,npoints1),normResq_i[:,npoints2//2, npoints3//2].squeeze())
    #     plt.tight_layout()
    #     plt.xlabel('$qi_{1}$ ')
    #     plt.ylabel('Probability Density')
    #     plt.title('Distribution of Resq_i (summing $2^{nd}$+$3^{rd}$ axis)')
    #     plt.grid(True)
    #     plt.show()
    # if plot_figs == 1:
    #     plt.plot(np.linspace(-qi3_range,qi3_range,npoints3)*1000,np.apply_over_axes(np.sum, normResq_i, [0,1]).squeeze())
    #     plt.tight_layout()
    #     plt.xlabel('$qi_{3}$ * 1000')
    #     plt.ylabel('Probability Density')
    #     plt.title('Distribution of Resq_i (summing $1^{st}$+$2^{nd}$ axis)')
    #     plt.grid(True)
    #     plt.show()

      

    # %%%%%%%%%%%%  For test purposes, plots %%%%%%%%%%%

    if plot_figs == 1:
        import matplotlib.ticker as mticker
        formatter = mticker.ScalarFormatter(useMathText=True)
        # Scatter plot  with shadows
        # not feasible/relevant  if Nrays > 1 million
        # a poltting range is required (same in all directions)
        plot_half_range = 0.0075
        print(Nrays,'N Nrays')

        slicenr =int( np.round(npoints1/2))
        im1 = Resq_i[slicenr]
        x1, y1 = np.meshgrid(np.linspace(-qi2_range, qi2_range, npoints2), np.linspace(-qi3_range, qi3_range, npoints2))
        fig2, ax2 = plt.subplots(figsize=(8,6))

        c1 = ax2.pcolormesh(y1, x1, im1, vmin=0, vmax=1)
        ax2.set_title('Slice through center in $q_{rock_{prime}}$', fontsize = 15)
        ax2.set_xlabel('$q_{roll}$', fontsize=13)
        ax2.set_ylabel('$q_{2\\theta}$', fontsize=13)
        # set the limits of the plot to the limits of the data
        ax2.ticklabel_format(style='sci', scilimits = (-3,-3), useMathText = True)
        ax2.axis([x1.min(), x1.max(), y1.max(), y1.min()])
        fig2.colorbar(c1, ax=ax2)
        plt.show()

        slicenr =int( np.round(npoints2/2))
        im2 = Resq_i[:,slicenr,:]
        y2, x2 = np.meshgrid(np.linspace(qi3_range, -qi3_range, npoints3),np.linspace(-qi1_range, qi1_range, npoints1))
        fig3, ax3 = plt.subplots(figsize=(8,6))
        qi3_range
        c2 = ax3.pcolormesh(x2, y2, im2, vmin=0, vmax=1)
        ax3.set_title('Slice through center in $q_{roll}$', fontsize = 15)
        ax3.set_xlabel('$q_{rock,prime}$', fontsize=13)
        ax3.set_ylabel('$q_{2\\theta}$', fontsize=13)
        # set the limits of the plot to the limits of the data
        ax3.axis([x2.min(), x2.max(), y2.max(), y2.min()])
        ax3.ticklabel_format(style='sci', scilimits = (-4,-4), axis = 'x', useMathText=True)
        ax3.ticklabel_format(style='sci', scilimits = (-3,-3), axis = 'y', useMathText=True)
        fig3.colorbar(c2, ax=ax3)
        plt.show()

        slicenr =int( np.round(npoints3/2))
        im3 = Resq_i[:,:,slicenr]
        y3, x3 = np.meshgrid(np.linspace(qi2_range, -qi2_range, npoints2),np.linspace(-qi1_range, qi1_range, npoints1))
        fig4, ax4 = plt.subplots(figsize=(8,6))
        c3 = ax4.pcolormesh(x3, y3, im3, vmin=0, vmax=1)
        ax4.set_title('Slice through center in $q_{2\\theta}$', fontsize = 15)
        ax4.set_xlabel('$q_{rock,prime}$', fontsize=13)
        ax4.set_ylabel('$q_{roll}$', fontsize=13)
        # set the limits of the plot to the limits of the data
        ax4.axis([x3.min(), x3.max(), y3.max(), y3.min()])
        ax4.ticklabel_format(style='sci', scilimits = (-4,-4), axis = 'x', useMathText=True)
        ax4.ticklabel_format(style='sci', scilimits = (-3,-3), axis = 'y', useMathText=True)
        fig4.colorbar(c3, ax=ax4)
        plt.show()

        project = np.zeros([npoints1,npoints3])
        for i in range(npoints1):
            for j in range(npoints3):
                project[i,j] += np.sum(Resq_i[i,:,j])
        y4, x4 = np.meshgrid(np.linspace(qi3_range, -qi3_range, npoints3),np.linspace(-qi1_range, qi1_range, npoints1))
        fig5, ax5 = plt.subplots(figsize=(8,8))
        c4 = ax5.pcolormesh(x4, y4, project, vmin=0, vmax=1)
        ax5.set_title('Projection of Resq_i', fontsize = 15)
        ax5.set_xlabel('$q_{rock,prime}$', fontsize=13)
        ax5.set_ylabel('$q_{2\\theta}$', fontsize=13)
        # set the limits of the plot to the limits of the data
        ax5.axis([x4.min(), x4.max(), y4.max(), y4.min()])
        ax5.ticklabel_format(style='sci', scilimits = (-4,-4), axis = 'x', useMathText=True)
        ax5.ticklabel_format(style='sci', scilimits = (-3,-3), axis = 'y', useMathText=True)
        ax5.set_xticks([-10e-4, -5e-4, 0, 5e-4, 10e-4])
        fig5.colorbar(c4, ax=ax5)
        plt.show()

        if Nrays < 100001:
            
            from mpl_toolkits.mplot3d import Axes3D
            # in crystal system
            x, y, z = qrock, qroll_plot, qpar
            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(111,projection='3d')
            ax.plot(x, z, 'yo', markersize=1.0, zdir='y', zs = plot_half_range)
            ax.plot(y, z, 'ro', markersize=1.0, zdir='x', zs = plot_half_range)
            ax.plot(x, y, 'co', markersize=1.0, zdir='z', zs = -plot_half_range)
            ax.scatter(x, y, z, s=1.0)
            ax.set_xlabel('$\hat{q}_{rock}$', fontsize=14)
            ax.set_ylabel('$\hat{q}_{roll}$', fontsize=14)
            ax.set_zlabel('$\hat{q}_{par}$', fontsize=14)
            ax.xaxis.set_major_formatter(formatter)
            ax.yaxis.set_major_formatter(formatter)
            ax.zaxis.set_major_formatter(formatter)
            ax.ticklabel_format(style='sci', scilimits = (-3,-3))
            # ax.set_xticks([-4e-3, -3e-3, -2e-3, -1e-3, 0, 1e-3, 2e-3, 3e-3, 4e-3])
            # ax.set_yticks([-8e-3, -6e-3, -4e-3, -2e-3, 0, 2e-3, 4e-3, 6e-3, 8e-3])
            # ax.set_zticks([-8e-3, -6e-3, -4e-3, -2e-3, 0, 2e-3, 4e-3, 6e-3, 8e-3])
            ax.set_xlim([-plot_half_range,plot_half_range])
            ax.set_ylim([plot_half_range,-plot_half_range])
            ax.set_zlim([plot_half_range,-plot_half_range])
            ax.view_init(200, 310)
            plt.title('Crystal coordinate system')
            # plt.savefig('test im1.jpg')
            plt.show()

            # in image system
            x1, y1, z1 = qrock_prime, qroll, q2th
            fig1 = plt.figure(figsize=(8, 6))
            ax1 = fig1.add_subplot(111,projection='3d')
            ax1.plot(x1, z1, 'yo', markersize=1.0, zdir='y', zs = plot_half_range)
            ax1.plot(y1, z1, 'ro', markersize=1.0, zdir='x', zs = plot_half_range)
            ax1.plot(x1, y1, 'co', markersize=1.0, zdir='z', zs = -plot_half_range)
            ax1.scatter(x1, y1, z1, s=1.0)

            ax1.set_xlabel('$\hat{q}^{\prime}_{rock}$', fontsize=14)
            ax1.set_ylabel('$\hat{q}_{roll}$', fontsize=14)
            ax1.set_zlabel('$\hat{q}_{2\\theta}$', fontsize=14)
            ax1.xaxis.set_major_formatter(formatter)
            ax1.yaxis.set_major_formatter(formatter)
            ax1.zaxis.set_major_formatter(formatter)
            ax1.ticklabel_format(style='sci', scilimits = (-3,-3))
            # ax1.set_xticks([-8e-3, -6e-3, -4e-3, -2e-3, 0, 2e-3, 4e-3, 6e-3, 8e-3])
            # ax1.set_yticks([-8e-3, -6e-3, -4e-3, -2e-3, 0, 2e-3, 4e-3, 6e-3, 8e-3])
            # ax1.set_zticks([-8e-3, -6e-3, -4e-3, -2e-3, 0, 2e-3, 4e-3, 6e-3, 8e-3])
            ax1.set_xlim([-plot_half_range,plot_half_range])
            ax1.set_ylim([plot_half_range,-plot_half_range])
            ax1.set_zlim([plot_half_range,-plot_half_range])
            ax1.view_init(200, 310)
            plt.title('Imaging coordinate system')
            # plt.savefig('test im2.jpg')
            plt.show()
