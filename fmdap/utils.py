import numpy as np

def dist_func(d, spatial_corr, formulation=3):

    f = np.zeros(np.size(d))
    
    if spatial_corr <= 0:
        return f
      
    if (formulation == 1):
        #- Gaussian distribution; integral=1.2533; support > 5 (val=4e-6 at 5)
        f = np.exp(-0.5 * (d / spatial_corr)**2)
      
    elif (formulation == 2 ):
        #- Exponential; integral=1; support inf (not for localization) (val 7e-3 at 5)
        f = np.exp(-1. * (d / spatial_corr))
      
    elif (formulation == 3) :
        #- Gaspari&Cohn 1999; integral=1.2250; support=3.5 (similar to Gauss)
        R = spatial_corr * 1.7386
        d = np.atleast_1d(d)
        r1 = ((d / R))
        r2 = ((d / R)**2)
        r3 = ((d / R)**3)
        ii = (np.where(d<=R))
        f[ii] = 1.0 + r2[ii]*(-r3[ii]/4.0 + r2[ii]/2.0) + r3[ii]*(5.0/8.0) - r2[ii]*(5.0/3.0)
        ii = np.where((d>R) & (d <= R*2))
        f[ii] = r2[ii]*(r3[ii]/12.0 - r2[ii]/2.0) + r3[ii]*(5.0/8.0) + r2[ii]*(5.0/3.0) - r1[ii]*5.0 + 4.0 - (2.0/3.0)/r1[ii]

    return f

def dist_in_meters(dfs, pt):
    ec   = dfs.get_element_coords()
    xe   = ec[:,0]
    ye   = ec[:,1]
    xp   = pt[0]
    yp   = pt[1]
    if dfs.is_geo:        
        d = get_dist_geo(xe, ye, xp, yp)
    else:   
        d = np.sqrt(np.square(xe - xp) + np.square(ye - yp))
    return d

def get_dist_geo(lon, lat, lon1, lat1):
    # assuming input in degrees!
    R = 6371e3 # Earth radius in metres
    dlon = np.deg2rad(lon1 - lon)
    #dlon = dlon % (2*np.pi)
    if(np.any(dlon>np.pi)): 
        dlon = dlon - 2*np.pi
    if(np.any(dlon<-np.pi)): 
        dlon = dlon + 2*np.pi    
    dlat = np.deg2rad(lat1 - lat)
    x = dlon*np.cos(np.deg2rad((lat+lat1)/2))
    y = dlat
    d = R * np.sqrt(np.square(x) + np.square(y) )
    return d
