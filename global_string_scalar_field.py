# Writen by Josh Sharkey, November 2023
# solve static field equation for a global string using 4th order Runge-Kutta
# static field equations solved as in:
#   A. Vilenkin & E. P. S Shellard
#   1994
#   Cambridge University Press
#   "Cosmic Strings and Other Topological Deffects"
#   Section 4.1.1

import numpy as np
from functools import partial
import matplotlib.pyplot as plt

def system_of_equations(r,f_g,n,lam):
    # reduction of static field equation for global string to system of 1st order ODEs
    #
    # INPUTS:
    #   r: scalar, r
    #   f_g: 2D vector, [f,g] at given r value
    #   n: scalar, parameter of equation
    #   lam: scalar, parameter of equation
    #
    # OUTPUT:
    #   [f_derivative, g_derivative]: vector, valkue of derivatives of f and g
    #
    # For use with Runge-Kutta algorithm
    # Best to use as partial function with n and lam fixed
    
    f=f_g[0]
    g=f_g[1]
    
    f_derivative=g
    g_derivative= ( (n**2)*f/(r**2) ) + ( (lam*f/2)*( (f**2)-1 ) ) - (g/r)
    
    return [f_derivative,g_derivative]

def RK4_step(r,f_g,system,step_size):
    # one RK4 step
    #
    # INPUTS:
    #   r
    #   f_g = [f(r), g(r)]
    # system = system(r, f_g) = function for system of equations
    # step_size = RK step size
    #
    # OUTPUTS:
    # f_g_next = [f(r+h), g(r+h)], where h = step_size
    
    # extraction of variables for convenience
    f=f_g[0]
    g=f_g[1]
    h=step_size
    
    #RK4 components
    k_1=system(r,[f,g])
    k_2=system(r+(h/2),[f+(h*k_1[0]/2),g+(h*k_1[1]/2)])
    k_3=system(r+(h/2),[f+(h*k_2[0]/2),g+(h*k_2[1]/2)])
    k_4=system(r+h,[f+(h*k_3[0]),g+(h*k_3[1])])
    
    f_g_new=[
        f + (h/6)*( k_1[0] + (2*k_2[0]) + (2*k_3[0]) + k_4[0]),
        g + (h/6)*( k_1[1] + (2*k_2[1]) + (2*k_3[1]) + k_4[1])
        ]
    
    return f_g_new

def RK4_full(r_0,f_g_0,system,step_size,r_max):
    # Impliment RK4
    #
    # INPUTS:
    # r_0: starting value of r
    # f_g_0 = [f(r_0),g(r_0)], i.e. initial conditions
    # system = system(r, f_g)
    # step_size = step size
    # r_max = cut off value of r
    #
    # OUTPUT
    # [r_grid, f_g_list]:
    # r_grid: list of r values where [f,g] evaluated
    # f_g_list: [ [f_0, f_1, f_2 ... ] , [g_0, g_1, g_2 ... ] ]
    
    r_grid=[x for x in np.arange(r_0,r_max,step_size)]
    
    f_g_list=[[],[]]
    f_g_list[0].append(f_g_0[0])
    f_g_list[1].append(f_g_0[1])
    f_g_current=f_g_0
    
    for r in r_grid:
        f_g_next=RK4_step(r,f_g_current,system,step_size)
        f_g_current=f_g_next
        f_g_list[0].append(f_g_current[0])
        f_g_list[1].append(f_g_current[1])
    
    r_grid.append(r_max)
    
    return[r_grid,f_g_list]

def get_g0 (r_0,f_0,n):
    # g = f' initial condition at r_0
    # INPUTS:
    #   r_0 = r initial condition
    #   f_0 = f initial condition
    #   n = winding number
    
    return f_0*n/r_0

def find_f0(r_0,r_max,step_size,system,get_g0,f_0_min,f_0_max):
    # find initial condition for f0
    # use binary search to probe solutions either side of bifurcation point
    # test if f>1 or f<1 at end r_max
    #
    # INPUTS
    #   r_0 = start value of r
    #   r_max = end value of r
    #   step_size = step_size
    #   system = system of 1st order DEs
    #   get_g0 = initial condition on g (given f_0 and r_0)
    #   f_0_min = minimum value of f_0 to try
    #   f_0_max = maximum value of f_0 to try
    #
    # OUTPUTS
    #   f_0 = initial condition on f
    
    # loop control parameters
    f_0_found=False
    loop_count=0
    loop_cut_off=50
    
    while((f_0_found == False) and (loop_count<loop_cut_off)):
        loop_count=loop_count+1
    
        f_0_new = 0.5*(f_0_max + f_0_min)
        g_0_new=get_g0(r_0,f_0_new)
        [r_grid,f_g_list]=RK4_full(r_0, [f_0_new,g_0_new], system, step_size, r_max)
        max_index=len(f_g_list[0])
        f_new_at_r_max=f_g_list[0][max_index-1]
        
        if (f_new_at_r_max < 1):
            f_0_min=f_0_new
        else:
            f_0_max=f_0_new
    
    f_0=f_0_min    
    
    return f_0

def get_monotomic_region(r_list,f_list):
    # return region over which solution is monotomic
    #
    # INPUT
    #   r_list: assending list of r values
    #   f_list: corresponding list of f values
    # OUTPUT
    #   mon_r_list: truncated r list
    #   mon_f_list: truncated f list such that f[i+1]>f[i]
    
    # initalize empty arrays
    mon_r_list=[]
    mon_f_list=[]
    monotomic_end=False
    
    # append first value to arrays
    mon_r_list.append(r_list[0])
    mon_f_list.append(f_list[0])
    
    for index in range(1,len(r_list)):
        if ((f_list[index]>=f_list[index-1]) and (monotomic_end==False)):
            mon_f_list.append(f_list[index])
            mon_r_list.append(r_list[index])
        else:
            monotomic_end=True
    
    return [mon_r_list,mon_f_list]

if __name__ == "__main__":
    # to impliment the code, the only parameters you need to specify are:
    #   n, winding number
    #   lam, lambda
    #   step-size, the step size for the RK4 algorithm
    #   r_max, the maximum value of r the system of equations is solved up to
    
    # parameter set up:
    n=1 # winding number
    lam=1 # lambda
    
    # RK4 set up:
    step_size=1e-3
    r_max=50
    system=partial(system_of_equations,n=n,lam=lam)
    get_g0_given_n=partial(get_g0,n=n)
        
    # initial conditions
    # NB: found using empirically tested method, do not need to specify yourself
    r_0 = step_size # starting value of r
    f_0 = find_f0(r_0,r_max,step_size,system,get_g0_given_n,(r_0/10)**n,((r_0/10)**n)*10) # f initial condition
    g_0 = get_g0_given_n(r_0,f_0) # g = f' initial condition
    
    # call rk4
    [r_grid,f_g_list]=RK4_full(r_0, [f_0,g_0], system, step_size, r_max)
  
    # get region where solution is monotomic
    [r_grid,f_grid]=get_monotomic_region(r_grid, f_g_list[0])
    
    # plot results
    fig,ax = plt.subplots(1,1,dpi=100) #,figsize=(32,8))
    ax.plot(r_grid,f_grid,color="blue")
    ax.set_xlabel("r")
    ax.set_ylabel("f (dimensionless)")
    ax.set_title("solving static field equation for global string with RK4")
    plt.savefig("solving_static_field_equation_for_global_string_with_rk4.png")