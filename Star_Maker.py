import numpy as np
from matplotlib import pyplot as plt
from scipy import integrate as integrate
# these are right
h_bar = 6.626e-34 / (2 * np.pi)		# h bar
m_e = 9.10938e-31					# electron mass
m_p = 1.67e-27						# proton mass
k_b = 1.38e-23						# boltzmann's const.
G = 6.67e-11						# Newton's grav. const.
c = 3.0e8							# speed of light
sigma_sb = 5.67e-8					# Stefan-Boltzmann const.
a = 4 * sigma_sb / c 				# radiation const. (this is right)
M_sun = 1.98847e30
L_sun = 3.916e26
R_sun = 6.96342e8

# monotomic ideal gas
gamma =	(5 / 3)						# Adiabatic index: I *think* the this is 5/3 for an ideal gas

### Stellar composition ###
# these are close enough
X = 0.71
X_cno = 0.03 * X
Y = 0.271
Z = 0.0122
mu_weight = 1 / (2 * X + 0.75 * Y + 0.5 * Z)	# temporary mean molecular weight

# Stop integrating when d_tau < threshold
threshold = 1e-3

# this is right
def energy_gen(rho, T, mode = "pp"):
	E_pp = 1.07e-7
	E_cno = 8.24e-26

	if mode == "pp":
		return E_pp * (rho * 1e-5) * (X ** 2) * ((T * 1e-6) ** 4)
	elif mode == "cno":
		return E_cno * (rho * 1e-5) * X * X_cno * ((T * 1e-6) ** 19.9)

# Not yet modified for He and C stars, only works for H stars
# *** Is it just T or T / 10^4 or something??? ***
# this one is right
def opacity(rho, T):
	H_es = 0.02 * (1 + X)
	H_ff = 1.0e24 * (Z + 0.0001) * ((rho * 1e-3) ** 0.7) * (T ** -3.5)
	H_minus = 2.5e-32 * (Z / 0.02) * ((rho * 1e-3) ** 0.5) * (T ** 9)

	kappa_inv = (1 / H_minus) + (1 / max(H_es, H_ff))
	return 1 / kappa_inv

# this one is right
def temp_gradient(rho, T, M, L, r):
	ideal_gas_P = k_b * T * rho / (mu_weight * m_p)	
	degen_P = (3 * (np.pi ** 2)) ** (2 / 3)			
	degen_P *= (h_bar ** 2) / (5 * m_e)				
	degen_P *= ((rho / m_p) ** (5 / 3))			

	rad_P = a * (T ** 4) / 3						
	pressure = ideal_gas_P + degen_P + rad_P

	# opacity
	kappa = opacity(rho, T)

	# Handle temperature gradients
	convection = (1 - (1 / gamma)) * G * M * rho * T / (pressure * (r ** 2))
	radiation = 3 * kappa * rho * L / (16 * np.pi * a * c * (T ** 3) * (r ** 2))
	


	if convection < radiation:
		dominant = "convection"
		T_grad = -convection
	else:
		dominant = "radiation"
		T_grad = -radiation

	return (T_grad, dominant)

# this one is right
def density_gradient(rho, T, M, L, r):
	T_grad = temp_gradient(rho, T, M, L, r)[0]

	dP_drho = (3 * (np.pi ** 2)) ** (2 / 3)
	dP_drho *= (h_bar ** 2) / (3 * m_e * m_p)
	dP_drho *= ((rho / m_p) ** (2 / 3))
	dP_drho += k_b * T / (mu_weight * m_p)

	dP_dT = rho * k_b / (mu_weight * m_p)
	dP_dT += 4 * a * (T ** 3) / 3

	rho_grad = (G * M * rho / (r ** 2)) + (dP_dT * T_grad)
	rho_grad /= -dP_drho

	return rho_grad

# this one is right
def ROC_func(x, r):
	# Components of the x-vector
	# this is right
	rho = x[0]
	temp = x[1]
	mass = x[2]
	lumin = x[3]
	tau = x[4]

	# Handle energy generation (for hydrogen!)
	# this is right
	energy = energy_gen(rho, temp, mode = "pp") + energy_gen(rho, temp, mode = "cno")

	# Handle opacity (for hydrogen stars!)
	kappa = opacity(rho, temp) 

	f0 = density_gradient(rho, temp, mass, lumin, r)
	f1, dominant = temp_gradient(rho, temp, mass, lumin, r)
	f2 = 4 * np.pi * (r ** 2) * rho
	f3 = 4 * np.pi * (r ** 2) * rho * energy
	f4 = kappa * rho

	return np.array([f0, f1, f2, f3, f4]), dominant


def forward_ODE(rho_c, T_c, dr = 1e3):
	r0 = 1e-3			# 1mm start radius

	# Inital mass, luminosity, epsilon, and tau
	M_c = 4 * np.pi * rho_c * (r0 ** 3) / 3
	energy_c = energy_gen(rho_c, T_c, mode = "pp") + energy_gen(rho_c, T_c, mode = "cno")
	opacity_c = opacity(rho_c, T_c)
	L_c = M_c * energy_c * r0
	tau_c = opacity_c * rho_c * r0	# Guess Tau(0) = 0,  idk

	x0 = np.array([rho_c, T_c, M_c, L_c, tau_c])

	r = [r0]
	x = [x0]

	# Use RK4 ODE solver to 
	d_tau = 1e5
	n = 0
	M = 0
	M_max = 1e35

	kappa_values = []
	dominance = []
	while d_tau > threshold and M < M_max:
		# Runge Kutta algorithm, hopefully I didn't screw this up
		k1, dominant = ROC_func(x[n],               r[n])
		k2 = ROC_func(x[n] + dr * k1 / 2, r[n] + dr / 2)[0]
		k3 = ROC_func(x[n] + dr * k2 / 2, r[n] + dr / 2)[0]
		k4 = ROC_func(x[n] + dr * k3,     r[n] + dr)[0]


		next_x = x[n] + (dr / 6) * (k1 + (2 * k2) + (2 * k3) + k4)
		x.append(next_x)
		next_r = r[n] + dr
		r.append(next_r)

		# Evaluate delta tau until  it is "small"
		# When delta tau < threshold we stop integration
		rho, T, M, L = x[n][:-1]
		rho_gradient = density_gradient(rho, T, M, L, r[n])
		kappa = opacity(rho, T)		
		d_tau = kappa * (rho ** 2) / np.abs(rho_gradient)


		if n == int(1e6) or n == int(2e6) or n == int(3e6):
			print("Warning: Integration taking too long... :(")
			print("delta Tau = %E" % d_tau)
			print("n = %i" % n)
			print("mass = %E" % M)
		elif n == int(2e7):
			exit()
		kappa_values.append(kappa)
		dominance.append(dominant)
		n += 1

	# Print results of integration to see if it's working
	print("delta Tau = %E" % d_tau)
	print("n = %i" % n)
	print("mass = %E" % M)
	print("")

	#plt.plot(r[1:], np.log10(kappa_values))
	#plt.show()
	return (np.array(r), np.array(x), np.array(dominance))


def find_error(L, T, R):
	L_blackbody = 4 * np.pi * sigma_sb * (R ** 2) * (T ** 4)
	numerator = L - L_blackbody
	norm = np.sqrt(L_blackbody * L)

	error = numerator / norm
	return error

def find_surface(tau, r):
	tau_infinity = tau[-1]
	dtau = tau_infinity - tau

	R = np.interp(2 / 3, dtau, r)
	return R

# this assumes that f(p1) > 0 and f(p2) < 0
def root_finder(p1, p2, T_c, closeness = 0.1):
	central_density = 0.5 * (p1 + p2)
	r, x, dominance = forward_ODE(central_density, T_c)
	
	# Find the surface!!
	tau = x[:, 4]
	R = find_surface(tau, r)
	T_surf = np.interp(R, r, x[:, 1])
	L = np.interp(R, r, x[:, 3])

	error = find_error(L, T_surf, R)

	print("Central Density: %E" % central_density)
	print("Error: %.2f" % error)
	print("")
	print("Surface Temp: %E" % T_surf)
	if np.abs(error) < closeness:
		return central_density
	elif error > 0:
		return root_finder(central_density, p2, T_c, closeness = closeness)
	elif error < 0:
		return root_finder(p1, central_density, T_c, closeness = closeness)

def multiplot(r, x, rho_c, T_c, dom = None):
	# Find the surface!!
	tau = x[:, 4]
	R = find_surface(tau, r)
	T_surf = np.interp(R, r, x[:, 1])
	M = np.interp(R, r, x[:, 2])
	L = np.interp(R, r, x[:, 3])
	
	error = find_error(L, T_surf, R)

	surface_idx = np.argmin(np.abs(r - R))
	print(surface_idx, r.size)
	radius = r[:surface_idx] * 1e-8
	density = x[:surface_idx, 0]
	temperature = x[:surface_idx, 1]
	mass_enc = x[:surface_idx, 2]
	luminosity = x[:surface_idx, 3]
	tau = x[:surface_idx, 4]

	T_grad_type = dom[:surface_idx]

	fig = plt.figure(figsize = (10, 5))
	ax1 = fig.add_subplot(2, 3, 1)
	ax1.set_title("Density")
	ax1.grid(b = True, axis = "both")
	ax1.plot(radius, density, color = "green")

	ax2 = fig.add_subplot(2, 3, 2)
	ax2.set_title("Temperature")
	ax2.grid(b = True, axis = "both")
	ax2.plot(radius, temperature, color = "red")
	ymin, ymax = ax2.get_ylim()

	ax2.fill_between(radius[T_grad_type == "convection"], ymin, ymax, color = "grey", alpha = 0.6)

	ax3 = fig.add_subplot(2, 3, 3)
	ax3.set_title("Enclosed Mass")
	ax3.grid(b = True, axis = "both")
	ax3.plot(radius, mass_enc, color = "blue")

	ax4 = fig.add_subplot(2, 3, 4)
	ax4.set_title("Luminosity")
	ax4.grid(b = True, axis = "both")
	ax4.plot(radius, luminosity, color = "orange")

	textblock = "$\\rho_c$=%.3f\n" % rho_c
	textblock += "T$_c$=%.3f\n" % T_c
	textblock += "M = %.3fM$_\odot$\n" % (M / M_sun)
	textblock += "R = %.3fR$_\odot$\n" % (R / R_sun)
	textblock += "L = %.3fL$_\odot$\n" % (L / L_sun)
	textblock += "T = %i" % int(T_surf)
	fig.text(0.5, 0.25, textblock, ha = "center", va = "center", fontsize = 14)

	ax6 = fig.add_subplot(2, 3, 6)
	ax6.set_title("Optical Depth")
	ax6.grid(b = True, axis = "both")
	ax6.plot(radius, tau, color = "black")

	title = "$\\rho_c$=%.3f" % rho_c
	title = "T$_c$=%.3f" % T_c

	#fig.suptitle(title, x = 0.5, y = 0.94, ha = "center", va = "center", fontsize = 16)
	fig.subplots_adjust(wspace = 0.3, hspace = 0.35)
	plt.show()


def make_star(T_c, rho_overshoot, rho_undershoot, show_plot = True):
	rho_c = root_finder(rho_overshoot, rho_undershoot, T_c, closeness = 0.05)
	r, x, dominance = forward_ODE(rho_c, T_c, dr = 1e3)

	# Find the surface!!
	tau = x[:, 4]
	R = find_surface(tau, r)
	T_surf = np.interp(R, r, x[:, 1])
	L = np.interp(R, r, x[:, 3])

	M = np.interp(R, r, x[:, 2])
	error = find_error(L, T_surf, R)

	print("Central Temp: %E" % T_c)
	print("Central Density: %E" % rho_c)
	print("Error: %.2f" % error)
	print("")
	print("Mass: %E" % M)
	print("Radius: %E" % R)
	print("Luminosity: %E" % L)
	print("Surface Temp: %E" % T_surf)

	print("")

	if show_plot:
		multiplot(r, x, rho_c, T_c, dom = dominance)

	return (rho_c, T_c, error, M, R, L, T_surf)

# Given a dictionary of where the values are all 1D array-like (with any lengths)
# and a filename, creates a CSV where the columns are these arrays
def makeCSV(data, filename):
    # Create an empty csv file
    data_file = open(filename, 'w+')
    
    dataset = {}
    for key in data:
        dataset.update({key: data[key]})
    
    # Create a header row
    num_rows = {}
    N_rows = 0
    for key in dataset:
        data_file.write(key + ',')
        if not hasattr(dataset[key], '__iter__') and type(dataset[key]) != str:
            num_rows.update({key: 0})
        else:
            current_length = len(dataset[key])
            num_rows.update({key: current_length})
            if current_length > N_rows:
                N_rows = current_length
    data_file.write('\n')
    
    # Add in all the data
    for i in range(N_rows):
        for key in dataset:
          value = dataset[key]
          if num_rows[key] > i:
              data_file.write(str(value[i]) + ',')
          elif i == 0:
              data_file.write(str(value) + ',')
          else:
              data_file.write(',')
        data_file.write('\n')
    data_file.close()

# Converts a CSV file to a dictionary of lists where the header row of the CSV becomes the keys
# of the dictionary and the columns are lists of values that go into their respective key
# If header_idx > 0, all lines before that will be ignored and the line with that index (starting at 0) 
# will be used as the header row
def unpackCSV(filename, empty_value = None, header_idx = 0):
                        
    # Get info from the file
    file_obj = open(filename, 'r')
    lines = file_obj.readlines()
    file_obj.close()
    dataDict = {}
    
    key = ''
    for char in lines[header_idx]:
        if char == ',' or char == '\n':
            dataDict[key] = []
            key = ''
        else:
            key += char
    
    keys = list(dataDict.keys())
    for line in lines[header_idx + 1:]:
        key_idx = 0
        value = ''
        lookfor_close_quote = ''
        for char in line:
            if lookfor_close_quote != '':
                if char == lookfor_close_quote:
                    lookfor_close_quote = ''
                elif char != ',':
                    value += char
            elif char == ',' or char == '\n':
                if key_idx < len(keys):
                    key = keys[key_idx]
                else:
                    continue
                if value == '':
                    if empty_value is not None:
                        dataDict[key].append(empty_value)
                else:        
                    dataDict[key].append(value)
                key_idx += 1
                value = ''
            elif char == '"' or char == "'":
                lookfor_close_quote = char
            else:
                value += char
    
    for key in dataDict.keys():
        dataDict[key] = np.array(dataDict[key])
    
    return dataDict

def Rate_of_Change(r, x):
	# Components of the x-vector
	# this is right
	rho = x[0]
	temp = x[1]
	mass = x[2]
	lumin = x[3]
	tau = x[4]

	# Handle energy generation (for hydrogen!)
	# this is right
	energy = energy_gen(rho, temp, mode = "pp") + energy_gen(rho, temp, mode = "cno")

	# Handle opacity (for hydrogen stars!)
	kappa = opacity(rho, temp) 

	f0 = density_gradient(rho, temp, mass, lumin, r)
	f1, dominant = temp_gradient(rho, temp, mass, lumin, r)
	f2 = 4 * np.pi * (r ** 2) * rho
	f3 = 4 * np.pi * (r ** 2) * rho * energy
	f4 = kappa * rho

	return np.array([f0, f1, f2, f3, f4])

T_c = 1.571e8
def try_ODEINT(rho_c):
	r0 = 1e-3

	# Inital mass, luminosity, epsilon, and tau
	M_c = 4 * np.pi * rho_c * (r0 ** 3) / 3
	energy_c = energy_gen(rho_c, T_c, mode = "pp") + energy_gen(rho_c, T_c, mode = "cno")
	opacity_c = opacity(rho_c, T_c)
	L_c = M_c * energy_c * r0
	tau_c = opacity_c * rho_c * r0	# Guess Tau(0) = 0,  idk

	x0 = np.array([rho_c, T_c, M_c, L_c, tau_c])

	
	N = 100000	
	r_list = np.linspace(r0, R_sun, N)

	x_soln = integrate.solve_ivp(Rate_of_Change, (r0, R_sun), x0, t_eval = r_list).y
	#plt.plot(r_list, x_soln[0, :])
	#plt.show()
	R = find_surface(x_soln[4, :], r_list)

	T_surf = np.interp(R, r_list, x_soln[1, :])
	L = np.interp(R, r_list, x_soln[3, :])

	error = find_error(L, T_surf, R)
	print(error)
	return error

# f(a) > 0 and f(b) < 0
def bisection(f, a, b, closeness = 0.05):

	c = (a + b) / 2
	f_next = f(c)
	if np.abs(f_next) < closeness:
		return c
	elif f_next > 0:
		return bisection(f, c, b, closeness = closeness)
	elif f_next < 0:
		return bisection(f, a, c, closeness = closeness)

def Shoot_Dir_Bisec(f, t0, T, y0, yT, v1, v2, kmax = 200, eps = 1.0e-6):
	v = None
	f1 = integrate.solve_ivp(Rate_of_Change, )

rho_c = bisection(try_ODEINT, 500e3, 0.5e3)
print(rho_c)




"""
#rho_c, T_c, error, M, R, L, T_surf = make_star(1e6, 4.0e5, 3.0e5)
rho_c, T_c = 1.622e5, 1.571e7
r, x, dominance = forward_ODE(rho_c, T_c, dr = 1000.0)
R = find_surface(x[:, 4], r)
T_surf = np.interp(R, r, x[:, 1])
L = np.interp(R, r, x[:, 3])

M = np.interp(R, r, x[:, 2])
error = find_error(L, T_surf, R)
print(error)
multiplot(r, x, rho_c, T_c, dom = dominance)

dataset = {
	"r": r,
	"rho": x[:, 0],
	"T": x[:, 1],
	"M": x[:, 2],
	"L": x[:, 3],
	"tau": x[:, 4],
	"dom": dominance
}
makeCSV(dataset, "welcome_to_PHYS375_stars.csv")
"""

"""
# Run the whole thing
# central temp
T_list = np.linspace(1.0e5, 1.0e6, 10)

dataDict = {
	"Central Temp": [],
	"Central Dens": [],
	"Error": [],
	"Mass": [],
	"Radius": [],
	"Luminosity": [],
	"Surface Temp": []
}

for T in T_list:
	p_predicted = 10 ** (13.518 - 1.321 * np.log10(T))
	p_under = 1.1 * p_predicted
	p_over = 0.9 * p_predicted
	T, p, E, M, R, L, T_surf = make_star(T, p_under, p_over, show_plot = False)
	dataDict["Central Temp"] += [T]
	dataDict["Central Dens"] += [p]
	dataDict["Error"] += [E]
	dataDict["Mass"] += [M]
	dataDict["Radius"] += [R]
	dataDict["Luminosity"] += [L]
	dataDict["Surface Temp"] += [T_surf]
makeCSV(dataDict, "Run_2021-04-03.csv")
"""
