import numpy as np
from matplotlib import pyplot as plt

h_bar = 6.626e-34 / (2 * np.pi)		# h bar
m_e = 9.10938e-31					# electron mass
m_p = 1.67e-27						# proton mass
k_b = 1.38e-23						# boltzmann's const.
G = 6.67e-11						# Newton's grav. const.
c = 3.0e8							# speed of light
sigma_sb = 5.67e-8					# Stefan-Boltzmann const.
a = 4 * sigma_sb / c 				# radiation const.


gamma =	(5 / 3)						# Adiabatic index: I *think* the this is 5/3 for an ideal gas

### Stellar composition ###
X = 0.71
X_cno = 0.096e-2
Y = 0.271
Z = 0.0122
mu_weight = 1 / (2 * X + 0.75 * Y + 0.5 * Z)	# temporary mean molecular weight

# Other stuff idk
threshold = 1e-3

def energy_gen(rho, T, mode = "pp"):
	E_pp = 1.07e-7
	E_cno = 8.24e-26

	if mode == "pp":
		return E_pp * (rho * 1e-5) * (X ** 2) * ((T * 1e-6) ** 4)
	elif mode == "cno":
		return E_cno * (rho * 1e-5) * X * X_cno * ((T * 1e-6) ** 19.9)

# Not yet modified for He and C stars, only works for H stars
# *** Is it just T or T / 10^4 or something??? ***
def opacity(rho, T):
	H_es = 0.02 * (1 + X)
	H_ff = 1.0e24 * (Z + 0.0001) * ((rho * 1e-3) ** 0.7) * (T ** -3.5)
	H_minus = 2.5e-32 * (Z / 0.02) * ((rho * 1e-3) ** 0.5) * (T ** 9)

	kappa = (1 / H_minus) + (1 / max(H_es, H_ff))
	return 1 / kappa

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
	T_grad = -min(convection, radiation)

	return T_grad

def density_gradient(rho, T, M, L, r):
	T_grad = temp_gradient(rho, T, M, L, r)

	dP_drho = (3 * (np.pi ** 2)) ** (2 / 3)
	dP_drho *= (h_bar ** 2) / (m_e * m_p)
	dP_drho *= ((rho / m_p) ** (2 / 3))
	dP_drho += k_b * T / (mu_weight * m_p)

	dP_dT = rho * k_b / (mu_weight * m_p)
	dP_dT += 4 * a * (T ** 3) / 3

	rho_grad = (G * M * rho / (r ** 2)) + (dP_dT * T_grad)
	rho_grad /= -dP_drho

	return rho_grad

def ROC_func(x, r):
	# Components of the x-vector
	rho = x[0]
	temp = x[1]
	mass = x[2]
	lumin = x[3]
	tau = x[4]

	# Handle energy generation (for hydrogen!)
	energy = energy_gen(rho, temp, mode = "pp") + energy_gen(rho, temp, mode = "cno")

	# Handle opacity (for hydrogen stars!)
	kappa = opacity(rho, temp) 

	f0 = density_gradient(rho, temp, mass, lumin, r)
	f1 = temp_gradient(rho, temp, mass, lumin, r)
	f2 = 4 * np.pi * (r ** 2) * rho
	f3 = 4 * np.pi * (r ** 2) * rho * energy
	f4 = kappa * rho

	return np.array([f0, f1, f2, f3, f4])

def forward_ODE(rho_c, T_c, dr = 1e3):
	r0 = 1e-3			# 1m start radius

	# Inital mass, luminosity, epsilon, and tau
	M_c = 4 * np.pi * rho_c * (r0 ** 3) / 3
	energy_c = energy_gen(rho_c, T_c, mode = "pp") + energy_gen(rho_c, T_c, mode = "cno")
	opacity_c = opacity(rho_c, T_c)
	L_c = M_c * energy_c
	tau_c = opacity_c * rho_c * r0	# Guess Tau(0) = 0,  idk

	x0 = np.array([rho_c, T_c, M_c, L_c, tau_c])

	r = [r0]
	x = [x0]

	# Use RK4 ODE solver to 
	d_tau = 1e5
	n = 0
	M = 0
	M_max = 1e35
	while d_tau > threshold and M < M_max:
		# Runge Kutta algorithm, hopefully I didn't screw this up
		k1 = ROC_func(x[n],               r[n])
		k2 = ROC_func(x[n] + dr * k1 / 2, r[n] + dr / 2)
		k3 = ROC_func(x[n] + dr * k2 / 2, r[n] + dr / 2)
		k4 = ROC_func(x[n] + dr * k3,     r[n] + dr)
		next_x = x[n] + (dr / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
		x.append(next_x)
		next_r = r[n] + dr
		r.append(next_r)

		# Evaluate delta tau until it is "small"
		# When delta tau < threshold we stop integration
		rho, T, M, L = x[n][:-1]
		rho_gradient = density_gradient(rho, T, M, L, r[n])
		kappa = opacity(rho, T)
		d_tau = kappa * (rho ** 2) / np.abs(rho_gradient)

		if n > 1e6:
			print("STOPPING: Integration taking too long... :(")
			print("delta Tau = %E" % d_tau)
			print("n = %i" % n)
			print("mass = %E" % M)
			exit()
		n += 1

	# Print results of integration to see if it's working
	print("delta Tau = %E" % d_tau)
	print("n = %i" % n)
	print("mass = %E" % M)
	print("")

	return (np.array(r), np.array(x))


def find_error(L, T, R):
	numerator = L - 4 * np.pi * sigma_sb * (R ** 2) * (T ** 4)
	norm = np.sqrt(4 * np.pi * sigma_sb * (R ** 2) * (T ** 4) * L)

	error = numerator / norm
	return error

# this assumes that f(p1) > 0 and f(p2) < 0
def root_finder(p1, p2, T_c, closeness = 0.1):
	central_density = 0.5 * (p1 + p2)

	r, x = forward_ODE(central_density, T_c)
	
	# Find the surface!!
	tau = x[:, 4]
	tau_infinity = x[-1, 4]
	dtau = tau_infinity - tau
	print(tau_infinity)
	#plt.plot(r, tau)
	#plt.plot(r, dtau)
	#plt.show()
	surface_condition = np.abs(dtau - 0.6666666)
	plt.plot(r, surface_condition)
	plt.show()
	surface_idx = np.argmin(surface_condition)

	print("Surface Index: %i" % surface_idx)
	print("Tau Inf - Tau = %.6f" % dtau[surface_idx])
	R = r[surface_idx]
	T_surf = x[surface_idx, 1]
	L = x[surface_idx, 3]

	error = find_error(L, T_surf, R)

	print("Central Density: %E" % central_density)
	print("Error: %.2f" % error)
	print("")
	print("Surface Temp: %E" % T_surf)
	if np.abs(error) < closeness:
		return p
	elif error > 0:
		return root_finder(central_density, p2, T_c, closeness = closeness)
	elif error < 0:
		return root_finder(p1, central_density, T_c, closeness = closeness)


# Run the whole thing
if __name__ == "__main__":
	# central temp
	T_c = 1.571e7
	rho_c = root_finder(T_c, 1.3081e5, 1.2698e5, closeness = 0.05)
	r, x = forward_ODE(rho_c, T_c)

	# Find the surface!!
	tau = x[:, 4]
	tau_infinity = x[-1, 4]
	dtau = tau_infinity - tau
	surface_idx = np.argmin(np.abs(dtau - (2 / 3)))

	R = r[surface_idx]
	T_surf = x[surface_idx, 1]
	L = x[surface_idx, 3]
	M = x[surface_idx, 2]
	error = find_error(L, T_surf, R)

	print("Central Temp: %E" % T_c)
	print("Central Density: %E" % rho_c)
	print("Error: %.2f" % error)
	print("")
	print("Mass: %E" % M)
	print("Radius: %E" % R)
	print("Luminosity: %E" % L)
	print("Surface Temp: %E" % T_surf)



"""
rho_c = 1e5

r, x = forward_ODE(rho_c, T_c)

density = x[:, 0]
temperature = x[:, 1]
mass = x[:, 2]
luminosity = x[:, 3]
optical_depth = x[:, 4]

print(mass[-1], luminosity[-1], temperature[-1], r[-1] * 1e-3)
print(error_func(density[0], temperature[-1], luminosity[-1], r[-1]))

fig = plt.figure(figsize = (8, 5))
ax = fig.add_subplot(1, 1, 1)

ax.grid(b = True, axis = "both")
ax.plot(r, density)

plt.show()
"""
