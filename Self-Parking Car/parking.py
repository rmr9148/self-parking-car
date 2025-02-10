import math
import time
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy.interpolate import CubicSpline
import sys

# initial states
initial_state = [0, 8, 0, 0]
final_state = [0, 0, 0, 0]

# graph values
x_graph = [-15, -4, -4, 4, 4, 15]
y_graph = [3, 3, -1, -1, 3, 3]

# defaults
generations = 0
pop_size = 200
opv_size = 10
binary_code_size = 7
mutation_prob = 0.005
k_val = 200
convert_power = (2 << (binary_code_size - 1)) - 1

# constraints
cost_tolerance = 0.1
max_pop_size = 500
max_gen_num = 1200

# bounds
gamma_ub = .524
gamma_lb = -.524
beta_ub = 5
beta_lb = -5

def create():
    population = []
    for i in range(pop_size):
        gamma_beta = []
        for j in range(opv_size):
            gamma = np.random.randint(0, np.power(2, binary_code_size) - 1)
            beta = np.random.randint(0, np.power(2, binary_code_size) - 1)
            gamma_beta.append(gamma)
            gamma_beta.append(beta)
        population.append(gamma_beta)

    return population

def interpolate(individual):
    time_range = range(0, opv_size)
    gamma, beta = [], []

    for i in range(opv_size * 2):
        if (i % 2) == 0:
            new_indiv = (individual[i] / convert_power) * (gamma_ub - gamma_lb) + gamma_lb
            gamma.append(new_indiv)
        else:
            new_indiv = (individual[i] / convert_power) * (beta_ub - beta_lb) + beta_lb
            beta.append(new_indiv)

    gamma_prime = CubicSpline(time_range, gamma, bc_type='natural')
    beta_prime = CubicSpline(time_range, beta, bc_type='natural')

    time_range = np.linspace(0, 10, (pop_size // 2))
    gamma_in_range = gamma_prime(time_range)
    beta_in_range = beta_prime(time_range)

    return gamma_in_range, beta_in_range

def distance(state):
    x = state[0]
    y = state[1]
    if x <= -4 and y <= 3:
        return k_val + abs(9 - (y * y))
    elif -4 < x < 4 and y <= -1:
        return k_val + abs(1 - (y * y))
    elif x >= 4 and y <= 3:
        return k_val + abs(9 - (y * y))
    return 0

def cost(gamma_interpolate, beta_interpolate, current, final_state):
    infeasibility = 0

    for i in range(pop_size // 2):
        current = state_calc(current, gamma_interpolate[i], beta_interpolate[i])
        infeasibility += distance(current)

    dist = sp.spatial.distance.euclidean(current, final_state)

    return dist + infeasibility

def state_calc(state, gamma, beta):
    h = 0.1
    x, y, a, v = state

    x += (h * (v * math.cos(a)))
    y += (h * (v * math.sin(a)))
    a += h * gamma
    v += h * beta

    return np.array([x, y, a, v])

def fitness_calc(pop):
    fitness = np.zeros([pop_size])

    for i in range(pop_size):
        gamma_new, beta_new = interpolate(pop[i])
        fitness[i] = (1 / (1 + cost(gamma_new, beta_new, initial_state, final_state)))

    return fitness

def decode(n):
    m = n >> 1
    while m:
        n ^= m
        m >>= 1
    return n

def mutate(bit_string):
    result = ''
    for bit in bit_string:
        rand = np.random.random()
        if rand <= mutation_prob:
            if bit == '1':
                result += '0'
            else:
                result += '1'
        else:
            result += bit
    return result

def crossover(parents_1, parents_2):
    cross = np.random.randint(1, 19)
    child1, child2 = [], []

    for i in range(opv_size * 2):
        par1_val = parents_1[i]
        par2_val = parents_2[i]
        par1_gray = bin(par1_val ^ (par1_val >> 1))[2:]
        par2_gray = bin(par2_val ^ (par2_val >> 1))[2:]

        if i == cross:
            cross = np.random.randint(1, binary_code_size - 1)
            child1_val = int(mutate(par1_gray[:cross] + par2_gray[cross:]), 2)
            child2_val = int(mutate(par2_gray[:cross] + par1_gray[cross:]), 2)

            child1.append(decode(child1_val))
            child2.append(decode(child2_val))
        elif i > cross:
            child1_val = int(mutate(par2_gray), 2)
            child2_val = int(mutate(par1_gray), 2)

            child1.append(decode(child1_val))
            child2.append(decode(child2_val))
        else:
            child1_val = int(mutate(par1_gray), 2)
            child2_val = int(mutate(par2_gray), 2)

            child1.append(decode(child1_val))
            child2.append(decode(child2_val))

    return np.array(child1), np.array(child2)

def next_gen(pop, elite, prob):
    parents = np.random.choice(np.arange(0, pop_size), [(pop_size // 2) - (1 - (pop_size + 1) % 2), 2], p=prob)
    children = [elite]
    for pair in parents:
        par1 = pop[pair[0]]
        par2 = pop[pair[1]]
        child1, child2 = crossover(par1, par2)
        children.append(child1)
        children.append(child2)

    return children

def plot_settings(y_vals, x_vals, title, x_axis, y_axis):
    plt.figure(figsize=(10, 8))
    plt.plot(x_vals, y_vals, 'black')
    plt.title(title)
    plt.axis('square')
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.savefig(title + '.png')
    plt.grid()
    plt.show()
    return

def main(args):

    # Genetic Algorithm
    
    global elitist
    begin_time = time.time()
    fin_time = begin_time

    pop = create()

    gen_count, elite_gen_count = 0, 0
    elitist = None
    score = -1
    tolerance = 10000
    has_parked = False
    while not has_parked:
        if (elite_gen_count != 0 and elite_gen_count % 50 == 0) or (gen_count >= 1200):
            print('Agent is not learning or max generations has been hit; Restarting')
            pop = create()

            gen_count = 0
            elitist = None
            score = -1
            elite_gen_count = 0
            tolerance = 10000
            has_parked = False

        fitness = fitness_calc(pop)
        max_val = np.argmax(fitness)

        if score < fitness[max_val]:
            elitist = pop[max_val]
            tolerance = (1 / fitness[max_val]) - 1
            elite_gen_count = 0
        else:
            elite_gen_count += 1

        result_string = "Generation " + str(gen_count) + " : J = " + str(tolerance)
        print(result_string)

        pop = next_gen(pop, elitist, np.divide(fitness, np.sum(fitness)))
        gen_count += 1

        fin_time = time.time()
        has_parked = (tolerance < cost_tolerance) or ((fin_time - begin_time) / 60 >= 7)

    print(str((fin_time - begin_time) / 60) + ' minutes')

    # Graph Printing

    gamma_new, beta_new = interpolate(elitist)
    new_vals = np.linspace(0, 10, 100)
    new_history = np.linspace(0, 10, 101)
    current = np.copy(initial_state)
    state_vals = np.array([current])
    control_vals = []
    for i, j in zip(gamma_new, beta_new):
        current = state_calc(current, i, j)
        state_vals = np.vstack((state_vals, current))
        control_vals.append(i)
        control_vals.append(j)

    np.savetxt('controls.dat', np.array(control_vals), fmt='%.18f')

    plt.figure(figsize=(10, 8))
    plt.plot(state_vals[:, 0], state_vals[:, 1], 'blue')
    plt.plot(x_graph, y_graph, 'black')
    plt.axis('square')
    plt.title('Path')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig('path.png')
    plt.grid()
    plt.show()

    plot_settings(state_vals[:, 0], new_history, "x-history", "Time (s)", "x (ft)")
    plot_settings(state_vals[:, 1], new_history, "y-history", "Time (s)", "y (ft)")
    plot_settings(state_vals[:, 2], new_history, "alpha-history", "Time (s)", "alpha (radians)")
    plot_settings(state_vals[:, 3], new_history, "v-history", "Time (s)", "v (ft/s)")
    plot_settings(gamma_new, new_vals, "gamma-history", "Time (s)", "gamma (radians/s)")
    plot_settings(beta_new, new_vals, "beta-history", "Time (s)", "beta (ft/s^2)")

    final_state = state_vals[-1]
    print("Final state values:")
    print("x_f = " + str(final_state[0]))
    print("y_f = " + str(final_state[1]))
    print("alpha_f = " + str(final_state[2]))
    print("v_f = " + str(final_state[3]))

if __name__ == "__main__":
    main(sys.argv)