import numpy as np
from copy import deepcopy as deep_copy


def rafael_bambirra_pso(nvar, ncal, type, function, chi, w):

    class Problem:

        def __init__(self, nvar, ncal, type, function, chi, w):
            #
            # particle parameters
            self.c1 = 1.4
            self.c2 = 1.4
            self.chi = chi
            self.w = w
            #
            # problem parameters
            self.dimension = nvar
            self.n_cal = ncal
            self.lower_bound = -100.0
            self.upper_bound = 100.0
            self.type_PSO = type         # (Global / Local)
            self.function = function     # (Sphere / Rastrigin)
            #
            # population parameters
            self.g_best_position = []
            self.g_best_value = np.inf   # default
            if np.sqrt(self.n_cal) > 200:
                self.swarm_size = 200
            else:
                self.swarm_size = int(np.sqrt(self.n_cal))
            self.n_iter = self.n_cal//self.swarm_size
            if self.type_PSO == 'Global':
                self.neighbours = self.swarm_size
            elif self.type_PSO == 'Local':
                self.neighbours = 2

        def cost_function(self, x):
            #
            # ============== Sphere function ================
            if self.function == 'Sphere':
                s = 0
                f_star = 0  # -1400
                for i in range(self.dimension):
                    s = s + (x[i]) ** 2
                fval = s + f_star
                return fval
            #
            # ============== Rastrigin function ================
            if self.function == 'Rastrigin':
                s = 0
                f_star = 0
                for i in range(self.dimension):
                    z = 5.12 * ((x[i] - 0) / 100)
                    s = s + z**2 - 10 * np.cos(2 * np.pi * z) + 10
                fval = s + f_star
                return fval

    class Particle(Problem):

        def __init__(self):
            Problem.__init__(self, nvar, ncal, type, function, chi, w)
            self.position = np.random.rand(self.dimension) * \
                (self.upper_bound - self.lower_bound) + self.lower_bound
            self.fitness = Problem.cost_function(self, x=self.position)

        def calc_fitness(self):
            self.fitness = Problem.cost_function(self, x=self.position)
            return self.fitness

        def move(self, velocity):
            self.position = self.position + velocity
            for i in range(self.dimension):
                flag = False
                while flag is False:    # make the particle bounce
                    if self.position[i] > self.upper_bound:
                        self.position[i] = self.upper_bound - (self.position[i] - self.upper_bound)
                    if self.position[i] < self.lower_bound:
                        self.position[i] = self.lower_bound - (self.position[i] - self.lower_bound)
                    if self.lower_bound < self.position[i] < self.upper_bound:
                        flag = True
            return self.position

        def update_velocity(self, velocity, p_best_position, g_best_position):
            r1 = np.random.rand(self.dimension)
            r2 = np.random.rand(self.dimension)
            cognitive = self.c1 * r1 * (p_best_position - self.position)
            social = self.c2 * r2 * (g_best_position - self.position)
            velocity = self.chi * (self.w * velocity + cognitive + social)
            return velocity

    class PSO(Problem):

        def __init__(self):
            Problem.__init__(self, nvar, ncal, type, function, chi, w)
            self.swarm = np.zeros([self.swarm_size, self.dimension])
            self.fitness = np.zeros(self.swarm_size) + np.inf
            self.velocity = np.zeros([self.swarm_size, self.dimension])
            self.personal_best = np.zeros(self.swarm_size) + np.inf
            self.personal_best_pos = np.zeros([self.swarm_size, self.dimension])
            if self.type_PSO == 'Global':
                self.global_best = np.inf
                self.global_best_pos = np.zeros(self.dimension)
            elif self.type_PSO == 'Local':
                self.global_best = np.zeros(self.swarm_size) + np.inf
                self.global_best_pos = np.zeros([self.swarm_size, self.dimension])
            #
            # initialize swarm
            for i in range(self.swarm_size):
                part = Particle()
                self.swarm[i, :] = part.position
                self.fitness[i] = part.fitness
                self.personal_best_pos[i, :] = self.swarm[i, :]
                self.personal_best[i] = self.fitness[i]
            self.global_best, self.global_best_pos = \
                self.update_best(g_best_value=self.global_best, g_best_position=self.global_best_pos)
            self.optimize()

        def optimize(self):
            for i in range(self.n_iter):
                # print('iteration = ' + str(i))
                # Update position of the particles
                for k in range(self.swarm_size):
                    part = Particle()
                    part.position = self.swarm[k, :]
                    if self.type_PSO == 'Global':
                        self.velocity[k, :] = part.update_velocity(velocity=self.velocity[k, :],
                                                                   p_best_position=self.personal_best_pos[k, :],
                                                                   g_best_position=self.global_best_pos)
                    elif self.type_PSO == 'Local':
                        self.velocity[k, :] = part.update_velocity(velocity=self.velocity[k, :],
                                                                   p_best_position=self.personal_best_pos[k, :],
                                                                   g_best_position=self.global_best_pos[k, :])
                    part.position = part.move(self.velocity[k, :])
                    self.swarm[k, :] = part.position
                    self.fitness[k] = part.calc_fitness()
                    if self.fitness[k] < self.personal_best[k]:
                        self.personal_best[k] = self.fitness[k]
                        self.personal_best_pos[k, :] = self.swarm[k, :]
                self.global_best, self.global_best_pos = \
                    self.update_best(g_best_value=self.global_best, g_best_position=self.global_best_pos)
            return self.global_best, self.global_best_pos

        def update_best(self, g_best_value, g_best_position):
            if self.type_PSO == 'Global':
                for i in range(self.swarm_size):
                    if self.fitness[i] < g_best_value:
                        g_best_value = deep_copy(self.fitness[i])
                        g_best_position = deep_copy(self.swarm[i, :])
            elif self.type_PSO == 'Local':
                aux_g_best_value = deep_copy(g_best_value)
                aux_1 = -1
                aux_2 = 1
                for i in range(self.swarm_size):
                    if aux_2 >= self.swarm_size:
                        aux_2 = 0
                    vector = [aux_g_best_value[aux_1], self.fitness[i], aux_g_best_value[aux_2]]
                    g_best_value[i] = deep_copy(np.min(vector))
                    if np.argmin(vector) == 0:
                        g_best_position[i, :] = deep_copy(g_best_position[aux_1, :])
                    elif np.argmin(vector) == 1:
                        g_best_position[i, :] = deep_copy(self.swarm[i, :])
                    elif np.argmin(vector) == 2:
                        g_best_position[i, :] = deep_copy(g_best_position[aux_2, :])
                    aux_1 += 1
                    aux_2 += 1
            return g_best_value, g_best_position

    prob = Problem(nvar, ncal, type, function, chi, w)

    p = PSO()

    print('========== Parameters of PSO ===========')
    print('dimension = ' + str(prob.dimension))
    print('n_cal = ' + str(prob.n_cal))
    print('pop size = ' + str(prob.swarm_size))
    print('chi = ' + str(prob.chi))
    print('weight inertia = ' + str(prob.w))
    print('type_PSO = ' + prob.type_PSO)
    print('function = ' + str(prob.function))
    print('================ BEST ==================')
    if prob.type_PSO == 'Global':
        print('global best =' + str(p.global_best))
        print('and its position = ' + str(p.global_best_pos))
        gbest = p.global_best
        gbest_position = p.global_best_pos
    elif prob.type_PSO == 'Local':
        print('global best =' + str(np.min(p.global_best)))
        print('and its position = ' + str(p.global_best_pos[np.argmin(p.global_best)]))
        gbest = np.min(p.global_best)
        gbest_position = p.global_best_pos[np.argmin(p.global_best)]
    return gbest, gbest_position


f, x = rafael_bambirra_pso(nvar=10, ncal=100000, type='Global', function='Rastrigin', chi=0.9, w=0.9)
