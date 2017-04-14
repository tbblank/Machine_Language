from random import Random;
from time import time;
from math import sin;
from math import sqrt;
from math import fabs;
from inspyred import ec;
from inspyred.ec import terminators;
import inspyred;

#Generator
def gen(random, args):
	#size = args.get('num_inputs', 100)
	return [(random.randint(-500, 500), random.randint(-500, 500)) for i in range(100)]


#Evaluator
def eval(candidates, args):
	fitness = []
	for cs in candidates:
		fit = 418.9829*2 - (-1*cs[0]*sin(sqrt(fabs(cs[0]))) + -1*cs[1]*sin(sqrt(fabs(cs[1]))))
		fitness.append(fit)
	return fitness

#Mutator
def mutate(random, candidates, args):
	mut_rate = .1 #args.setdefault('mutation_rate', 0.1)
	#bounder = args['_ec'].bounder
	index = 0
	for i, j in candidates:
		if random.random() < mut_rate:
			x = int(i + random.gauss(0,1))
			y = int(j + random.gauss(0,1))
			print("Mutated candidate #: ", index, " ", (i,j), " to: ", (x,y))
			candidates[index] = (x,y)
			index += 1
	return candidates
	
	
#   Main
rand = Random()
rand.seed(int(time()))
candidates = gen(rand,1)
print("Generating candidates...\n\n", candidates)
print("\n\nEvaluating the fitness of the candidates...\n\n", eval(candidates,1), "\n\n\n")
mutate(rand, candidates, 1)

evo = inspyred.ec.EvolutionaryComputation(rand)
evo.selector = inspyred.ec.selectors.tournament_selection
evo.variator = [inspyred.ec.variators.uniform_crossover, mutate]
evo.terminator = terminators.evaluation_termination

# I tried to make the below work, but kept running into problems with the evolve method using tuples
# with the evaluating method.  I believe the preceeding functions are correct, with some possible tweaks
# to the evaluator method to make the evolve method work.

#res = evo.evolve(generator=gen, evaluator=eval, pop_size=1, max_evaluations=20000, mutation_rate=.25)
