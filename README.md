# GeneticAlgorithm

Basic implementation of a Genetic algorithm in Python.

The algorithm is tested on an optimization problem that consists of finding an x &isin; [-1,2], which maximizes the
function f(x) = x sen(10  &pi; x) + 1, that is, finding x<sub>o</sub> such that f(x<sub>o</sub>) &geq; f(x), for all x &isin; [-1,2].

<b>Encoding:</b> In this step the chromosomes are encoded as sequences of digits
binary and have a fixed size 'm', where 'm' is the number of bits needed to
encode a real number in the range [-1,2]. The accuracy required through representation
chromosomal is 6 decimal places.

<b>Population size:</b> The population size is 100 individuals, initially
randomly generated.

<b>Number of generations:</b> as a stopping criterion,
the maximum number of 200 generations has been established.

<b>Fitness function:</b> This function, for optimization problems, is the objective function itself
of the problem. Fitness (chromosome<sub>i</sub>) = f (x<sub>i</sub>) = x sen (10 &pi; x<sub>i</sub>) + 1, where x<sub>i</sub> is the real value
represented by chromosome i.

<b>Selection Method:</b> Selection by Roulette and Elitism.

<b>Genetic Operators:</b> Crossover of a point and simple mutation.
