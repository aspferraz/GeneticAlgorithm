# GeneticAlgorithm

Basic implementation of a Genetic algorithm in Python.

The algorithm is tested on an optimization problem that consists of finding an x &isin; [-1,2], which maximizes the
function f(x) = x sen(10  &pi; x) + 1, that is, finding x<sub>o</sub> such that f(x<sub>o</sub>) &geq; f(x), for all x &isin; [-1,2].

<b>Encoding:</b> In this step the chromosomes are encoded as sequences of digits
binary and have a fixed size 'm', where 'm' is the number of bits needed to
encode a real number in the range [-1,2]. The accuracy required through representation
chromosomal is 6 decimal places.
