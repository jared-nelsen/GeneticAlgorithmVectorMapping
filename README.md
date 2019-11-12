# Genetic Algorithm Vector Mapping

##### A project studying Genetic Algorithms as a training algorithm for neural networks

In this project, I built a Genetic Algorithm that trains a neural network to map arbitrary data vectors together. I asked a simple question ***how well can a neural network be trained to be a mapping operator between two sets of random data.*** This project is purely an ***experiment*** for use in learning how to use Genetic Algorithms as a training algorithm for neural networks. 

##### Preliminary results

Based on trials of randomized data the algorithm proves to converge towards a mapping function between generated data. Just like any other approach to training a neural network, the algorithm may never arrive at a perfect mapping function because there might not be a perfect one, or that it is too highly dimensional to approximate in any reasonable amount of time. However, the algorithm proves to steadily approach an approximation. Different runs produce different results: one run may converge all the way to a near-optimal approximation and another run might flatline along the way. In the future I might experiment with the algorithm parameters on one single dataset to find the best overall parameters. 

##### Recommendations

In order to understand the source code and the methodology it is first necessary to understand Genetic Algorithms. I've included several sections below outlining traditional training approaches and Genetic Algorithms in general.

My recommended reading path:

> 1. Read the summary and outline in the sections below to learn more about Genetic Algorithms
> 2. Follow this path through the source code:
>    1. gavm.py
>    2. DataFrame.py
>    3. MappingOperator.py
>    4. EvaluationModule.py
>    5. GeneticAlgorithm.py

##### Experimenting with GAVM

To run GAVM:

> 1. Change directories to src
> 2. Run this command in a terminal: ***python gavm.py***

To experiment with the parameters of the algorithm, alter the source code values in EvaluationModule.py and rerun GAVM.

I've tuned the current incarnation of EvaluationModule.py with the most effective parameters I've found this far. Note that the dimensions of the training data that is generated are currently very small for experimentation's sake.

***Happy experimenting!***

### Genetic Algorithms: A New Approach to Training Neural Networks

Traditionally, neural networks are trained using the backpropagation algorithm. Backpropagation is a well studied algorithm and is used as the default training algorithm for most major neural network frameworks. 

The task of training a neural network as a learning system can be considered as the ***optimization*** of weight values that constitute the underlying mapping function, if there is one, between a set of inputs and outputs. In other words, neural networks ***learn to associate*** inputs and outputs. There is a vast body of literature on why this is useful in a number of different domains so I won't go into that here. What we care about is what is going on mathematically during the learning process.

Backpropagation is a method of making weight adjustments in the network based on ***calculated*** error as measured using a cost function. Common cost functions like mean squared error give exact measures of how off the association is and gives the backpropagation algorithm information about which weights to adjust and by how much. But is this the best way to learn an association? Backpropagation comes with its own set of challenges like the vanishing gradients problem and uncertainty about what learning rate to use. Can we do better?

Traditional backpropagation using gradient descent:

![Traditional Gradient Descent](https://miro.medium.com/max/1005/1*_6TVU8yGpXNYDkkpOfnJ6Q.png)

#### Optimization as a Search Problem

What if there was a more abstract way to treat the ***optimization*** of weights in a neural network? We can steal an idea from another area of machine learning: ***combinatorial optimization.*** In a nutshell combinatorial optimization is the practice of finding a ***single optimal solution*** out of a ***set of solutions.*** If one were to consider a set of weights of a neural network to be a ***configuration*** of weights then a question might arise: "Is this configuration of weights the ***best configuration*** out of the set of ***all possible configurations?***" By asking this question, we reframe the problem of *training* the weights of the network into a problem of *searching* for the best possible weights.

> "All is for the best in this best of all possible worlds!" *Pangloss, Candide*.

##### How then do we search?

> In Computer Science, a search algorithm is any algorithm which retrieves information from within some data structure or calculated discrete or continuous search space.

To be pedantic, even though we are using search to reframe our thought process, we are still trying to learn an underlying mapping function between input and outputs. We are just ***traversing*** the search space in a different manner. To further reframe our thoughts, instead of calling a the measure of how well the neural network fits the underlying data a ***loss function*** we call it an ***heuristic***. Our heuristic will guide us through the search space by giving our algorithm an objective measure of how good or how poor a solution is.

##### The Motivation

There is a traditional scenario that every Computer Science student learns about when learning about optimization:

> Imagine a mountain climber, we will call him Gerald, who, like most sensible mountain climbers, wants to get to the top of any mountain he climbs. Unfortunately, Gerald recently went blind because he ate too many potato chips (Google it). However, Gerald is a smart and persistent chap and can figure out when he takes a step is if he is higher than we was before taking it. This is a great talent and his mom is very proud.

I'll be willing to bet more than a few tater tots that Gerald can get to the top of this mountain using his *heuristic*:

![I'll bet Gerald can get to the top of this mountain:](https://upload.wikimedia.org/wikipedia/commons/0/05/Hill_climb.png)

> Upon reaching the top of of this easy mountain, Gerald calls his mom and tells her about his great success. His mom is very proud and offers to buy him some more potato chips if he can best himself and climb a more challenging mountain:

Can Gerald climb this more challenging mountain with his simple *heuristic*?

![Gerald's next mountain](https://upload.wikimedia.org/wikipedia/commons/7/7e/Local_maximum.png)

> Well it depends on where he starts, right? If he starts to the far right of the mountain he will get to the top by choosing at each step to ascend. What if he started to the left of the mountain? Gerald, as smart as he is with his heuristic will end up at the top of the smaller false peak and inevitably call his mom. Unfortunately, Gerald failed to get to the true ***global optimum*** and ended up at a ***local optimum***. If you were his mom would you buy him some potato chips then?

Jokes aside, how would our optimizer handle this problem?

##### Adding a Crucial Element

How can we solve this problem of converging to a local optimum? The simple answer is to give our optimizer a better heuristic to search with. In my example, the character always ***exploits*** the current state by always ascending the mountain, which is in actuality an objective function. It is clear that our searcher will need to ***explore*** the search space more!

![Exploration](https://d9hhrg4mnvzow.cloudfront.net/try.neuroshell.com/features/8f5ea36d-genetic-optimization_0cv0bb0cv0bb00000001o.jpg)

However, it is not always clear when to ***explore*** and when to ***exploit***. What would you say is the right balance between exploring and exploiting? What would happen if we explored all the time? It is clear that we would never reach any optimum! And we have already surmised what would happen if we were to exploit all the time. Let's try to answer this by observing a natural phenomenon: ***natural selection***!

#### Learning With Natural Processes

Oftentimes Computer Scientists use observation as a means to solve problems. There are many techniques in computing that have been developed by observing the natural world. Techniques such as Simulated Annealing, Ant Colony Optimization, and Particle Swarm Optimization are among the more well known candidates.

> See this link for a great list of metaphor based metaheuristics: https://en.wikipedia.org/wiki/List_of_metaphor-based_metaheuristics

However, I looked to a technique called a ***Genetic Algorithm*** for this project. A Genetic Algorithm mimics biological evolution to explore search spaces. The idea behind a GA is to model a balance between exploration and exploitation by with two key components:

> ***Mutation*** is the ***exploratory*** element of the algorithm. It mimics the mutations biological creatures develop that allow them to adapt to new environments. Just like in the natural world, mutation occurs very ***infrequently***. The thinking goes that a creature (a solution) that only mutates itself will likely not survive for long. However, some mutation might offer a competitive advantage for a creature in the quest for survival. If we relate this back to what is going on in the optimization of a search space we can notice an allegory: exploration, just like biological mutation, can be beneficial in avoiding local optima in the search space.
>
> ***Crossover*** is the ***exploitative*** element of the algorithm. It mimics the biological phenomenon of reproduction. Luckily, when two humans reproduce the child is definitely going to be human too. This is good because the child's parents were successful enough to reproduce. In other words, they were a good solution to being human. It is likely their child will be too. This is an analog to our simplistic heuristic of always improving on our current solution in the search space because it will lead to immediate results. This is the driving force in optimization. It happens ***often***.

##### Key Components

There are two other key components of the Genetic Algorithm:

> 1. An ***encoding*** is a representation of a solution in the search space. Encodings are powerful because they are flexible. For example, in the ***Traveling Salesman Problem*** an encoding would be a route through the cities in the graph. ***Simulation parameters*** can be represented as an encoding of set of real valued numbers. ***Instructions*** can be represented as encodings using binary A ***neural network*** can be represented as a set of real valued weights. As long as there are ***discrete components represented as arrangements*** then a GA might be a good choice as an optimization algorithm. Encodings are arranged as a ***population*** of candidate solutions.
> 2. A ***fitness function***, or heuristic, is a measure of how 'good' a solution is. In the Traveling Salesman problem a good fitness function would be ***overall route length***. In a simulation a good fitness function might be a ***number of how well an object performs in the simulation***. In procedure generation a good fitness function might be ***how far off*** following the generated instructions will leave you. In a neural network a ***loss function*** is a good fitness function. A ***measure of how optimal*** a solution is makes a good fitness function.

#### Genetic Algorithm Pseudocode

The pseudocode behind the algorithm is:

```pyt
while generationCount < maxGenerations and not optimalSolutionFound
	selectFitessMembersFromPopulationToCrossOver()
	crossoverSelectedMembers()
	mutateSelectedMembers()
	evaluatePopulationMembersAgainstFitnessFunction()
```

Using this algorithm, the ***average fitness of the population*** gets better and better. At each generation there is a higher likelihood of reaching a new ***global best solution*** which is represented as a member of the population.

A candidate solution populations average fitness over time:

![](http://www.scielo.br/img/revistas/lajss/v13n15//1679-7825-lajss-13-15-02922-gf4.jpg)
