# Genetic Algorithm Vector Mapping



### Traditional Neural Network Training Approach

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

To be pedantic, even though we are using search to reframe our thought process, we are still trying to learn an underlying mapping function between input and outputs. We are just ***traversing*** the search space in a different manner. To further reframe our thoughts, instead of calling a the measure of how well the neural network fits the underlying data a ***loss function*** we call it an ***heuristic***. Our heuristic will guide us through the search space by giving our algorithm an objective measure of how well or how poor a solution is.

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