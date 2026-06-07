# Markov Chain State Evolution (Module Alpha)

For this assignment, we have chosen **Module Alpha**, implementing a Markov Chain using a transition probability matrix over a 365-day timeline.

The simulation models three macroeconomic states: Bull Market (State 0), Stagnation (State 1), and Catastrophic Recession (State 2). Under normal conditions, the baseline transition matrix strongly favors remaining in a Bull Market (85% retention) and provides modest chances of transitioning into other states.

The baseline transition matrix is:

[0.85, 0.12, 0.03]
[0.10, 0.75, 0.15]
[0.05, 0.20, 0.75]

However, an unexpected Black Swan shock is introduced from day 180 through day 189. During this 10-day crisis window, the probability mass for Bull Market and Stagnation transitions shifts dramatically, sending 80% of their mass directly into Catastrophic Recession, heavily skewing the macro environment. After the shock ends, baseline behavior is restored.

If the Exercise 2 cash-flow model were tied to this volatile Markov environment, it would likely collapse during the recession spike. As the probability of a Catastrophic Recession surges during the Black Swan event, market demand would plummet while regulatory penalty rates would likely maximize, devastating revenue generation. Since the production asset cost in the cash-flow model already has high variance and heavy tail risks, this sudden shock would drastically amplify downside losses, dragging the VaR95 deeply into the negative and bankrupting the simulated business.
