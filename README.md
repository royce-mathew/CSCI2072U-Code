# 2072U-Course-Codes
# Table of Contents
1. **[Lecture 2](#lecture-2)**
   - [Bisection](#bisection)
2. **[Lecture 3/4](#lecture-3--4)**
   - [Newton Raphson Theorum](#newton-raphson-theorum)
   - [Secant Method](#secant-method)
   
This repository contains final versions of codes we've written during class, as well as other relevant codes.
## Lecture 2
### [Bisection](https://github.com/royce-mathew/CSCI2072U-Code/blob/main/bisection_code.py)
[Pseudocode - Page 33](https://learn.ontariotechu.ca/courses/21707/files/2796036?module_item_id=499488)
#### Remarks
- Upper bound for eror: $x^{(k)} −x^∗| ≤ \epsilon^{(k)} = |b^{(k)} −a^{(k)}|$
- Error on kth iteration: $\epsilon^{(k)} = \frac{\epsilon^{(0)}}{2^k}$
- Linear Convergence.
- Straight line on a semilog plot
- Works only for one unknown
<br/><br/>

#### Questions
1. **Under what conditions does the algorithm converge?**
- Bisection converges to some $x^∗$ such that $$f(x^*) = 0, x^\* \in [a, b]$$
if f is continuous and f (a)f (b) < 0. If there are two or more solutions, we don’t know to which one it will converge to.
2. **How accurate will the result be?**
- Gives us $x^*$ up to machine precision.
3. **How fast does it converge?**
- The error $|x^* − x(k)|$ decreases by a factor of $\frac{1}{2}$ in each iteration.

## Lecture 3 / 4
### Notes / topics covered
- In general, to solve and find isolated solutions we need the same number of equations as unknowns. 
- Intermediate Value Theorum
- Lecture 4 shows the comparisons between Bisection and Newton / Secant methods
### [Newton Raphson Theorum](https://github.com/royce-mathew/CSCI2072U-Code/blob/main/NetwonRaphson_plot.py)
[Pseudocode - Page 22](https://learn.ontariotechu.ca/courses/21707/files/2809429?module_item_id=501240)

#### Remarks
- The newton raphson methods needs the derivative to function correctly. When you don't have derivative or can't calculate, look at secant or bisection.
- Need one/two initial guesses.
- Error Estimate: $\epsilon^{(k+1)} \approx (\epsilon^{(k)})^2$
- Quadratic Convergence
- Steeper than linear on semilog plot.
<br/><br/>

#### Questions
1. **Under what conditions does the algorithm converge?**
- Newton-Raphson iteration converges if $x_0$ is sufficiently close to
$x^∗$. Usually, we do not know a priori how close is close enough and must resort to trial and error. 
2. **How accurate will the result be?**
- Gives us $x^*$ up to machine precision.
3. **How fast does it converge?**
- In Newton-Raphson iterations, the error is approximately squared in each iteration (provided it is small enough!). $$\epsilon_0, \epsilon_0^2, \epsilon_0^4, \epsilon_0^8, ...$$

### [Secant Method](https://github.com/royce-mathew/CSCI2072U-Code/blob/main/Assignment3_Solutions/Secant.py)
Secant methods needs two initial guesses: $x_{(0)}$ and $x_{(1)}$
#### Remarks
- This method uses a `finite difference` approximation to $f′$.
- Need one/two initial guesses.
- Asymptotically (meaning if $|x_k − x^*|$ is small enough) the secant method converges as fast as Newton-Raphson method does.
- The secant method has extensions to problems with more than 1 unknown, but in this case Newton method tends to be less cumbersome.
- The secant method is a `second order recurrence relation`. It relates the next approximation to the two previous approximations
- If we can find an $a$ and $b$ such that $x^∗ \in [a, b]$, then $x_0 = a$ and $x_1 = b$ is a good starting point.
- Error Estimate: $\epsilon^{(k+1)} \approx (\epsilon^{(k)})^2$
