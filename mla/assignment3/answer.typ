#import "@template/setting:0.1.0": *

#show: doc => assignment(
		title: [ Machine Learning \ Home Assignment 3 ],
		doc
)

= Preprocessing (33 points)

== Importance of Preprocessing (6 points)

We have the following data points:

=== a)

#align(center)[
#table(
    columns: 4,
    table.header[*Person*][*Age in years*][*Income in thousands of USD*][*Paied off*],
    [A], [47], [35], [yes],
    [B], [22], [40], [no],
    [C], [21], [36], [-],
)
]

$
d_A = sum_(i=1)^(2) (x_i - y_i)^2 = (21-47) ^ 2 + (36-35) ^ 2 = 677
$

$
d_B = (21-22) ^ 2 + (36-40) ^ 2 = 17
$

Therefore we get $d_A > d_B$ and we can conclude the BoL should not give credit
to C, according to the nearest neighbor algorithm.

=== b)

#align(center)[
#table(
    columns: 4,
    table.header[*Person*][*Age in years*][*Income in USD*][*Paied off*],
    [A], [47], [35000], [yes],
    [B], [22], [40000], [no],
    [C], [21], [36000], [-],
)
]

$
d_A = sum_(i=1)^(2) (x_i - y_i)^2 = (21-47) ^ 2 + (36000-35000) ^ 2 = 1000676
$

$
d_B = (21-22) ^ 2 + (36000-40000) ^ 2 = 16000001
$

Therefore we get $d_A < d_B$ and we can conclude the BoL should give credit to
C, according to the nearest neighbor algorithm.

== Input Centering (9 points)

=== a)

Considering the following equations:

$ z_n = x_n - macron(x), forall n = 1, ..., N $
$ macron(x) = frac(1, N) X^T 1 $
$ gamma = 1 - frac(1, N) 1 1^T $

We can show that:

$ Z = gamma X $

Indeed:

$ z_n = x_n - macron(x) forall n = 1, ..., N $

Turning this into a matrix form, we get:

$ Z &= X - 1 macron(x)^T \
&= X - 1 (frac(1, N) X^T 1)^T $

Remembering that $(A B)^T = B^T A^T$, we get:

$ Z &= X - frac(1, N) 1 1^T X \
&= I X - frac(1, N) 1 1^T X \
&= (I - frac(1, N) 1 1^T) X \
&= gamma X $

Therefore, we have shown that $Z = gamma X$.

=== b)

Considering that $Z$ is a $N times D$ matrix and $"rank"(Z) = "rank"(Z^T)$.\
Citing the rank-nullity theorem@rank_nullity_theorem, we have that given a matrix of dimension $d
times d$ $A$: 
$ "rank"(A) + "rank"(ker(A)) = d. $

Citing a property of the rank@rank, given two matrixes $A, B$: 
$ "rank"(A B) <= min("rank"(A), "rank"(B)). $

Therefore, we have that $"rank"(gamma) + "rank"(ker(gamma)) = N$. And since
$"rank"(ker(gamma)) = 1$@centering_matrix, we have $"rank"(gamma) = N - 1$.

Therefore, $"rank"(Z) = "rank"(gamma X) <= min("rank"(X), N - 1) < N$.

== Input Whitening (18 points)

=== a)

Given the following:

$ "Var"(hat(x)_1) = "Var"(hat(x)_2) = 1 $
$ EE[hat(x)_1] = EE[hat(x)_2] = 0 $
$ x_1 = hat(x)_1 $
$ x_2 = sqrt(1 - epsilon^2) hat(x)_1 + epsilon hat(x)_2 "for" epsilon in [-1, 1] $
$ "Cov"(hat(x)_1, hat(x)_2) = 0 $

The last equation is given by the fact that the two variables are independent.\
Therefore we already have the variance of $x_1$: $"Var"(x_1) = "Var"(hat(x)_1) = 1$.\
The variance of $x_2$ is given by:

$ "Var"(x_2) &= sqrt(1 - epsilon^2)^2 "Var"(hat(x)_1) + epsilon^2 "Var"(hat(x)_2) \
 &= 1 - epsilon^2 + epsilon^2 \
 &= 1 $

Finally, the covariance between $x_1$ and $x_2$ is given by:

$ "Cov"(x_1, x_2) &= sqrt(1 - epsilon^2) "Cov"(hat(x)_1, hat(x)_1) + epsilon "Cov"(hat(x)_1, hat(x)_2) \
&= sqrt(1 - epsilon^2) "Var"(hat(x)_1) \
&= sqrt(1 - epsilon^2) $

=== b)

Given the following:

$ x = (x_1, x_2)^T $
$ hat(x) = (hat(x)_1, hat(x)_2)^T $
$ f(hat(x)) = hat(w)_1 hat(x)_1 + hat(w)_2 hat(x)_2 $

Follows equivalent statements one below the other:

$ w_1 x_1 + w_2 x_2 = hat(w)_1 hat(x)_1 + hat(w)_2 hat(x)_2 $
$ w_1 hat(x)_1 + w_2 (sqrt(1 - epsilon^2) hat(x)_1 + epsilon hat(x)_2) = hat(w)_1 hat(x)_1 + hat(w)_2 hat(x)_2 $
$ w_1 hat(x)_1 + w_2 sqrt(1 - epsilon^2) hat(x)_1 + w_2 epsilon hat(x)_2 = hat(w)_1 hat(x)_1 + hat(w)_2 hat(x)_2 $
$ cases(
    (w_1 + w_2 sqrt(1 - epsilon^2)) hat(x)_1 = hat(w)_1 hat(x)_1 \
    w_2 epsilon hat(x)_2 = hat(w)_2 hat(x)_2
) $
$ cases(
    w_1 + w_2 sqrt(1 - epsilon^2) = hat(w)_1 \
    w_2 epsilon = hat(w)_2
) $

And so we we arrive to the final conclusion that $f$ is linear in the correlated
inputs:
$ cases(
    w_1 = hat(w)_1 - hat(w)_2 / epsilon sqrt(1 - epsilon^2) \
    w_2 = hat(w)_2 / epsilon
) $

=== c)

Given target function:

$ f(hat(x)) = hat(x)_1 + hat(x)_2 $

The constraint $C$:

$ w_1^2 + w_2^2 <= C $

If we perform regression with the correlated inputs $x$, then let's find the
minimum value of $C$ such that the constraint is satisfied.
First of all let's compute the values of $w_1$ and $w_2$ considering the
previous results:

$ hat(w)_1 &= 1 \
 hat(w)_2 &= 1 $

Therefore:

$ w_1 &= 1 - 1 / epsilon sqrt(1 - epsilon^2) \
 w_2 &= 1 / epsilon $

Now we can compute the value of $C$:

$ C &= w_1^2 + w_2^2 \
 &= (1 - 1 / epsilon sqrt(1 - epsilon^2))^2 + (1 / epsilon)^2 \
 &= 1 + frac(1 - epsilon^2, epsilon^2) - frac(2, epsilon) sqrt(1 - epsilon^2) + 1 / epsilon^2 \
 &= 2 / epsilon^2 - (2 sqrt(1 - epsilon^2)) / epsilon $

=== d)

Let's compute the following limit:

$ lim_(epsilon -> 0) C = 
lim_(epsilon -> 0) (2 / epsilon^2 - (2 sqrt(1 - epsilon^2)) / epsilon) = 
oo $

= Competition Design to Find Defective Products (24 points)

==

Follows the theorem of generalization bound for selection from finite
$cal(H)$:

$ PP(L(hat(h)^*_S) <= hat(L)(hat(h)^*_S, S) + sqrt(ln(M / delta) / (2n))) >= 1 - delta $ <hp-bound>

Let's repeat our hypothesis:

$ M = 20 $
$ delta = 2 $

We are looking for the minimum value of $n$ such that the follwing inequality is satisfied:

$ sqrt(ln(M / delta) / (2n)) <= 0.04 $

Therefore:

$ ln(20 / 2) / (2 dot 0.04^2) = 312.5 < 313 = n $

==

Given the following:

$ n = 1800 $
$ delta = 2 $

We are looking for the maximum value of $M$ such that the following inequality
is satisfied:

$ sqrt(ln(M / delta) / (2n)) <= 0.04 $

Therefore:

$ 2 exp(0.04^2 dot 2 dot 1800) tilde 634.7 > 634 = M $

= Combining Multiple Confidence Intervals (22 points)

Given the following:

$ i &in I = {1, 2, 3} \
S_i &= S \
"CI"_i &= [l_i, u_i] & "w.p." 1 - delta_i \
0.99 &= product_I (1 - delta_i) \
delta &= delta_i = delta_j & forall i, j in I
$

Let's compute the value of $delta$:

$ 1 - root(3, 0.99) tilde 0.0033 < 0.004 = delta $

We could compute a more precise value for $delta$ by I only need to show how do
answer this question. \
Alex can choose any combination of the confidence intervals endpoints such that\
$l_("chosen") <= u_("chosen")$, because any such combination is a valid 
(at least 99)-CI. Therefore, he should choose the combination that minimizes the
length of the CI:

$ "CI" &= [max(l_i), min(u_i)] $

= Early Stopping (21 points)

== Neural network with early stopping (21 points)

#quote(attribution: [Wikipedia@bias])[
Statistical bias, in the mathematical field of statistics, is a systematic
tendency in which the methods used to gather data and generate statistics
present an inaccurate, skewed or biased depiction of reality.
]

=== Predefined Stopping

The $S_("val")$ has no influence on the choice of the target
function $h_(t^*)$, so the bias is not present.

=== Non-adaptive Stopping

It is chosen the target function $h_(t^*)$ that minimizes the
validation error $hat(L)(h_(t^*))$. Therefore, the dataset is used to choose the
best target function, which lead the final model to be biased by the validation
set $S_("val")$.

=== Adaptive Stopping

$h_(t^*)$ is chosen in the sequence of hypotesis $h_1, h_2, h_3, ..., h_t$. 
While the target function is not chosen based on the validation set, 
the sequence stops when the validation does not improve anymore for a certain 
number of steps. 
Therefore, the sequence of models is biased by the validation set
$S_("val")$. \
As a counterexample to the claim that the bias is not present,
let's consider the case in which a different validation set $S'_("val")$ is
used. 
Then, it would produce the sequence of hypotesis models 
$h_1, h_2, h_3, ..., h_j$ and $j  eq.not t$. 
Let $j > t$ and $h_j$ be the best model, thus the final model choice differs
from the one obtained with the original validation set $S_("val")$. Therefore,
we can conclude that the final model is biased by the validation set
$S_("val")$.

==

I have already cited the theorem of generalization bound (see @hp-bound), so 
follows the solution for the two cases.

=== Predefined Stopping

We have $M = 1$, because we are only considering the final model:

$ PP(L(hat(h)^*_S) <= hat(L)(hat(h)^*_S, S) + sqrt(ln(1 / delta) / (2n))) >= 1 - delta $

=== Non-adaptive Stopping

We have $M = T$, where $T$ is the number of epochs and so the number of models
to consider:

$ PP(L(hat(h)^*_S) <= hat(L)(hat(h)^*_S, S) + sqrt(ln(T / delta) / (2n))) >= 1 - delta $

#bibliography("biblio.bib")
