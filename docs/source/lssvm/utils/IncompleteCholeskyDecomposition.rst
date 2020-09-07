.. _seclabel-ICHOLDoc:

Incomplete Cholesky Decomposition of The Kernel Matrix
=======================================================

**Cholesky decomposition** of the **kernel matrix** is 
equivalent to the **Gram-Schmidt orthogonalisation** (QR) of the **feature space** 
matrix. The incomplete Cholesky decomposition can be used to obtain a low-rank 
approximation of the kernel matrix. 


The incomplete Cholesky decomposition is implemented in the 
:cpp:class:`template<class TKernel, typename T>IncCholesky` class.



Gram-Schmidt orthonormalisation 
-------------------------------

Given a set of **linearly independent** vectors :math:`\{\mathbf{u}_i\}_{i=1}^{N},
\mathbf{u}_i \in \mathbb{R}^d` 
of a **finate dimensional** inner product space :math:`V`, the GS process generates 
an orthonormal basis :math:`\{\mathbf{q}_i\}_{i=1}^{N},\mathbf{q}_i \in \mathbb{R}^d` 
by orthogonalising each vector one by one to all the earlier vectors. 

The first basis vector is just one of the vectors normalised to unit length as 
:math:`\mathbf{q}_1 := \mathbf{u}_1/\|\mathbf{u}_1\|`. The :math:`i`-th basis 
vector is obtained by selecting :math:`\mathbf{u}_i` i.e. one of the remaining 
:math:`N-i` vectors, taking its (vector) projection to the orthogonal complement 
of the sub-space spanned by the :math:`\mathbf{q}_1,\dots,\mathbf{q}_{i-1}` 
vectors, i.e. subtracting its (vector) projections to all the previously
generated orthonormal vectors :math:`\mathbf{q}_1,\dots,\mathbf{q}_{i-1}` 

.. math::
    :label: GS_qi_0
    
     P_{V_{i-1}}^{\perp}(\mathbf{u}_i) & = 
     \mathbf{u}_i - \sum_{j=1}^{i-1} P_{\mathbf{q}_j}(\mathbf{u}_i) = 
     \mathbf{u}_i - \sum_{j=1}^{i-1} \frac{\mathbf{q}_j^T \mathbf{u}_i}{\|\mathbf{q}_j\|^2}\mathbf{q}_j =
     \mathbf{u}_i - \sum_{j=1}^{i-1} [\mathbf{q}_j^T \mathbf{u}_i] \mathbf{q}_j = 
     \mathbf{u}_i - \sum_{j=1}^{i-1} [\mathbf{q}_j\mathbf{q}_j^T] \mathbf{u}_i  \\ & =
     \left[ \mathbf{I} -  Q_{i-1}Q_{i-1}^T \right] \mathbf{u}_i

where :math:`V_{i-1}` is the sub-space spanned by the :math:`\mathbf{q}_1,\dots,
\mathbf{q}_{i-1}` vectors, :math:`Q_{i-1} \in \mathbb{R}^{d\times i-1}` matrix 
with :math:`\mathbf{q}_1,\dots,\mathbf{q}_{i-1}` orthogonal vectors as its columns
and :math:`I \in \mathbb{R}^{d \times d}` identity matrix.
For the final form of :math:`\mathbf{q}_i` one needs to normalise this 
projection as 

   
.. math::
   :label: GS_qi_1
   
     \mathbf{q}_i = \frac{ \left[ \mathbf{I} -  Q_{i-1}Q_{i-1}^T \right] \mathbf{u}_i }{\nu_i}
  
where :math:`\nu_i = \left\| \left[ \mathbf{I} -  Q_{i-1}Q_{i-1}^T \right] \mathbf{u}_i \right\|`
:cite:`shawe2004kernel`.


According to Eq. :eq:`GS_qi_1`, the input data vector :math:`\mathbf{u}_i` can 
be written as 

.. math::
    :label: GS_R_1
    
     \mathbf{u}_i & = Q_{i-1}Q_{i-1}^T\mathbf{u}_i + \nu_i \mathbf{q}_i \\ & =
       \begin{bmatrix}
           \mathbf{q}_1,\dots, \mathbf{q}_{i-1}, \mathbf{q}_i 
       \end{bmatrix}_{(d\times i)}
       \begin{bmatrix}
           \mathbf{q}_1^T \mathbf{u}_i \\ 
           \vdots \\ 
           \mathbf{q}_{i-1}^T \mathbf{u}_i \\ 
           \nu_i
       \end{bmatrix}_{(i\times 1)} 
       \\ & =
       \underbrace{
         \begin{bmatrix}
             \mathbf{q}_1,\dots, \mathbf{q}_{i-1}, \mathbf{q}_i, \dots, \mathbf{q}_N  
         \end{bmatrix}_{(d\times N)}
       }_{\mathbf{Q}}  
       \underbrace{
         \begin{bmatrix}
             \mathbf{q}_1^T \mathbf{u}_i \\ 
             \vdots \\ 
             \mathbf{q}_{i-1}^T \mathbf{u}_i \\ 
             \nu_i \\
             \mathbf{0}_{N-i}
         \end{bmatrix}_{(N\times 1)}
         }_{\mathbf{R}_{\cdot,i}}  
       
therefore the input data matrix :math:`X \in \mathbb{R}^{(N\times d)}` with the 
:math:`\{\mathbf{u}_i\}_{i=1}^{N},\mathbf{x}_i \in \mathbb{R}^d` vectors as rows
can be written as 

.. math::
    
    X^T = \begin{bmatrix}
            \mathbf{u}_1, \dots, \mathbf{u}_N 
          \end{bmatrix}_{(d\times N)} 
        = Q_{(d\times N)} R_{N\times N}
        
where the :math:`Q \in \mathbb{R}^{(d\times N)}` matrix contains the 
:math:`\{\mathbf{q}_i\}_{i=1}^{N},\mathbf{q}_i \in \mathbb{R}^d` orthonormal 
(i.e. :math:`\mathbf{q}_i^T\mathbf{q}_j = \delta_{ij}`) basis vectors as columns 
and the :math:`i`-th column of the :math:`R \in \mathbb{R}^{(N\times N)}` upper 
triangular matrix :math:`R_{\cdot,i}` contains the projections of the :math:`i`-th input data vector 
:math:`\mathbf{u}_i` onto these basis vectors (that is zero for :math:`j>i`).
One can interpret the :math:`R_{\cdot,i}, i=\{1,\dots,N\}` columns of the matrix 
:math:`R` as the representation of the 
:math:`\{\mathbf{u}_i\}_{i=1}^{N},\mathbf{x}_i \in \mathbb{R}^d` input data 
vectors in the :math:`\{\mathbf{q}_i\}_{i=1}^{N},\mathbf{q}_i \in \mathbb{R}^d` 
basis.

So the input data vectors processed one after the other generating the corresponding 
subsequent orthonormal basis vectors.
When the input data vector :math:`\mathbf{u}_i` processed is **not linearly independent** 
form the previously processed ones, the corresponding residual norm
:math:`\nu_i` (the length of the projection of the :math:`\mathbf{u}_i` vector to the 
orthogonal complement of the space spanned by the :math:`\mathbf{q}_1,\dots,\mathbf{q}_{i-1}`
vectors) becomes zero (since :math:`\mathbf{u}_i` can be expressed as 
:math:`\mathbf{u}_i=\sum_{j=1}^{i-1}[\mathbf{q}_j\mathbf{q}_j^T]\mathbf{u}_i`).

In general, the residual norm :math:`\nu_i` indicates how independent the corresponding 
input data :math:`\mathbf{u}_i` from the previously processed ones. 
Changing the order, in which the input data is processed, by selecting the one 
with the largest residual norm to be processed at the next step (**pivoting**) and 
and eventually ignoring those with small residual norms (**partial, incomplete**) 
leads to incomplete and pivoted versions of the algorithm.

Cholesky decomposition of the kernel matrix 
-------------------------------------------

Let the :math:`\varphi(\cdot):\mathbb{R}^d\to\mathbb{R}^{n_h}` is the mapping to 
the high(even infinite)-dimensional feature space and the :math:`\Phi \in \mathbb{R}^{(N\times n_h)}`
matrix is the feature space matrix with rows of the feature maps of the input data 
as

.. math::

    \Phi = \begin{bmatrix}
             \varphi(\mathbf{u}_1)^T \\
             \vdots \\
             \varphi(\mathbf{u}_N)^T \\
           \end{bmatrix}_{(N\times n_h)} 

If :math:`\Phi^T_{(n_h\times N)}=Q_{(n_h\times N)}R_{(N\times N)}` is the 
QR decomposition (Gram–Schmidt orthogonalization of the columns of 
:math:`\Phi^T` i.e. the feature maps of the input data) of this feature space 
matrix, then the kernel matrix 

.. math::

    \Omega = \Phi \Phi^T = [QR]^T QR = R^T \underbrace{Q^TQ}_{I_{(N\times N)}} R = R^TR

where the fact that the matrix :math:`Q` builds up from mutually orthonormal 
columns i.e. :math:`\mathbf{q}_i^T\mathbf{q}_j = \delta_{ij} \to QQ^T=I` was used.
Therefore, **performing QR decomposition of the feature space** 
:math:`\mathbf{ \{\varphi(\mathbf{u}_i)\}_{i=1}^N }` 
**is equivalent to the Cholesky decomposition of the corresponding kernel matrix** 
:math:`\mathbf{ \Omega_{ij}=\varphi(\mathbf{u}_i)^T\varphi(\mathbf{u}_j),  i,j={1,\dots,N} }`.

Computing the :math:`j,i`-th element of the :math:`R \in \mathbb{R}^{(N\times N)}` 
upper triangular matrix:
 
 - according to Eq. :eq:`GS_R_1`, computing the :math:`R_{\cdot,i}, i`-th column 
   of the :math:`R \in \mathbb{R}^{(N\times N)}` upper triangular matrix involves 
   the computation of the (scalar) projections of the feature map of the 
   :math:`i`-th input data :math:`\varphi(\mathbf{u}_i)` onto all the previously 
   generated :math:`\mathbf{q}_1,\dots,\mathbf{q}_{i-1}` basis vectors: 
   
   .. math::
       
        R_{\cdot,i} = 
                \begin{bmatrix}
                    \mathbf{q}_1^T \varphi{(\mathbf{u}_i)} \\ 
                    \vdots \\ 
                    \mathbf{q}_{i-1}^T \varphi{(\mathbf{u}_i)} \\ 
                    \nu_i \\
                    \mathbf{0}_{N-i}
                \end{bmatrix}_{(N\times 1)}

   
 - furthermore, these basis vectors :math:`\mathbf{q}_j, j=1,\dots,i-1` can be 
   expressed as Eqs. :eq:`GS_qi_1` and :eq:`GS_qi_0` 
   :math:`\nu_j\mathbf{q}_j = \varphi{(\mathbf{u}_j)} - \sum_{t=1}^{j-1} [\mathbf{q}_t\mathbf{q}_t^T] \varphi{(\mathbf{u}_j)}`

The :math:`j,i`-th element of the :math:`R, j<i`, i.e. the :math:`j<i`-th 
elements the column vector written in the first point above and be expressed by 
using the second point as 

.. math::
    :label: Ichol_eq_1
    
    R_{ji} & = \mathbf{q}_j^T \varphi{(\mathbf{u}_i)}  \\
           & = \frac{1}{\nu_j}\left[ 
                 \varphi{(\mathbf{u}_j)} - \sum_{t=1}^{j-1} [\mathbf{q}_t \mathbf{q}_t^T] \varphi{(\mathbf{u}_j)}
               \right]^T \varphi{(\mathbf{u}_i)} \\ 
           & = \frac{1}{\nu_j}\left[ 
                 \varphi{(\mathbf{u}_j)}^T - \sum_{t=1}^{j-1} \varphi{(\mathbf{u}_j)}^T \mathbf{q}_t \mathbf{q}_t^T
               \right] \varphi{(\mathbf{u}_i)} \\ 
           & = \frac{1}{\nu_j}\left[ 
                 \varphi{(\mathbf{u}_j)}^T \varphi{(\mathbf{u}_i)} - 
                 \sum_{t=1}^{j-1} \varphi{(\mathbf{u}_j)}^T \mathbf{q}_t \mathbf{q}_t^T \varphi{(\mathbf{u}_i)}
               \right]  \\ 
           & = \frac{1}{\nu_j}\left[ 
                 \underbrace{
                     \varphi{(\mathbf{u}_j)}^T \varphi{(\mathbf{u}_i)}
                 }_{\Omega_{ji}}    
                 -
                 \sum_{t=1}^{j-1} 
                   \underbrace{ [\mathbf{q}_t^T \varphi{(\mathbf{u}_j)}] }_{R_{tj}} 
                   \underbrace{ [\mathbf{q}_t^T \varphi{(\mathbf{u}_i)}] }_{R_{ti}}
               \right]  \\ 
           & = \frac{1}{\nu_j}\left[ \Omega_{ji} - \sum_{t=1}^{j-1} R_{tj}R_{ti} \right]


where :math:`j<i`. Since :math:`\nu_j` is given at Eq. :eq:`GS_qi_1`

.. math::
    :label: Ichol_eq_2

    \nu_j^2 & = \left\| \varphi(\mathbf{u}_j) - \sum_{t=1}^{j-1} \mathbf{q}_t \mathbf{q}_t^T \varphi(\mathbf{u}_j) \right\|^2  \\
            & = \left[ \varphi(\mathbf{u}_j) - \sum_{t=1}^{j-1} \mathbf{q}_t \mathbf{q}_t^T \varphi(\mathbf{u}_j) \right]^T 
                \left[ \varphi(\mathbf{u}_j) - \sum_{t=1}^{j-1} \mathbf{q}_t \mathbf{q}_t^T \varphi(\mathbf{u}_j) \right] \\
            & = \left[ \varphi(\mathbf{u}_j)^T - \sum_{t=1}^{j-1} \varphi(\mathbf{u}_j)^T \mathbf{q}_t \mathbf{q}_t^T \right]    
                \left[ \varphi(\mathbf{u}_j) - \sum_{t=1}^{j-1} \mathbf{q}_t \mathbf{q}_t^T \varphi(\mathbf{u}_j) \right] \\
            & = \varphi(\mathbf{u}_j)^T \varphi(\mathbf{u}_j) 
              - \sum_{t=1}^{j-1} \varphi(\mathbf{u}_j)^T \mathbf{q}_t \mathbf{q}_t^T \varphi(\mathbf{u}_j) \\
            & = \varphi(\mathbf{u}_j)^T \varphi(\mathbf{u}_j) 
              - \sum_{t=1}^{j-1} 
                 \underbrace { [\mathbf{q}_t^T \varphi(\mathbf{u}_j) ] }_{R_{tj}} \underbrace { [\mathbf{q}_t^T \varphi(\mathbf{u}_j) ] }_{R_{tj}} \\
            & = \Omega_{jj} - \sum_{t=1}^{j-1} R_{tj}^2
            
where the mutually orthonormal property :math:`\mathbf{q}_k^T\mathbf{q}_l = \delta_{kl}` 
of the basis vectors :math:`\mathbf{q}_k,\dots,\mathbf{q}_{j-1}` was used to obtain 
the 4-th equation.

(Also note, that according to :math:`\Omega \approx R^TR`, actually :math:`\sum_{t=1}^{j-1} R_{tj}^2` 
is the approximation of the :math:`\Omega_{jj}, j=1,\dots,N` diagonal elements 
at the :math:`j-1`-th step of the algorithm.)


The algorithm
~~~~~~~~~~~~~

Given the :math:`\{\mathbf{x}_i\}_{i=1}^{N},\mathbf{x}_i \in \mathbb{R}^d` input 
data set with the kernel function :math:`K(\mathbf{x_i},\mathbf{x_j})=\varphi(\mathbf{x})_i^T\varphi(\mathbf{x}_j)`
with :math:`\varphi(\cdot):\mathbb{R}^d\to\mathbb{R}^{n_h}` mapping to the feature space.
One can formulate the algorithm for the incomplete Cholesky decomposition of 
the kernel matrix :math:`\Omega \approx \tilde{\Omega} = G^TG` with 
:math:`\Omega \in \mathbb{R}^{(N\times N)}, \Omega_{ij} = K(\mathbf{x_i},\mathbf{x_j})`
symmetric, positive semi-definite matrix and :math:`G \in \mathbb{R}^{(R\times N)}
, R \leq N` upper triangular matrix by combining Eqs. :eq:`Ichol_eq_1` and 
:eq:`Ichol_eq_2`. The following algorithm generates the rows of :math:`G` one by 
one as: 

  0. Initialise the squared diagonal elements of the :math:`\tilde{\Omega} = G^TG` 
     as :math:`\nu_k^2 = \Omega_{kk} = K(\mathbf{x}_k,\mathbf{x}_k), k=1,\dots,N`:
     
      * these elements will be updated after the computation of the :math:`j`-th 
        row of the :math:`G`, :math:`G_{j\cdot}` as 
        
        .. math::
          
          \nu_k^2 = \nu_k^2 - G_{jk}^2, k=j+1,\dots,N   
        
        according to Eq. :eq:`Ichol_eq_2`.
        
      * the :math:`\nu_k^2` value is the squared residual norm of the feature 
        map of the :math:`k`-th input data i.e. :math:`\varphi(\mathbf{x}_k)`. 
        These will be used to:
        
        - the maximum of these will be used to select the next, :math:`j+1`-th 
          input data (more exactly its feature map) to orthogonalise (to all the 
          previously generated basis vectors). **This will greedily select input 
          data to minimise the residual norm** i.e. the projection of the 
          feature maps of the input data onto the orthogonal complement of the 
          currently generated sub-space spanned by the underlying basis vectors.  
        - these will be used to compute the approximation error after the 
          :math:`j`-th step as 
          
          .. math::
           
            \|\Omega-\tilde{\Omega}\|_1 & = \|\Omega-G^TG\|_1 
               = \text{tr}(\Omega) - \text{tr}(G^TG) 
               = \text{tr}(\Omega) - \text{tr}(G^TG) \\
             & = \text{tr}(\Omega) - \sum_{k=1}^{N} \sum_{t=1}^{j} G_{tk}^2
               = \sum_{k=1}^{N}\Omega_{kk} - \sum_{k=1}^{N} \sum_{t=1}^{j} G_{tk}^2 \\
             & = \sum_{k=1}^{N} \left[ \Omega_{kk} - \sum_{t=1}^{j} G_{tk}^2 \right]
               = \sum_{k=1}^{N} \nu_k^2 
        
          which will be normalised as 
          :math:`\eta := \|\Omega-\tilde{\Omega}\|_1/N =  \left[ \sum_{k=1}^{N} \nu_k^2 \right]/N = \left[ \sum_{k=j+1}^{N} \nu_k^2 \right]/N \in [0,1]`
          where the last equality took into account that 
          :math:`\Omega_{kk}=\tilde{\Omega}_{kk}, \text{ for } k \leq j`. 
           
      * these residual norms will give the diagonal elements of the :math:`G` matrix 
        (see below) 
       
  1. The first row, i.e. :math:`j=1` of :math:`G`:
      
      * set :math:`G_{jj} = \sqrt{\nu_j^2}`
      * then for :math:`i=j+1,\dots,N`:
        
        a. set :math:`G_{ji} = \Omega_{ji} = K(\mathbf{x}_j,\mathbf{x}_i)`
        b. update the residual norm as :math:`\nu_i^2 = \nu_i^2 - G_{ji}^2`
        
      * while performing the above two sub-steps for all :math:`i=j+1,\dots,N`:
      
        a. find the next pivot i.e. the index of the next input data feature map to orthogonalise 
           i.e. find :math:`i` with the maximum residual norm :math:`\texttt{pivot}=\arg\max_i(\nu_i^2)`
        b. compute the sum of the squared residual norms for the approximation 
           error computation which is :math:`\eta = \left[ \sum_i \nu_i^2 \right]/N`
           at the end of this :math:`j`-th step    
      
      * increase the index j of the generated rows of the matrix :math:`G` to j+1
      
 
  2. The :math:`1<j\leq N`-th row of :math:`G`:
      
      * swap the:
        
        - :math:`j`-th column of the current :math:`G` matrix with the 
          :math:`\texttt{pivot}`-th column, i.e. the input data index selected 
          during the previous step
        - :math:`j`-th element of the squared residual norms with the 
          :math:`\texttt{pivot}`-th element i.e. :math:`\nu_{\texttt{pivot}}^2 \text{ and } \nu_j^2`
          
      * set :math:`G_{jj} = \sqrt{\nu_j^2}`
      * then for :math:`i=j+1,\dots,N`:
        
        a. set :math:`G_{ji} = \Omega_{ji} - \sum_{t=1}^{j-1} G_{tj}G_{ti}`
        b. update the residual norm as :math:`\nu_i^2 = \nu_i^2 - G_{ji}^2`
      
      * all steps are the same as in case of 1.  
    
  3. Repeat step 2. as long as :math:`\eta > \epsilon` and :math:`j < \texttt{itr}_{\text{max}}`
     where :math:`\epsilon` is the required approximation error. 
     The :math:`\texttt{itr}_{\text{max}}` is the maximum iteration number which is 
     the maximum number of rows of :math:`G` to be generated. This is equal to the 
     rank of the approximated kernel matrix :math:`\tilde{\Omega} = G^TG` 
     which is equal to the rank of the projected input data feature map.
     
Notes on the approximation error
--------------------------------

Based on some things form :cite:`bach2005predictive`.

As it has already been discussed, the incomplete Cholesky decomposition selects 
the data vector to orthogonalise in the next step (pivot) based on the norm of 
their projections to the orthogonal complement of the 
actual sub-space spanned by the actual set of bias vectors. This will remove the 
highest residual norm at each step by including the corresponding data vector 
into the orthogonalisation. 

In turn the individual residual norms (square) after completing the :math:`k`-th step
(i.e. the norm of the projection of the individual data vectors to the orthogonal 
complement of the :math:`k` basis vectors), is equal to the difference  
between the corresponding real and approximated diagonal elements of the kernel 
matrix :math:`\nu_j^2=\Omega_{jj} - \tilde{\Omega}_{jj} = \varphi(\mathbf{x}_j)^T\varphi(\mathbf{x}_j) -  \sum_{t=1}^{k} G_{tj}^{(k)2})`
Therefore, by selecting the data vector for the :math:`k+1`-th step to orthogonalise 
with the highest residual norm i.e. removing the highest residual norm in the next 
step can be interpreted as the minimisation of :math:`\texttt{trace}\{\Omega-\tilde{\Omega}\}`
by removing the highest individual contribution at each step.

Let the :math:`\|A\|_1` denote the sum of the singular values of the matrix :math:`A`. 
If :math:`A` is a square, symmetric matrix then its singular values are its eigenvalues.
Furthermore, the sum of the eigenvalues of this matrix is equal to its trace.
The kernel matrix :math:`\Omega` is a Gram matrix i.e. square, symmetric, positive 
definite matrix i.e. :math:`\Omega \succeq 0`. Moreover, :math:`\Omega - \tilde{\Omega}^{(k)} \succeq 0`
since :math:`\tilde{\Omega}^{(k)} = G^{(k)T}G \preceq \Omega` for each :math:`k`
steps (:math:`\succeq \text{ means that } \mathbf{x}^TA\mathbf{x} \geq 0, \forall \mathbf{x}`). 

.. note::

    Therefore, :math:`\|\Omega-\tilde{\Omega}^{(k)}\|_1= \sum_{i} \lambda_i = \texttt{trace}\{ \Omega-\tilde{\Omega}^{(k)} \}`
    So minimising the trace of :math:`\|\Omega-\tilde{\Omega} \|` is maximising 
    :math:`\texttt{trace}\{ \tilde{\Omega} \} = \sum_j \lambda_j`. Good feeling that 
    :math:`\texttt{trace}\{ \tilde{\Omega} \} \leq \texttt{trace}\{ \Omega \}`
    this can be seen from either that the diagonals of the approximation approaching 
    the real one or from the fact that all the kernel, approximated kernel and their difference 
    are positive semi-definite i.e. non negative eigenvalues and tehir sums...)  
    since :math:`\tilde{\Omega}^{(k)} = G^{(k)T}G \preceq \Omega` the eigenvalues of 
    :math:`\mathbf{x}^T \left[ \Omega - \tilde{\Omega} \right] \mathbf{x} =  \mathbf{x}^T \Omega \mathbf{x} - \mathbf{x}^T \tilde{\Omega}\mathbf{x} = \lambda \mathbf{x}`)



So back to the original, the goal of minimising the :math:`\| \Omega-\tilde{\Omega}^{(k)} \|_1` 
which is the sum of the eigenvalues. After the :math:`k`-th step, 
:math:`\texttt{trace} \{ \tilde{\Omega}^{(k)} \} = \sum_{i=1}^{N} \sum_{t=1}^{k} G_{ti}^{(k)^2}`
The gain in the :math:`k+1`-th step to this is :math:`\sum_{i=1}^{N} G_{k+1,i}^{(k+1)^2}` i.e. 
the squared sum of the last added, :math:`k+1`-th row of the :math:`G` matrix.
According to Eqs. :eq:`Ichol_eq_1` and :eq:`Ichol_eq_2` these elements can be computed 
as (in the :math:`k+1`-th step by selecting the :math:`p`-th data vector out of 
the :math:`N-k` remaining) 

.. math:: 
  
    \sum_{i=1}^{N} G_{k+1,i} & \underbrace {=}_{G_{ji}=0, \forall i<j} \sum_{i=k+1}^{N} G_{k+1,i}^2 \\
      & = G_{k+1,i=k+1}^2 + \sum_{i=k+2}^{N} G_{k+1,i}^2 = 
      \underbrace{ \nu_p^2 }_{\Omega_{pp} -  \sum_{t=1}^{k} G_{t,p}^2 } + 
      \frac{1}{\nu_p^2} \sum_{i=k+2} \left [ \Omega_{k+1,i} - \sum_{t=1}^{k} G_{t,p} G{t,i}   \right]^2
     
So we should chose :math:`p` after the :math:`k`-th step out of the :math:`N-k` remaining
such that  the above sum is maximal. We should compute the sum for all possible values 
of :math:`p` that would not be feasible since it would include the computation of all 
possible :math:`k+1`-th row of the matrix :math:`G` (at each step). 
Instead, a lower bound of this sum i.e. :math:`\nu_p^2` is used. Since these 
diagonals are known after the completing the :math:`k`-th step, one can chose 
the pivot :math:`p` such that :math:`p = \arg \max_p{\nu_p^2}` (and hope that 
it also will lead to a maximum of the above sum, but no guarantee). One could 
also chose more than one :math:`p` say :math:`\kappa` that are the :math:`\kappa`  
maximal :math:`\nu_{p_{1,\dots,\kappa}}^2`. Then compute the sum for all and chose 
the one that maximal. If :math:`\kappa` is low then it might help.

   


     
     
Summary and remarks 
-------------------

As it was shown, performing Cholesky decomposition of the kernel matrix is 
equivalent to performing QR decomposition of the feature map matrix. During 
this process, the feature map of the input data set will be expressed in a new 
orthonormal basis. The set of these basis vectors is greedily generated by 
choosing the feature map at each step that has the maximum residual norm:
the residual norm is the projection to the orthogonal complement of the sub-space 
spanned by the current set of basis vectors. The higher this residual norm the 
more independent from the corresponding feature map form those processed previously:
 
 - it becomes zero: if the corresponding input data feature map can be expressed 
   as linear combination of the previous i.e. this vectors lies into the sub-space 
   spanned by the current set of basis vectors (sum of the projections to the current 
   set of basis vectors already equals to the vector)
 - it's equal to the norm of the vector itself: this vector is linearly independent
   from all the previous i.e. this vector lies entirely into the orthogonal 
   complement of the sub-space spanned by the current set of basis vectors (sum 
   of the projections to the current set of basis vectors is zero i.e. rthogonal 
   to all of these basis vectors)

The rank of the feature map of the input data is equal to the rank of the kernel 
matrix. Although the kernel matrix might have even full rank, its spectrum decays
rapidly in case of many kernels and can be well approximated by low rank 
(:math:`\ll \texttt{full rank}`) matrices :cite:`smola2000sparse` :cite:`williams2000effect`
:cite:`bach2002kernel`. This happens when the data lives "near" a low-dimensional 
subspace in feature space which means that even with a low-number (low-dimensional) of
basis vectors a small sum residual norm ("near") can be achieved (the sum of the 
vector projections to these basis vectors already approximate very well the feature 
map vectors). Incomplete Cholesky decomposition can be used in such cases to 
obtain a low-rank approximation of the kernel matrix that can help e.g. to reduce 
the size of the eigenvalue problem in which the full kernel matrix is involved or 
deal with the full kernel matrix that would not fit into memory through an 
appropriate approximation. 


Note, that the :math:`G_{\cdot i} \in \mathbb{R}^{R}, i=1,\dots,N` columns of the 
:math:`G \in \mathbb{R}^{(R\times N)}` matrix are the projections of the feature map 
of the input data set :math:`\left\{ \varphi(\mathbf{x}_i) \right\}_{i=1}^N` 
onto the low (:math:`R`) dimensional sub-space spanned by the :math:`R` orthonormal 
basis vectors. Moreover, this sub-space is obtained such that the residual norm of 
the feature maps (i.e. the norm of the projection to the orthogonal complement of 
the sub-space) is minimised (at least its maximum value is reduced to zero at 
each step). Therefore, one can view the columns of :math:`G` as a low dimensional 
representation of the input data feature map.

.. note::

    One could see similarities here with:
      
      - KPCA: that will select directions to project the feature map of the input 
        data such that the variance of the projection is maximised. The optimal 
        solution to this maximisation problem is given by the leading eigenvectors 
        of the kernel matrix i.e. using the eigenvectors as directions that corresponds 
        to the highest eigenvalues.
      
      - as it was shown in :cite:`girolami2002orthogonal`, by selecting the 
        directions to project as the largest sum of the eigenvalue and eigenvector 
        pairs one can obtain a new set of data that will maximise the quadratic 
        Renyi entropy (see also in :cite:`jenssen2007kernel`)
           
      - one could keep in mind here that incomplete Cholesky is a kind of trace 
        maximisation of the approximated :math:`\tilde{\Omega} = G^TG` kernel matrix
        (not a global trace maximisation because we use just a lower bound...!).
        And keeping mind that the trace is the sum of the eigenvalues so it's
        related somehow to the PCA ??? and related somehow to the Renyi entropy
        maximisatio (but in sub-set selection i.e. sum) ???. Or not.
        Also note, in :cite:`jenssen2007kernel` section 3.1. somethind interestuon.
        
        ??? Maximising the Reny entropy of a Kernel matrix (or a sub- as in sub-set selection)
        means computing the sum of the elements (which are projections to each other).  
        
        
        ?? Also, note, :math:`\|A\|_F=\sqrt{ \texttt{trace} \{ A^TA\} }` so 
           :math:`\| G \|_F=\sqrt{ \texttt{trace} \{ G^TG \} }` so by maximising 
           the :math:`\texttt{trace} \{ \tilde{Omega} \} = \texttt{trace} \{ G^TG \}`
           one maximising the Frobenius norm of :math:`G` which is the sum of the 
           squared elements of :math:`G`.
        


