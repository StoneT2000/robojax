'''Functions to operate on, or return, quaternions.
Quaternions here consist of 4 values ``w, x, y, z``, where ``w`` is the
real (scalar) part, and ``x, y, z`` are the complex (vector) part.
Note - rotation matrices here apply to column vectors, that is,
they are applied on the left of the vector.  For example:
>>> import numpy as np
>>> q = [0, 1, 0, 0] # 180 degree rotation around axis 0
>>> M = quat2mat(q) # from this module
>>> vec = np.array([1, 2, 3]).reshape((3,1)) # column vector
>>> tvec = np.dot(M, vec)
Terms used in function names:
* *mat* : array shape (3, 3) (3D non-homogenous coordinates)
* *aff* : affine array shape (4, 4) (3D homogenous coordinates)
* *quat* : quaternion shape (4,)
* *axangle* : rotations encoded by axis vector and angle scalar
'''

import math
import numpy as np
import jax
from functools import partial
import jax.numpy as jnp
from chex import Array

_MAX_FLOAT = np.maximum_sctype(np.float64)
_FLOAT_EPS = jnp.finfo(jnp.float64).eps


# def fillpositive(xyz, w2_thresh=None):
#     # TODO
#     ''' Compute unit quaternion from last 3 values
#     Parameters
#     ----------
#     xyz : iterable
#        iterable containing 3 values, corresponding to quaternion x, y, z
#     w2_thresh : None or float, optional
#        threshold to determine if w squared is really negative.
#        If None (default) then w2_thresh set equal to
#        ``-np.finfo(xyz.dtype).eps``, if possible, otherwise
#        ``-np.finfo(np.float64).eps``
#     Returns
#     -------
#     wxyz : array shape (4,)
#          Full 4 values of quaternion
#     Notes
#     -----
#     If w, x, y, z are the values in the full quaternion, assumes w is
#     positive.
#     Gives error if w*w is estimated to be negative
#     w = 0 corresponds to a 180 degree rotation
#     The unit quaternion specifies that np.dot(wxyz, wxyz) == 1.
#     If w is positive (assumed here), w is given by:
#     w = np.sqrt(1.0-(x*x+y*y+z*z))
#     w2 = 1.0-(x*x+y*y+z*z) can be near zero, which will lead to
#     numerical instability in sqrt.  Here we use the system maximum
#     float type to reduce numerical instability
#     Examples
#     --------
#     >>> import numpy as np
#     >>> wxyz = fillpositive([0,0,0])
#     >>> np.all(wxyz == [1, 0, 0, 0])
#     True
#     >>> wxyz = fillpositive([1,0,0]) # Corner case; w is 0
#     >>> np.all(wxyz == [0, 1, 0, 0])
#     True
#     >>> np.dot(wxyz, wxyz)
#     1.0
#     '''
#     # Check inputs (force error if < 3 values)
#     if len(xyz) != 3:
#         raise ValueError('xyz should have length 3')
#     # If necessary, guess precision of input
#     if w2_thresh is None:
#         try: # trap errors for non-array, integer array
#             w2_thresh = -np.finfo(xyz.dtype).eps * 3
#         except (AttributeError, ValueError):
#             w2_thresh = -_FLOAT_EPS * 3
#     # Use maximum precision
#     xyz = np.asarray(xyz, dtype=_MAX_FLOAT)
#     # Calculate w
#     w2 = 1.0 - np.dot(xyz, xyz)
#     if w2 < 0:
#         if w2 < w2_thresh:
#             raise ValueError('w2 should be positive, but is %e' % w2)
#         w = 0
#     else:
#         w = np.sqrt(w2)
#     return np.r_[w, xyz]

@jax.jit
def quat2mat(q: Array):
    ''' Calculate rotation matrix corresponding to quaternion
    Parameters
    ----------
    q : 4 element array-like
    Returns
    -------
    M : (3,3) array
      Rotation matrix corresponding to input quaternion *q*
    Notes
    -----
    Rotation matrix applies to column vectors, and is applied to the
    left of coordinate vectors.  The algorithm here allows quaternions that
    have not been normalized.
    References
    ----------
    Algorithm from http://en.wikipedia.org/wiki/Rotation_matrix#Quaternion
    Examples
    --------
    >>> import numpy as np
    >>> M = quat2mat([1, 0, 0, 0]) # Identity quaternion
    >>> np.allclose(M, np.eye(3))
    True
    >>> M = quat2mat([0, 1, 0, 0]) # 180 degree rotn around axis 0
    >>> np.allclose(M, np.diag([1, -1, -1]))
    True
    '''
    w, x, y, z = q
    Nq = w*w + x*x + y*y + z*z

    def comp():
        s = 2.0/Nq
        X = x*s
        Y = y*s
        Z = z*s
        wX = w*X; wY = w*Y; wZ = w*Z
        xX = x*X; xY = x*Y; xZ = x*Z
        yY = y*Y; yZ = y*Z; zZ = z*Z
        mat = jnp.array(
            [[ 1.0-(yY+zZ), xY-wZ, xZ+wY ],
                [ xY+wZ, 1.0-(xX+zZ), yZ-wX ],
                [ xZ-wY, yZ+wX, 1.0-(xX+yY) ]])
        return mat
    return jnp.where(Nq < _FLOAT_EPS, jnp.eye(3), comp())

@jax.jit
def mat2quat(M: Array):
    ''' Calculate quaternion corresponding to given rotation matrix
    Method claimed to be robust to numerical errors in `M`.
    Constructs quaternion by calculating maximum eigenvector for matrix
    ``K`` (constructed from input `M`).  Although this is not tested, a maximum
    eigenvalue of 1 corresponds to a valid rotation.
    A quaternion ``q*-1`` corresponds to the same rotation as ``q``; thus the
    sign of the reconstructed quaternion is arbitrary, and we return
    quaternions with positive w (q[0]).
    See notes.
    Parameters
    ----------
    M : array-like
      3x3 rotation matrix
    Returns
    -------
    q : (4,) array
      closest quaternion to input matrix, having positive q[0]
    References
    ----------
    * http://en.wikipedia.org/wiki/Rotation_matrix#Quaternion
    * Bar-Itzhack, Itzhack Y. (2000), "New method for extracting the
      quaternion from a rotation matrix", AIAA Journal of Guidance,
      Control and Dynamics 23(6):1085-1087 (Engineering Note), ISSN
      0731-5090
    Examples
    --------
    >>> import numpy as np
    >>> q = mat2quat(np.eye(3)) # Identity rotation
    >>> np.allclose(q, [1, 0, 0, 0])
    True
    >>> q = mat2quat(np.diag([1, -1, -1]))
    >>> np.allclose(q, [0, 1, 0, 0]) # 180 degree rotn around axis 0
    True
    Notes
    -----
    http://en.wikipedia.org/wiki/Rotation_matrix#Quaternion
    Bar-Itzhack, Itzhack Y. (2000), "New method for extracting the
    quaternion from a rotation matrix", AIAA Journal of Guidance,
    Control and Dynamics 23(6):1085-1087 (Engineering Note), ISSN
    0731-5090
    '''
    # Qyx refers to the contribution of the y input vector component to
    # the x output vector component.  Qyx is therefore the same as
    # M[0,1].  The notation is from the Wikipedia article.
    Qxx, Qyx, Qzx, Qxy, Qyy, Qzy, Qxz, Qyz, Qzz = M.flatten()
    # Fill only lower half of symmetric matrix
    K = jnp.array([
        [Qxx - Qyy - Qzz, 0,               0,               0              ],
        [Qyx + Qxy,       Qyy - Qxx - Qzz, 0,               0              ],
        [Qzx + Qxz,       Qzy + Qyz,       Qzz - Qxx - Qyy, 0              ],
        [Qyz - Qzy,       Qzx - Qxz,       Qxy - Qyx,       Qxx + Qyy + Qzz]]
        ) / 3.0
    # Use Hermitian eigenvectors, values for speed
    vals, vecs = jnp.linalg.eigh(K)
    # Select largest eigenvector, reorder to w,x,y,z quaternion
    q = vecs[[3, 0, 1, 2], jnp.argmax(vals)]
    # Prefer quaternion with positive w
    # (q * -1 corresponds to same rotation as q)
    return jnp.where(q[0] < 0, q * - 1, q)

@jax.jit
def qmult(q1: Array, q2: Array):
    ''' Multiply two quaternions
    Parameters
    ----------
    q1 : 4 element sequence
    q2 : 4 element sequence
    Returns
    -------
    q12 : shape (4,) array
    Notes
    -----
    See : http://en.wikipedia.org/wiki/Quaternions#Hamilton_product
    '''
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 + y1*w2 + z1*x2 - x1*z2
    z = w1*z2 + z1*w2 + x1*y2 - y1*x2
    return jnp.array([w, x, y, z])


def qconjugate(q: Array):
    ''' Conjugate of quaternion
    Parameters
    ----------
    q : 4 element sequence
       w, i, j, k of quaternion
    Returns
    -------
    conjq : array shape (4,)
       w, i, j, k of conjugate of `q`
    '''
    return jnp.array(q) * jnp.array([1.0, -1, -1, -1])

@jax.jit
def qnorm(q: Array):
    ''' Return norm of quaternion
    Parameters
    ----------
    q : 4 element sequence
       w, i, j, k of quaternion
    Returns
    -------
    n : scalar
       quaternion norm
    Notes
    -----
    http://mathworld.wolfram.com/QuaternionNorm.html
    '''
    return jnp.sqrt(jnp.dot(q, q))

@jax.jit
def qisunit(q: Array):
    ''' Return True is this is very nearly a unit quaternion '''
    return jnp.allclose(qnorm(q), 1)

@jax.jit
def qinverse(q: Array):
    ''' Return multiplicative inverse of quaternion `q`
    Parameters
    ----------
    q : 4 element sequence
       w, i, j, k of quaternion
    Returns
    -------
    invq : array shape (4,)
       w, i, j, k of quaternion inverse
    '''
    return qconjugate(q) / qnorm(q)

@partial(jax.jit, static_argnames=["dtype"])
def qeye(dtype=jnp.float64):
    ''' Return identity quaternion '''
    return jnp.array([1.0,0,0,0], dtype = dtype)

# @jax.jit
# # TODO
# def qexp(q: Array):
#     ''' Return exponential of quaternion
#     Parameters
#     ----------
#     q : 4 element sequence
#        w, i, j, k of quaternion
#     Returns
#     -------
#     q_exp : array shape (4,)
#         The quaternion exponential
#     Notes
#     -----
#     See:
#     * https://en.wikipedia.org/wiki/Quaternion#Exponential,_logarithm,_and_power
#     * https://math.stackexchange.com/questions/1030737/exponential-function-of-quaternion-derivation
#     '''
#     q = jnp.array(q)  # to ensure there is a dtype
#     w, v = q[0], q[1:]
#     norm = jnp.sqrt(np.dot(v, v))
#     result = jnp.zeros((4,), q.dtype)

#     if norm == 0.:
#         return qeye(q.dtype)

#     result[0] =  jnp.cos(norm)
#     result[1:] = jnp.sin(norm)/norm * v
#     return result * jnp.exp(w)

# @jax.jit
# # TODO
# def qlog(q: Array):
#     ''' Return natural logarithm of quaternion
#     Parameters
#     ----------
#     q : 4 element sequence
#        w, i, j, k of quaternion
#     Returns
#     -------
#     q_log : array shape (4,)
#         Natual logarithm of quaternion
#     Notes
#     -----
#     See: https://en.wikipedia.org/wiki/Quaternion#Exponential,_logarithm,_and_power
#     '''
#     q = jnp.array(q)  # To ensure there is a dtype
#     qnorm_ = qnorm(q)
#     if qnorm_ == 0.:
#         return qeye(q.dtype)

#     w, v = q[0], q[1:]
#     vnorm = jnp.sqrt(jnp.dot(v, v))
#     result = jnp.zeros((4,), q.dtype)

#     if vnorm == 0.:
#         return qeye(q.dtype)

#     result[0] =  jnp.log(qnorm_)
#     result[1:] = v/vnorm * jnp.arccos(w/qnorm_)
#     return result

# @jax.jit
# # TODO
# def qpow(q: Array, n: float):
#     r''' Return the `n` th power of quaternion `q`
#     Parameters
#     ----------
#     q : 4 element sequence
#        w, i, j, k of quaternion
#     n : int or float
#        A real number
#     Returns
#     -------
#     q_pow : array shape (4,)
#         The quaternion `q` to `n` th power.
#     Notes
#     -----
#     See:
#     https://en.wikipedia.org/wiki/Quaternion#Exponential,_logarithm,_and_power
#     '''
#     q = jnp.array(q)  # To ensure there is a dtype.
#     qnorm_ = qnorm(q)

#     if qnorm_ == 0.:
#         return qeye(q.dtype)

#     w, v = q[0], q[1:]

#     nnorm = jnp.sqrt(jnp.dot(v, v))
#     result = jnp.zeros((4,), q.dtype)

#     if nnorm == 0.:
#         return qeye(q.dtype)

#     theta = jnp.arccos(w/qnorm_)
#     n_hat = v/nnorm

#     result[0] = jnp.cos(n*theta)
#     result[1:] = n_hat * jnp.sin(n*theta)
#     return result *  jnp.power(qnorm_, n)

@partial(jax.jit, static_argnames=["is_normalized"])
def rotate_vector(v: Array, q: Array, is_normalized=True):
    ''' Apply transformation in quaternion `q` to vector `v`
    Parameters
    ----------
    v : 3 element sequence
        3 dimensional vector
    q : 4 element sequence
        w, i, j, k of quaternion
    is_normalized : {True, False}, optional
        If True, assume `q` is normalized.  If False, normalize `q` before
        applying.
    Returns
    -------
    vdash : array shape (3,)
       `v` rotated by quaternion `q`
    Notes
    -----
    See: http://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation#Describing_rotations_with_quaternions
    '''
    if not is_normalized:
        q = q / qnorm(q)
    varr = jnp.zeros((4,))
    varr[1:] = v
    return qmult(q, qmult(varr, qconjugate(q)))[1:]

@partial(jax.jit)
def nearly_equivalent(q1: Array, q2: Array, rtol=1e-5, atol=1e-8):
    ''' Returns True if `q1` and `q2` give near equivalent transforms
    `q1` may be nearly numerically equal to `q2`, or nearly equal to `q2` * -1
    (because a quaternion multiplied by -1 gives the same transform).
    Parameters
    ----------
    q1 : 4 element sequence
       w, x, y, z of first quaternion
    q2 : 4 element sequence
       w, x, y, z of second quaternion
    Returns
    -------
    equiv : bool
       True if `q1` and `q2` are nearly equivalent, False otherwise
    Examples
    --------
    >>> q1 = [1, 0, 0, 0]
    >>> nearly_equivalent(q1, [0, 1, 0, 0])
    False
    >>> nearly_equivalent(q1, [1, 0, 0, 0])
    True
    >>> nearly_equivalent(q1, [-1, 0, 0, 0])
    True
    '''
    q1 = jnp.array(q1)
    q2 = jnp.array(q2)
    return jnp.where(jnp.allclose(q1, q2, rtol, atol), True, jnp.allclose(q1 * -1, q2, rtol, atol))

@partial(jax.jit, static_argnames=["is_normalized"])
def axangle2quat(vector: Array, theta: float, is_normalized=False):
    ''' Quaternion for rotation of angle `theta` around `vector`
    Parameters
    ----------
    vector : 3 element sequence
       vector specifying axis for rotation.
    theta : scalar
       angle of rotation in radians.
    is_normalized : bool, optional
       True if vector is already normalized (has norm of 1).  Default
       False.
    Returns
    -------
    quat : 4 element sequence of symbols
       quaternion giving specified rotation
    Examples
    --------
    >>> q = axangle2quat([1, 0, 0], np.pi)
    >>> np.allclose(q, [0, 1, 0,  0])
    True
    Notes
    -----
    Formula from http://mathworld.wolfram.com/EulerParameters.html
    '''
    vector = jnp.array(vector)
    if not is_normalized:
        # Cannot divide in-place because input vector may be integer type,
        # whereas output will be float type; this may raise an error in versions
        # of numpy > 1.6.1
        vector = vector / jnp.sqrt(jnp.dot(vector, vector))
    t2 = theta / 2.0
    st2 = jnp.sin(t2)
    return jnp.concatenate([jnp.array([jnp.cos(t2)]),
                           vector * st2])

@partial(jax.jit, static_argnames=["identity_thresh"])
def quat2axangle(quat: Array, identity_thresh=None):
    ''' Convert quaternion to rotation of angle around axis
    Parameters
    ----------
    quat : 4 element sequence
       w, x, y, z forming quaternion.
    identity_thresh : None or scalar, optional
       Threshold below which the norm of the vector part of the quaternion (x,
       y, z) is deemed to be 0, leading to the identity rotation.  None (the
       default) leads to a threshold estimated based on the precision of the
       input.
    Returns
    -------
    theta : scalar
       angle of rotation.
    vector : array shape (3,)
       axis around which rotation occurs.
    Examples
    --------
    >>> vec, theta = quat2axangle([0, 1, 0, 0])
    >>> vec
    array([1., 0., 0.])
    >>> np.allclose(theta, np.pi)
    True
    If this is an identity rotation, we return a zero angle and an arbitrary
    vector:
    >>> quat2axangle([1, 0, 0, 0])
    (array([1., 0., 0.]), 0.0)
    If any of the quaternion values are not finite, we return a NaN in the
    angle, and an arbitrary vector:
    >>> quat2axangle([1, np.inf, 0, 0])
    (array([1., 0., 0.]), nan)
    Notes
    -----
    A quaternion for which x, y, z are all equal to 0, is an identity rotation.
    In this case we return a 0 angle and an arbitrary vector, here [1, 0, 0].
    The algorithm allows for quaternions that have not been normalized.
    '''
    Nq = jnp.sum(quat ** 2)
    # if not np.isfinite(Nq):
    #     return np.array([1.0, 0, 0]), float('nan')

    # if identity_thresh is None:
    #     try:
    #         identity_thresh = np.finfo(Nq.type).eps * 3
    #     except (AttributeError, ValueError): # Not a numpy type or not float
    identity_thresh = _FLOAT_EPS * 3

    bad_res = jnp.array([1.0, 0, 0]), 0.0
    ret = bad_res
    def normalize(quat):
        quat = jnp.where(Nq != 1, quat / jnp.sqrt(Nq), quat) # Normalize if not normalized
        xyz = quat[1:]
        len2 = jnp.sum(xyz ** 2)
        def clip(quat):
            # Make sure w is not slightly above 1 or below -1
            theta = 2 * jnp.arccos(jnp.clip(quat[0], -1, 1))
            return xyz / jnp.sqrt(len2), theta
        # if vec is nearly 0,0,0, return an identity rotation
        return jax.lax.cond(len2 < identity_thresh ** 2, lambda q: bad_res, clip, quat)
        # return quat
    # Return identity if results unreliable after normalization
    return jax.lax.cond(Nq < _FLOAT_EPS ** 2, lambda q: bad_res, normalize, quat)