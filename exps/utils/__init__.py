import random
import numbers
import numpy as np
import torch


def set_seed(seed):
    """ Random seed generation for PyTorch. See https://pytorch.org/docs/stable/notes/randomness.html
        for further details.
    Args:
        seed (int): the seed for pseudonumber generation.
    """

    if seed is not None:
        random.seed(seed)

        np.random.seed(seed)

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def view_as_windows(ten_in, window_shape, step=1):
    """Rolling window view of the input n-dimensional tensor
    Windows are overlapping views of the input array, with adjacent windows
    shifted by a single row or column (or an index of a higher dimension).
    Parameters
    ----------
    arr_in : torch.tensor
        N-d input tensor.
    window_shape : integer or tuple of length arr_in.ndim
        Defines the shape of the elementary n-dimensional orthotope
        (better know as hyperrectangle [1]_) of the rolling window view.
        If an integer is given, the shape will be a hypercube of
        sidelength given by its value.
    step : integer or tuple of length arr_in.ndim
        Indicates step size at which extraction shall be performed.
        If integer is given, then the step is uniform in all dimensions.
    Returns
    -------
    arr_out : torch.tensor
        (rolling) window view of the input tensor.
    Notes
    -----
    One should be very careful with rolling views when it comes to
    memory usage.  Indeed, although a 'view' has the same memory
    footprint as its base tensor, the actual tensor that emerges when this
    'view' is used in a computation is generally a (much) larger tensor
    than the original, especially for 2-dimensional tensor and above.
    For example, let us consider a 3 dimensional tensor of size (100,
    100, 100) of ``float64``. This tensor takes about 8*100**3 Bytes for
    storage which is just 8 MB. If one decides to build a rolling view
    on this tensor with a window of (3, 3, 3) the hypothetical size of
    the rolling view (if one was to reshape the view for example) would
    be 8*(100-3+1)**3*3**3 which is about 203 MB! The scaling becomes
    even worse as the dimension of the input tensor becomes larger.
    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Hyperrectangle
    Examples
    --------
    >>> import torch
    >>> A = torch.arange(4*4).reshape(4,4)
    >>> A
    array([[ 0,  1,  2,  3],
           [ 4,  5,  6,  7],
           [ 8,  9, 10, 11],
           [12, 13, 14, 15]])
    >>> window_shape = (2, 2)
    >>> B = view_as_windows(A, window_shape)
    >>> B[0, 0]
    array([[0, 1],
           [4, 5]])
    >>> B[0, 1]
    array([[1, 2],
           [5, 6]])
    >>> A = torch.arange(10)
    >>> A
    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    >>> window_shape = (3,)
    >>> B = view_as_windows(A, window_shape)
    >>> B.shape
    (8, 3)
    >>> B
    array([[0, 1, 2],
           [1, 2, 3],
           [2, 3, 4],
           [3, 4, 5],
           [4, 5, 6],
           [5, 6, 7],
           [6, 7, 8],
           [7, 8, 9]])
    >>> A = torch.arange(5*4).reshape(5, 4)
    >>> A
    array([[ 0,  1,  2,  3],
           [ 4,  5,  6,  7],
           [ 8,  9, 10, 11],
           [12, 13, 14, 15],
           [16, 17, 18, 19]])
    >>> window_shape = (4, 3)
    >>> B = view_as_windows(A, window_shape)
    >>> B.shape
    (2, 2, 4, 3)
    >>> B  # doctest: +NORMALIZE_WHITESPACE
    array([[[[ 0,  1,  2],
             [ 4,  5,  6],
             [ 8,  9, 10],
             [12, 13, 14]],
            [[ 1,  2,  3],
             [ 5,  6,  7],
             [ 9, 10, 11],
             [13, 14, 15]]],
           [[[ 4,  5,  6],
             [ 8,  9, 10],
             [12, 13, 14],
             [16, 17, 18]],
            [[ 5,  6,  7],
             [ 9, 10, 11],
             [13, 14, 15],
             [17, 18, 19]]]])
    """

    # -- basic checks on arguments
    if not torch.is_tensor(ten_in):
        raise TypeError("`ten_in` must be a torch tensor")

    ndim = ten_in.ndim

    if isinstance(window_shape, numbers.Number):
        window_shape = (window_shape,) * ndim
    if not (len(window_shape) == ndim):
        raise ValueError("`window_shape` is incompatible with `ten_in.shape`")

    if isinstance(step, numbers.Number):
        if step < 1:
            raise ValueError("`step` must be >= 1")
        step = (step,) * ndim
    if len(step) != ndim:
        raise ValueError("`step` is incompatible with `ten_in.shape`")

    ten_shape = torch.tensor(ten_in.shape)
    window_shape = torch.tensor(window_shape, dtype=ten_shape.dtype)

    if ((ten_shape - window_shape) < 0).any():
        raise ValueError("`window_shape` is too large")

    if ((window_shape - 1) < 0).any():
        raise ValueError("`window_shape` is too small")

    # -- build rolling window view
    slices = tuple(slice(None, None, st) for st in step)
    window_strides = torch.tensor(ten_in.stride())

    indexing_strides = ten_in[slices].stride()

    '''win_indices_shape = (((torch.tensor(ten_in.shape) - 
                           window_shape.clone().detach())
                          // torch.tensor(step)) + 1)'''  # next line was like this

    win_indices_shape = torch.div((torch.tensor(ten_in.shape) -
                           window_shape.clone().detach()), torch.tensor(step), rounding_mode='trunc') + 1

    new_shape = tuple(list(win_indices_shape) + list(window_shape))
    stride = tuple(list(indexing_strides) + list(window_strides))

    arr_out = torch.as_strided(ten_in, size=new_shape, stride=stride)
    return arr_out