
"""The metrics of computation graph implementation. 
   As the matter of efficient computation, we should run pure graph operations in training processes(only on GPU).
   For example, when we use keras fit_generate function, we must use graph implementation wrote here.
"""

import keras.backend as K
