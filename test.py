import numpy as np

a = np.arange(24).reshape(3,2,4)
print(np.sum(a, axis=0) == [[12,15,18,21],[24,27,30,33]], np.sum(a, axis=0).shape == {2,4}
)

print(np.sum(a, axis=0)) 


a = np.arange(24).reshape(2,3,4)
print(np.transpose(a)[0,0,0] == 0,
np.transpose(a)[1,0,0] == 1,
np.transpose(a)[0,1,0] == 4,
np.transpose(a)[0,0,1] == 12, np.transpose(a).shape)
