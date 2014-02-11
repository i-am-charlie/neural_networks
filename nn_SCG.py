"""
	this program implements a neural network using Scaled Conjugate Gradient
	Descent as the optimizer
	"""

import numpy as np
import matplotlib.pyplot as plt
import ScaledConjugateGradient as SCG

# Make some training data
n = 20
X = np.linspace(0.,20.0,n).reshape((-1,1)) - 10
T = 0.2 + 0.05 * X + 0.4 * np.sin(X) + 0.2 * np.random.normal(size=(n,1))

# Make some testing data
Xtest = X + 0.1*np.random.normal(size=(n,1))
Ttest = 0.2 + 0.05 * X + 0.4 * np.sin(Xtest) + 0.2 * np.random.normal(size=(n,1))

# Set parameters of neural network
nInputs = X.shape[1]
nHiddens = 20
nOutputs = T.shape[1]

# Initialize weights to uniformly distributed values between small uniformly-distributed between -0.1 and 0.1
V = np.random.uniform(-0.1,0.1,(nInputs+1,nHiddens))
W = np.random.uniform(-0.1,0.1,(1+nHiddens,nOutputs))

# Add constant column of 1's
def addOnes(A):
    return np.hstack((np.ones((A.shape[0],1)),A))

X1 = addOnes(X)
Xtest1 = addOnes(Xtest)
np.hstack((X1, T))



######################################################################
## Neural Net specific stuff

### gradientDescent functions require all parameters in a vector.
def pack(V,W):
    return np.hstack((V.flat,W.flat))
def unpack(w):
    '''Assumes V, W, nInputs, nHidden, nOuputs are defined in calling context'''
	V[:] = w[:(nInputs+1)*nHiddens].reshape((nInputs+1,nHiddens))
	W[:] = w[(nInputs+1)*nHiddens:].reshape((nHiddens+1,nOutputs))

### Function f to be minimized
def errorFunction(w):
    unpack(w)
	# Forward pass on training data
	Y = np.dot( addOnes(np.tanh(np.dot(X1,V))),  W )
	return 0.5 * np.mean((Y - T)**2)

### Gradient of f with respect to V,W
def errorGradient(w):
    unpack(w)
	Z = np.tanh(np.dot( X1, V ))
	Z1 = addOnes(Z)
	Y = np.dot( Z1, W )
	nSamples = X1.shape[0]
	nOutputs = T.shape[1]
	error = (Y - T) / (nSamples*nOutputs)
	dV = np.dot( X1.T, np.dot( error, W[1:,:].T) * (1-Z**2)) #/ (nSamples * nOutputs)
	dW = np.dot( Z1.T, error) #/ nSamples
	return pack(dV,dW)


# Initialize weights to uniformly distributed values between small uniformly-distributed between -0.1 and 0.1
V = np.random.uniform(-0.1,0.1,(nInputs+1,nHiddens))
W = np.random.uniform(-0.1,0.1,(1+nHiddens,nOutputs))

result = SCG.scg(pack(V,W), errorFunction, errorGradient,
                    xPrecision = 1.e-8,
				    fPrecision = 1.e-12,
					nIterations = 2000,
					ftracep = True)
result



# Now plot everything and take a look
fig = plt.figure(figsize=(10,15))
plt.subplot(3,1,1)
plt.plot(result['ftrace'])
plt.xlabel('Epochs')
plt.ylabel('Train RMSE')

plt.subplot(3,1,2)
Y = np.dot(addOnes(np.tanh(np.dot(X1,V))), W) 
Ytest = np.dot(addOnes(np.tanh(np.dot(Xtest1,V))), W)
plt.plot(X,T,'o-',Xtest,Ttest,'o-',Xtest,Ytest,'o-')
plt.xlim(-10,10)
plt.legend(('Training','Testing','Model'),loc='upper left')
plt.xlabel('$x$')
plt.ylabel('Actual and Predicted $f(x)$')
        
plt.subplot(3,1,3)
Z = np.tanh(np.dot(X1,V))
plt.plot(X,Z)
plt.xlabel('$x$')
plt.ylabel('Hidden Unit Outputs ($z$)');




""" Now with the steepest descent"""
# Initialize weights to uniformly distributed values between small normally-distributed between -0.1 and 0.1
nHiddens = 20
V = np.random.uniform(-0.1,0.1,(nInputs+1,nHiddens))
W = np.random.uniform(-0.1,0.1,(1+nHiddens,nOutputs))

rho = 0.2
ftrace = []
for i in xrange(20000):
    w = pack(V,W)
	w = w - rho * errorGradient(w)
	unpack(w)
	Y = np.dot(addOnes(np.tanh(np.dot(X1,V))), W)
	error = 0.5 * np.mean((Y-T)**2)
	ftrace.append(error)

fig = plt.figure(figsize=(10,15))
plt.subplot(3,1,1)
plt.plot(ftrace)
plt.xlabel('Epochs')
plt.ylabel('Train RMSE')

plt.subplot(3,1,2)
Y = np.dot(addOnes(np.tanh(np.dot(X1,V))), W) 
Ytest = np.dot(addOnes(np.tanh(np.dot(Xtest1,V))), W)
plt.plot(X,T,'o-',Xtest,Ttest,'o-',Xtest,Ytest,'o-')
plt.xlim(-10,10)
plt.legend(('Training','Testing','Model'),loc='upper left')
plt.xlabel('$x$')
plt.ylabel('Actual and Predicted $f(x)$')

plt.subplot(3,1,3)
Z = np.tanh(np.dot(X1,V))
plt.plot(X,Z)
plt.xlabel('$x$')
plt.ylabel('Hidden Unit Outputs ($z$)');

