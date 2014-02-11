import matplotlib.pyplot as plt
import numpy as np

X = np.linspace(1,10,20).reshape(-1,1) - 5  # -5 to center the X values
T = np.array([1,4,6,8,7,5,3,4,2,1,4,6,9,8,7,8,5,4,8,9]).reshape(-1,1)

plt.plot(X,T,'o-')

# now we make the lest squares estimator
X1 = np.hstack((np.ones((X.shape[0],1)), X))
W = np.linalg.solve(np.dot(X1.T,X1), np.dot(X1.T,T))
Y = np.dot(X1,W)

plt.plot(X,T,'o-')
plt.plot(X,Y,'ro-')

"""
	What followd is the creation of the neural network, we first initialize our weight matrices V and W to small random values, after assigning the number of input dimensions, number of hidden units, and the number of outputs
	"""
nInputs = X.shape[1]
nOutputs = T.shape[1]
nHiddens = 5
V = 0.1 * (2 * np.random.uniform(size=(1+nInputs, nHiddens)) - 1)
W = 0.1 * (2 * np.random.uniform(size=(1+nHiddens, nOutputs)) - 1)
# V,W


# now I just define a function that adds a column of ones to a matrix
## this will just be very handy and make life easier
def addOnes(M):
    return np.hstack(( np.ones((M.shape[0],1)), M ))


####
# Now repeat after me: forward pass, backward pass, frwrd pass, bckwrd pass...
####

nSamples = X.shape[0]
nInputs = X.shape[1]
nOutputs = T.shape[1]
nHiddens = 10

V = 0.1 * (2 * np.random.uniform(size=(1+nInputs, nHiddens)) - 1)
W = 0.1 * (2 * np.random.uniform(size=(1+nHiddens, nOutputs)) - 1)

rhoh = 0.3 / (nSamples * nOutputs)
rhoo = 0.1 / nSamples

nRepetitions = 5000
errorTrace = np.empty(nRepetitions)

for reps in range(nRepetitions):
    # forward pass
	Z = np.tanh(np.dot( X1, V ))
	Z1 = addOnes(Z)
	Y = np.dot( Z1, W )
				    
	# error
	error = Y - T
							    
	# backward pass
	V = V - rhoh * np.dot( X1.T, np.dot( error, W[1:,:].T) * (1 - Z**2))
	W = W - rhoo * np.dot( Z1.T, error)
											    
	# Keep track of training and testing Root-Mean-Square-Error (RMSE)
	rmse =  np.sqrt(np.mean(( Y - T )**2))
	errorTrace[reps] = rmse
	
fig = plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(errorTrace);
plt.subplot(1,2,2)
plt.plot(X,T)
plt.plot(X,Y)
plt.legend(('Data','Model'), loc='lower right');





"""
	Now this is cool... we can find out what the hidden units have learned. 
	We can plot their outputs for each input sample.  
	We can also plot their outputs multiplied by their output weights to see
	their full effect on the network's output.
	"""

plt.figure(figsize=(6,11))
plt.subplot(3,1,1)
plt.plot(X,T)
plt.plot(X,Y)
plt.subplot(3,1,2)
plt.plot(X,Z)
plt.subplot(3,1,3)
plt.plot(X,Z*W[1:,:].T);
