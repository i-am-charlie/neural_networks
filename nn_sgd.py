import numpy as np
import matplotlib.pyplot as plt

def addOnes(M):
    return np.hstack(( np.ones((M.shape[0],1)), M ))

X = np.linspace(1,10,20).reshape(-1,1) - 5  # -5 to center the X values
T = np.array([1,4,6,8,7,5,3,4,2,1,4,6,9,8,7,8,5,4,8,9]).reshape(-1,1)

nSamples = X.shape[0]
nInputs = X.shape[1]
nOutputs = T.shape[1]
nHiddens = 20

V = np.random.uniform(-0.1,0.1,(1+nInputs, nHiddens))
W = np.random.uniform(-0.1,0.1,(1+nHiddens, nOutputs))

rhoh = 0.01
rhoo = 0.01

nRepetitions = 6000
errorTrace = np.empty(nRepetitions)

X1 = addOnes(X)

for reps in range(nRepetitions):

    sampleOrder = np.random.permutation(nSamples)
	for xi in sampleOrder:
		        
		# forward pass
		Z = np.tanh(np.dot( X1[xi:xi+1,:], V ))  # xi:xi+1 preserves 2D matrix structure
		Z1 = addOnes(Z)
		Y = np.dot( Z1, W )
												    
		# error
		error = Y - T[xi:xi+1,:]
																	        
	
		# backward pass
		V = V - rhoh * np.dot( X1[xi:xi+1,:].T, np.dot( error, W[1:,:].T) * (1 - Z**2))
		W = W - rhoo * np.dot( Z1.T, error)

																						# Keep track of training and testing Root-Mean-Square-Error (RMSE)
		Y = np.dot(addOnes(np.tanh(np.dot(X1,V))), W) # calculate for all samples
		rmse =  np.sqrt(np.mean(( Y - T )**2))
		errorTrace[reps] = rmse

print "Final RMSE is",rmse
fig = plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(errorTrace);
plt.subplot(1,2,2)
plt.plot(X,T)
plt.plot(X,Y)
plt.legend(('Data','Model'), loc='lower right');



###
#### try with a different size step

rho0 = 0.01

rhos = []
for i in range(100):
    rho = rho0 / (1+i*rho0)
	    rhos.append(rho)
		plt.plot(rhos);



###
#### now we increment the t a little differently

V = np.random.uniform(-0.1,0.1,(1+nInputs, nHiddens))
W = np.random.uniform(-0.1,0.1,(1+nHiddens, nOutputs))

rho0 = 0.03

nRepetitions = 6000
errorTrace = np.empty(nRepetitions)

X1 = addOnes(X)
t = 0 # for rho decay
for reps in range(nRepetitions):

    sampleOrder = np.random.permutation(nSamples)
	for xi in sampleOrder:
		    
		# forward pass
		Z = np.tanh(np.dot( X1[xi:xi+1,:], V ))  # xi:xi+1 preserves 2D matrix structure
		Z1 = addOnes(Z)
		Y = np.dot( Z1, W )
												   
		# error
		error = Y - T[xi:xi+1,:]
																	    
		# backward pass
		rho = rho0 / (1+t*rho0)
		#t = t + 1
		V = V - rho * np.dot( X1[xi:xi+1,:].T, np.dot( error, W[1:,:].T) * (1 - Z**2))
		W = W - rho * np.dot( Z1.T, error)

   t = t + 1 # incrementing t here rather than inside the sample loop seems to work better
       
	# Keep track of training and testing Root-Mean-Square-Error (RMSE)
	Y = np.dot(addOnes(np.tanh(np.dot(X1,V))), W) # calculate for all samples
	rmse =  np.sqrt(np.mean(( Y - T )**2))
	errorTrace[reps] = rmse
	
print "Final RMSE is",rmse
fig = plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(errorTrace);
plt.subplot(1,2,2)
plt.plot(X,T)
plt.plot(X,Y)
plt.legend(('Data','Model'), loc='lower right');
