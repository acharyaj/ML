import numpy as np

def featureScaling(x):
	numFeatures = len(x) # Assumption is that x is a list of the form [list1, list2, ..., listn] where n = numVar is the number of features and length of each individual list is the number of training examples 	
	xa = np.array(x)
	minx = np.min(xa, 1)
	maxx = np.max(xa, 1)
		
	x_s = [((xa[i] - minx[i])/float(maxx[i] - minx[i]) - 0.5).tolist() for i in range(numFeatures)] # The tolist converts the output back to a list so that i/p and o/p are of consistent data type (list)
	return x_s

def computeInitialCentroids(x_s, numCentroids):
	y = np.rint(np.random.random(numCentroids)*len(x_s))
	y = [int(y[i]) for i in range(numCentroids)]
	centroids = [(np.array(x_s)[:,y[i]]).tolist() for i in range(numCentroids)] # The tolist converts the output back to a list so that i/p and o/p are of consistent data type (list)
	return centroids

def findClosestCentroids(x_s, centroids, numCentroids):
	numSamples = len(x_s[0])
	indx = []
	for i in range(numSamples):
		k = np.argmin([np.linalg.norm(np.array(x_s)[:,i] - np.array(centroids[j])) for j in range(numCentroids)])	
		indx.append(k)
		
	if np.max(indx) >= numCentroids:
		sys.exit('Error !!!')

	return indx

def computeCentroids(x_s, indx, numCentroids):
	numSamples = len(x_s[0])
	centroids = []
	for i in range(numCentroids):
		if i in indx: # in effect discards clusters that have no points 
			tmp = np.mean([np.array(x_s)[:,j] for j in range(numSamples) if indx[j] == i],0).tolist()
			centroids.append(tmp)
	
	return centroids 

def runKmeans(x):
	maxIters = 7
	numCentroids = 3
	
	x_s = featureScaling(x)
	centroids = computeInitialCentroids(x_s, numCentroids)

	for i in range(maxIters):
		indx = findClosestCentroids(x_s, centroids, numCentroids)
		centroids = computeCentroids(x_s, indx, numCentroids)
		numCentroids = len(centroids)

	cluster = {'indx':indx, 'numCentroids':numCentroids}
	return cluster
