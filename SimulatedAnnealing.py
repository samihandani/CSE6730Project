import random
import math
import matplotlib.pyplot as plt
import copy
import numpy as np

class simulatedAnnealing():
	def __init__(self, initTemp, dimSize, optimizeFunc, startPos):
		self.dimSize = dimSize
		self.initTemp = initTemp
		self.temp = self.initTemp
		self.iter = 0
		self.time = 100 #start at 200 to avoid complete random walk phase
		self.stepSize = 0.2
		self.currentScore = 1E100
		self.currentPos = []
		self.optimizeFunc = optimizeFunc

		self.currentPos = startPos.flatten()

		self.currentPos = np.array(self.currentPos)



	def update(self):
		potentialNewPos = self.getNeighbor()
		newScore = self.getScore(potentialNewPos)
		deltaE = newScore - self.currentScore
		if deltaE <= 0:
			self.currentPos = potentialNewPos
			self.currentScore = newScore
			print(self.iter, self.currentScore)
		else:
			probability = math.exp(-deltaE / self.temp * 0.1)
			if random.uniform(0, 1) < probability:
				self.currentPos = potentialNewPos
				self.currentScore = newScore
			
				print(self.iter, newScore, self.time, probability)


		#decay temperature as a function of time
		self.temp = self.initTemp * 0.98 ** self.time #exponential temp function
		# self.temp = self.initTemp/(1 + math.log(self.time + 1.1)) #logerithmic temp function
		# self.temp = self.initTemp - 1E1 * self.time
		self.time += 1
		self.iter += 1

	def getNeighbor(self):
		increasing = random.choice([True, False])
		dimToChange = random.choice(range(0, self.dimSize))
		newPos = copy.deepcopy(self.currentPos)
		if increasing:
			newPos[dimToChange] += self.stepSize
		else:
			newPos[dimToChange] -= self.stepSize
		return newPos


	def getScore(self, pos):
		'''Determines how close a particle is to the optima - lower is better
		'''
		return self.optimizeFunc(pos)

def closeToTwo(pos):
	return abs(pos[0] - 2)

if __name__ == '__main__':
	simAnn = simulatedAnnealing(closeToTwo, 1000000000, 1, (-100, 100))
	poses = []
	while  simAnn.time < 10000:
		simAnn.update()
		print(simAnn.time, simAnn.temp, simAnn.currentScore)
		print(simAnn.currentPos)
		poses.append(simAnn.currentPos)
	
	plt.plot(poses) # plotting by columns
	plt.show()
