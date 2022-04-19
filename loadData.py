import geopandas as gpd
import matplotlib.pyplot as plt
from sklearn import cluster
from sklearn.cluster import KMeans
from scipy import spatial
import numpy as np

from SimulatedAnnealing import simulatedAnnealing

class ElectionSim():
	def __init__(self, desiredElectionResult):
		#load shape file with geopandas
		self.shapefile = gpd.read_file("FL/FL.shp")

		self.numDistricts = int(max(self.shapefile["cd113"]))

		self.kmeans = KMeans(n_clusters=self.numDistricts, n_init=1, random_state=None, max_iter=100)

		self.desiredElectionResult = np.sort(desiredElectionResult)

		#extract centroids from shape file
		self.centers = []
		self.weights = []
		for index, line in self.shapefile.iterrows():
			geo = line['geometry']
			pop = line['TotPop']
			self.centers.append([geo.centroid.x, geo.centroid.y])
			self.weights.append(pop)

	def generateClusterCenters(self):
		#run k-means clustering
		
		weighted_kmeans = self.kmeans.fit(self.centers, sample_weight = self.weights)
		cluster_centers = weighted_kmeans.cluster_centers_

		return cluster_centers
	
	def getElectionResult(self, indexes):
		self.shapefile['sim district'] = indexes
		np_shapefile = self.shapefile.to_numpy()
		demReps = 0
		repReps = 0
		#count up the number of democratic and republican votes in each district
		for districtNum in range(self.numDistricts):
			district = np_shapefile[np.where(np_shapefile[:, 12] == districtNum)]
			republicanVotes = np.sum(district[:, 9])
			democraticVotes = np.sum(district[:, 10])
			if republicanVotes < democraticVotes:
				demReps += 1
			else:
				repReps += 1

		return demReps, repReps
	
	def getElectionRatios(self, indexes):
		self.shapefile['sim district'] = indexes
		np_shapefile = self.shapefile.to_numpy()
		ratios = []
		#count up the number of democratic and republican votes in each district
		for districtNum in range(self.numDistricts):
			district = np_shapefile[np.where(np_shapefile[:, 12] == districtNum)]
			republicanVotes = np.sum(district[:, 9])
			democraticVotes = np.sum(district[:, 10])
			ratios.append(republicanVotes / (republicanVotes + democraticVotes))

		return np.array(ratios)

	def runSimuluations(self, numSims):
		for i in range(numSims):
			cluster_centers = sim.generateClusterCenters()
			distances,indexes = spatial.cKDTree(cluster_centers).query(sim.centers) #Voronoi diagram estimation to form districts
			demReps, repReps = sim.getElectionResult(indexes)

			#record this election
			if demReps not in electionRecords:
				electionRecords[demReps] = 0

			electionRecords[demReps] += 1

			print(f"iteration: {i}: {demReps} democrats and {repReps} republicans elected")

		print(electionRecords)
		plt.bar(electionRecords.keys(), electionRecords.values(), color ='darkblue', width = 0.8)

		plt.xlabel("No. of Democratic Represenatives Elected")
		plt.ylabel(f"No. of simulations (out of {numSims})")

		#plot the simulated districts
		f, ax = plt.subplots(1)
		sim.shapefile.plot(ax=ax, column='sim district')

	def scoreDistrictBoundaries(self, cluster_centers):
		self.cluster_centers = cluster_centers.reshape((27, 2))
		_, indexes = spatial.cKDTree(self.cluster_centers).query(sim.centers) #Voronoi diagram estimation to form districts
		self.shapefile['sim district'] = indexes
		np_shapefile = self.shapefile.to_numpy()

		electionRatios = []
		#create a vector of district votes
		for districtNum in range(self.numDistricts):
			district = np_shapefile[np.where(np_shapefile[:, 12] == districtNum)]
			republicanVotes = np.sum(district[:, 9])
			democraticVotes = np.sum(district[:, 10])
			if democraticVotes + republicanVotes > 0:
				electionRatios.append(republicanVotes / (democraticVotes + republicanVotes))
			else:
				electionRatios.append(100) #high number to disencentivise - far from goal

		electionRatios = np.array(electionRatios)
		electionRatios = np.sort(electionRatios)
		
		distFromDesired = 0
		for i in range(len(electionRatios)):
			distFromDesired += np.abs(electionRatios[i] - self.desiredElectionResult[i])
		distFromDesired /= len(electionRatios)
		# distFromDesired = np.linalg.norm(electionRatios - self.desiredElectionResult)

		return distFromDesired
			

#simulate a certain number of elections
electionRecords = dict()
numDistricts = 27
numSims = 1000
desired = np.ones(27) * 0.6
sim = ElectionSim(desired)
cluster_centers = sim.generateClusterCenters()
simAnn = simulatedAnnealing(400, numDistricts * 2, sim.scoreDistrictBoundaries, cluster_centers)
for _ in range(2000):
	simAnn.update()
cluster_centers = simAnn.currentPos.reshape((27, 2))
distances,indexes = spatial.cKDTree(cluster_centers).query(sim.centers) #Voronoi diagram estimation to form districts

ratios = sim.getElectionRatios(indexes)
# sim.shapefile['sim district'] = indexes.flatten()
print(f"average dist: {np.average(np.abs(0.5 - ratios))}")

ind = np.linspace(0, numDistricts, num=27, endpoint=False)
print(f"ind: {ind}")


f, ax = plt.subplots(1)
ax.bar(ind, ratios, 0.35, color='r')
ax.bar(ind, 1-ratios, 0.35,bottom=ratios, color='b')
ax.set_xlabel('District')
ax.set_ylabel("% of Vote")
ax.set_title('Estimated Vote for prescribed 60%')
ax.set_xticks(ind)
ax.set_yticks(np.arange(0, 1, 0.05))


# f, ax2 = plt.subplots(2)
# sim.shapefile.plot(ax=ax2, column='sim district')
# ax2.scatter([c[0] for c in cluster_centers], [c[1] for c in sim.cluster_centers], marker="o", color='black')
# sim.runSimuluations(100)




# shapefile.boundary.plot(ax=ax, color=(0,0,0)) #plot boundaries (makes graph messy)

#plot centroids of census districts
# ax.scatter([c[0] for c in centers], [c[1] for c in centers], marker="o", color='r')

#plot means from k-means clustering
# 

plt.show()
