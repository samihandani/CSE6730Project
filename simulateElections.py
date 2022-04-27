import geopandas as gpd
import matplotlib.pyplot as plt
from sklearn import cluster
from sklearn.cluster import KMeans
from scipy import spatial
import numpy as np

from SimulatedAnnealing import simulatedAnnealing

class ElectionSim():
	def __init__(self):
		#load shape file with geopandas
		self.shapefile = gpd.read_file("FL/FL.shp")
		self.numDistricts = int(max(self.shapefile["cd113"]))
		self.kmeans = KMeans(n_clusters=self.numDistricts, n_init=4, random_state=None, max_iter=1000)
		self.numDistricts = 27 #number of districts in florida

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
	
	def getElectionRatios(self, simInd=None):
		if simInd is not None: #we're using a simulated map
			self.shapefile['sim district'] = simInd
		np_shapefile = self.shapefile.to_numpy()
		np_shapefile[:, 2] = np_shapefile[:,2].astype(str).astype(int) #needed if plotting election results
		np_shapefile[:, 12] = np_shapefile[:,12].astype(str).astype(int) #needed if plotting election results
		
		ratios = []
		#count up the number of democratic and republican votes in each district
		for districtNum in range(0, self.numDistricts):
			if simInd is None:
				district = np_shapefile[np.where(np_shapefile[:, 2] == districtNum + 1)]
			else:
				district = np_shapefile[np.where(np_shapefile[:, 12] == districtNum)]
			republicanVotes = np.sum(district[:, 9])
			democraticVotes = np.sum(district[:, 10])
			if republicanVotes + democraticVotes == 0:
				ratios.append(np.nan) #empty district
			else:
				ratios.append(republicanVotes / (republicanVotes + democraticVotes))
		ratios = np.array(ratios)
		ratios.sort()
		return ratios

	def getPopulationRatios(self, simInd=None):
		if simInd is not None: #we're using a simulated map
			self.shapefile['sim district'] = simInd
		np_shapefile = self.shapefile.to_numpy()
		np_shapefile[:, 2] = np_shapefile[:,2].astype(str).astype(int) #needed if plotting election results
		np_shapefile[:, 12] = np_shapefile[:,12].astype(str).astype(int) #needed if plotting election results
		
		popRatios = []
		totalPop = np.sum(np_shapefile[:,3])
		#count up the number of democratic and republican votes in each district
		for districtNum in range(0, self.numDistricts):
			if simInd is None:
				district = np_shapefile[np.where(np_shapefile[:, 2] == districtNum + 1)]
			else:
				district = np_shapefile[np.where(np_shapefile[:, 12] == districtNum)]
			districtTotalPop = np.sum(district[:, 3])
			popRatios.append(districtTotalPop / totalPop)
		
		if simInd is None:
			print(popRatios)
		popRatios = np.array(popRatios)
		popRatios.sort()

		return popRatios

	def runKMeansSimuluations(self, numSims):
		electionRecords = dict()
		for i in range(numSims):
			cluster_centers = sim.generateClusterCenters()
			distances,indexes = spatial.cKDTree(cluster_centers).query(sim.centers) #Voronoi diagram estimation to form districts
			demReps, repReps = sim.getElectionResult(indexes)

			#record this election
			if demReps not in electionRecords:
				electionRecords[demReps] = 0

			electionRecords[demReps] += 1

			if i % 100 == 0:
				print(f"running K-Means: {i} / {numSims}")

		return electionRecords

	def scoreDistrictBoundaries(self, cluster_centers):
		self.cluster_centers = cluster_centers.reshape((27, 2))
		_, indexes = spatial.cKDTree(self.cluster_centers).query(self.centers) #Voronoi diagram estimation to form districts
		# self.shapefile['sim district'] = indexes
		# np_shapefile = self.shapefile.to_numpy()
		# print(indexes)

		voteRatios = self.getElectionRatios(indexes)
		popRatios = self.getPopulationRatios(indexes)

		distFromDesiredVote = 0
		for i in range(len(voteRatios)):
			if voteRatios[i] == np.nan:
				distFromDesiredVote += 100 #heavily disencentivize empty districts
			else:
				distFromDesiredVote += np.abs(voteRatios[i] - self.desiredElectionResult[i]) ** 4
		distFromDesiredVote /= len(voteRatios)
		# distFromDesiredVote = np.linalg.norm(voteRatios - self.desiredElectionResult)

		idealPopRatio = 1.0 / self.numDistricts #each district should have equal population
		distFromDesiredPop = 0
		for i in range(len(popRatios)):
			distFromDesiredPop += np.abs(popRatios[i] - idealPopRatio) ** 4 * 3
		distFromDesiredPop /= len(popRatios)

		distFromDesired = distFromDesiredPop + distFromDesiredVote

		return distFromDesired * 1000

	def runSimulatedAnnealing(self, desired, numSims):
		self.desiredElectionResult = np.sort(desired)
		#simulate a certain number of elections
		cluster_centers = sim.generateClusterCenters()
		simAnn = simulatedAnnealing(400, self.numDistricts * 2, sim.scoreDistrictBoundaries, cluster_centers)
		for _ in range(numSims):
			simAnn.update()
		cluster_centers = simAnn.currentPos.reshape((27, 2))
		distances,indexes = spatial.cKDTree(cluster_centers).query(sim.centers) #Voronoi diagram estimation to form districts

		return indexes



if __name__ == "__main__":
	sim = ElectionSim()

	#first run k-means
	kMeansSims = 0
	electionRecords = sim.runKMeansSimuluations(kMeansSims)
	plt.bar(electionRecords.keys(), electionRecords.values(), color ='darkblue', width = 0.8)

	plt.xlabel("No. of Democratic Represenatives Elected")
	plt.ylabel(f"No. of simulations (out of {kMeansSims})")

	#next, create a gerrymandered distrct using simulated annealing
	numSimulatedAnnealingIters = 1500
	ind = sim.runSimulatedAnnealing(np.ones(27) * 0.5, numSimulatedAnnealingIters)
	sim.shapefile['sim district'] = ind.flatten().astype(np.object0)
	voteRatios = sim.getElectionRatios(ind)
	popRatios = sim.getPopulationRatios(ind)
	print(f"vote ratios: {voteRatios}")
	print(f"pop ratios: {popRatios}")

	#plot the raio of votes from simulated annealing
	district_nums = np.linspace(0, sim.numDistricts, num=sim.numDistricts, endpoint=False)
	f, ax = plt.subplots(1)
	ax.bar(district_nums, voteRatios, 0.35, color='r')
	ax.bar(district_nums, 1-voteRatios, 0.35,bottom=voteRatios, color='b')
	ax.set_xlabel('District')
	ax.set_ylabel("% of Vote")
	ax.set_title('Estimated Vote for Simulated Annealing Gerrymandered District Map')
	# ax.set_xticks(district_nums)
	ax.set_yticks(np.arange(0, 1, 0.05))

	f, ax2 = plt.subplots(1)
	ax2.bar(district_nums, popRatios, 0.35, color='k')
	ax2.set_xlabel('District')
	ax2.set_ylabel("% of Population")
	ax2.set_title('Estimated Population per District for Gerrymandered District Map')
	# ax.set_xticks(district_nums)
	ax2.set_yticks(np.arange(0, 1, 0.05))

	#plot the map from simulated annealing
	f, ax3 = plt.subplots(1)
	sim.shapefile['sim district'] = sim.shapefile.to_numpy()[:, 12].astype(str).astype(int).astype(str)
	sim.shapefile.plot(ax=ax3, column='sim district')
	ax3.set_title("Gerrymandered District using Simulated Annealing")

	#plot the raio of votes and population from 2010 district map
	f, ax4 = plt.subplots(1)
	voteRatios = sim.getElectionRatios(None)
	popRatios = sim.getPopulationRatios(None)

	ax4.bar(district_nums, voteRatios, 0.35, color='r')
	ax4.bar(district_nums, 1-voteRatios, 0.35,bottom=voteRatios, color='b')
	ax4.set_xlabel('District')
	ax4.set_ylabel("% of Vote")
	ax4.set_title('Estimated Vote for 2010 Florida District Map')
	# ax4.set_xticks(district_nums)
	ax4.set_yticks(np.arange(0, 1, 0.05))

	f, ax5 = plt.subplots(1)
	ax5.bar(district_nums, popRatios, 0.35, color='k')
	ax5.set_xlabel('District')
	ax5.set_ylabel("% of Population")
	ax5.set_title('Estimated Population per District for 2010 Florida District Map')
	# ax5.set_xticks(district_nums)
	ax5.set_yticks(np.arange(0, 1, 0.05))

	#plot from 2010 district map
	f, ax6 = plt.subplots(1)
	sim.shapefile['cd113'] = sim.shapefile.to_numpy()[:, 2].astype(str).astype(np.intc).astype(str)
	sim.shapefile.plot(ax=ax6, column='cd113')
	ax6.set_title("2010 Florida House Districts")

	plt.show()