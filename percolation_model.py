"""
Agent based percolation model of innovation
Nathan Goldschlag
December 10, 2013
Version 1.0
Written in Python 2.7

This python file executes an agent based percolation model that extends the percolation model in Silverberg and Verspagen 2007. The model incorporates patents which can: 1) provide additional resources to innovators to invest in R&D and 2) block other firms from exploring the technology space. 

ABSTRACT: An agent-based computational percolation model is proposed as a method of understanding the innovation process and the effects of patents in incentivizing and stifling innovative search. Traditional economic models of the innovation process and patenting struggle to capture the interdependencies among different types of technologies and the path dependence of innovative search. This research addresses this gap by proposing a model that directly incorporates path dependence, localized search, uncertainty, and interdependence of technological innovations. The model is capable of replicating several empirical observations including the skewed distribution of innovation value as well as the temporal clustering of radical innovations. Simulation results are used to investigate the balance between the incentivizing effects of monopoly privilege on one hand, and the stifling effects of forcing other firms to ‘invent-around’. The results suggest that patents can improve innovative performance when firms have less monopoly power and the technology space is difficult to navigate. With a simpler technology space, patents can also improve innovative performance if monopoly rents are moderate and monopoly power is weak. However, there are large areas of the parameter space for which innovative performance without patents is significantly higher than with patents, primarily when monopoly power is strong and when the technology space is simple.

Silverberg, G. and B. Verspagen. 2007. "Self-Organization of R&D Search in Complext Technology Spaces". Journal of Economic Interaction and Coordination, 2:195-210.

CRs:
    Improve structure of firm data file
"""

## USEFUL LIBRARIES
import random, os, csv
import time
#import sys
#import gc
import numpy as np
import math
import scipy as sp
import matplotlib.pyplot as plt
import pylab as py 
#from images2gif import writeGif # Used for gifs
#from PIL import Image # Used for gifs

t0 = time.clock()

## PARAMETERS
savePlots = False # Whether plots are saved after created
doGifs = False # Whether to save snapshots used to create percolation gifs
gifFreq = 20 # Frequency of snapshots for gifs
saveParams = True # Whether a text file is saved with the parameters from the run
saveTSDataFiles = False # Whether a file is created to hold time series data
saveRunDataFiles = True # Whether a file is created to hold run data
saveInnovDataFiles = True # Whether a file is created to hold innovation sizes
saveFirmDataFiles = False # Whether a csv is written with firm level data from the model
numRuns = 120 # Number of times the model is executed
numSteps = 3000 # Number of steps in each run
maxHeight = 100 # Maximum height allowed for each run, substitute for numSteps
numCol = 50 # Number of columns in the lattice
numRow = 100 # Number of initial rows in the lattice
numFirms = 20 # Number of firms to start
r = 2 # Search radius
probMove = 1 # Probability a firm moves
piDecayRate = 0.01 # Rate at which profits from an innovation decay
resistMu = 4 # Mean of resistance for lognormal dist
resistSigma = 1.5 # Standard deviation of resistance for lognormal dist
baseRnD = 1 # Fixed portion of the firm's budget
monopProfit = 1 # Monopoly profits recieved for a patented innovation each step during its patent life and the initial profit for an innovation prior to decay
path = "C:\\Users\\ngoldschlag\\Documents\\Graduate Classes\\Python Library\\" # path that plots will be saved to, used to clean up gif plots
doPatent = True # Whether the model uses patents
probPatent = 0.05 # Probability a firm will create a patent for an innovation
patentLife = 5 # Number of periods a patent lasts
patentRadius = 1 # Radius a patent covers
randomSeed = 123456789 # Seed used to initialize the random number generator

runDesc = "Test 1.b" # Description of run to be written to the beginning of the parameter file

## SIMULATION TESTS
doSigTest = False # Runs the model across different values for resistSigma 
doDecayTest = True # Runs the model for different values of piDecayRate
doMonoPTest = False # Runs the model for different values of monopProfit
doMuTest = False  # Runs the model for different values of resistMu

## CLASS DEFINITIONS
class Firm(object):
    """
    Attributes: id (numeric), patents (list), position (x,y coordinate), innovations (list), rndBudget (profits from previous step, applied to rnd next step), patent profits (numeric total of profits recieved for its patents), totalProfits (numeric total of profits recieved)
    
    Holds firm objects which perform innovative search in the technology lattice. Firms are initially assigned positions on the lattice and then perform innovative search extracting and expending resources to move upward through the lattice. 
    """
    def __init__(self, id, pos):
        self.id = id
        self.patents = []
        self.position = pos
        self.innovations = []
        self.rndBudget = 0
        self.patentProfits = 0
        self.totalProfits = 0
    
    def getPosition(self):
        """
        Parameters: (Firm object)
        Returns firm's position on the lattice.
        """
        return self.position
    
    def getRow(self):
        """
        Parameters: (Firm object)
        Returns row, the first element of the firm's position.
        """
        return self.position[0]
    
    def getPatents(self):
        """
        Parameters: (Firm object)
        Returns list of the firm's patents, lists contains not only the original focus cell being patented but also any cells covered by the patent radius.
        """
        return self.patents
    
    def getNeighborCols(self):
        """
        Parameters: (Firm object)
        Returns list of neighboring columns within the search radius r around the firm's current position.
        """
        col = 0
        nghbr = []
        while col-r<=r:
            if self.position[1] - r + col > (numCol-1):
                nghbr.append((self.position[1] - r + col - numCol))
            elif self.position[1] - r + col < 0:
                nghbr.append((self.position[1] - r + col + numCol))
            else:
                nghbr.append((self.position[1] - r + col))
            col +=1
        return nghbr

    def getMove(self):
        """
        Parameters: (Firm object)
        Returns a new position the firm will move to by calculating the probability the firm will move to a position on the BPF within the firm's search radius. Firms weight the height of potential cells on the BPF of columns within their search radius, and when patents are used firms also consider the ratio of cells covered by other firms' patents.
        """
        # Create list containing the BPF for columns within the firm's search radius, height of cells on the BPF for those columns is irrelevant
        localBPF = []
        nghbr = self.getNeighborCols()
        for col in nghbr:
            height = 0
            for row in range(len(L)):
                if L[row,col].value ==2:
                    height = row
            if L[height,col].value ==2:
                if doPatent:
                    if L[height,col].patent == False:
                        localBPF.append([height, col])
                    elif (height,col) in self.patents:
                        localBPF.append([height, col])
                else:
                    localBPF.append([height, col])
        # If any of the BPF is returned calculate the probabilities of moving to each column then generate a random number to determine which column the firm moves to
        if localBPF:
            moveExp = []
            if doPatent:
                # Calculate the percent of neighboring positions that are patented by other firms for each position the firm will potentially move to
                perPat = []
                for i in localBPF:
                    numNgbPat = 0
                    neighbors = getNeighbors(i)
                    for ngb in neighbors:
                        if L[tuple(ngb)].patent == True:
                            if ngb not in self.patents:
                                numNgbPat += 1
                    perPat.append(float(numNgbPat)/float(len(neighbors)))
                # Evaluate the height and ratio of patented cells of each cell the firm will potentially move to
                for i in range(len(localBPF)):
                    if localBPF[i][0]-self.position[0] > 10:
                        moveExp.append(round(math.exp(10*(1-perPat[i])),2))
                    else:
                        moveExp.append(round(math.exp((localBPF[i][0]-self.position[0])*(1-perPat[i])),2))
            else:
                # Evaluate the height of each cell the firm will potentially move to
                for i in localBPF:
                    if i[0]-self.position[0] > 10:
                        moveExp.append(round(math.exp(10),2))
                    else:
                        moveExp.append(round(math.exp(i[0]-self.position[0]),2))
            # Generate the probabilities of moving to each cell 
            probMoveSum = sum(moveExp)
            if probMoveSum: # With patents probMoveSum may be zero
                probMove = map(lambda x: x/probMoveSum, moveExp)
                cols = []
                for i in localBPF:
                    cols.append(i[1])
                y = np.random.choice(cols, p=probMove)
                newpos = [x for x in localBPF if x[1]==y][0]
                return newpos
            else:
                return None
        else:
            return None
    
    def getNeighbors(self):
        """
        Parameters: (Firm object)
        Returns list of neighboring cells in a square around the firm's position with radius r. If patents are active patented cells are removed.
        """
        # Returns a list of positions around the firm in square radius of r
        neighbors = []
        pos = self.position
        if pos[0]+r > len(L):
            addRows(L,20)
        for col in range(2*r+1):
            row = 0
            while row <= 2*r: 
                # Bounded vertically
                if pos[0]-r+row >= 0:
                    # Jump east
                    if pos[1]-r+col > (numCol-1):
                        neighbors.append([pos[0]-r+row, pos[1]-r+col - numCol])
                    else:
                        neighbors.append([pos[0]-r+row,pos[1]-r+col])
                row +=1
        neighbors.remove([pos[0],pos[1]])
        if doPatent:
            # Remove any cells that are patented by other firms
            newNeighbors = []
            for i in neighbors:
                if L[tuple(i)].patent == False:
                    newNeighbors.append(i)
                elif i in self.patents:
                    newNeighbors.append(i)
            neighbors = newNeighbors
        return neighbors

    def chainReac(self,rndPos):
        """
        Parameters: (Firm object, cell position)
        Takes given cell position, finds connected cells with a value of 1, flips those cells to value 2 and potentially patents that cell. Cycles through all connected cells with value 1 until no longer connections exist. Once all connections are exhausted innovation sizes are captured as the height difference for each affected column.  
        """
        # Given a position that is changing from 1 to 2, this method analyzes neigboring positions to determine if they should also flip from 1 to 2
        global innovSizeCounter
        BPF = getBPF()
        cons = []
        updated = []
        L[rndPos].updateCon(1)
        for i in L[rndPos].conOnes:
            cons.append(i)
        while cons:
            # Take the first connection in the list and update its list of connected positions with value = 1
            L[cons[0]].updateCon(1)
            # Set the value of that first connection to 2
            L[cons[0]].value = 2
            if doPatent:
                if L[cons[0]].patent == False:
                    self.innovations.append(cons[0])
                    L[cons[0]].innovFirm = self
                    self.genPatent(cons[0])
                elif cons[0] in self.patents:
                    self.innovations.append(cons[0])
                    L[cons[0]].innovFirm = self
                else:
                    L[cons[0]].patentFirm.innovations.append(cons[0])
                    L[cons[0]].innovFirm = L[cons[0]].patentFirm
            else:
                self.innovations.append(cons[0])
                L[cons[0]].innovFirm = self
            # For each of that first connection's neighboring 1s, if its not already in the list of connections to be iterated through, and it hasn't been updated already, then add it to the list of connections to be processed
            for i in L[cons[0]].conOnes:
                if i not in cons:
                    if i not in updated:
                        cons.append(i)
            updated.append(cons[0])
            cons.remove(cons[0])
        if updated:
            # Determine the size of each innovation (jump in rows) for the columns affected by the chain reaction (NOT limited by radius as in S&V 2007)
            innov = []
            # Populate the list of innovations with the increase in row between the BPF and the positions updated by the chain reaction
            cols = list(range(numCol))
            for i in cols:
                if [x for x in updated if x[1] == i]: # Updated cell for that col
                    if [x for x in BPF if x[1] == i]: # BPF for that col
                        # If the updated positions are below the BPF append zero
                        if max([x for x in updated if x[1] == i])[0] - [x for x in BPF if x[1] == i][0][0] > 0:
                            innov.append(max([x for x in updated if x[1] == i])[0] - [x for x in BPF if x[1] == i][0][0])
                    # If there is no BPF for that col, meaning there are no innovs yet for that col, then increase in row is just the innovation's row
                    else:
                        innov.append(max([x for x in updated if x[1] == i])[0])
            for i in innov:
                innovSizes[innovSizeCounter] = run, step, i ##BUG## Reading run/step right?
                innovSizeCounter += 1

    def genPatent(self,rndPos):
        """
        Parameters: (Firm object, cell position)
        Takes given cell position, determines if firm patents cell, and if so patents that and all other cells in patent radius.  
        """
        # Firm potentially creates patent for a cell and the surrounding cells
        ## Allows for partial radius of patent, no overlap but odd shapped patent radius
        if random.random() < probPatent:
            global totalPatents 
            totalPatents += 1
            neighbors = getPatRadius(rndPos)
            neighbors.append(rndPos)
            for i in neighbors:
                if L[tuple(i)].patent == False and L[tuple(i)].patentFirm == 0: # patentFirm == 0 ensures that the cell has never been patented before
                    L[tuple(i)].patent = True
                    L[tuple(i)].patentFirm = self
                    L[tuple(i)].patentLife = patentLife
                    L[tuple(i)].profitDuration = 0 # Reset profit decay counter, patented cell may have been innov by another firm, increasing the decay 
                    self.patents.append(i)

    def updateProfits(self):
        """
        Parameters: (Firm object)
        Updates profits for a firm. If profits are active special processing is done to separate profits accruing to other firms if an innovation is patented by a different firm. 
        """
        # Updates the firm's profits based on its list of innovations
        if doPatent:
            firmInnovsPat = [x for x in self.innovations if L[x].patent == True]
            firmInnovsNoPat = [x for x in self.innovations if L[x].patent == False]
            for i in firmInnovsNoPat:
                if L[i].profitDuration < len(profitStream):
                    newProfits = profitStream[L[i].profitDuration] # profitStream holds an ordered list of the decaying profit valuations
                    self.rndBudget += newProfits
                    self.totalProfits += newProfits
                    L[i].profitDuration += 1
            for i in firmInnovsPat:
                # Firms can patent the innovations of other firms
                if L[i].patentFirm != self:
                    if L[i].profitDuration < patentLife:
                        newProfits = monopProfit
                        L[i].patentFirm.rndBudget += newProfits
                        L[i].patentFirm.patentProfits += newProfits
                        L[i].patentFirm.totalProfits += newProfits
                        L[i].profitDuration += 1
                    elif L[i].profitDuration < patentLife + len(profitStream):
                        newProfits = profitStream[L[i].profitDuration - patentLife] # profitStream holds an ordered list of the decaying profit valuations
                        L[i].patentFirm.rndBudget += newProfits
                        L[i].patentFirm.totalProfits += newProfits
                        L[i].profitDuration += 1
                else:
                    if L[i].profitDuration < patentLife:
                        newProfits = monopProfit
                        self.rndBudget += newProfits
                        self.totalProfits += newProfits
                        self.patentProfits += newProfits
                        L[i].profitDuration += 1                        
                    elif L[i].profitDuration < patentLife + len(profitStream):
                        newProfits = profitStream[L[i].profitDuration - patentLife] # profitStream holds an ordered list of the decaying profit valuations
                        self.rndBudget += newProfits
                        self.totalProfits += newProfits
                        L[i].profitDuration += 1
        else: 
            for i in self.innovations:
                if L[i].profitDuration < len(profitStream):
                    newProfits = profitStream[L[i].profitDuration] # profitStream holds an ordered list of the decaying profit valuations
                    self.rndBudget += newProfits
                    self.totalProfits += newProfits
                    L[i].profitDuration += 1
        
class Point(object):
    """
    Attributes: value (0, 1, or 2), innovFirm (0 or firm object that discovered cell), resistance (numeric), position (row, column), conOnes (list of connected cells with value=1), conTwos (list of connected cells with value=2), profitDuration (counter for profitStream), patent (True/False), patentFirm  (0 or firm object that patented cell), patentLife (counter for remaining life of patent)
    
    Point objects are cells in the lattice. Each point object holds data for a given position on the lattice. 
    """
    def __init__(self):
        self.value = 0
        self.innovFirm = 0
        self.resistance = random.lognormvariate(resistMu,resistSigma)
        self.position = []
        self.conOnes = []
        self.conTwos = []
        self.profitDuration = 0
        self.patent = False
        self.patentFirm = 0
        self.patentLife = 0
    
    def updateCon(self,val):
        """
        Parameters: (Point Object, val=(1 or 2))
        Given a cell it updates the list of connected cells with the given value (1 or 2), looking north, south, east, and west.
        """
        # Check N, S, E, W and update list of connected points
        # North
        if L[(self.position[0]+1,self.position[1])].value == val:
            if val == 1:
                if (self.position[0]+1,self.position[1]) not in self.conOnes:
                    self.conOnes.append((self.position[0]+1,self.position[1]))
            if val == 2:
                if (self.position[0]+1,self.position[1]) not in self.conTwos:
                    self.conTwos.append((self.position[0]+1,self.position[1]))
        # West
        if L[(self.position[0],self.position[1]-1)].value == val:
            if val == 1:
                if (self.position[0],self.position[1]-1) not in self.conOnes:
                    self.conOnes.append((self.position[0],self.position[1]-1))
            if val == 2:
                if (self.position[0],self.position[1]-1) not in self.conTwos:
                    self.conTwos.append((self.position[0],self.position[1]-1))
        # Special treatement for E on E boundary
        if self.position[1] == (numCol-1):
            if L[(self.position[0],0)].value == val:
                if val == 1:
                    if (self.position[0],0) not in self.conOnes:
                        self.conOnes.append((self.position[0],0))
                if val == 2:
                    if (self.position[0],0) not in self.conTwos:
                        self.conTwos.append((self.position[0],0))
        else:
            if L[(self.position[0],self.position[1]+1)].value == val:
                if val == 1:
                    if (self.position[0],self.position[1]+1) not in self.conOnes:
                        self.conOnes.append((self.position[0],self.position[1]+1))
                if val == 2:
                    if (self.position[0],self.position[1]+1) not in self.conTwos:
                        self.conTwos.append((self.position[0],self.position[1]+1))
        # Special treatement for S on S boundary
        if self.position[0] != 0:
            if L[(self.position[0]-1,self.position[1])].value == val:
                if val == 1:
                    if (self.position[0]-1,self.position[1]) not in self.conOnes:
                        self.conOnes.append((self.position[0]-1,self.position[1]))
                if val == 2:
                    if (self.position[0]-1,self.position[1]) not in self.conTwos:
                        self.conTwos.append((self.position[0]-1,self.position[1]))

## FUNCTION DEFINITIONS

def genLattice():
    """
    Parameters: ()
    Creates the lattice representing the technology space.
    """
    # Creates a lattice that will act as the technology space
    # First create a list of point objects as the first row, then create the array
    row = []
    for i in range(numCol):
        row.append(Point())
    M = np.array([row])
    # Then loop through and create the remaining rows of point objects, appending each to the array
    for i in range(numRow-1):
        row = []
        for i in range(numCol):
            row.append(Point())
        M = np.append(M, [row], axis=0)
    # Write the array position to each of the point objects in the array
    for i, x in np.ndenumerate(M):
        M[i].position = i
    return M

def addRows(M,n):
    """
    Parameters: (Lattice, n=Number of cells to add)
    Given a lattice, adds a given number of rows. Returns the modified lattice. 
    """
    # Returns modified lattice with n new rows appended to the end
    # Create each row, append it to the matrix, then write the position again to all points
    for i in range(n):
        row = []
        for i in range(numCol):
            row.append(Point())
            row[-1].position = (len(M),i)
        M = np.append(M, [row], axis=0)
    return M

def getBPF():
    """
    Parameters: ()
    Returns the BPF of the lattice, L.
    """
    # Returns a list of the highest discovered and viable (value=2) position across all columns in the lattice L
    BPF = []
    for col in range(len(L[0])):
        height = 0
        for row in range(len(L)):
            if L[row,col].value ==2:
                height = row
        if L[height,col].value ==2:
            BPF.append([height, col])
    return BPF

def getNeighbors(pos):
    """
    Parameters: (position)
    Given a position, returns the neighboring cells in radius r. Patented cells are not removed, but the given cell is not returned in the list. 
    """
    # Returns a list of positions around the firm in square radius of r
    neighbors = []
    if pos[0]+r > len(L):
        addRows(L,20)
    for col in range(2*r+1):
        row = 0
        while row <= 2*r: 
            # Bounded vertically
            if pos[0]-r+row >= 0:
                # Jump east
                if pos[1]-r+col > (numCol-1):
                    neighbors.append([pos[0]-r+row, pos[1]-r+col - numCol])
                else:
                    neighbors.append([pos[0]-r+row,pos[1]-r+col])
            row +=1
    neighbors.remove([pos[0],pos[1]])
    return neighbors

def getPatRadius(pos):
    """
    Parameters: (position)
    Given a position, returns the neighboring cells within the patent radius. Patented cells are not removed. The given cell is NOT returned in the list. 
    """
    # Returns a list of positions around the given position in square radius of r
    neighbors = []
    if pos[0]+patentRadius > len(L):
        addRows(L,20)
    for col in range(2*patentRadius+1):
        row = 0
        while row <= 2*patentRadius: 
            # Bounded vertically
            if pos[0]-patentRadius+row >= 0:
                # Jump east
                if pos[1]-patentRadius+col > (numCol-1):
                    neighbors.append([pos[0]-patentRadius+row, pos[1]-patentRadius+col - numCol])
                elif pos[1]-patentRadius+col < 0:
                    neighbors.append([pos[0]-patentRadius+row,numCol + pos[1]-patentRadius+col])
                else:
                    neighbors.append([pos[0]-patentRadius+row,pos[1]-patentRadius+col])
            row +=1
    if [pos[0],pos[1]] in neighbors:
        neighbors.remove([pos[0],pos[1]])
    return neighbors

def getCluster():
    """
    Parameters: ()
    Calculates and returns the cluster index for firms the lattice. 
    """
    # Returns an index that captures how clustered firms are across the technology space
    firmPositions = map(lambda Firm: Firm.getPosition(), firms)
    colByFirm = []
    for i in range(numCol):
        colByFirm.append(math.pow(len([x for x in firmPositions if x[1] == i]),2))
    return sum(colByFirm)

def getPerBPFPat():
    """
    Parameters: ()
    Calculates and returns the percent of the space under the BPF that is patented. 
    """
    # Returns a percent of the cells under the BPF that are patented
    BPF = getBPF()
    numPats = 0
    # For each column in the lattice count the number of patented and nonpatented cells
    for col in range(numCol):
        BPFPos = [x for x in BPF if x[1] == col]
        # If there is a position on the BPF for that column then count the number of patented cells in that column
        if BPFPos:
            for i in range(BPFPos[0][0]):
                if L[(i,col)].patent == True:
                    numPats += 1
    totalBPF = sum([x[0] for x in BPF])
    if totalBPF:
        return float(numPats)/float(totalBPF)
    else:
        return 0

## PLOTTERS
def plotValue(step):
   # Plots snapshot of value for gif
    K = np.zeros((len(L),numCol))
    for i, x in np.ndenumerate(L):
        K[i] = L[i].value
    firmPositions = map(lambda Firm: Firm.getPosition(), firms)
    for i in firmPositions:
        K[i] = 1.5
    py.figure(figsize=(8,8))
    py.imshow(K, origin='lower', interpolation='nearest')
    yticks = range(0,len(L),(len(L)/15))
    xticks = range(0,len(L[0]),(len(L[0])/15))
    py.xticks(xticks)
    py.yticks(yticks)
    py.grid(color='k',linewidth=2)
    py.title("Value Plot")
    fileName = "plotValue_{0}.png".format(float(step)/float(numSteps))
    plt.savefig(fileName)

def plotBPF(step):
    # Plots snapshot of BPF for gif
    K = np.zeros((len(L),numCol))
    BPF = getBPF()
    for i, x in np.ndenumerate(K):
        if [x for x in BPF if x[1]==i[1]]:
            if [x for x in BPF if x[1]==i[1]][0][0] >= i[0]:
                K[i] = 2
    py.figure(figsize=(8,8))
    py.imshow(K, origin='lower', interpolation='nearest')
    yticks = range(0,len(L),(len(L)/15))
    xticks = range(0,len(L[0]),(len(L[0])/15))
    py.xticks(xticks)
    py.yticks(yticks)
    py.grid(color='k',linewidth=2)
    py.title("BPF Plot")
    fileName = "plotBPF_{0}.png".format(float(step)/float(numSteps))
    plt.savefig(fileName)

def plotValueAndResist(step):
    # Plots snapshot of value and resistance for gif
    K = np.zeros((len(L),numCol))
    for i, x in np.ndenumerate(L):
        K[i] = L[i].value
    firmPositions = map(lambda Firm: Firm.getPosition(), firms)
    for i in firmPositions:
        K[i] = 1.5
    py.figure(figsize=(30,30))
    py.subplot(121)
    py.imshow(K, origin='lower', interpolation='nearest')
    py.colorbar()
    yticks = range(0,len(L),(len(L)/15))
    xticks = range(0,len(L[0]),(len(L[0])/15))
    py.xticks(xticks)   
    py.yticks(yticks)
    py.grid(color='k',linewidth=2)
    py.title("Value Plot")

    K = np.zeros((len(L),numCol))
    for i, x in np.ndenumerate(L):
        K[i] = L[i].resistance
    py.subplot(122)
    py.imshow(K, origin='lower', interpolation='nearest')
    py.colorbar()
    yticks = range(0,len(L),(len(L)/15))
    xticks = range(0,len(L[0]),(len(L[0])/15))
    py.xticks(xticks)
    py.yticks(yticks)
    py.grid(color='k',linewidth=2)
    py.title("Resistance Plot")
    fileName = "plotVandR_{0}.png".format(float(step)/float(numSteps))
    plt.savefig(fileName)

def plotPats(step):
    # Plots snapshot of value and patents for gif
    K = np.zeros((len(L),numCol))
    for i, x in np.ndenumerate(L):
        K[i] = L[i].value
        if L[i].patent:
            K[i] = 0.5
    firmPositions = map(lambda Firm: Firm.getPosition(), firms)
    for i in firmPositions:
        K[i] = 1.5
    py.figure(figsize=(8,8))
    py.imshow(K, origin='lower', interpolation='nearest')
    yticks = range(0,len(L),(len(L)/15))
    xticks = range(0,len(L[0]),(len(L[0])/15))
    py.xticks(xticks)
    py.yticks(yticks)
    py.grid(color='k',linewidth=2)
    py.title("Value and Patents Plot")
    fileName = "plotPat_{0}.png".format(float(step)/float(numSteps))
    plt.savefig(fileName)

def genGifs(type,run):
    # Creates gifs from images saved via plotValue(), plotBPF(), and plotValueAndResist(); images2gif.py file must be available in the default directory
    file_names = sorted( (fn for fn in os.listdir('.') if fn.startswith("plot{0}".format(type))))    
    images = [Image.open(fn) for fn in file_names]    
    size = (500,500)
    for im in images:
        im.thumbnail(size, Image.ANTIALIAS)    
    print writeGif.__doc__    
    filename = "{0}_Run{1}_Percol_{2}_Gif.GIF".format(filetime,run,type)
    writeGif(filename, images, duration=0.5)

def cleanGifPlots(type):
    # Deletes the files created via plotValue(), plotBPF(), and plotValueAndResist()
    file_names = sorted( (fn for fn in os.listdir('.') if fn.startswith("plot{0}".format(type))))    
    for i in file_names:
        if os.path.isfile(path+i):
            os.remove(path+i)

## SIMULATION 
# Define the file time used in plotters and create the files used to save data
filetime = time.strftime("%Y-%m-%d %H%M_%S")
if doPatent:
    dataFileName1 = filetime + "_PercolTSData_Pat.csv"
    dataFileName2 = filetime + "_Innov_Size_Data_Pat.csv"
    dataFileName3 = filetime + "_PercolRunData_Pat.csv"
    paramFileName = filetime + '_PercolParams_Pat.txt'
else:
    dataFileName1 = filetime + "_PercolTSData_NoPat.csv"
    dataFileName2 = filetime + "_Innov_Size_Data_NoPat.csv"
    dataFileName3 = filetime + "_PercolRunData_NoPat.csv"
    paramFileName = filetime + '_PercolParams_NoPat.txt'

# Write run description to the parameter file
if saveParams:
    writer = open(paramFileName, 'a')
    writer.write(runDesc + '\n' + '\n')
    
# Initialize the random number generator with the parameterized seed
random.seed(randomSeed)

# Define profit stream list
profitStream = [monopProfit]
while profitStream[-1] > 0.01:
    profitStream.append(round(monopProfit * math.exp(-piDecayRate*len(profitStream)),3))
print "profitStream", profitStream

# Simulation tests, lists of parameter values to be used during specific simulation tests
if doSigTest:
    testResistSigma = [0.2, 0.2, 0.2, 0.4, 0.4, 0.4, 0.6, 0.6, 0.6, 0.8, 0.8, 0.8, 1.0, 1.0, 1.0, 1.2, 1.2, 1.2, 1.4, 1.4, 1.4, 1.6, 1.6, 1.6, 1.8, 1.8, 1.8, 2.0, 2.0, 2.0]
if doDecayTest:
    testPiDecay = [0.005, 0.005, 0.005, 0.01, 0.01, 0.01, 0.015, 0.015, 0.015, 0.02, 0.02, 0.02, 0.025, 0.025, 0.025, 0.03, 0.03, 0.03, 0.035, 0.035, 0.035, 0.04, 0.04, 0.04, 0.045, 0.045, 0.045, 0.05, 0.05, 0.05]*4
    #testPiDecay = [0.01, 0.01, 0.01, 0.02, 0.02, 0.02, 0.03, 0.03, 0.03, 0.04, 0.04, 0.04, 0.05, 0.05, 0.05, 0.06, 0.06, 0.06, 0.07, 0.07, 0.07, 0.08, 0.08, 0.08, 0.09, 0.09, 0.09, 0.1, 0.1, 0.1]*4
if doMonoPTest:
    testPiDecay = [0.01, 0.01, 0.01, 0.02, 0.02, 0.02, 0.03, 0.03, 0.03, 0.04, 0.04, 0.04, 0.05, 0.05, 0.05, 0.06, 0.06, 0.06, 0.07, 0.07, 0.07, 0.08, 0.08, 0.08, 0.09, 0.09, 0.09, 0.1, 0.1, 0.1]*6
if doMuTest:
    testResistMu = [2, 2, 2, 2.2, 2.2, 2.2, 2.4, 2.4, 2.4, 2.6, 2.6, 2.6, 2.8, 2.8, 2.8, 3.0, 3.0, 3.0, 3.2, 3.2, 3.2, 3.4, 3.4, 3.4, 3.6, 3.6, 3.6, 3.8, 3.8, 3.8]*4

innovSizes = {} # Innovation sizes held in a dictionary with attributes run, step, and innovation size
innovSizeCounter = 0 # Counter for keys in the dictionary

# Run the simulation numRuns times
for run in range(numRuns):
    print "Run", run
    
    # Modify simulation parameters according to the test being performed, if any
    if doSigTest:
        resistSigma = testResistSigma[run]
        print "resistSigma", resistSigma
    if doDecayTest:
        piDecayRate = testPiDecay[run]
        print "piDecayRate", piDecayRate
        # Define profit stream
        profitStream = [monopProfit]
        while profitStream[-1] > 0.01:
            profitStream.append(round(monopProfit * math.exp(-piDecayRate*len(profitStream)),3))
        if run < 29:
            patentLife = 5
        if run > 29:
            patentLife = 100
        if run > 59:
            patentLife = 200
        if run > 89:
            doPatent = False
        print "patentLife", patentLife
        print "doPatent", doPatent
    if doMuTest:
        resistMu = testResistMu[run]
        print "resistMu", resistMu
        if run < 29:
            patentLife = 5
        if run > 29:
            patentLife = 100
        if run > 59:
            patentLife = 200
        if run > 89:
            doPatent = False
        print "patentLife", patentLife
        print "doPatent", doPatent
    if doMonoPTest:
        piDecayRate = testPiDecay[run]
        print "piDecayRate", piDecayRate
        # Define profit stream
        profitStream = [monopProfit]
        while profitStream[-1] > 0.01:
            profitStream.append(round(monopProfit * math.exp(-piDecayRate*len(profitStream)),3))
        if run < 29:
            monopProfit = 1
        if run > 29:
            monopProfit = 3
        if run > 59:
            monopProfit = 5
        if run > 89:
            monopProfit = 1
            doPatent = False
        if run > 119:
            monopProfit = 3
            doPatent = False
        if run > 149:
            monopProfit = 5
            doPatent = False
        print "monopProfit", monopProfit
        print "doPatent", doPatent
        
    # Define lists to hold simulation data
    BPFChangeTS = []
    clusterIndexTS = []
    meanBPFTS = []
    maxBPFTS = [0] # Starts with 0 so that while loop has a begining value to compare to maxHeight
    stdProfitTS = []
    if doPatent:
        numPatentsTS = []
        perBPFPatTS = []
        totalPatents = 0
        totalPatentsTS = []
    
    # Create lattice; must be named "L" since it is referenced as "L" throughout the methods
    L = genLattice()
    
    # Capture statistics about initial resistance values for the initial lattice, written to parameter file
    resistValues = []
    for i, x in np.ndenumerate(L):
        resistValues.append(L[i].resistance)
    maxResist = max(resistValues)
    meanResist = np.mean(resistValues)
    minResist = min(resistValues)
        
    # Create firms
    firms = []
    for id in range(numFirms):
        # If the number of firms is equal to the number of columns assign them sequentially to each column
        if numFirms == numCol:
            pos = (0,id)
            firms.append(Firm(id,pos))
        # Otherwise randomly assign the firms to columns
        else:
            col = random.randint(0, (numCol-1))
            pos = (0,col)
            firms.append(Firm(id, pos))
    
    # Turn each position where the firms are instantiated to discovered
    firmPositions = map(lambda Firm: Firm.getPosition(), firms)
    for i in firmPositions:
        L[i].value = 2
        L[i].resistance = 0
    
    # Create the firm data file for each step for the run
    if saveFirmDataFiles:
        headers = []
        headers.append('steps')
        for frm in range(numFirms):
            headers.append('firm{0}totalProfits'.format(frm))
            headers.append('firm{0}patentProfits'.format(frm))
            headers.append('firm{0}height'.format(frm))
            headers.append('firm{0}numInnovs'.format(frm))
            headers.append('firm{0}numPats'.format(frm))
        if doPatent:
            firmFileName = filetime + "_Firm_Data_Run{0}_Pat.csv".format(run)
        else:
            firmFileName = filetime + "_Firm_Data_Run{0}_NoPat.csv".format(run)
        inputFile = open(firmFileName, 'wb')
        writer = csv.writer(inputFile, lineterminator='\n')
        writer.writerow(headers)
        inputFile.close()
    
    # Start step counter
    step = -1
    # Run the simulation steps
    while maxBPFTS[-1] < maxHeight:
        step += 1
        # Print info to console while model is running
        if step%(numSteps/10)==0:
            print "Step:", step
            print "maxBPF", maxBPFTS[-1]
            t = time.clock() - t0
            print "Minutes Lapsed:", (t/60)
        # Remove that first element from when maxBPFTS was initialized
        if step==0:
            del maxBPFTS[0]
        # Capture plots for gifs
        if doGifs:
            if step%(numSteps/gifFreq)==0:
                plotValue(step)
                plotBPF(step)
                plotValueAndResist(step)
                plotPats(step)
        
        # Create snapshot of BPF for comparison after round of R&D
        preBPF = getBPF()
        
        # Shuffle the list of firms, which is then used as the activation order
        random.shuffle(firms)
       
        for i in range(len(firms)):
            
            # Set active firm
            firm = firms[i]
            
            # Firm chooses new position (potentially the same position)
            if random.random() < probMove:
                newPosition = firm.getMove()
                if newPosition:
                    firm.position = newPosition
        
            # Firm selects position to perform R&D on
            # With patents there is a chance that there are no eligible positions near the firm to perform R&D, in which case rndPos is set to zero and the firm does not perform R&D that step
            gN = firm.getNeighbors()
            if gN:
                rndPos = tuple(random.choice(gN))
            else:
                rndPos = 0
                
            # If there is a position returned that the firm can perform R&D on firm performs R&D
            if rndPos:
                # Firm performs R&D, reducing the resistance of the cell
                if L[rndPos].resistance > 0:
                    L[rndPos].resistance =  L[rndPos].resistance - random.random()*(baseRnD + firm.rndBudget)
                    
                    # Reset rndBudget to zero since all of the rndBudget is applied
                    firm.rndBudget = 0
                    # If R&D effort makes the resistance value <= 0 then set value to 1 (discovered but not yet viable)
                    if L[rndPos].resistance <= 0:
                        # Cell value is set to 2 automatically if it is on the baseline and firm potentially creates patent
                        if rndPos[0]==0:
                            L[rndPos].value = 2
                            firm.innovations.append(rndPos)
                            L[rndPos].innovFirm = firm
                            if doPatent:
                                firm.genPatent(rndPos)
                            
                        # If newly discovered position is connected to the baseline by an unbroken chain of positions with value = 2 (discovered and viable), then set value to 2
                        else:
                            L[rndPos].value = 1
                            L[rndPos].updateCon(2)
                            if L[rndPos].conTwos:
                                L[rndPos].value = 2
                                firm.innovations.append(rndPos)
                                L[rndPos].innovFirm = firm
                                if doPatent:
                                    firm.genPatent(rndPos)
                                firm.chainReac(rndPos)
                            # Firm can still attempt to patent an invention that is not yet viable
                            else:
                                if doPatent:
                                    firm.genPatent(rndPos)
            
            # Check highest firm to see if new rows must be added
            if len(L) - max(map(lambda Firm: Firm.getRow(), firms)) < 30:
                L = addRows(L,30)
        
        # Calculate profits for each firm
        for frm in firms:
            frm.updateProfits()
        
        # Reduce the life of existing patents, and remove patents that are no longer valid
        latticePatents = [x for i, x in np.ndenumerate(L) if x.patent] 
        for pats in latticePatents:
            pats.patentLife -= 1
            if pats.patentLife == 0:
                pats.patent = False
                
        # Compare pre and post BPFs and capture data
        postBPF = getBPF()
        BPFDiff = []
        for i in postBPF:
            # If there is an entry in the preBPF for that column, append difference
            if [x for x in preBPF if x[1] == i[1]]:
                BPFDiff.append(i[0] - [x for x in preBPF if x[1] == i[1]][0][0])
            # If it is a new position on BPF starting at row = zero append one
            elif i[0] == 0:
                BPFDiff.append(1)
            # If it is a new position on BPF starting at row > zero append row + 1
            else:
                BPFDiff.append(i[0]+1)
        # Append time series lists
        BPFChangeTS.append(sum(BPFDiff))
        clusterIndexTS.append(getCluster())
        meanBPFTS.append(round(np.mean([x[0] for x in postBPF]),2))
        maxBPFTS.append(max([x[0] for x in postBPF]))
        stdProfitTS.append(np.std([x.totalProfits for x in firms]))
        if doPatent:
            numPatentsTS.append(sum([len(x.patents) for x in firms]))
            perBPFPatTS.append(getPerBPFPat())
            totalPatentsTS.append(totalPatents)

        # Write out firm level data for each step 
        if saveFirmDataFiles:
            # Write subsequent runs to the first file 
            inputFile = open(firmFileName, 'rb')
            outputFile = open("outputFile.csv", 'wb')
            reader = csv.reader(inputFile, lineterminator='\n')
            writer = csv.writer(outputFile, lineterminator='\n')
            # Write in all existing rows
            for row in range(step+1):
                newRow = reader.next()
                writer.writerow(newRow)
            # Write in new row
            newRow = []
            newRow.append(step)
            for frms in range(numFirms): 
                # Number of columns = number of firms * number of types of data
                firmWrite = [x for x in firms if x.id == frms][0]
                newRow.append(firmWrite.totalProfits) 
                newRow.append(firmWrite.patentProfits)
                newRow.append(firmWrite.position[0])
                newRow.append(len(firmWrite.innovations))
                if firmWrite.patents:
                    newRow.append(len([x for x in firmWrite.patents if L[tuple(x)].patent == True]))
                else:
                    newRow.append(0)
            writer.writerow(newRow)
            inputFile.close()
            outputFile.close()
            os.remove(path+firmFileName)
            os.rename("outputFile.csv", firmFileName)
    
    # Calculate log-log coefficient of innovation sizes for run
    # Filter the dictionary for the innovSizes for the run
    innovSizesRun = []
    innovKeys = [k for k, v in innovSizes.iteritems() if v[0] == run]
    if innovKeys:
        for i in innovKeys:
            innovSizesRun.append(innovSizes[i][2])
    if innovSizesRun:
        inputBins = [0, 1.5]
        maxInnovSizes = max(innovSizesRun)
        while inputBins[-1] < maxInnovSizes:
            inputBins.append(inputBins[-1]*1.5)
        hist, bins = np.histogram(innovSizesRun, inputBins)
        inputBins.remove(0)
        
        lbins = list(np.log(inputBins))
        lhist = list(np.log(hist))
        for i in range(len(lhist)):
            if lhist[i] == float('-inf'):
                lhist[i] = 0
        
        coeff = np.polyfit(lbins,lhist,1)
        polynomial = np.poly1d(coeff)
        ys = polynomial(lbins)
        logCoef = round(polynomial[1],2)
    else:
        logCoef = None
    
    # Plot gifs for the run
    if doGifs:
        genGifs("Value",run)
        cleanGifPlots("Value")
        genGifs("BPF",run)
        cleanGifPlots("BPF")
        genGifs("VandR",run)
        cleanGifPlots("VandR")
        genGifs("Pat",run)
        cleanGifPlots("Pat")
        
    # Store time series data for each run
    if saveTSDataFiles:
        # If its the first run create the initial headers and rows
        if run == 0:
            # Write initial run to the first file
            headers = ['run', 'step', 'resistMu', 'resistSigma', 'doPatent', 'piDecayRate', 'patentLife', 'patentRadius', 'BPFChange', 'clusterIndex', 'meanBPF', 'maxBPF', 'stdProfit', 'perBPFPat', 'totalPatents']
            inputFile = open(dataFileName1, 'wb')
            steps = list(range(step))
            writer = csv.writer(inputFile, lineterminator='\n')
            writer.writerow(headers)
            if doPatent:
                for i in range(len(steps)):
                    writer.writerow([run, i, resistMu, resistSigma, doPatent, piDecayRate, patentLife, patentRadius, BPFChangeTS[i], clusterIndexTS[i], meanBPFTS[i], maxBPFTS[i], stdProfitTS[i], perBPFPatTS[i], totalPatentsTS[i]])
            else:
                for i in range(len(steps)):
                    writer.writerow([run, i, resistMu, resistSigma, doPatent, piDecayRate, patentLife, patentRadius, BPFChangeTS[i], clusterIndexTS[i], meanBPFTS[i], maxBPFTS[i], stdProfitTS[i]])
            inputFile.close()
        else:
            # Write subsequent runs to the first file 
            inputFile = open(dataFileName1, 'rb')
            outputFile = open("outputFile.csv", 'wb')
            reader = csv.reader(inputFile, lineterminator='\n')
            writer = csv.writer(outputFile, lineterminator='\n')
            row_count = sum(1 for row in csv.reader( open(dataFileName1)))
            for i in range(row_count):
                row = reader.next()
                writer.writerow(row)
            if doPatent:
                for i in range(step):
                    writer.writerow([run, i, resistMu, resistSigma, doPatent, piDecayRate, patentLife, patentRadius, BPFChangeTS[i], clusterIndexTS[i], meanBPFTS[i], maxBPFTS[i], stdProfitTS[i], perBPFPatTS[i], totalPatentsTS[i]])
            else:
                for i in range(step):
                    writer.writerow([run, i, resistMu, resistSigma, doPatent, piDecayRate, patentLife, patentRadius, BPFChangeTS[i], clusterIndexTS[i], meanBPFTS[i], maxBPFTS[i], stdProfitTS[i]])
            inputFile.close()
            outputFile.close()
            os.remove(path+dataFileName1)
            os.rename("outputFile.csv", dataFileName1)
                 
    # Store run data for each run
    if saveRunDataFiles:
        # If its the first run create the initial headers and rows
        if run == 0:
            # Write initial run to the first file
            headers = ['run', 'resistMu', 'resistSigma', 'monopProfit', 'doPatent', 'piDecayRate', 'patentLife', 'patentRadius', 'BPFChange', 'clusterIndex', 'meanBPF', 'maxBPF', 'steps', 'stdProfit', 'logCoef', 'perBPFPat', 'totalPatents']
            inputFile = open(dataFileName3, 'wb')
            writer = csv.writer(inputFile, lineterminator='\n')
            writer.writerow(headers)
            if doPatent:
                writer.writerow([run, resistMu, resistSigma, monopProfit, doPatent, piDecayRate, patentLife, patentRadius, np.mean(BPFChangeTS), np.mean(clusterIndexTS), meanBPFTS[-1], maxBPFTS[-1], step, np.mean(stdProfitTS), logCoef, np.mean(perBPFPatTS), totalPatents])
            else:
                writer.writerow([run, resistMu, resistSigma, monopProfit, doPatent, piDecayRate, patentLife, patentRadius, np.mean(BPFChangeTS), np.mean(clusterIndexTS), meanBPFTS[-1], maxBPFTS[-1], step, np.mean(stdProfitTS), logCoef])
            inputFile.close()
        else:
            # Write subsequent runs to the first file 
            inputFile = open(dataFileName3, 'rb')
            outputFile = open("outputFile.csv", 'wb')
            reader = csv.reader(inputFile, lineterminator='\n')
            writer = csv.writer(outputFile, lineterminator='\n')
            for i in range(run+1):
                row = reader.next()
                writer.writerow(row)
            if doPatent:
                writer.writerow([run, resistMu, resistSigma, monopProfit, doPatent, piDecayRate, patentLife, patentRadius, np.mean(BPFChangeTS), np.mean(clusterIndexTS), meanBPFTS[-1], maxBPFTS[-1], step, np.mean(stdProfitTS), logCoef, np.mean(perBPFPatTS), totalPatents])
            else:
                writer.writerow([run, resistMu, resistSigma, monopProfit, doPatent, piDecayRate, patentLife, patentRadius, np.mean(BPFChangeTS), np.mean(clusterIndexTS), meanBPFTS[-1], maxBPFTS[-1], step, np.mean(stdProfitTS), logCoef])
            inputFile.close()
            outputFile.close()
            os.remove(path+dataFileName3)
            os.rename("outputFile.csv", dataFileName3)

    # Create a file to save the parameters used, subsequent runs appended to the same text file
    if saveParams:
        writer = open(paramFileName, 'a')
        writer.write('Whether patents are used, doPatent' + '\t' + str(doPatent) + '\n')
        writer.write('Probablity a firm will generate a patent for an innovation, probPatent' + '\t' + str(probPatent) + '\n')
        writer.write('How many steps patents last, patentLife' + '\t' + str(patentLife) + '\n')
        writer.write('The radius of patents, patentRadius' + '\t' + str(patentRadius) + '\n')
        writer.write('Random seed' + '\t' + str(randomSeed) + '\n')
        writer.write('Whether the test of log normal std dev was run, doSigTest' + '\t' + str(doSigTest) + '\n')
        writer.write('Whether the test of profit decay was run, doDecayTest' + '\t' + str(doDecayTest) + '\n')
        writer.write('Number of times the model is run, numRuns' + '\t' + str(numRuns) + '\n')
        writer.write('Maximum height allowed for each run, maxHeight' + '\t' + str(maxHeight) + '\n')
        writer.write('Number of columns, numCol' + '\t' + str(numCol) + '\n')
        writer.write('Number of initial rows, numRow' + '\t' + str(numRow) + '\n')
        writer.write('Mean of lognormal distribution of resistance, resistMu' + '\t' + str(resistMu) + '\n')
        writer.write('Std Dev of lognormal distribution of resistance, resistSigma' + '\t' + str(resistSigma) + '\n')
        writer.write('Mean, min, and max of initial resistance values' + '\t' + str(round(meanResist,2)) + ', ' + str(round(minResist,2)) + ', ' + str(round(maxResist,2)) + '\n')
        writer.write('Number of firms, numFirms' + '\t' + str(numFirms) + '\n')
        writer.write('Local search radius, r' + '\t' + str(r) + '\n')
        writer.write('Exponential rate of decay for innovation profits, piDecayRate' + '\t' + str(piDecayRate) + '\n')
        writer.write('Monopoly profit for patents and initial pre-decay profit, monopProfit' + '\t' + str(monopProfit) + '\n')
        writer.write('Base RnD strength, baseRnD' + '\t' + str(baseRnD) + '\n' + '\n')
        writer.write('Results'+ '\n')
        writer.write('Number of innovations, len(innovSizesRun)' + '\t' + str(len(innovSizesRun)) + '\n')
        writer.write('Ending Mean BPF Height, meanBPFTS[-1]' + '\t' + str(meanBPFTS[-1]) + '\n')
        writer.write('Ending Max BPF Height, maxBPFTS[-1]' + '\t' + str(maxBPFTS[-1]) + '\n')
        writer.write('Number of steps it took to reach maxHeight, step' + '\t' + str(step) + '\n')
        writer.write('Ending Std of Profit, stdProfitTS[-1]' + '\t' + str(stdProfitTS[-1]) + '\n')
        writer.write('Run Time, minutes lapsed' + '\t' + str((time.clock() - t0)/60) + '\n' + '\n')
        writer.close()

# Store innovation sizes
if saveInnovDataFiles:
    # Write initial run to the second file 
    headers = ['id', 'run', 'step', 'innovSize']
    inputFile = open(dataFileName2, 'wb')
    writer = csv.writer(inputFile, lineterminator='\n')
    writer.writerow(headers)
    for i in range(len(innovSizes)):
        writer.writerow([i, innovSizes[i][0], innovSizes[i][1], innovSizes[i][2]])
    inputFile.close()


t = time.clock() - t0
print "Minutes Lapsed:", (t/60)