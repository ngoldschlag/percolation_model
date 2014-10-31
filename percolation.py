# -*- coding: cp1252 -*-
"""
Agent based percolation model of innovation
Nathan Goldschlag
October 19, 2014
Version 1.1
Written in Python 2.7

This python file executes an agent based percolation model that extends the percolation model 
in Silverberg and Verspagen 2007. The model incorporates patents which can: 1) provide additional 
resources to innovators to invest in R&D and 2) block other firms from exploring the technology space.

To execute the model modify the tests to run in main()
Below is a list of valid simulation test names.
['test1a', 'test1b', 'test2a', 'test2b', 'test3a', 'test3b', 'test4a', 'test4b', 'typical','microsim']
"""

## IMPORT LIBRARIES
import random
import time, os, sys
import numpy as np
import math
import matplotlib.pyplot as plt
import pylab as py 
import pandas as pd
import multiprocessing

t0 = time.time()

## CLASS DEFINITIONS
class Firm(object):
    """
    Attributes: id (numeric), patents (list), position (x,y coordinate), innovations (list), rndBudget (profits from previous step, applied to rnd next step), patent profits (numeric total of profits received for its patents), totalProfits (numeric total of profits received)
    
    Holds firm objects which perform innovative search in the technology lattice. Firms are initially assigned positions on the lattice and then perform innovative search extracting and expending resources to move upward through the lattice. 
    """
    def __init__(self, id, pos, simulation):
        self.id = id
        self.patents = []
        self.position = pos
        self.innovations = []
        self.rndBudget = 0
        self.patentProfits = 0
        self.totalProfits = 0
        # inherited simulation level parameters
        self.sim = simulation
    
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
        Returns list of neighbouring columns within the search radius r around the firm's current position.
        """
        r = self.sim.r
        numCol = self.sim.numCol
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

    def getMove(self,L):
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
                if self.sim.doPatent:
                    if L[height,col].patent == False:
                        localBPF.append([height, col])
                    elif (height,col) in self.patents:
                        localBPF.append([height, col])
                else:
                    localBPF.append([height, col])
        # If any of the BPF is returned calculate the probabilities of moving to each column then generate a random number to determine which column the firm moves to
        if localBPF:
            moveExp = []
            if self.sim.doPatent:
                # Calculate the percent of neighboring positions that are patented by other firms for each position the firm will potentially move to
                perPat = []
                for i in localBPF:
                    numNgbPat = 0
                    neighbors = self.sim.getNeighbors(i,L)
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
    
    def getFirmNeighbors(self,L):
        """
        Parameters: (Firm object)
        Returns list of neighboring cells in a square around the firm's position with radius r. If patents are active patented cells are removed.
        """
        r = self.sim.r
        numCol = self.sim.numCol
        # Returns a list of positions around the firm in square radius of r
        neighbors = []
        pos = self.position
        if pos[0]+r > len(L):
            self.sim.addRows(L,20)
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
        if self.sim.doPatent:
            # Remove any cells that are patented by other firms
            newNeighbors = []
            for i in neighbors:
                if L[tuple(i)].patent == False:
                    newNeighbors.append(i)
                elif i in self.patents:
                    newNeighbors.append(i)
            neighbors = newNeighbors
        return neighbors

    def chainReac(self,rndPos,L,run,step):
        """
        Parameters: (Firm object, cell position)
        Takes given cell position, finds connected cells with a value of 1, flips those cells to value 2 and potentially patents that cell. Cycles through all connected cells with value 1 until no longer connections exist. Once all connections are exhausted innovation sizes are captured as the height difference for each affected column.  
        """
        # Given a position that is changing from 1 to 2, this method analyzes neighbouring positions to determine if they should also flip from 1 to 2
        BPF = self.sim.getBPF(L)
        cons = []
        updated = []
        L[rndPos].updateCon(1,L)
        for i in L[rndPos].conOnes:
            cons.append(i)
        while cons:
            # Take the first connection in the list and update its list of connected positions with value = 1
            L[cons[0]].updateCon(1,L)
            # Set the value of that first connection to 2
            L[cons[0]].value = 2
            if self.sim.doPatent:
                if L[cons[0]].patent == False:
                    self.innovations.append(cons[0])
                    L[cons[0]].innovFirm = self
                    self.genPatent(cons[0],L)
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
            cols = list(range(self.sim.numCol))
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
                self.sim.innovSizes[self.sim.innovSizeID] = run, step, i ##BUG## Reading run/step right?
                self.sim.innovSizeID += 1

    def genPatent(self,rndPos,L):
        """
        Parameters: (Firm object, cell position)
        Takes given cell position, determines if firm patents cell, and if so patents that and all other cells in patent radius.  
        """
        # Firm potentially creates patent for a cell and the surrounding cells
        ## Allows for partial radius of patent, no overlap but odd shapped patent radius
        if random.random() < self.sim.probPatent:
            self.sim.totalPatents += 1
            neighbors = self.sim.getPatRadius(rndPos,L)
            neighbors.append(rndPos)
            for i in neighbors:
                if L[tuple(i)].patent == False and L[tuple(i)].patentFirm == 0: # patentFirm == 0 ensures that the cell has never been patented before
                    L[tuple(i)].patent = True
                    L[tuple(i)].patentFirm = self
                    L[tuple(i)].patentLife = self.sim.patentLife
                    L[tuple(i)].profitDuration = 0 # Reset profit decay counter, patented cell may have been innov by another firm, increasing the decay 
                    self.patents.append(i)

    def updateProfits(self,L):
        """
        Parameters: (Firm object)
        Updates profits for a firm. If profits are active special processing is done to separate profits accruing to other firms if an innovation is patented by a different firm. 
        """
        profitStream = self.sim.profitStream
        patentLife = self.sim.patentLife
        # Updates the firm's profits based on its list of innovations
        if self.sim.doPatent:
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
                        newProfits = self.sim.monopProfit
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
                        newProfits = self.sim.monopProfit
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
    def __init__(self,simulation):
        self.value = 0
        self.innovFirm = 0
        self.resistance = random.lognormvariate(simulation.resistMu,simulation.resistSigma)
        self.position = []
        self.conOnes = []
        self.conTwos = []
        self.profitDuration = 0
        self.patent = False
        self.patentFirm = 0
        self.patentLife = 0
        # inherited simulation level parameters
        self.sim = simulation
        
    
    def updateCon(self,val,L):
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
        # Special treatment for E on E boundary
        if self.position[1] == (self.sim.numCol-1):
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
        # Special treatment for S on S boundary
        if self.position[0] != 0:
            if L[(self.position[0]-1,self.position[1])].value == val:
                if val == 1:
                    if (self.position[0]-1,self.position[1]) not in self.conOnes:
                        self.conOnes.append((self.position[0]-1,self.position[1]))
                if val == 2:
                    if (self.position[0]-1,self.position[1]) not in self.conTwos:
                        self.conTwos.append((self.position[0]-1,self.position[1]))


class Simulation(object):
    """    
    Holds simulation objects which execute instances of the simulation. Each set of simulation parameters are unique to the simulation, allowing multiple simulations to be executed simultaneously. 
    
    Simulation parameters:
    numRuns - Number of times the model is executed
    numSteps - Number of steps in each run
    numCol - Number of columns in the initial lattice
    numRow - Number of initial rows in the initial lattice
    numFirms - Number of firms to start
    doPatent - Whether the model uses patents
    r - Firm search radius
    resistMu -  Mean of resistance for lognormal dist
    resistSigma - Standard deviation of resistance for lognormal dist
    baseRnD - Fixed portion of the firm's budget
    monopProfit - Monopoly profits received for a patented innovation each step during its patent life and the initial profit for an innovation prior to decay
    probPatent - Probability a firm will create a patent for an innovation
    patentRadius - Radius a patent covers
    patentLife - Number of steps a patent lasts
    piDecayRate - Rate at which profits from an innovation decay
    """
    def __init__(self, simTest, randomSeed):
        self.simTest = simTest
        self.filetime = time.strftime("%Y-%m-%d %H%M_%S")
        self.randomSeed = randomSeed # Seed used to initialize the random number generator
        ## SIMULATION TESTS
        if simTest in ['test1a', 'test1b', 'test2a', 'test2b', 'test3a', 'test3b', 'test4a', 'test4b']:
            self.testDoPatent = dict(dict.fromkeys(range(0,60),True).items() + 
                                dict.fromkeys(range(60,90),False).items())
            self.doPatent = True
            self.numRuns = 90 
            self.numSteps = 3000 
            self.numCol = 50 
            self.numRow = 100 
            self.numFirms = 20 
            self.r = 2 
            self.testResistMu = dict(dict.fromkeys(range(0,90),2).items())
            self.resistMu = float()
            self.resistSigma = 1.5 
            self.baseRnD = 1 
            self.testMonopProfit = dict(dict.fromkeys(range(0,90),1).items())
            self.monopProfit = int()
            self.probPatent  = 0.05 
            self.patentRadius = 1
            self.testPatentLife = dict(dict.fromkeys(range(0,30),5).items() + 
                                    dict.fromkeys(range(30,60),200).items() + 
                                    dict.fromkeys(range(60,90),None).items())
            self.patentLife = int()
            testPiDecay = sorted([x / 100.0 for x in range(1,11)]*3)
            self.testPiDecayRate = dict(dict(zip(range(0,30),testPiDecay)).items() + 
                                    dict(zip(range(30,60),testPiDecay)).items() + 
                                    dict(zip(range(60,90),testPiDecay)).items())
            self.piDecayRate = float()
        if simTest=='test1b':
            self.testResistMu = dict(dict.fromkeys(range(0,90),4).items()) 
        if simTest=='test2a':
            self.patentRadius = 3
        if simTest=='test2b':
            self.testResistMu = dict(dict.fromkeys(range(0,90),4).items()) 
            self.patentRadius = 3
        if simTest=='test3a':
            testResistMu = [x for x in sorted([x / 10.0 for x in range(20,40)]*3) if x*10%2==0]
            self.testPiDecayRate = dict(dict.fromkeys(range(0,90),0.01).items())
            self.testResistMu = dict(dict(zip(range(0,30),testResistMu)).items() + 
                                dict(zip(range(30,60),testResistMu)).items() + 
                                dict(zip(range(60,90),testResistMu)).items())
        if simTest=='test3b':
            testResistMu = [x for x in sorted([x / 10.0 for x in range(20,40)]*3) if x*10%2==0]
            self.testPiDecayRate = dict(dict.fromkeys(range(0,90),0.1).items())
            self.testResistMu = dict(dict(zip(range(0,30),testResistMu)).items() + 
                                dict(zip(range(30,60),testResistMu)).items() + 
                                dict(zip(range(60,90),testResistMu)).items())
        if simTest in ['test4a','test4a']:
            self.numRuns = 180
            self.testResistMu = dict(dict.fromkeys(range(0,180),2).items())
            self.testDoPatent = dict(dict.fromkeys(range(0,90),True).items() + 
                                dict.fromkeys(range(90,180),False).items())
            self.testMonopProfit = dict(dict.fromkeys(range(0,30),1).items() + 
                                    dict.fromkeys(range(30,60),3).items() +
                                    dict.fromkeys(range(60,90),5).items() +
                                    dict.fromkeys(range(90,120),1).items() +
                                    dict.fromkeys(range(120,150),3).items() + 
                                    dict.fromkeys(range(150,180),5).items() ) 
            self.testPatentLife = dict(dict.fromkeys(range(0,180),100).items())
            testPiDecay = sorted([x / 100.0 for x in range(1,11)]*3)
            self.testPiDecayRate = dict(dict(zip(range(0,30),testPiDecay)).items() + 
                                    dict(zip(range(30,60),testPiDecay)).items() + 
                                    dict(zip(range(60,90),testPiDecay)).items() +
                                    dict(zip(range(90,120),testPiDecay)).items() + 
                                    dict(zip(range(120,150),testPiDecay)).items() + 
                                    dict(zip(range(150,180),testPiDecay)).items())
        if simTest=='test4b':
            self.testResistMu = dict(dict.fromkeys(range(0,180),4).items()) 
        if simTest=='typical':
            self.testDoPatent = [False]
            self.doPatent = False
            self.numRuns = 1
            self.numSteps = 5000 
            self.numCol = 50 
            self.numRow = 100 
            self.numFirms = 20 
            self.r = 2 
            self.testResistMu = [2]
            self.resistMu = float()
            self.resistSigma = 1.5 
            self.baseRnD = 1 
            self.testMonopProfit = [1]
            self.monopProfit = int()
            self.probPatent = 'N/A'
            self.patentRadius = 'N/A'
            self.testPatentLife = ['N/A']
            self.testPiDecayRate = [0.5]
            self.piDecayRate = float()
        if simTest == 'microSim':
            self.doPatent = True 
            self.numRuns = 10
            self.numSteps = 500 
            self.numCol = 50 
            self.numRow = 100 
            self.numFirms = 20 
            self.r = 2 
            self.resistMu = 2 
            self.resistSigma = 1.5 
            self.baseRnD = 1 
            self.monopProfit = 1 
            self.probPatent = 0.05 
            self.patentRadius = int() 
            self.patentLife = int() 
            self.piDecayRate = float() 
        self.innovSizes = {} # Innovation sizes held in a dictionary with attributes run, step, and innovation size
        self.innovSizeID = 0 # Counter for keys in the dictionary
        self.profitStream = []
        self.saveParams = True # Whether a text file is saved with the parameters from the run
        self.saveStepData = False # Whether a file is created to hold time series data
        self.saveRunData = True # Whether a file is created to hold run data
        self.saveInnovData = False # Whether a file is created to hold innovation sizes
        self.saveFirmStepData = False # Whether a csv is written with firm level data from the model
        self.saveDataPath = 'C:/Users/ngold/Documents/python_library/work_percolation/'
    
     ## FUNCTION DEFINITIONS

    def genLattice(self):
        """
        Parameters: ()
        Creates the lattice representing the technology space.
        """
        # Creates a lattice that will act as the technology space
        # First create a list of point objects as the first row, then create the array
        row = []
        for i in range(self.numCol):
            row.append(Point(self))
        M = np.array([row])
        # Then loop through and create the remaining rows of point objects, appending each to the array
        for i in range(self.numRow-1):
            row = []
            for i in range(self.numCol):
                row.append(Point(self))
            M = np.append(M, [row], axis=0)
        # Write the array position to each of the point objects in the array
        for i, x in np.ndenumerate(M):
            M[i].position = i
        return M

    def addRows(self,M,n):
        """
        Parameters: (Lattice, n=Number of cells to add)
        Given a lattice, adds a given number of rows. Returns the modified lattice. 
        """
        # Returns modified lattice with n new rows appended to the end
        # Create each row, append it to the matrix, then write the position again to all points
        for i in range(n):
            row = []
            for i in range(self.numCol):
                row.append(Point(self))
                row[-1].position = (len(M),i)
            M = np.append(M, [row], axis=0)
        return M

    def getBPF(self,L):
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

    def getNeighbors(self,pos,L):
        """
        Parameters: (position)
        Given a position, returns the neighboring cells in radius r. Patented cells are not removed, but the given cell is not returned in the list. 
        """
        r  = self.r
        # Returns a list of positions around the firm in square radius of r
        neighbors = []
        if pos[0]+r > len(L):
            self.addRows(L,20,self.resistMu)
        for col in range(2*r+1):
            row = 0
            while row <= 2*r: 
                # Bounded vertically
                if pos[0]-r+row >= 0:
                    # Jump east
                    if pos[1]-r+col > (self.numCol-1):
                        neighbors.append([pos[0]-r+row, pos[1]-r+col - self.numCol])
                    else:
                        neighbors.append([pos[0]-r+row,pos[1]-r+col])
                row +=1
        neighbors.remove([pos[0],pos[1]])
        return neighbors

    def getPatRadius(self,pos,L):
        """
        Parameters: (position)
        Given a position, returns the neighboring cells within the patent radius. Patented cells are not removed. The given cell is NOT returned in the list. 
        """
        # Returns a list of positions around the given position in square radius of r
        neighbors = []
        if pos[0]+self.patentRadius > len(L):
            self.addRows(L,20,self.resistMu)
        for col in range(2*self.patentRadius+1):
            row = 0
            while row <= 2*self.patentRadius: 
                # Bounded vertically
                if pos[0]-self.patentRadius+row >= 0:
                    # Jump east
                    if pos[1]-self.patentRadius+col > (self.numCol-1):
                        neighbors.append([pos[0]-self.patentRadius+row, pos[1]-self.patentRadius+col - self.numCol])
                    elif pos[1]-self.patentRadius+col < 0:
                        neighbors.append([pos[0]-self.patentRadius+row,self.numCol + pos[1]-self.patentRadius+col])
                    else:
                        neighbors.append([pos[0]-self.patentRadius+row,pos[1]-self.patentRadius+col])
                row +=1
        if [pos[0],pos[1]] in neighbors:
            neighbors.remove([pos[0],pos[1]])
        return neighbors

    def getCluster(self,firms):
        """
        Parameters: ()
        Calculates and returns the cluster index for firms the lattice. 
        """
        # Returns an index that captures how clustered firms are across the technology space
        firmPositions = map(lambda Firm: Firm.getPosition(), firms)
        colByFirm = []
        for i in range(self.numCol):
            colByFirm.append(math.pow(len([x for x in firmPositions if x[1] == i]),2))
        return sum(colByFirm)

    def getPerBPFPat(self,L):
        """
        Parameters: ()
        Calculates and returns the percent of the space under the BPF that is patented. 
        """
        # Returns a percent of the cells under the BPF that are patented
        BPF = self.getBPF(L)
        numPats = 0
        # For each column in the lattice count the number of patented and nonpatented cells
        for col in range(self.numCol):
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

    def doSimulation(self):
        ## SIMULATION 
        # Initialize dataframes to store data 
        if self.saveParams:
            paramDFCols = ['run','numRuns', 'numSteps', 'numCol', 'numRow', 'numFirms', 'r', 'piDecayRate', 'monopProfit', 'baseRnD', 'resistMu', 'resistSigma', 'initialMeanResist', 'initialMaxResist', 'initialMinResist', 'doPatent', 'probPatent', 'patentLife', 'patentRadius', 'randomSeed', 'simTest', 'runTime']
            paramDF = pd.DataFrame(columns=paramDFCols)
        if self.saveRunData:
            runDFCols = ['run', 'BPFChange', 'clusterIndex', 'meanBPF', 'maxBPF', 'numInnov', 'steps', 'stdProfit', 'logCoef', 'perBPFPat', 'totalPatents']
            runDF = pd.DataFrame(columns=runDFCols)
        if self.saveStepData:
            stepDFCols = ['run', 'step', 'resistMu', 'resistSigma', 'doPatent', 'piDecayRate', 'patentLife', 'patentRadius', 'BPFChange', 'clusterIndex', 'meanBPF', 'maxBPF', 'stdProfit', 'numPats', 'perBPFPat', 'totalPatents']
            stepDF = pd.DataFrame(columns=stepDFCols)
        if self.saveFirmStepData:
            firmDFCols = ['run', 'step', 'firmID', 'totaProfits', 'patentProfits', 'height', 'numInnovs', 'numPats']
            firmDF = pd.DataFrame(columns=firmDFCols)


        # Initialize the random number generator with the parameterized seed
        random.seed(self.randomSeed)

        # Run the simulation numRuns times
        for run in range(self.numRuns):
            if run%200==0:
                t = time.time() - t0
                print 'Worker', multiprocessing.current_process().name, 'Run:', run, 'Minutes Lapsed:', (t/60)
                
            # Modify simulation parameters according to the test being performed, if any
            if self.simTest in ['test1a', 'test1b', 'test2a', 'test2b', 'test3a', 'test3b', 'test4a', 'test4b', 'typical']:
                self.doPatent = self.testDoPatent[run]
                self.resistMu = self.testResistMu[run]
                self.monopProfit = self.testMonopProfit[run]
                self.piDecayRate = self.testPiDecayRate[run]
                self.patentLife = self.testPatentLife[run]
                # Define profit stream
                self.profitStream = [self.monopProfit]
                while self.profitStream[-1] > 0.01:
                    self.profitStream.append(round(self.monopProfit * math.exp(-self.piDecayRate*len(self.profitStream)),3))
                    
            if self.simTest in ['microSim']:
                self.piDecayRate = random.uniform(0.01,0.1)
                self.resistMu = random.uniform(2.0,4.0)
                self.doPatent = random.choice([True, False])
                self.patentLife = random.randint(1,100)
                self.patentRadius = random.randint(1,3)
                # Define profit stream
                self.profitStream = [self.monopProfit]
                while self.profitStream[-1] > 0.01:
                    self.profitStream.append(round(self.monopProfit * math.exp(-self.piDecayRate*len(self.profitStream)),3))
                
            # Define lists to hold step data, to be aggregated in the runDF
            self.BPFChangeTS = []
            self.clusterIndexTS = []
            self.meanBPFTS = []
            self.maxBPFTS = [] 
            self.stdProfitTS = []
            if self.doPatent:
                self.numPatentsTS = []
                self.perBPFPatTS = []
                self.totalPatents = 0
                self.totalPatentsTS = []
            else:   
                self.numPatents = ['N/A']
                self.perBPFPat = ['N/A']
                self.totalPatents = 'N/A'
                self.totalPatents = ['N/A']
            
            # Create lattice; must be named "L" since it is referenced as "L" throughout the methods
            L = self.genLattice()

            # Capture statistics about initial resistance values for the initial lattice, written to parameter file
            resistValues = []
            for i, x in np.ndenumerate(L):
                resistValues.append(L[i].resistance)
            maxResist = max(resistValues)
            meanResist = np.mean(resistValues)
            minResist = min(resistValues)
                    
            # Create firms
            firms = []
            for id in range(self.numFirms):
                # If the number of firms is equal to the number of columns assign them sequentially to each column
                if self.numFirms == self.numCol:
                    pos = (0,id)
                    firms.append(Firm(id,pos,self))
                # Otherwise randomly assign the firms to columns
                else:
                    col = random.randint(0, (self.numCol-1))
                    pos = (0,col)
                    firms.append(Firm(id,pos,self))
            
            # Turn each position where the firms are instantiated to discovered
            firmPositions = map(lambda Firm: Firm.getPosition(), firms)
            for i in firmPositions:
                L[i].value = 2
                L[i].resistance = 0
            
            # Run the simulation steps
            for step in range(self.numSteps):
                # Print info to console while model is running
                #if self.numSteps > 1000:
                #    if step%1000==0:
                #        t = time.time() - t0
                #        print 'Worker', multiprocessing.current_process().name, 'Run:',run,'Step:', step, 'Minutes Lapsed:', (t/60)
                
                # Create snapshot of BPF for comparison after round of R&D
                preBPF = self.getBPF(L)
                
                # Shuffle the list of firms, which is then used as the activation order
                random.shuffle(firms)
               
                for i in range(len(firms)):
                    # Set active firm
                    firm = firms[i]
                    
                    # Firm chooses new position (potentially the same position)
                    newPosition = firm.getMove(L)
                    if newPosition:
                        firm.position = newPosition
                
                    # Firm selects position to perform R&D on
                    # With patents there is a chance that there are no eligible positions near the firm to perform R&D, in which case rndPos is set to zero and the firm does not perform R&D that step
                    gN = firm.getFirmNeighbors(L)
                    if gN:
                        rndPos = tuple(random.choice(gN))
                    else:
                        rndPos = 0
                        
                    # If there is a position returned that the firm can perform R&D on firm performs R&D
                    if rndPos:
                        # Firm performs R&D, reducing the resistance of the cell
                        if L[rndPos].resistance > 0:
                            L[rndPos].resistance =  L[rndPos].resistance - random.random()*(self.baseRnD + firm.rndBudget)
                            
                            # Reset rndBudget to zero since all of the rndBudget is applied
                            firm.rndBudget = 0
                            # If R&D effort makes the resistance value <= 0 then set value to 1 (discovered but not yet viable)
                            if L[rndPos].resistance <= 0:
                                # Cell value is set to 2 automatically if it is on the baseline and firm potentially creates patent
                                if rndPos[0]==0:
                                    L[rndPos].value = 2
                                    firm.innovations.append(rndPos)
                                    L[rndPos].innovFirm = firm
                                    if self.doPatent:
                                        firm.genPatent(rndPos,L)
                                    
                                # If newly discovered position is connected to the baseline by an unbroken chain of positions with value = 2 (discovered and viable), then set value to 2
                                else:
                                    L[rndPos].value = 1
                                    L[rndPos].updateCon(2,L)
                                    if L[rndPos].conTwos:
                                        L[rndPos].value = 2
                                        firm.innovations.append(rndPos)
                                        L[rndPos].innovFirm = firm
                                        if self.doPatent:
                                            firm.genPatent(rndPos,L)
                                        firm.chainReac(rndPos,L,run,step)
                                    # Firm can still attempt to patent an invention that is not yet viable
                                    else:
                                        if self.doPatent:
                                            firm.genPatent(rndPos,L)
                    
                    # Check highest firm to see if new rows must be added
                    if len(L) - max(map(lambda Firm: Firm.getRow(), firms)) < 30:
                        L = self.addRows(L,30)
                
                # Calculate profits for each firm
                for frm in firms:
                    frm.updateProfits(L)
                
                # Reduce the life of existing patents, and remove patents that are no longer valid
                latticePatents = [x for i, x in np.ndenumerate(L) if x.patent] 
                for pats in latticePatents:
                    pats.patentLife -= 1
                    if pats.patentLife == 0:
                        pats.patent = False
                        
                # Compare pre and post BPFs and capture data
                postBPF = self.getBPF(L)
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
                self.BPFChangeTS.append(sum(BPFDiff))
                self.clusterIndexTS.append(self.getCluster(firms))
                self.meanBPFTS.append(round(np.mean([x[0] for x in postBPF]),2))
                self.maxBPFTS.append(max([x[0] for x in postBPF]))
                self.stdProfitTS.append(np.std([x.totalProfits for x in firms]))
                if self.doPatent:
                    self.numPatentsTS.append(sum([len(x.patents) for x in firms]))
                    self.perBPFPatTS.append(self.getPerBPFPat(L))
                    self.totalPatentsTS.append(self.totalPatents)
                
                # Calculate step data and append to step DF
                if self.saveStepData:
                    if self.doPatent:
                        row = pd.Series([run, step, self.resistMu, self.resistSigma, self.doPatent, self.piDecayRate, self.patentLife, self.patentRadius, self.BPFChangeTS[-1], self.clusterIndexTS[-1], self.meanBPFTS[-1], self.maxBPFTS[-1], self.stdProfitTS[-1], self.numPatentsTS[-1], self.perBPFPatTS[-1], self.totalPatentsTS[-1]], index=stepDFCols)
                        stepDF = stepDF.append(row, ignore_index=True)  
                    else:
                        row = pd.Series([run, step, self.resistMu, self.resistSigma, self.doPatent, self.piDecayRate, self.patentLife, self.patentRadius, self.BPFChangeTS[-1], self.clusterIndexTS[-1], self.meanBPFTS[-1], self.maxBPFTS[-1], self.stdProfitTS[-1], 'N/A', 'N/A', 'N/A'], index=stepDFCols)
                        stepDF = stepDF.append(row, ignore_index=True)  
                # Calculate firm data and append firm DF
                if self.saveFirmStepData:
                    for frm in range(self.numFirms):
                        if self.doPatent:
                            numPats = len(frm.patents)
                        else:
                            numPats = 'N/A'
                        row = pd.Series([run, step, frm, frm.totalProfits, frm.patentProfits, frm.position, len(frm.innovations), numPats], index=firmDFCols)
                        firmDF = firmDF.append(row, ignore_index=True)
                
                ### End of STEP ###
            
            ### End of RUN ###
            
            # Calculate log-log coefficient of innovation sizes for run
            # Filter the dictionary for the innovSizes for the run
            innovSizesRun = []
            innovKeys = [k for k, v in self.innovSizes.iteritems() if v[0] == run]
            if innovKeys:
                for i in innovKeys:
                    innovSizesRun.append(self.innovSizes[i][2])
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
            
            # Store run data for each run
            if self.saveRunData:
                if self.doPatent:
                    row = pd.Series([run, np.mean(self.BPFChangeTS), np.mean(self.clusterIndexTS), self.meanBPFTS[-1], self.maxBPFTS[-1], len(innovSizesRun), step, np.mean(self.stdProfitTS), logCoef, np.mean(self.perBPFPatTS), self.totalPatents], index=runDFCols)
                    runDF = runDF.append(row, ignore_index=True)
                else:
                    row = pd.Series([run, np.mean(self.BPFChangeTS), np.mean(self.clusterIndexTS), self.meanBPFTS[-1], self.maxBPFTS[-1], len(innovSizesRun), step, np.mean(self.stdProfitTS), logCoef, 'N/A', 'N/A'], index=runDFCols)
                    runDF = runDF.append(row, ignore_index=True)
            # Create a file to save the parameters used, subsequent runs appended to the same text file
            if self.saveParams:
                runTime = (time.time() - t0)/60
                row = pd.Series([run, self.numRuns, self.numSteps, self.numCol, self.numRow, self.numFirms, self.r, self.piDecayRate, self.monopProfit, self.baseRnD, self.resistMu, self.resistSigma, meanResist, maxResist, minResist, self.doPatent, self.probPatent, self.patentLife, self.patentRadius, self.randomSeed, self.simTest, runTime], index=paramDFCols)
                paramDF = paramDF.append(row, ignore_index=True)
                
        # Write data to CSVs
        if self.saveFirmStepData:
            firmDF.to_csv(self.saveDataPath+self.filetime+ '_'+multiprocessing.current_process().name+'_'+'percol_firmdata.csv',index=False)
        if self.saveStepData:
            stepDF.to_csv(self.saveDataPath+self.filetime+ '_'+multiprocessing.current_process().name+'_'+'percol_stepdata.csv',index=False)
        if self.saveRunData:
            runDF.to_csv(self.saveDataPath+self.filetime+ '_'+multiprocessing.current_process().name+'_'+'percol_rundata.csv',index=False)
        if self.saveParams:
            paramDF.to_csv(self.saveDataPath+self.filetime+ '_'+multiprocessing.current_process().name+'_'+'percol_params.csv',index=False)
        if self.saveInnovData:
            innovDF = pd.DataFrame.from_dict(self.innovSizes,orient='index')
            innovDF.columns = ['run','step','innovSize']
            innovDF.to_csv(self.saveDataPath+self.filetime+ '_'+multiprocessing.current_process().name+'_'+'percol_innovdata.csv',index=False)

        t = time.time() - t0
        print "Minutes Lapsed:", (t/60)
     
def main():
    """
    validTests = ['test1a', 'test1b', 'test2a', 'test2b', 'test3a', 'test3b', 'test4a', 'test4b', 'typical', 'microSim']
    """
    testsToRun = ['microSim','microSim','microSim','microSim','microSim']
    print "Simulation Tests to Run", testsToRun
    
    ## Run simulations
    # Initialize simulation objects
    sims = []
    for test in testsToRun:
        sims.append(Simulation(test,randomSeed=123456789))
    # Kick off simulations
    processes = []
    for s in range(len(sims)):
        w = multiprocessing.Process(name='sim_{0}_{1}'.format(sims[s].simTest,s), target=sims[s].doSimulation, args=(),)
        w.start()
        processes.append(w)
        time.sleep(2)
    
    
    for p in processes:
        p.join()
    print 'Workers:',processes,'finished and joined'
if __name__=="__main__":
    main()
