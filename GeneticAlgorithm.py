from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import time
import sys
import math
import random
import copy
##############Static Objective######################

def StaticObjectives(assignment, mu, numController,numSwitch,switchLoad):
    controllerLoad = []
    totalLoad = 0
    for key in assignment:
        load = 0
        for value in key:
            load += switchLoad[value]
        totalLoad += load
        controllerLoad.append(load)

    totalLoad = totalLoad/numController

    obj1 = 0
    obj2 = 0
    print(controllerLoad)
    for i, key in enumerate(assignment):
        temp = abs(controllerLoad[i] - totalLoad)
        obj2 += temp**2
        for value in key:
            temp = mu - controllerLoad[i]
            obj1 += 1/temp

    obj1 = obj1/numSwitch
    obj2 = obj2/numController
    print("Th time request spent in the system")
    print(obj1)
    print("Th variance in the controller loads")
    print(obj2)
    return obj1, obj2

##############GeneticAlgorithm######################

def InitailAssignemnt(mu,numController, numSwitch, switchLoad, beforeAssign):
    iniAssign = []
    remainLoad = []
    for i in range(0, numController):
        remainLoad.append(mu)
        iniAssign.append([])
    for i in range(0, numSwitch):
        tempNum = random.randint(0, numController-1)
        while remainLoad[tempNum] < switchLoad [i]:
            tempNum = random.randint(0, numController - 1)
        iniAssign[tempNum].append(i)
        remainLoad[tempNum] -= switchLoad[i]
    # print(remainLoad)
    # print(iniAssign)

    fitness = GetFitness(iniAssign,numSwitch, switchLoad,beforeAssign)
    return [iniAssign, fitness]


def GetSwitchPosition(swichID, assignment):

    for key, value in enumerate(assignment):
        if swichID in value:
            return key


def GetFitness(iniAssign, numSwitch, switchLoad,beforeAssign):

    obj = 0

    totalSwitchLoad = 0
    for i in switchLoad:
        totalSwitchLoad += i
    # iniAssign m*n matrix [[3, 11], [4, 5, 10, 12], [1, 2, 9, 13], [0, 6, 7, 8]]

    for i in range(numSwitch):
        beforePosition =GetSwitchPosition(i, beforeAssign)
        newPosition = GetSwitchPosition(i, iniAssign)
        if beforePosition != newPosition:
            obj += switchLoad[i]

    return obj/totalSwitchLoad


def TournamentSelection(population):

    candidate =[]

    random1 = random.randint(0,len(population)-1)
    candidate.append(population[random1])
    random2 = random.randint(0,len(population)-1)
    while(random2 == random1):
        random2 = random.randint(0, len(population) - 1)
    candidate.append(population[random2])
    random3 = random.randint(0,len(population)-1)
    while (random3 == random2) or (random3 == random1):
        random3 = random.randint(0, len(population) - 1)
    candidate.append(population[random3])

    low = 99999999
    lowAssignment=[]
    for key in candidate:
        if key[1] < low:
            low = key[1]
            lowAssignment = key

    return lowAssignment


def TwoPointCrossover(fatherAssign,motherAssign, numController, numSwitch):

    child1, child2 = [], []
    for i in range(numController):
        child1.append([])
        child2.append([])
    random1 = random.randint(0, numSwitch - 1)
    random2 = random.randint(0, numSwitch - 1)
    while (random2 == random1):
        random2 = random.randint(0, numSwitch - 1)
    if random1<random2:
        small = random1
        large = random2
    else:
        small = random2
        large = random1

    # print ("small ", small)
    # print ("large ", large)
    # print ("father before ", fatherAssign)
    # print ("mother before ", motherAssign)


    for i in range(numSwitch):

        if i <= small or i >=large:
           child1[GetSwitchPosition(i,fatherAssign[0])].append(i)
           child2[GetSwitchPosition(i,motherAssign[0])].append(i)
        else:
            child1[GetSwitchPosition(i, motherAssign[0])].append(i)
            child2[GetSwitchPosition(i, fatherAssign[0])].append(i)

    # print("father after ", child1)
    # print("mother after ", child2)

    return child1,child2


def Mutation(child1,child2,mutationRate,numSwitch,numController):
    if mutationRate> random.random():
        switchRandom = random.randint(0, numSwitch - 1)
        controllerRandom = random.randint(0, numController - 1)
        switchPosition = GetSwitchPosition(switchRandom, child1)

        child1[switchPosition].remove(switchRandom)
        child1[controllerRandom].append(switchRandom)

    if mutationRate> random.random():
        switchRandom = random.randint(0, numSwitch - 1)
        controllerRandom = random.randint(0, numController - 1)
        switchPosition = GetSwitchPosition(switchRandom, child2)

        child2[switchPosition].remove(switchRandom)
        child2[controllerRandom].append(switchRandom)

    return child1, child2


def OverLoaded(assignment,mu,switchLoad):

    for key in assignment:
        load = 0
        for value in key:
            load+= switchLoad[value]
        if load > mu:
            return True

    return False


def RemoveHighest(population):
    high = -9999
    hightIndex = []
    for key in population:
        if key[1] > high:
            high = key[1]
            hightIndex = key

    population.remove(hightIndex)


def GetAverageFitness(population):
    fitness = 0

    for key in population:
        fitness+=key[1]

    return fitness/len(population)


def UpdatePopulation(child1,child2,switchLoad, population, beforeAssign,mu):

     if OverLoaded(child1,mu,switchLoad)==False:
        RemoveHighest (population)
        population.append([child1,GetFitness(child1, numSwitch, switchLoad, beforeAssign)])
     if OverLoaded(child2,mu,switchLoad)==False:
        RemoveHighest(population)
        population.append([child2, GetFitness(child2, numSwitch, switchLoad, beforeAssign)])


def GetMinimumAssignment(population):
    minimumFitness = 999999
    minimumAssignment =[]
    for key in population:
        if key[1]<minimumFitness:
            minimumFitness =key[1]
            minimumAssignment = key

    return minimumAssignment


def GeneticAlgorithm (mu,numController, numSwitch, switchLoad,beforeAssign):


    #Maximum number of assignment is m^n
    popuNumber = 200
    population = []
    generation = 400
    mutationRate =0.02
    geniIdex = 0

    while (len(population)!=popuNumber):

        key = InitailAssignemnt(mu, numController, numSwitch, switchLoad,beforeAssign)
        if key not in population:
            population.append(key)

    while geniIdex != generation:
        # print(GetAverageFitness(population))
        fatherAssign = TournamentSelection(population)
        motherAssign = TournamentSelection(population)
        child1, child2 = TwoPointCrossover(fatherAssign,motherAssign, numController, numSwitch)
        child1, child2 = Mutation(child1, child2, mutationRate, numSwitch, numController)
        UpdatePopulation (child1,child2,switchLoad, population, beforeAssign, mu)
        geniIdex += 1

    # print(GetAverageFitness(population))
    # print(population)

    finalAssignment = GetMinimumAssignment(population)
    print("Switch Migration Cost:")
    print(finalAssignment[1])

    obj1,obj2 = StaticObjectives(finalAssignment[0], mu, numController, numSwitch, switchLoad)
    return(finalAssignment[1], obj1,obj2)

##############ModifiedGreedyAlgorithm######################

def GetDelta(beforeAssign,numController,switchLoad):
    totalLoad = 0
    sum = 0
    for key in beforeAssign:
        for value in key:
            totalLoad+= switchLoad[value]
    totalLoad = totalLoad /numController

    for key in beforeAssign:
        avgLoad = 0
        for value in key:
            avgLoad+= switchLoad[value]

        sum += abs(avgLoad-totalLoad)

    return sum/numController


def GetMaximumController(beforeAssign,switchLoad):

    maxLoad = -9999
    maxIndex = -1

    for i, key in enumerate(beforeAssign):
        load = 0
        for value in key:
            load += switchLoad[value]
        if load> maxLoad:
            maxLoad = load
            maxIndex = i
    return maxIndex


def GetMinimumController(beforeAssign, switchLoad):
    minLoad = 999999
    minIndex = -1

    for i, key in enumerate(beforeAssign):
        load = 0
        for value in key:
            load += switchLoad[value]
        if load < minLoad:
            minLoad = load
            minIndex = i
    return minIndex


def GetMinimumSwitch(controllerList, switchLoad):
    minLoad = 9999999
    minIndex = -1

    for key in controllerList:
        if switchLoad[key] < minLoad:
            minLoad = switchLoad[key]
            minIndex = key

    return minIndex


def ModifiedGreedyAlgorithm (mu, numController, numSwitch, switchLoad, beforeAssign):

    threshold = 100
    maxCount = 1000
    count = 0

    delta = GetDelta(beforeAssign, numController, switchLoad)
    deepCopyBefore = copy.deepcopy(beforeAssign)
    bestAssignment = copy.deepcopy(beforeAssign)
    bestValue = 99999
    while delta > threshold and count < maxCount:
        maxIndex = GetMaximumController(beforeAssign, switchLoad)
        minIndex = GetMinimumController(beforeAssign, switchLoad)
        minSwitch = GetMinimumSwitch(beforeAssign[maxIndex], switchLoad)
        beforeAssign[maxIndex].remove(minSwitch)
        beforeAssign[minIndex].append(minSwitch)
        delta = GetDelta(beforeAssign, numController, switchLoad)
        temp = GetFitness(beforeAssign, numSwitch, switchLoad, deepCopyBefore)
        if temp < bestValue:
            bestValue = temp
            bestAssignment = beforeAssign
        count += 1

    print("Switch Migration Cost:")
    print(GetFitness(bestAssignment, numSwitch, switchLoad, deepCopyBefore))

    obj1, obj2 = StaticObjectives(bestAssignment, mu, numController, numSwitch, switchLoad)
    return GetFitness(bestAssignment, numSwitch, switchLoad, deepCopyBefore),obj1,obj2

##############SimulatedAnnealingAlgorithm######################

def GetNewAssignment(beforeAssign,numSwitch,mu,switchLoad):

    newAssign = copy.deepcopy(beforeAssign)
    if random.random() > 0.5:
        random1 = random.randint(0, numSwitch - 1)
        random2 = random.randint(0, numSwitch - 1)
        while (random2 == random1):
            random2 = random.randint(0, numSwitch - 1)

        controller1 = GetSwitchPosition(random1, beforeAssign)
        controller2 = GetSwitchPosition(random2, beforeAssign)
        newAssign[controller1].remove(random1)
        newAssign[controller1].append(random2)
        newAssign[controller2].remove(random2)
        newAssign[controller2].append(random1)
    else:
        random1 = random.randint(0, numSwitch - 1)
        random2 = random.randint(0, numSwitch - 1)
        while (random2 == random1):
            random2 = random.randint(0, numSwitch - 1)
        random3 = random.randint(0, numSwitch - 1)
        while (random3 == random2) or (random3 == random1) :
            random3 = random.randint(0, numSwitch - 1)

        controller1 = GetSwitchPosition(random1, beforeAssign)
        controller2 = GetSwitchPosition(random2, beforeAssign)
        controller3 = GetSwitchPosition(random3, beforeAssign)

        newAssign[controller1].remove(random1)
        newAssign[controller1].append(random2)
        newAssign[controller2].remove(random2)
        newAssign[controller2].append(random3)

        newAssign[controller3].remove(random3)
        newAssign[controller3].append(random1)



    while OverLoaded(newAssign,mu,switchLoad)== True:

        newAssign = GetNewAssignment(beforeAssign, numSwitch,mu,switchLoad)

    return newAssign


def SimulatedAnnealing (mu, numController, numSwitch, switchLoad, beforeAssign):
    tStart = 30
    tEnd = 1e-8
    iteration = 50
    decreaseRate = 0.98

    t = tStart

    iniAssign = []
    remainLoad = []
    for i in range(0, numController):
        remainLoad.append(mu)
        iniAssign.append([])
    for i in range(0, numSwitch):
        tempNum = random.randint(0, numController - 1)
        while remainLoad[tempNum] < switchLoad[i]:
            tempNum = random.randint(0, numController - 1)
        iniAssign[tempNum].append(i)
        remainLoad[tempNum] -= switchLoad[i]

    bestAssign = []
    bestFitness = GetFitness(iniAssign, numSwitch, switchLoad, beforeAssign)

    while True:

        if t <= tEnd:
            break
        for count in range(iteration):
            # print("IniAssign assign ", iniAssign)
            newAssign = GetNewAssignment(iniAssign,numSwitch,mu,switchLoad)
            # print("new assign", newAssign)
            newFitness = GetFitness(newAssign, numSwitch, switchLoad, beforeAssign)
            iniFitness = GetFitness(iniAssign, numSwitch, switchLoad, beforeAssign)
            delt = newFitness-iniFitness
            if delt <= 0:
                iniAssign = newAssign
                if newFitness < bestFitness:
                    bestAssign = copy.deepcopy(newAssign)
                    bestFitness = copy.deepcopy(newFitness)

            elif delt > 0:
                p = math.exp(-delt / t)
                r = random.uniform(0, 1)
                if r < p:
                    iniAssign = copy.deepcopy(newAssign)

        t = t * decreaseRate


    print("Switch Migration Cost:")
    print(bestFitness)

    obj1, obj2 =StaticObjectives(bestAssign, mu, numController, numSwitch, switchLoad)
    return bestFitness, obj1, obj2


##############SimulatedAnnealingAlgorithm######################

def MigrationTrigger(mu, numController, numSwitch, switchLoad, beforeAssign,threshold):
    controllerLoad = []
    returnMatrix = []
    OM_S, IM_S = set(), set()
    triggerFlag = False
    for key in beforeAssign:
        keyLoad = 0
        for i in key:
            keyLoad += switchLoad[i]
        controllerLoad.append(keyLoad)
    for i, key1 in enumerate(controllerLoad):
        row = []
        for j, key2 in enumerate(controllerLoad):
            if key2 == 0:
                temp = float("inf")
            else:
                temp = key1/key2
            if temp > threshold:
                triggerFlag = True
                OM_S.add(i)
                IM_S.add(j)
            row.append(temp)
        returnMatrix.append(row)

    return triggerFlag,OM_S,IM_S, returnMatrix, controllerLoad


def MigrationProbility(switchID, controllerID,switchLoad, returnMatrix, controllerLoad):
    avarage_load = sum(controllerLoad)/len(controllerLoad)
    top = abs(avarage_load-controllerLoad[controllerID]-switchLoad[switchID])
    bot = (controllerLoad[controllerID]-avarage_load)
    if bot == 0:
        temp = float("inf")
    else:
        temp = top/bot

    return 1-temp


def MigrationCost(controllerLoad,maxSwitchID,beforeAssign,switchLoad,controllerID,ci):
    avg = sum(controllerLoad)/len(controllerLoad)
    before = 0
    after = 0
    # print("in id",controllerID)
    # print("out id", ci)
    for key in controllerLoad:
        before += (key-avg)**2
        if key == controllerID:
            after += (key-switchLoad[maxSwitchID] - avg) ** 2
        elif key == ci:
            after += (key + switchLoad[maxSwitchID] - avg) ** 2
        else:
            after += (key-avg)**2
    before = before/len(controllerLoad)
    after = after/len(switchLoad)

    rec = abs(after-before)/(controllerLoad[ci]+switchLoad[maxSwitchID])
    return rec


def UpdateAssignment(P,tempAssignment):
    print(P)
    for key in P:
        if -1 in key:
            continue
        tempAssignment[key[0]].remove(key[1])
        tempAssignment[key[2]].append(key[1])
    return tempAssignment


def SwitchMiragtion(OM_S,IM_S, switchLoad, returnMatrix, controllerLoad, tempAssignment, mu):

    for controllerID in OM_S:
        P = []
        maxPValue = -9999
        maxSwitchID = -1
        for switchID in tempAssignment[controllerID]:
            pValue = MigrationProbility (switchID, controllerID, switchLoad, returnMatrix, controllerLoad)
            if pValue > maxPValue:
                maxPValue = pValue
                maxSwitchID = switchID
        # print (maxPValue, maxSwitchID)
        for ci in IM_S:
            maxCi = -1
            maxTValue= -9999
            # print(controllerLoad[ci] + switchLoad[maxSwitchID])
            if controllerLoad[ci] + switchLoad[maxSwitchID] < mu:
                rec = MigrationCost(controllerLoad,maxSwitchID,tempAssignment,switchLoad,controllerID,ci)
                if rec > maxTValue:
                    maxCi = ci
                    maxTValue = rec
        P.append([controllerID,maxSwitchID, maxCi])
        tempAssignment = UpdateAssignment(P, tempAssignment)

        # print("Before Assign ", beforeAssign)
        # print("After Assign ", tempAssignment)

    return tempAssignment


def SMDMAlgorithm (mu, numController, numSwitch, switchLoad, beforeAssign):
    threshold = 1.25
    triggerFlag, OM_S, IM_S, returnMatrix, controllerLoad = MigrationTrigger(mu, numController, numSwitch, switchLoad, beforeAssign, threshold)

    tempAssignment = copy.deepcopy(beforeAssign)

    maxCount = 100
    count = 0
    while triggerFlag and count < maxCount:
        tempAssignment = SwitchMiragtion(OM_S,IM_S, switchLoad, returnMatrix, controllerLoad, tempAssignment,mu)
        triggerFlag, OM_S, IM_S, returnMatrix, controllerLoad = MigrationTrigger(mu, numController, numSwitch, switchLoad, tempAssignment, threshold)
        count += 1
        print("Count", count)


    obj1,obj2 = StaticObjectives(tempAssignment,mu,numController,numSwitch,switchLoad)
    fitness = GetFitness(tempAssignment,numSwitch,switchLoad,beforeAssign)
    if fitness==0:
        print("23333333")
    # print("Before Assign ", beforeAssign)
    # print("After Assign ", tempAssignment)
    return fitness, obj1, obj2

if __name__ == "__main__":

    # Topology information
    mu = 55000  # maximum load of controller
    numController = 4
    numSwitch = 14
    switchLoad = []

    # if len(sys.argv) != 3:
    #     print("Please enter the number of controllers and switches")
    #     exec()
    #     if int(sys.argv[1]) < 2:
    #         print("The number of controllers must be greater than 2")
    #         exec()
    #     else:
    #         numController = int(sys.argv[1])
    #         numSwitch = int(sys.argv[2])

    gneticMigrationgGroup, gneticTimeGroup, gneticVarianceGroup = [], [], []
    modifiedMigrationGroup, modifiedTimeGroup, modifiedVarianceGroup = [], [], []
    simulatedMigrationGroup, simulatedTimeGroup, simulatedVarianceGroup = [], [], []
    SMDMMigrationGroup, SMDMTimeGroup, SMDMVarianceGroup = [], [], []
    iteration = 0
    while iteration < 100:
        ##########Initial Switch Load############
        c = 0
        switchLoad =[]
        for i in range(1, numSwitch + 1):
            switchLoad.append(i * 1500)

        ##########Initial Assignment############

        beforeAssign = []
        remainLoad = []

        for i in range(0, numController):
            remainLoad.append(mu)
            beforeAssign.append([])
        for i in range(0, numSwitch):
            tempNum = random.randint(0, numController - 1)
            while remainLoad[tempNum] < switchLoad[i]:
                tempNum = random.randint(0, numController - 1)
            beforeAssign[tempNum].append(i)
            remainLoad[tempNum] -= switchLoad[i]

        SMDMBefore = copy.deepcopy(beforeAssign)
        GeneticBefore = copy.deepcopy(beforeAssign)
        ModifiedBefore = copy.deepcopy(beforeAssign)
        SimulatedBefore = copy.deepcopy(beforeAssign)

        print("######SMDM Algorithm #####")  # https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7883881
        SMDMMi, SMDMTime, SMDMVariance = SMDMAlgorithm(mu, numController, numSwitch, switchLoad, SMDMBefore)

        print("######Genetic Algorithm#####")
        geMi, geTime, geViriance = GeneticAlgorithm(mu, numController, numSwitch, switchLoad, GeneticBefore)


        print("######Modified Greedy Algorithm #####")
        moMo, moTime, moViriance = ModifiedGreedyAlgorithm(mu, numController, numSwitch, switchLoad, ModifiedBefore)

        print("######Simulated Annealing Algorithm #####")
        siMo, siTime, siViriance = SimulatedAnnealing(mu, numController, numSwitch, switchLoad, SimulatedBefore)

        if (SMDMTime>0):
            SMDMMigrationGroup.append(SMDMMi)
            SMDMTimeGroup.append(SMDMTime)
            SMDMVarianceGroup.append(SMDMVariance)

            gneticMigrationgGroup.append(geMi)
            gneticTimeGroup.append(geTime)
            gneticVarianceGroup.append(geViriance)

            modifiedMigrationGroup.append(moMo)
            modifiedTimeGroup.append(moTime)
            modifiedVarianceGroup.append(moViriance)

            simulatedMigrationGroup.append(siMo)
            simulatedTimeGroup.append(siTime)
            simulatedVarianceGroup.append(siViriance)

            iteration+=1


    ############ Figures ####################

    plt.figure(figsize=(15, 12))
    plt.subplot(1, 1, 1)
    x = []
    for i in range(100):
        x.append(i)
    print(x)
    plt.plot(x, modifiedMigrationGroup, color="g", linestyle="-", marker="v", linewidth=1, label="Modified Greedy Algorithm")
    plt.plot(x, gneticMigrationgGroup, color="y", linestyle="-", marker=".", linewidth=1, label="Genetic Algorithm")
    plt.plot(x, simulatedMigrationGroup, color="b", linestyle="-", marker="s", linewidth=1, label="Simulated Annealing Algorithm ")
    plt.plot(x, SMDMMigrationGroup, color="r", linestyle="-", marker="^", linewidth=1,
             label="SMDM Algorithm ")

    plt.xlabel("Iterations", fontsize=16)
    plt.ylabel("The switch migration cost", fontsize=16)
    plt.title("Comparison For the Switch Migration Cost", fontsize=16)


    def to_percent(temp, position):
        return '%1.0f' % (100 * temp) + '%'


    plt.legend()
    plt.savefig('MigrationFigure.png', bbox_inches='tight')
    plt.show()

    plt.subplot(1, 1, 1)

    plt.plot(x, modifiedTimeGroup, color="g", linestyle="-", marker="v", linewidth=1, label="Modified Greedy Algorithm")
    plt.plot(x, gneticTimeGroup, color="y", linestyle="-", marker=".", linewidth=1, label="Genetic Algorithm")
    plt.plot(x, simulatedTimeGroup, color="b", linestyle="-", marker="s", linewidth=1, label="Simulated Annealing Algorithm ")
    plt.plot(x, SMDMTimeGroup, color="r", linestyle="-", marker="^", linewidth=1,
             label="SMDM Algorithm ")

    plt.xlabel("Iterations", fontsize=16)
    plt.ylabel("The time requests spent in the system", fontsize=16)
    plt.title("Comparison For the Time Requests Spent", fontsize=16)


    def to_percent(temp, position):
        return '%1.0f' % (100 * temp) + '%'


    plt.legend()
    plt.savefig('TimeFigure.png', bbox_inches='tight')
    plt.show()

    plt.subplot(1, 1, 1)

    plt.plot(x, modifiedVarianceGroup, color="g", linestyle="-", marker="v", linewidth=1, label="Modified Greedy Algorithm")
    plt.plot(x, gneticVarianceGroup, color="y", linestyle="-", marker=".", linewidth=1, label="Genetic Algorithm")
    plt.plot(x, simulatedVarianceGroup, color="b", linestyle="-", marker="s", linewidth=1, label="Simulated Annealing Algorithm ")
    plt.plot(x, SMDMVarianceGroup, color="r", linestyle="-", marker="^", linewidth=1,
             label="SMDM Algorithm ")
    plt.xlabel("Iterations", fontsize=16)
    plt.ylabel("The variance of controller loads", fontsize=16)
    plt.title("Comparison For the Variance of Controller Loads", fontsize=16)


    def to_percent(temp, position):
        return '%1.0f' % (100 * temp) + '%'


    plt.legend()
    plt.savefig('VarianceFigure.png', bbox_inches='tight')
    plt.show()

    plt.subplot(1, 1, 1)
    index = sorted(range(len(modifiedMigrationGroup)), key=lambda k: modifiedMigrationGroup[k])
    x_Modified, y_Modified = [], []
    for i in index:
        x_Modified.append(modifiedMigrationGroup[i])
        y_Modified.append(modifiedTimeGroup[i])
    plt.plot(x_Modified, y_Modified, color="g", linestyle="-", marker="v", linewidth=1,
             label="Modified Greedy Algorithm")

    index = sorted(range(len(gneticMigrationgGroup)), key=lambda k: gneticMigrationgGroup[k])
    x_Genetic, y_Genetic = [], []
    for i in index:
        x_Genetic.append(gneticMigrationgGroup[i])
        y_Genetic.append(gneticTimeGroup[i])

    plt.plot(x_Genetic, y_Genetic, color="y", linestyle="-", marker=".", linewidth=1, label="Genetic Algorithm")

    index = sorted(range(len(simulatedMigrationGroup)), key=lambda k: simulatedMigrationGroup[k])
    x_Simulated, y_Simulated = [], []
    for i in index:
        x_Simulated.append(simulatedMigrationGroup[i])
        y_Simulated.append(simulatedTimeGroup[i])

    plt.plot(x_Simulated, y_Simulated, color="b", linestyle="-", marker="s", linewidth=1,
             label="Simulated Annealing Algorithm ")

    index = sorted(range(len(SMDMMigrationGroup)), key=lambda k: SMDMMigrationGroup[k])
    x_SMDM, y_SMDM= [], []
    for i in index:

        x_SMDM.append(SMDMMigrationGroup[i])
        y_SMDM.append(SMDMTimeGroup[i])

    plt.plot(x_SMDM, y_SMDM, color="r", linestyle="-", marker="^", linewidth=1,
             label="SMDM Algorithm ")

    plt.xlabel("The switch migration cost", fontsize=16)
    plt.ylabel("The time requests spent in the system", fontsize=16)
    plt.title("Trade-off between the time requests spent and the switch migration cost ", fontsize=16)


    def to_percent(temp, position):
        return '%1.0f' % (100 * temp) + '%'


    plt.legend()
    plt.savefig('MigrationVsTime.png', bbox_inches='tight')
    plt.show()





    plt.subplot(1, 1, 1)

    index = sorted(range(len(modifiedMigrationGroup)), key=lambda k: modifiedMigrationGroup[k])
    x_Modified, y_Modified = [], []
    for i in index:
        x_Modified.append(modifiedMigrationGroup[i])
        y_Modified.append(modifiedVarianceGroup[i])
    plt.plot(x_Modified, y_Modified, color="g", linestyle="-", marker="v", linewidth=1,
             label="Modified Greedy Algorithm")
    # plt.plot(modifiedMigrationGroup, modifiedVarianceGroup, color="g", linestyle="-", marker="v", linewidth=1,
    #          label="Modified Greedy Algorithm")

    index = sorted(range(len(gneticMigrationgGroup)), key=lambda k: gneticMigrationgGroup[k])
    x_Genetic, y_Genetic = [], []
    for i in index:
        x_Genetic.append(gneticMigrationgGroup[i])
        y_Genetic.append(gneticVarianceGroup[i])
    plt.plot(x_Genetic, y_Genetic, color="y", linestyle="-", marker=".", linewidth=1, label="Genetic Algorithm")

    # plt.plot(gneticMigrationgGroup, gneticVarianceGroup, color="y", linestyle="-", marker=".", linewidth=1,
    #          label="Genetic Algorithm")

    index = sorted(range(len(simulatedMigrationGroup)), key=lambda k: simulatedMigrationGroup[k])
    x_Simulated, y_Simulated = [], []
    for i in index:
        x_Simulated.append(simulatedMigrationGroup[i])
        y_Simulated.append(simulatedVarianceGroup[i])

    plt.plot(x_Simulated, y_Simulated, color="b", linestyle="-", marker="s", linewidth=1,
             label="Simulated Annealing Algorithm ")
    # plt.plot(simulatedMigrationGroup, simulatedVarianceGroup, color="b", linestyle="-", marker="s", linewidth=1,
    #          label="Simulated Annealing Algorithm ")

    index = sorted(range(len(SMDMMigrationGroup)), key=lambda k: SMDMMigrationGroup[k])
    x_SMDM, y_SMDM = [], []
    for i in index:
        x_SMDM.append(SMDMMigrationGroup[i])
        y_SMDM.append(SMDMVarianceGroup[i])

    plt.plot(x_SMDM, y_SMDM, color="r", linestyle="-", marker="^", linewidth=1,
             label="SMDM Algorithm ")

    # plt.plot(SMDMMigrationGroup, SMDMVarianceGroup, color="r", linestyle="-", marker="^", linewidth=1,
    #          label="SMDM Algorithm ")

    plt.xlabel("The switch migration cost", fontsize=16)
    plt.ylabel("The variance of controller loads", fontsize=16)
    plt.title("Trade-off between the variance of controller loads and the switch migration cost ", fontsize=16)


    def to_percent(temp, position):
        return '%1.0f' % (100 * temp) + '%'


    plt.legend()
    plt.savefig('MigrationVsVariance.png', bbox_inches='tight')
    plt.show()



    plt.subplot(1, 1, 1)

    index = sorted(range(len(modifiedTimeGroup)), key=lambda k: modifiedTimeGroup[k])
    x_Modified, y_Modified = [], []
    for i in index:
        x_Modified.append(modifiedTimeGroup[i])
        y_Modified.append(modifiedVarianceGroup[i])
    plt.plot(x_Modified, y_Modified, color="g", linestyle="-", marker="v", linewidth=1,
             label="Modified Greedy Algorithm")

    # plt.plot(modifiedTimeGroup, modifiedVarianceGroup, color="g", linestyle="-", marker="v", linewidth=1,
    #          label="Modified Greedy Algorithm")

    index = sorted(range(len(gneticTimeGroup)), key=lambda k: gneticTimeGroup[k])
    x_Genetic, y_Genetic = [], []
    for i in index:
        x_Genetic.append(gneticTimeGroup[i])
        y_Genetic.append(gneticVarianceGroup[i])
    plt.plot(x_Genetic, y_Genetic, color="y", linestyle="-", marker=".", linewidth=1, label="Genetic Algorithm")

    # plt.plot(gneticTimeGroup, gneticVarianceGroup, color="y", linestyle="-", marker=".", linewidth=1,
    #          label="Genetic Algorithm")

    index = sorted(range(len(simulatedTimeGroup)), key=lambda k: simulatedTimeGroup[k])
    x_Simulated, y_Simulated = [], []
    for i in index:
        x_Simulated.append(simulatedTimeGroup[i])
        y_Simulated.append(simulatedVarianceGroup[i])

    plt.plot(x_Simulated, y_Simulated, color="b", linestyle="-", marker="s", linewidth=1,
             label="Simulated Annealing Algorithm ")
    # plt.plot(simulatedTimeGroup, simulatedVarianceGroup, color="b", linestyle="-", marker="s", linewidth=1,
    #          label="Simulated Annealing Algorithm ")

    index = sorted(range(len(SMDMTimeGroup)), key=lambda k: SMDMTimeGroup[k])
    x_SMDM, y_SMDM = [], []
    for i in index:
        x_SMDM.append(SMDMTimeGroup[i])
        y_SMDM.append(SMDMVarianceGroup[i])

    plt.plot(x_SMDM, y_SMDM, color="r", linestyle="-", marker="^", linewidth=1,
             label="SMDM Algorithm ")
    # plt.plot(SMDMTimeGroup, SMDMVarianceGroup, color="r", linestyle="-", marker="^", linewidth=1,
    #          label="SMDM Algorithm ")

    plt.xlabel("The time requests spent in the system", fontsize=16)
    plt.ylabel("The variance of controller loads", fontsize=16)
    plt.title("Trade-off between the variance of controller loads and the time requests spent", fontsize=16)


    def to_percent(temp, position):
        return '%1.0f' % (100 * temp) + '%'


    plt.legend()
    plt.savefig('TimeVsVariance.png', bbox_inches='tight')
    plt.show()

