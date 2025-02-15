package org.sbpo2025.challenge;

import org.apache.commons.lang3.time.StopWatch;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.*;

public class ChallengeSolver {
    private final long MAX_RUNTIME = 30000; // milliseconds; 30 s
    private final int NUM_THREADS = Runtime.getRuntime().availableProcessors();
    protected List<Map<Integer, Integer>> orders;
    protected List<Map<Integer, Integer>> aisles;
    protected int nItems;
    protected int waveSizeLB;
    protected int waveSizeUB;
    private Map<Integer, Set<Integer>> orderToAislesMap = new HashMap<>();

    public ChallengeSolver(
            List<Map<Integer, Integer>> orders, List<Map<Integer, Integer>> aisles, int nItems, int waveSizeLB, int waveSizeUB) {
        this.orders = orders;
        this.aisles = aisles;
        this.nItems = nItems;
        this.waveSizeLB = waveSizeLB;
        this.waveSizeUB = waveSizeUB;
    }

    public static class Individual {
        protected ArrayList<Boolean> genome;
        protected double fitness;

        public Individual(ArrayList<Boolean> genome, double fitness) {
            this.genome = genome;
            this.fitness = fitness;
        }

    }
    public void reduceAisles(Individual individual) {
        ChallengeSolution solution = decodeIndividual(individual);
        Set<Integer> selectedAisles = new HashSet<>(solution.aisles());
        Set<Integer> selectedOrders = new HashSet<>(solution.orders());

        List<Integer> aislesList = new ArrayList<>(selectedAisles);
        Collections.shuffle(aislesList); // Randomiza a ordem de remoção

        for (int aisle : aislesList) {
            Set<Integer> tempAisles = new HashSet<>(selectedAisles);
            tempAisles.remove(aisle);
            ChallengeSolution newSolution = new ChallengeSolution(selectedOrders, tempAisles);

            if (isSolutionFeasible(newSolution, false)) {
                selectedAisles.remove(aisle);
            }
        }

        // Atualiza o genoma do indivíduo
        for (int i = 0; i < aisles.size(); i++) {
            individual.genome.set(orders.size() + i, selectedAisles.contains(i));
        }

        // Atualiza o fitness do indivíduo
        individual.fitness = computeObjectiveFunction(decodeIndividual(individual));
    }

    public Individual geneticAlgorithm(int populationSize,int maxIterations) {
        //Generates the initial population
        ArrayList<Individual> population = this.generateInitialPopulation(populationSize);

        //Evaluates the population
        double bestScore;
        ChallengeSolution bestSolution;

        for (Individual individual : population) {
            ChallengeSolution solution = this.decodeIndividual(individual);
            individual.fitness = this.computeObjectiveFunction(solution);
        }

        population.sort((a, b) -> Double.compare(b.fitness, a.fitness));

        bestScore = population.get(0).fitness;
        bestSolution = this.decodeIndividual(population.get(0));

        //System.out.println("Initial best score: " + bestScore);
        //System.out.println("is feaseble?" + isSolutionFeasible(bestSolution, true));

        int currentIteration = 0;
        //Main loop
        StopWatch cronometro= new StopWatch();
        cronometro.start();
        while (cronometro.getDuration().getSeconds()<30 && currentIteration < maxIterations) {
            currentIteration++;

            //Crossover
            population = crossOverPopulation(population);

            for (Individual individual : population) {
                ChallengeSolution solution = this.decodeIndividual(individual);
                individual.fitness = this.computeObjectiveFunction(solution);
            }

            population.sort((a, b) -> Double.compare(b.fitness, a.fitness));

            //Selects the next generation
            chooseNextGeneration(population, populationSize, PopulationSelectionType.ELITIST);

            if (population.get(0).fitness > bestScore) {
                bestScore = population.get(0).fitness;
               // System.out.println("New best score achieved in iteration " + currentIteration + " best score: " + bestScore);
                //System.out.println("New solution: " + bestSolution);
                //System.out.println("is feasible? " + isSolutionFeasible(bestSolution, true));
            }

        }
        return population.get(0);
    }

    public ChallengeSolution solve(StopWatch stopWatch) {
        final int populationSize = 100;
        final int maxIterations = Integer.MAX_VALUE;
        Individual bestIndividual = geneticAlgorithm(populationSize, maxIterations);

        if(bestIndividual != null) {

            System.out.println("Melhor individuo: " + decodeIndividual(bestIndividual) +" Score:" + bestIndividual.fitness);
        }
        System.out.println("Levou:"+stopWatch.getDuration().getSeconds());
        return this.decodeIndividual(bestIndividual);

    }

    public ArrayList<Individual> generateInitialPopulation(int size) {
        ArrayList<Individual> population = new ArrayList<>();
        Random rand = new Random();

        List<Integer> orderIndexes = new ArrayList<>();
        for (int i = 0; i < orders.size(); i++) {
            orderIndexes.add(i);
        }


        for (int i = 0; i < size; i++) {
            ArrayList<Boolean> genome = new ArrayList<>(Collections.nCopies(orders.size() + aisles.size(), false));
            int pickedUnits = 0;
            List<Integer> localOrderIndexes = new ArrayList<>(orderIndexes);
            List<Integer> localAisleIndexes = new ArrayList<>();
            List<Map<Integer, Integer>> unitsTakenFromAisles = new ArrayList<>(Collections.nCopies(aisles.size(), new HashMap<>()));


            while (!localOrderIndexes.isEmpty()) {
                int orderIndex = rand.nextInt(localOrderIndexes.size());
                int pickedOrder = localOrderIndexes.get(orderIndex);
                int orderUnitsSum = orders.get(pickedOrder).values().stream().mapToInt(Integer::intValue).sum();

                localOrderIndexes.remove(orderIndex);

                if (pickedUnits + orderUnitsSum > waveSizeUB) continue;

                int satisfiedItems = 0;
                int totalItems = orders.get(pickedOrder).size();

                for (Map.Entry<Integer, Integer> entry : orders.get(pickedOrder).entrySet()) {
                    int item = entry.getKey();
                    int quantity = entry.getValue();

                    for (int aisleIndex = 0; aisleIndex < this.aisles.size(); aisleIndex++) {
                        Map<Integer, Integer> aisle = this.aisles.get(aisleIndex);
                        if (aisle.containsKey(item)) {
                            int amountTakenFromItemInAisle = unitsTakenFromAisles.get(aisleIndex).getOrDefault(item, 0);
                            int amountAvailableInAisle = aisle.get(item);
                            if (amountTakenFromItemInAisle == amountAvailableInAisle) continue;
                            if (!localAisleIndexes.contains(aisleIndex)) localAisleIndexes.add(aisleIndex);
                            int newItemInAisleAmount = Math.min(amountAvailableInAisle, amountTakenFromItemInAisle + quantity);
                            unitsTakenFromAisles.get(aisleIndex).put(item, newItemInAisleAmount);
                            genome.set(orders.size() + aisleIndex, true);
                            if (amountTakenFromItemInAisle + quantity <= amountAvailableInAisle) {
                                satisfiedItems++;
                                break;
                            }
                            quantity -= amountAvailableInAisle - amountTakenFromItemInAisle;
                        }
                    }
                }
                if (satisfiedItems == totalItems) {
                    pickedUnits += orderUnitsSum;
                    genome.set(pickedOrder, true);
                }
            }
            population.add(new Individual(genome, -1));
        }
        return population;
    }

    public ChallengeSolution decodeIndividual(Individual individual) {
        Set<Integer> selectedOrders = new HashSet<>();
        Set<Integer> selectedAisles = new HashSet<>();

        for (int i = 0; i < orders.size(); i++) {
            if (individual.genome.get(i)) {
                selectedOrders.add(i);
            }
        }

        for (int i = 0; i < aisles.size(); i++) {
            if (individual.genome.get(orders.size() + i)) {
                selectedAisles.add(i);
            }
        }

        return new ChallengeSolution(selectedOrders, selectedAisles);
    }

    public void reduceAislesOptimized(Individual individual) {
        ChallengeSolution solution = decodeIndividual(individual);
        Set<Integer> selectedAisles = new HashSet<>(solution.aisles());
        Set<Integer> selectedOrders = new HashSet<>(solution.orders());

        List<Integer> ordersList = new ArrayList<>(selectedOrders);
        Collections.shuffle(ordersList); // Randomiza a seleção dos pedidos a serem removidos

        int ordersToRemove = ordersList.size() / 2; // Remove aproximadamente metade dos pedidos
        Set<Integer> removedOrders = new HashSet<>(ordersList.subList(0, ordersToRemove));

        // Remove os pedidos selecionados
        selectedOrders.removeAll(removedOrders);

        // Identifica corredores associados aos pedidos removidos e os elimina
        Set<Integer> aislesToRemove = new HashSet<>();
        for (int aisle : selectedAisles) {
            if (!isAisleNeeded(aisle, selectedOrders)) {
                aislesToRemove.add(aisle);
            }
        }
        selectedAisles.removeAll(aislesToRemove);

        // Atualiza o genoma do indivíduo
        for (int i = 0; i < individual.genome.size(); i++) {
            if (i < ordersList.size()) {
                individual.genome.set(i, selectedOrders.contains(i));
            } else {
                individual.genome.set(i, selectedAisles.contains(i - ordersList.size()));
            }
        }

        // Atualiza o fitness do indivíduo
        individual.fitness = computeObjectiveFunction(decodeIndividual(individual));
    }

    private boolean isAisleNeeded(int aisle, Set<Integer> orders) {
        // Implementação para verificar se o corredor ainda é necessário
        for (int order : orders) {
            if (orderRequiresAisle(order, aisle)) {
                return true;
            }
        }
        return false;
    }

    private boolean orderRequiresAisle(int order, int aisle) {
        // Verifica se o pedido requer um corredor específico
        return getAislesForOrder(order).contains(aisle);
    }

    private Set<Integer> getAislesForOrder(int order) {
        // Retorna os corredores associados a um pedido
        return orderToAislesMap.getOrDefault(order, Collections.emptySet());
    }

    public void initializeOrderToAislesMap(Map<Integer, Set<Integer>> mapping) {
        this.orderToAislesMap = new HashMap<>(mapping);
    }


    public Individual crossover(Individual parent1, Individual parent2) {
        Random rand = new Random();
        ArrayList<Boolean> childGenome = new ArrayList<>(Collections.nCopies(orders.size() + aisles.size(), false));
        Set<Integer> selectedOrders = new HashSet<>();
        Set<Integer> selectedAisles = new HashSet<>();
        int totalUnits = 0;

        // Seleciona pedidos aleatoriamente de ambos os pais
        for (int i = 0; i < orders.size(); i++) {
            if ((parent1.genome.get(i) || parent2.genome.get(i)) && rand.nextBoolean()) {
                int orderUnits = orders.get(i).values().stream().mapToInt(Integer::intValue).sum();
                if (totalUnits + orderUnits <= waveSizeUB) {
                    selectedOrders.add(i);
                    totalUnits += orderUnits;
                }
            }
        }

        // Se totalUnits for menor que waveSizeLB, adiciona mais pedidos
        List<Integer> remainingOrders = new ArrayList<>();
        for (int i = 0; i < orders.size(); i++) {
            if (!selectedOrders.contains(i)) remainingOrders.add(i);
        }
        Collections.shuffle(remainingOrders);
        for (int i : remainingOrders) {
            int orderUnits = orders.get(i).values().stream().mapToInt(Integer::intValue).sum();
            if (totalUnits + orderUnits <= waveSizeUB) {
                selectedOrders.add(i);
                totalUnits += orderUnits;
                if (totalUnits >= waveSizeLB) break;
            }
        }

        // Ativa corredores correspondentes aos pedidos selecionados
        for (int orderIndex : selectedOrders) {
            childGenome.set(orderIndex, true);
            for (Map.Entry<Integer, Integer> entry : orders.get(orderIndex).entrySet()) {
                int item = entry.getKey();
                for (int aisleIndex = 0; aisleIndex < aisles.size(); aisleIndex++) {
                    if (aisles.get(aisleIndex).containsKey(item)) {
                        selectedAisles.add(aisleIndex);
                    }
                }
            }
        }

        for (int aisleIndex : selectedAisles) {
            childGenome.set(orders.size() + aisleIndex, true);
        }
        ChallengeSolution atual = new ChallengeSolution(selectedOrders, selectedAisles);
        Individual child = new Individual(childGenome, computeObjectiveFunction(atual));
        mutate(child,1);
        return child;
    }


    public void mutate(Individual individual,int type) {
        Random rand = new Random();
        if (rand.nextDouble() > 0.1) return;
        if(type == 0) {
            int numMutations = Math.max(1, individual.genome.size() / 10);
            Set<Integer> mutationPoints = new HashSet<>();

            while (mutationPoints.size() < numMutations) {
                mutationPoints.add(rand.nextInt(individual.genome.size()));
            }

            for (int mutationPoint : mutationPoints) {
                individual.genome.set(mutationPoint, !individual.genome.get(mutationPoint));
            }
        }
        else if (type == 1) {
            reduceAisles(individual);
        }

    }

    /*
     * Recieves an already sorted population
     * Returns a population of size n * 1.5
     */
    public ArrayList<Individual> crossOverPopulation(ArrayList<Individual> population) {
        ArrayList<Individual> newPopulation = new ArrayList<>();
        Random rand = new Random();
        while (!population.isEmpty()) {
            Individual parent1 = population.remove(0);
            Individual parent2;
            newPopulation.add(parent1);
            //biased for loop
            if (population.isEmpty()) break;
            for (int i = 0; i < population.size(); i++) {
                if (rand.nextDouble() < 0.5 || i == population.size() - 1) {
                    parent2 = population.remove(i);
                    newPopulation.add(parent2);
                    newPopulation.add(crossover(parent1, parent2));
                }
            }
        }
        return newPopulation;

    }

    public Individual putMoreOrders(Individual individual) {
        Map<Integer, Integer> usedItems = new HashMap<>();
        int totalItems = 0;
        
        for (int i = 0; i < nItems; i++) {
            usedItems.put(i, 0);
        }
        
        for (int i = 0; i < orders.size(); i++) {
            if (individual.genome.get(i)) {
                Map<Integer, Integer> order = orders.get(i);
                for (Map.Entry<Integer, Integer> entry : order.entrySet()) {
                    usedItems.put(entry.getKey(), usedItems.get(entry.getKey()) + entry.getValue());
                    totalItems += entry.getValue();
                }
            }
        }

        Map<Integer, Integer> availableItems = new HashMap<>();
        
        for (int i = 0; i < nItems; i++) {
            availableItems.put(i, 0);
        }
        
        for (int i = 0; i < aisles.size(); i++) {
            if (individual.genome.get(orders.size() + i)) {
                Map<Integer, Integer> aisle = aisles.get(i);
                for (Map.Entry<Integer, Integer> entry : aisle.entrySet()) {
                    availableItems.put(entry.getKey(), availableItems.get(entry.getKey()) + entry.getValue());
                }
            }
        }

        for (Map.Entry<Integer, Integer> entry : usedItems.entrySet()) {
            availableItems.put(entry.getKey(), availableItems.get(entry.getKey()) - entry.getValue());
        }
        
        boolean added;
        do {
            added = false;
            
            for (int i = 0; i < orders.size(); i++) {
                if (!individual.genome.get(i)) {
                    Map<Integer, Integer> order = orders.get(i);
                    boolean canBeAdded = true;
                    int orderTotalItems = 0;
                    Map<Integer, Integer> tempAvailableItems = new HashMap<>(availableItems);
                    
                    for (Map.Entry<Integer, Integer> entry : order.entrySet()) {
                        if (tempAvailableItems.get(entry.getKey()) < entry.getValue()) {
                            canBeAdded = false;
                            break;
                        }
                        tempAvailableItems.put(entry.getKey(), tempAvailableItems.get(entry.getKey()) - entry.getValue());
                        orderTotalItems += entry.getValue();
                    }
                    
                    if (canBeAdded && (totalItems + orderTotalItems) <= waveSizeUB) {
                        individual.genome.set(i, true);
                        totalItems += orderTotalItems;
                        availableItems = tempAvailableItems;
                        added = true;
                    }
                }
            }
        } while (added);
        
        individual.fitness = this.computeObjectiveFunction(decodeIndividual(individual));

        return individual;
    }
    
    
    
    

    /*
     * Get the remaining time in seconds
     */
    protected long getRemainingTime(StopWatch stopWatch) {
        return Math.max(
                TimeUnit.SECONDS.convert(MAX_RUNTIME - stopWatch.getTime(TimeUnit.MILLISECONDS), TimeUnit.MILLISECONDS),
                0);
    }

    protected boolean isSolutionFeasible(ChallengeSolution challengeSolution, boolean print) {
        Set<Integer> selectedOrders = challengeSolution.orders();
        Set<Integer> visitedAisles = challengeSolution.aisles();
        if (selectedOrders == null || visitedAisles == null || selectedOrders.isEmpty() || visitedAisles.isEmpty()) {
            return false;
        }

        int[] totalUnitsPicked = new int[nItems];
        int[] totalUnitsAvailable = new int[nItems];

        // Calculate total units picked
        for (int order : selectedOrders) {
            for (Map.Entry<Integer, Integer> entry : orders.get(order).entrySet()) {
                totalUnitsPicked[entry.getKey()] += entry.getValue();
            }
        }

        // Calculate total units available
        for (int aisle : visitedAisles) {
            for (Map.Entry<Integer, Integer> entry : aisles.get(aisle).entrySet()) {
                totalUnitsAvailable[entry.getKey()] += entry.getValue();
            }
        }

        // Check if the total units picked are within bounds
        int totalUnits = Arrays.stream(totalUnitsPicked).sum();
        if (totalUnits < waveSizeLB) {
            if (print) {
                System.out.println("Motive: totalUnits < waveSizeLB");
            }
            return false;
        }
        if (totalUnits > waveSizeUB) {
            if (print) {
                System.out.println("Motive: totalUnits > waveSizeUB");
            }
            return false;
        }

        // Check if the units picked do not exceed the units available
        for (int i = 0; i < nItems; i++) {
            if (totalUnitsPicked[i] > totalUnitsAvailable[i]) {
                if (print) {
                    System.out.println("Motive: More Picked Items than offered by Aisles");
                }
                return false;
            }
        }

        return true;
    }

    protected double computeObjectiveFunction(ChallengeSolution challengeSolution) {
        Set<Integer> selectedOrders = challengeSolution.orders();
        Set<Integer> visitedAisles = challengeSolution.aisles();
        if (selectedOrders == null || visitedAisles == null || selectedOrders.isEmpty() || visitedAisles.isEmpty()) {
            return 0.0;
        }

        int penalty = 0;
        if (!isSolutionFeasible(challengeSolution, false)) {
            penalty -= applyPenalty(challengeSolution);
        }

        int totalUnitsPicked = 0;

        // Calculate total units picked
        for (int order : selectedOrders) {
            totalUnitsPicked += orders.get(order).values().stream()
                    .mapToInt(Integer::intValue)
                    .sum();
        }

        // Calculate the number of visited aisles
        int numVisitedAisles = visitedAisles.size();

        // Objective function: total units picked / number of visited aisles
        return (double) (totalUnitsPicked / numVisitedAisles) + penalty;
    }

    protected int applyPenalty(ChallengeSolution challengeSolution) {
        int penalty = 0;
        Set<Integer> selectedOrders = challengeSolution.orders();
        Set<Integer> visitedAisles = challengeSolution.aisles();

        int[] totalUnitsPicked = new int[nItems];
        int[] totalUnitsAvailable = new int[nItems];

        // Calculate total units picked
        for (int order : selectedOrders) {
            for (Map.Entry<Integer, Integer> entry : orders.get(order).entrySet()) {
                totalUnitsPicked[entry.getKey()] += entry.getValue();
            }
        }

        // Calculate total units available
        for (int aisle : visitedAisles) {
            for (Map.Entry<Integer, Integer> entry : aisles.get(aisle).entrySet()) {
                totalUnitsAvailable[entry.getKey()] += entry.getValue();
            }
        }

        // Check if the total units picked are within bounds
        int totalUnits = Arrays.stream(totalUnitsPicked).sum();
        if (totalUnits < waveSizeLB) {
            penalty += (waveSizeLB - totalUnits) * nItems;
        }

        if (totalUnits > waveSizeUB) {
            penalty += (totalUnits - waveSizeLB) * nItems;
        }

        // Check if the units picked do not exceed the units available
        for (int i = 0; i < nItems; i++) {
            if (totalUnitsPicked[i] > totalUnitsAvailable[i]) {
                penalty += (totalUnitsPicked[i] - totalUnitsAvailable[i]) * nItems;
            }
        }

        return penalty;

    }

    public enum PopulationSelectionType {
        ELITIST, TOURNAMENT
    }

    public static void chooseNextGeneration(ArrayList<Individual> population, int populationSize, PopulationSelectionType selectionType) {
        if (selectionType == PopulationSelectionType.ELITIST) {
            population.subList(populationSize, population.size()).clear();
        } else if (selectionType == PopulationSelectionType.TOURNAMENT) {
            Random rand = new Random();
            while (population.size() > populationSize) {
                int index1 = rand.nextInt(population.size());
                int index2 = rand.nextInt(population.size());
                if (population.get(index1).fitness > population.get(index2).fitness) {
                    population.remove(index2);
                } else {
                    population.remove(index1);
                }
            }
        }
    }
}
