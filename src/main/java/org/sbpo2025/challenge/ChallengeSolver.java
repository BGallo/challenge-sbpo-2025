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

public class ChallengeSolver {
    private final long MAX_RUNTIME = 10000; // milliseconds; 30 s

    protected List<Map<Integer, Integer>> orders;
    protected List<Map<Integer, Integer>> aisles;
    protected int nItems;
    protected int waveSizeLB;
    protected int waveSizeUB;

    public ChallengeSolver(
            List<Map<Integer, Integer>> orders, List<Map<Integer, Integer>> aisles, int nItems, int waveSizeLB, int waveSizeUB) {
        this.orders = orders;
        this.aisles = aisles;
        this.nItems = nItems;
        this.waveSizeLB = waveSizeLB;
        this.waveSizeUB = waveSizeUB;
    }

    public class Individual {
        protected ArrayList<Boolean> genome;
        protected double fitness;

        public Individual(ArrayList<Boolean> genome, double fitness) {
            this.genome = genome;
            this.fitness = fitness;
        }
    }

    public ChallengeSolution solve(StopWatch stopWatch) {
        final int populationSize = 40;
        final int maxIterations = Integer.MAX_VALUE;

        //Generates the initial population
        ArrayList<Individual> population = this.generateInitialPopulation(populationSize);
        
        //Evaluates the population
        double bestScore = Integer.MIN_VALUE;
        ChallengeSolution bestSolution = null;

        for(Individual individual: population){
            ChallengeSolution solution = this.decodeIndividual(individual);
            double fitness = this.computeObjectiveFunction(solution);
            individual.fitness = fitness;   
        }

        population.sort((a, b) -> Double.compare(b.fitness, a.fitness));

        bestScore = population.get(0).fitness;
        bestSolution = this.decodeIndividual(population.get(0));
        System.out.println("Initial best score: " + bestScore);
        System.out.println("Is Initial best score feasible: " + isSolutionFeasible(bestSolution, true));
        //System.out.println("Initial Best solution: " + bestSolution);

        int currentIteration = 0;
        //Main loop
        while(stopWatch.getTime() < MAX_RUNTIME && currentIteration < maxIterations){
            currentIteration++;

            //One point crossover
            population = crossOverPopulation(population);

            for(Individual individual: population){
                ChallengeSolution solution = this.decodeIndividual(individual);
                double fitness = this.computeObjectiveFunction(solution);
                individual.fitness = fitness;   
            }

            population.sort((a, b) -> Double.compare(b.fitness, a.fitness));

            //Elitist population selection
            population.subList(populationSize, population.size()).clear();

            if(population.get(0).fitness > bestScore){
                bestScore = population.get(0).fitness;
                bestSolution = this.decodeIndividual(population.get(0));
            }
        }

        System.out.println("Best score: " + bestScore);
        //System.out.println("Best solution: " + bestSolution);
        System.out.println("Is Best solution feasible: " + isSolutionFeasible(bestSolution, true));
        return bestSolution;
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


            while (localOrderIndexes.size() > 0) {
                int orderIndex = rand.nextInt(localOrderIndexes.size());
                int pickedOrder = localOrderIndexes.get(orderIndex);
                int orderUnitsSum = orders.get(pickedOrder).values().stream().mapToInt(Integer::intValue).sum();
                
                localOrderIndexes.remove(orderIndex);

                if (pickedUnits + orderUnitsSum > waveSizeUB) continue;

                int satisfiedItems = 0;
                int totalItems = orders.get(pickedOrder).size();

                for(Map.Entry<Integer, Integer> entry: orders.get(pickedOrder).entrySet()){
                    int item = entry.getKey();
                    int quantity = entry.getValue();

                    for(int aisleIndex = 0; aisleIndex < this.aisles.size(); aisleIndex++){
                        Map<Integer, Integer> aisle = this.aisles.get(aisleIndex);
                        if(aisle.containsKey(item)){
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

    //One point crossover
    public Individual crossover(Individual parent1, Individual parent2) {
        Random rand = new Random();
        int crossoverPoint = rand.nextInt(parent1.genome.size());
        ArrayList<Boolean> childGenome = new ArrayList<>();
        for (int i = 0; i < parent1.genome.size(); i++) {
            if (i < crossoverPoint) {
                childGenome.add(parent1.genome.get(i));
            } else {
                childGenome.add(parent2.genome.get(i));
            }
        }
        Individual child =  new Individual(childGenome, -1);
        mutate(child);
        return child;
    }


    public void mutate(Individual individual) {
        Random rand = new Random();
        if (rand.nextDouble() > 0.1) return;
        int numMutations = Math.max(1, individual.genome.size() / 10);
        Set<Integer> mutationPoints = new HashSet<>();
        
        while (mutationPoints.size() < numMutations) {
            mutationPoints.add(rand.nextInt(individual.genome.size()));
        }
        
        for (int mutationPoint : mutationPoints) {
            individual.genome.set(mutationPoint, !individual.genome.get(mutationPoint));
        }
    }

    /*
     * Recieves an already sorted population
     * Returns a population of size n * 1.5
     */
    public ArrayList<Individual> crossOverPopulation(ArrayList<Individual> population) {
        ArrayList<Individual> newPopulation = new ArrayList<>();
        Random rand = new Random();
        while(population.size() > 0){
            Individual parent1 = population.remove(0);
            Individual parent2;
            newPopulation.add(parent1);
            //biased for loop
            if (population.size() == 0) break;
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

   /*  public ArrayList<ArrayList<Boolean>> generateInitialPopulation(int size){

        ArrayList<ArrayList<boolean>> population;
        Map<Integer,Integer> itemSums = new Map<>();
        int index = 0;
        for(Map<Integer,Integer> order:this.orders){
            int total = order.values().stream().mapToInt(Integer::intValue).sum();
            itemSums.put(index++,total);
        }
        for(int i = 0; i < size; i++){
            int waveSize = 0;
            List<Map.Entry<K,V>> sorted = itemSums.entrySet().stream().sorted(Map.Entry.comparingByValue()).collect(Collectors.toList());

            ArrayList<boolean> individual;
            while (waveSize < this.waveSizeUB){
                Random random = new Random();
                if(random.nextInt()){
                    int order = random.nextInt(sorted.lenght);
                    int selectedOrderIndex = sorted[order].getKey();
                    Map<Integer, Integer> selectedOrder = this.orders[selectedOrderIndex];
                }
            }
        }

        ArrayList<ArrayList<Boolean>> population = new ArrayList<>();
        Random rand = new Random();

        for (int i = 0; i < size; i++) {
            ArrayList<Boolean> individual = new ArrayList<>(Collections.nCopies(orders.size() + aisles.size(), false));
            Set<Integer> selectedOrders = new HashSet<>();
            Set<Integer> selectedAisles = new HashSet<>();
            Map<Integer, Integer> itemAvailability = new HashMap<>();
            int totalItems = 0;
            boolean foundItem = true;
            int oldTotalItems = 0;
            System.out.println("i:"+i);
            while (totalItems < waveSizeUB && foundItem) {
                int orderIndex = rand.nextInt(orders.size());
                if (selectedOrders.contains(orderIndex)) continue;

                Map<Integer, Integer> order = orders.get(orderIndex);
                int orderItems = order.values().stream().mapToInt(Integer::intValue).sum();

                if (totalItems + orderItems > waveSizeUB) continue;

                selectedOrders.add(orderIndex);
                individual.set(orderIndex, true);
                oldTotalItems=totalItems;
                totalItems += orderItems;
                System.out.println(totalItems);

                if(totalItems==oldTotalItems){
                    System.out.println("chegou aqui");
                    foundItem = false;
                    break;
                };
                System.out.println(foundItem);
                for (Map.Entry<Integer, Integer> entry : order.entrySet()) {
                    int item = entry.getKey();
                    int quantity = entry.getValue();
                    int remainingNeeded = quantity - itemAvailability.getOrDefault(item, 0);

                    if (remainingNeeded > 0) {
                        for (int aisleIndex = 0; aisleIndex < aisles.size(); aisleIndex++) {
                            if (selectedAisles.contains(aisleIndex)) continue;

                            Map<Integer, Integer> aisle = aisles.get(aisleIndex);
                            if (aisle.containsKey(item)) {
                                selectedAisles.add(aisleIndex);
                                individual.set(orders.size() + aisleIndex, true);
                                int aisleQuantity = aisle.get(item);
                                itemAvailability.put(item, itemAvailability.getOrDefault(item, 0) + aisleQuantity);

                                if (itemAvailability.get(item) >= quantity) break;
                            }
                        }
                    }
                }
            }
            population.add(individual);
        }
        return population;
    } */

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
            if(print){
                System.out.println("Motive: totalUnits < waveSizeLB");
            }
            return false;
        }
        if (totalUnits > waveSizeUB) {
            if(print){
                System.out.println("Motive: totalUnits < waveSizeLB");
            }
            return false;
        }

        // Check if the units picked do not exceed the units available
        for (int i = 0; i < nItems; i++) {
            if (totalUnitsPicked[i] > totalUnitsAvailable[i]) {
                if(print){
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

    protected int applyPenalty(ChallengeSolution challengeSolution){
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
            penalty += waveSizeLB - totalUnits;
        }

        if (totalUnits > waveSizeUB) {
            penalty += totalUnits - waveSizeLB;
        }

        // Check if the units picked do not exceed the units available
        for (int i = 0; i < nItems; i++) {
            if (totalUnitsPicked[i] > totalUnitsAvailable[i]) {
                penalty += totalUnitsPicked[i] - totalUnitsAvailable[i]; 
            }
        }

        return penalty/10;

    }
}
