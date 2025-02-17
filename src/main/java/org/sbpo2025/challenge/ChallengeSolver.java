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
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class ChallengeSolver {
    private final long MAX_RUNTIME = 30000; // milliseconds; 30 s

    protected List<Map<Integer, Integer>> orders;
    protected List<Map<Integer, Integer>> aisles;
    protected int nItems;
    protected int waveSizeLB;
    protected int waveSizeUB;

    public ChallengeSolver(
            List<Map<Integer, Integer>> orders, List<Map<Integer, Integer>> aisles, int nItems, int waveSizeLB,
            int waveSizeUB) {
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

    public ChallengeSolution solve(StopWatch stopWatch) {
        final int populationSize = 50;
        final int maxIterations = Integer.MAX_VALUE;

        // Generates the initial population
        ArrayList<Individual> population = this.generateInitialPopulation(populationSize);

        // Evaluates the population
        double bestScore;
        ChallengeSolution bestSolution;

        population.sort((a, b) -> Double.compare(b.fitness, a.fitness));

        bestScore = population.get(0).fitness;
        bestSolution = this.decodeGenome(population.get(0).genome);

        System.out.println("Initial best score: " + bestScore);
        System.out.println("Initial Solution: " + bestSolution);
        System.out.println("is feaseble?" + isSolutionFeasible(bestSolution, true));

        int currentIteration = 0;
        // Main loop
        while (stopWatch.getTime() < MAX_RUNTIME && currentIteration < maxIterations) {
            currentIteration++;

            // One point smart crossover
            population = crossOverPopulation(population);

            population.sort((a, b) -> Double.compare(b.fitness, a.fitness));

            // Selects the next generation
            chooseNextGeneration(population, populationSize, PopulationSelectionType.TOURNAMENT);

            if (population.get(0).fitness > bestScore) {
                bestScore = population.get(0).fitness;
                bestSolution = this.decodeGenome(population.get(0).genome);
                System.out.println("New best score: " + bestScore + " at iteration " + currentIteration);
                System.out.println("Solution: " + bestSolution);
                System.out.println("is feaseble?" + isSolutionFeasible(bestSolution, true));
            }
        }

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
            List<Map<Integer, Integer>> unitsTakenFromAisles = new ArrayList<>(
                    Collections.nCopies(aisles.size(), new HashMap<>()));

            while (!localOrderIndexes.isEmpty()) {
                int orderIndex = rand.nextInt(localOrderIndexes.size());
                int pickedOrder = localOrderIndexes.get(orderIndex);
                int orderUnitsSum = orders.get(pickedOrder).values().stream().mapToInt(Integer::intValue).sum();

                localOrderIndexes.remove(orderIndex);

                if (pickedUnits + orderUnitsSum > waveSizeUB)
                    continue;

                int satisfiedItems = 0;
                int totalItems = orders.get(pickedOrder).size();

                ArrayList<Integer> selectedAisles = new ArrayList<>();
                for (Map.Entry<Integer, Integer> entry : orders.get(pickedOrder).entrySet()) {
                    int item = entry.getKey();
                    int quantity = entry.getValue();

                    for (int aisleIndex = 0; aisleIndex < this.aisles.size(); aisleIndex++) {
                        Map<Integer, Integer> aisle = this.aisles.get(aisleIndex);
                        if (aisle.containsKey(item)) {
                            int amountTakenFromItemInAisle = unitsTakenFromAisles.get(aisleIndex).getOrDefault(item, 0);
                            int amountAvailableInAisle = aisle.get(item);
                            if (amountTakenFromItemInAisle == amountAvailableInAisle)
                                continue;
                            if (!localAisleIndexes.contains(aisleIndex))
                                localAisleIndexes.add(aisleIndex);
                            int newItemInAisleAmount = Math.min(amountAvailableInAisle,
                                    amountTakenFromItemInAisle + quantity);
                            unitsTakenFromAisles.get(aisleIndex).put(item, newItemInAisleAmount);

                            selectedAisles.add(aisleIndex);

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
                    for (int aisle : selectedAisles) {
                        genome.set(orders.size() + aisle, true);
                    }
                } else {
                    selectedAisles.clear();
                }
            }

            if (pickedUnits < waveSizeLB) {
                i--;
                continue;
            }

            double fitness = computeObjectiveFunction(decodeGenome(genome));
            population.add(new Individual(genome, fitness));
        }

        return population;
    }

    public ChallengeSolution decodeGenome(ArrayList<Boolean> genome) {
        Set<Integer> selectedOrders = new HashSet<>();
        Set<Integer> selectedAisles = new HashSet<>();

        for (int i = 0; i < orders.size(); i++) {
            if (genome.get(i)) {
                selectedOrders.add(i);
            }
        }

        for (int i = 0; i < aisles.size(); i++) {
            if (genome.get(orders.size() + i)) {
                selectedAisles.add(i);
            }
        }

        return new ChallengeSolution(selectedOrders, selectedAisles);
    }

    // two point proportional crossover
    public Individual crossover(Individual parent1, Individual parent2) {
        Random rand = new Random();
        ArrayList<Boolean> childGenome = new ArrayList<>(Collections.nCopies(orders.size() + aisles.size(), false));

        int orderCrossoverPoint = rand.nextInt(this.orders.size());
        int aisleCrossoverPoint = (orderCrossoverPoint * this.aisles.size()) / this.orders.size();

        for (int i = 0; i < parent1.genome.size(); i++) {
            Individual parent;
            if (i < this.orders.size()) {
                parent = i < orderCrossoverPoint ? parent1 : parent2;
            } else {
                parent = i < orders.size() + aisleCrossoverPoint ? parent1 : parent2;
            }

            boolean parentGene = parent.genome.get(i);
            childGenome.set(i, parentGene);

        }

        Individual child = new Individual(childGenome, -1);

        mutate(child);
        return child;
    }

    public void mutate(Individual individual) {
        Random rand = new Random();
        if (rand.nextFloat() > 0.1) {
            // Mess with orders
            if (rand.nextBoolean()) {
                // Remove a random order
                if (rand.nextBoolean()) {
                    if (individual.genome.subList(0, this.orders.size()).stream().anyMatch(b -> b)) {

                        int orderIndex = rand.nextInt(orders.size());

                        while (!individual.genome.get(orderIndex)) {
                            orderIndex = rand.nextInt(orders.size());
                        }

                        individual.genome.set(orderIndex, false);
                    }
                    // Try to add an order without breaking the rules
                } else {
                    ArrayList<Integer> localOrderIndexes = IntStream.rangeClosed(0, orders.size() - 1).boxed()
                            .collect(Collectors.toCollection(ArrayList::new));
                    localOrderIndexes.sort((a, b) -> rand.nextInt(3) - 1);
                    List<Integer> selectedOrders = IntStream.range(0, this.orders.size())
                            .filter(i -> individual.genome.get(i)).boxed().collect(Collectors.toList());
                    List<Integer> selectedAisles = IntStream.range(0, this.aisles.size())
                            .filter(i -> individual.genome.get(i + this.orders.size())).boxed()
                            .collect(Collectors.toList());

                    while (!localOrderIndexes.isEmpty()) {
                        int orderIndex = localOrderIndexes.remove(0);

                        if (selectedOrders.contains(orderIndex))
                            continue;

                        boolean canAdd = true;
                        int totalQuantity = 0;
                        for (Map.Entry<Integer, Integer> entry : this.orders.get(orderIndex).entrySet()) {
                            int item = entry.getKey();
                            int quantity = entry.getValue();

                            totalQuantity += quantity;

                            int demand = selectedOrders.stream().mapToInt(i -> orders.get(i).getOrDefault(item, 0))
                                    .sum();
                            int supply = selectedAisles.stream().mapToInt(i -> aisles.get(i).getOrDefault(item, 0))
                                    .sum();

                            if (demand + quantity > supply) {
                                canAdd = false;
                                break;
                            }
                        }

                        int pickedUnits = selectedOrders.stream()
                                .mapToInt(i -> orders.get(i).values().stream().mapToInt(Integer::intValue).sum()).sum();
                        canAdd = canAdd && pickedUnits + totalQuantity <= waveSizeUB;

                        if (canAdd) {
                            individual.genome.set(orderIndex, true);
                            break;
                        }

                    }
                }
            // Mess with aisles
            } else {
                //Add a random aisle
                if(rand.nextBoolean()) {
                    if (individual.genome.subList(this.orders.size(),individual.genome.size()).stream().anyMatch(b -> !b)) {

                        int aisleIndex = rand.nextInt(aisles.size());

                        while (!individual.genome.get(aisleIndex)) {
                            aisleIndex = rand.nextInt(orders.size());
                        }

                        individual.genome.set(aisleIndex, true);
                    }
                // Try to remove an aisle without breaking the rules
                } else {

                }
            }
        }
        double fitness = computeObjectiveFunction(decodeGenome(individual.genome));
        individual.fitness = fitness;
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

            // biased for loop
            if (population.isEmpty())
                break;
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

    public static enum PopulationSelectionType {
        ELITIST, TOURNAMENT
    }

    public static void chooseNextGeneration(ArrayList<Individual> population, int populationSize,
            PopulationSelectionType selectionType) {
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
