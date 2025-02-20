package org.sbpo2025.challenge;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.BitSet;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.concurrent.ThreadLocalRandom;
import java.util.concurrent.TimeUnit;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import org.apache.commons.lang3.time.StopWatch;

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
        protected BitSet genome;
        protected double fitness;
        protected HashMap<Integer, Integer> offeredItems;
        protected HashMap<Integer, Integer> pickedItems;
        protected int totalPickedItems;

        public Individual(BitSet genome, double fitness) {
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
        System.out.println("is feaseble?" + isSolutionFeasible(bestSolution, true));

        int currentIteration = 0;
        int iterationsSinceLastImprovement = 0;
        int genomeSize = orders.size() + aisles.size();
        // Main loop
        while (stopWatch.getTime() < MAX_RUNTIME && currentIteration < maxIterations) {
            currentIteration++;

            int factor = Math.max(Math.min((int) ((Math.sqrt(iterationsSinceLastImprovement / 5000.0)) * genomeSize), genomeSize / 4), 1);

            // One point smart crossover
            population = crossOverPopulation(population, factor);

            population.sort((a, b) -> Double.compare(b.fitness, a.fitness));

            System.out.println("Iteration: " + currentIteration + " AVG: " + population.stream().mapToDouble(i -> i.fitness).average().getAsDouble());

            // Selects the next generation
            chooseNextGeneration(population, populationSize, PopulationSelectionType.BEST_THEN_RANDOM);

            if (population.get(0).fitness > bestScore) {
                bestScore = population.get(0).fitness;
                bestSolution = this.decodeGenome(population.get(0).genome);
                System.out.println("New best solution found: " + bestScore + " in iteration: " + currentIteration);
                iterationsSinceLastImprovement = 0;
            }

            iterationsSinceLastImprovement++;
        }

        System.out.println("Final best score: " + bestScore);
        System.out.println("is feaseble?" + isSolutionFeasible(bestSolution, true));
        return bestSolution;
    }

    public ArrayList<Individual> generateInitialPopulation(int size) {
        List<Integer> orderIndexes = new ArrayList<>();
        for (int i = 0; i < orders.size(); i++) {
            orderIndexes.add(i);
        }

        return IntStream.range(0, size).parallel().mapToObj(_i -> createIndividual(orderIndexes))
                .collect(Collectors.toCollection(ArrayList::new));
    }

    public Individual createIndividual(List<Integer> orderIndexes) {
        boolean added = false;

        while (!added) {
            added = true;

            BitSet genome = new BitSet(orders.size() + aisles.size());

            int pickedUnits = 0;
            List<Integer> localOrderIndexes = new ArrayList<>(orderIndexes);
            List<Integer> localAisleIndexes = new ArrayList<>();

            List<Map<Integer, Integer>> unitsTakenFromAisles = new ArrayList<>(aisles.size());
            for (int i = 0; i < aisles.size(); i++) {
                unitsTakenFromAisles.add(new HashMap<>());
            }

            while (!localOrderIndexes.isEmpty()) {
                int orderIndex = ThreadLocalRandom.current().nextInt(localOrderIndexes.size());
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
                            int amountTakenFromItemInAisle = unitsTakenFromAisles.get(aisleIndex).getOrDefault(
                                    item,
                                    0);
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
                added = false;
                continue;
            }

            // Final touches
            double fitness = computeObjectiveFunction(decodeGenome(genome));
            Individual individual = new Individual(genome, fitness);
            calcOfferAndDemand(individual);
            return individual;
        }

        return null;
    }

    public void calcOfferAndDemand(Individual individual) {
        HashMap<Integer, Integer> pickedItems = new HashMap<>();
        for (int i = 0; i < orders.size(); i++) {
            if (individual.genome.get(i)) {
                for (Map.Entry<Integer, Integer> entry : orders.get(i).entrySet()) {
                    int item = entry.getKey();
                    int quantity = entry.getValue();
                    pickedItems.put(item, pickedItems.getOrDefault(item, 0) + quantity);
                }
            }
        }

        HashMap<Integer, Integer> offeredItems = new HashMap<>();
        for (int i = 0; i < aisles.size(); i++) {
            if (individual.genome.get(orders.size() + i)) {
                for (Map.Entry<Integer, Integer> entry : aisles.get(i).entrySet()) {
                    int item = entry.getKey();
                    int quantity = entry.getValue();
                    offeredItems.put(item, offeredItems.getOrDefault(item, 0) + quantity);
                }
            }
        }

        individual.offeredItems = offeredItems;
        individual.pickedItems = pickedItems;
        individual.totalPickedItems = pickedItems.values().stream().mapToInt(Integer::intValue).sum();
    }

    public ChallengeSolution decodeGenome(BitSet genome) {
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
    public Individual crossover(Individual parent1, Individual parent2, int factor) {
        Random rand = new Random();
        BitSet childGenome = new BitSet(this.orders.size() + this.aisles.size());

        int crossoverPoint = rand.nextInt(this.orders.size());
        ArrayList<Integer> choosenOrders = new ArrayList<>();

        for (int i = 0; i < this.orders.size(); i++) {
            Individual parent = i < crossoverPoint ? parent1 : parent2;
            
            boolean parentGene = parent.genome.get(i);
            childGenome.set(i, parentGene);
            if(parentGene) choosenOrders.add(i);
        }

        //build demand
        HashMap<Integer, Integer> pickedItems = new HashMap<>();
        for (int i = 0; i < this.orders.size(); i++) {
            if (childGenome.get(i)) {
                for (Map.Entry<Integer, Integer> entry : this.orders.get(i).entrySet()) {
                    int item = entry.getKey();
                    int quantity = entry.getValue();
                    pickedItems.put(item, pickedItems.getOrDefault(item, 0) + quantity);
                }
            }
        }

        //build matching aisles for offer
        ArrayList<Integer> aislesIndexes = IntStream.range(0, this.aisles.size()).boxed().collect(Collectors.toCollection(ArrayList::new));
        Collections.shuffle(aislesIndexes);

        ArrayList<Integer> choosenAisles = new ArrayList<>();
        
        boolean added = true;
        for (Map.Entry<Integer, Integer> entry : pickedItems.entrySet()) {
            int item = entry.getKey();
            int quantity = entry.getValue();
            int _i = 0;
            while (quantity > 0 && _i < aislesIndexes.size()) {
                Integer aislesIndex = aislesIndexes.get(_i);
                Map<Integer, Integer> aisle = this.aisles.get(aislesIndex);

                if (aisle.containsKey(item)) {
                    int quantityInAisle = aisle.get(item);
                    int quantityToTake = Math.min(quantity, quantityInAisle);
                    quantity -= quantityToTake;
                    choosenAisles.add(aislesIndex);
                }
                
                _i++;
            }

            if(quantity > 0) {
                added = false;
                break;
            }  
        }

        Individual child;
        if(added) {
            for (int i = 0; i < this.aisles.size(); i++) {
                if (choosenAisles.contains(i)) {
                    childGenome.set(this.orders.size() + i, true);
                }
            }

            child = new Individual(childGenome, -1);
        } else {
            child = new Individual(parent1.genome, -1);
        }

        calcOfferAndDemand(child);
        mutate(child, factor);

        return child;
    }

    public void mutate(Individual individual, int factor) {
        Random rand = new Random();
        if (rand.nextFloat() < 0.25) {
            // Mess with orders
            if (rand.nextBoolean()) {
                // Remove a random order
                if (rand.nextFloat() < 0.25) {
                    while (factor > 0) {
                        int firstSetBit = individual.genome.nextSetBit(0);
                        boolean anyMatch = (firstSetBit != -1 && firstSetBit < orders.size());

                        if (anyMatch) {
                            int orderIndex = rand.nextInt(orders.size());

                            while (!individual.genome.get(orderIndex)) {
                                orderIndex = rand.nextInt(orders.size());
                            }

                            Map<Integer, Integer> order = orders.get(orderIndex);

                            int totalItems = 0;

                            for (Map.Entry<Integer, Integer> entry : order.entrySet()) {
                                int item = entry.getKey();
                                int quantity = entry.getValue();
                                totalItems += quantity;
                                individual.pickedItems.put(item, individual.pickedItems.get(item) - quantity);
                            }

                            individual.totalPickedItems -= totalItems;
                            individual.genome.set(orderIndex, false);
                        }
                        factor--;
                    }
                    // Try to add an order without breaking the rules
                } else {
                    List<Integer> orderIndexes = IntStream.range(0, orders.size()).boxed().collect(Collectors.toList());

                    Collections.shuffle(orderIndexes);

                    while (!orderIndexes.isEmpty() && factor > 0) {
                        int orderIndex = orderIndexes.remove(0);
                        Map<Integer, Integer> order = orders.get(orderIndex);

                        int orderSum = order.values().stream().mapToInt(Integer::intValue).sum();

                        if (orderSum + individual.totalPickedItems > waveSizeUB) {
                            continue;
                        }

                        boolean canAddOrder = true;
                        for (Map.Entry<Integer, Integer> entry : order.entrySet()) {
                            int item = entry.getKey();
                            int quantity = entry.getValue();
                            if (individual.offeredItems.getOrDefault(item, 0)
                                    - individual.pickedItems.getOrDefault(item, 0) < quantity) {
                                canAddOrder = false;
                                break;
                            }
                        }

                        if (canAddOrder) {
                            for (Map.Entry<Integer, Integer> entry : order.entrySet()) {
                                int item = entry.getKey();
                                int quantity = entry.getValue();
                                individual.pickedItems.put(item,
                                        individual.pickedItems.getOrDefault(item, 0) + quantity);
                            }

                            individual.genome.set(orderIndex, true);
                            individual.totalPickedItems += orderSum;
                        }
                        factor--;
                    }
                }
                // Mess with aisles
            } else {
                // Add a random aisle
                if (rand.nextFloat() < 0.25) {

                    while (factor > 0) {
                        int start = orders.size();
                        int end = orders.size() + aisles.size();
                        boolean anyFalse = individual.genome.nextClearBit(start) < end;

                        if (anyFalse) {

                            int aislesIndex = rand.nextInt(aisles.size());

                            while (individual.genome.get(aislesIndex + orders.size())) {
                                aislesIndex = rand.nextInt(aisles.size());
                            }

                            Map<Integer, Integer> aisle = aisles.get(aislesIndex);

                            for (Map.Entry<Integer, Integer> entry : aisle.entrySet()) {
                                int item = entry.getKey();
                                int quantity = entry.getValue();
                                individual.offeredItems.put(item,
                                        individual.offeredItems.getOrDefault(item, 0) + quantity);
                            }

                            individual.genome.set(aislesIndex + orders.size(), true);
                        }
                        factor--;
                    }
                    // Try to remove an aisle without breaking the rules
                } else {
                    List<Integer> aisleIndexes = IntStream.range(0, aisles.size())
                            .filter(i -> individual.genome.get(i + orders.size())).boxed().collect(Collectors.toList());

                    Collections.shuffle(aisleIndexes);

                    while (!aisleIndexes.isEmpty() && factor > 0) {
                        int aisleIndex = aisleIndexes.remove(0);
                        Map<Integer, Integer> aisle = aisles.get(aisleIndex);

                        boolean canRemoveAisle = true;
                        for (Map.Entry<Integer, Integer> entry : aisle.entrySet()) {
                            int item = entry.getKey();
                            int quantity = entry.getValue();
                            if (individual.pickedItems.getOrDefault(item, 0) > individual.offeredItems.get(item)
                                    - quantity) {
                                canRemoveAisle = false;
                                break;
                            }
                        }

                        if (canRemoveAisle) {
                            for (Map.Entry<Integer, Integer> entry : aisle.entrySet()) {
                                int item = entry.getKey();
                                int quantity = entry.getValue();
                                individual.offeredItems.put(item, individual.offeredItems.get(item) - quantity);
                            }

                            individual.genome.set(aisleIndex + orders.size(), false);
                        }
                        factor--;
                    }

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
    public ArrayList<Individual> crossOverPopulation(ArrayList<Individual> population, int factor) {
        Collections.shuffle(population);
        final int halfPopulationSize = population.size() / 2;

        return IntStream.range(0, halfPopulationSize).parallel().mapToObj(i -> {
            Individual parent1 = population.get(i);
            Individual parent2 = population.get(i + halfPopulationSize);

            return Arrays.asList(crossover(parent1, parent2, factor), crossover(parent2, parent1, factor));
        }).flatMap(List::stream).collect(Collectors.toCollection(ArrayList::new));
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
        ELITIST, TOURNAMENT, BEST_THEN_RANDOM
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
        } else if (selectionType == PopulationSelectionType.BEST_THEN_RANDOM) {
            List<Individual> bestIndividuals = new ArrayList<>(population.subList(0, 5));

            List<Individual> remaining = new ArrayList<>(population.subList(5, population.size()));
            Collections.shuffle(remaining);

            int required = populationSize - bestIndividuals.size();
            List<Individual> newPopulation = new ArrayList<>(populationSize);
            newPopulation.addAll(bestIndividuals);
            newPopulation.addAll(remaining.subList(0, Math.min(required, remaining.size())));

            population.clear();
            population.addAll(newPopulation);
        }
    }
}
