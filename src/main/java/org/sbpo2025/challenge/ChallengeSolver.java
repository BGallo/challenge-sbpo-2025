package org.sbpo2025.challenge;

import org.apache.commons.lang3.time.StopWatch;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.TimeUnit;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class ChallengeSolver {
    private final long MAX_RUNTIME = 60;

    protected List<Map<Integer, Integer>> orders;
    protected List<Map<Integer, Integer>> aisles;
    protected int nItems;
    protected int waveSizeLB;
    protected int waveSizeUB;

    protected final int maxPercentage = 40;
    protected final int minPercentage = 10;
    protected int currentPercentage = 10;

    protected final int maxNoImprovementIterations = 100;

    protected HashMap<Neighborhood, Double> weightedAddAislesNeighborhoods = new HashMap<>() {{
            put(new AddBestAisles(), 1.0);
    }};

    protected HashMap<Neighborhood, Double> weightedRemoveAislesNeighborhoods = new HashMap<>() {{
            put(new RemoveRandomAisles(), 1.0);
    }};

    protected HashMap<Neighborhood, Double> weightedAddOrdersNeighborhoods = new HashMap<>() {{
            put(new AddBestOrders(), 1.0);
    }};

    protected HashMap<Neighborhood, Double> weightedRemoveOrdersNeighborhoods = new HashMap<>() {{
            put(new RemoveRandomOrders(), 1.0);
    }};

    protected List<Integer> bestOrdersByItemNumber = new ArrayList<>();
    protected List<Integer> bestAislesByItemNumber = new ArrayList<>();


    public ChallengeSolver(
            List<Map<Integer, Integer>> orders, List<Map<Integer, Integer>> aisles, int nItems, int waveSizeLB, int waveSizeUB) {
        this.orders = orders;
        this.aisles = aisles;
        this.nItems = nItems;
        this.waveSizeLB = waveSizeLB;
        this.waveSizeUB = waveSizeUB;

        bestOrdersByItemNumber = IntStream.range(0, orders.size())
            .boxed()
            .sorted((i, j) -> {
                int sumI = orders.get(i).values().stream().mapToInt(Integer::intValue).sum();
                int sumJ = orders.get(j).values().stream().mapToInt(Integer::intValue).sum();
                return Integer.compare(sumJ, sumI);
            })
            .collect(Collectors.toList());

        bestAislesByItemNumber = IntStream.range(0, aisles.size())
            .boxed()
            .sorted((i, j) -> {
                int sumI = aisles.get(i).values().stream().mapToInt(Integer::intValue).sum();
                int sumJ = aisles.get(j).values().stream().mapToInt(Integer::intValue).sum();
                return Integer.compare(sumJ, sumI);
            })
            .collect(Collectors.toList());
    }

    class ALNSSolution {
        List<Integer> selectedAisles = new ArrayList<>();
        List<Integer> selectedOrders = new ArrayList<>();
        Map<Integer, Integer> itensLeftInAisles = new HashMap<>();
        double objectiveValue;
        int totalItemsPicked;

        public ALNSSolution() {
            this.objectiveValue = 0;
        }

        public ALNSSolution(List<Integer> selectedOrders, List<Integer> selectedAisles) {
            this.selectedOrders = selectedOrders;
            this.selectedAisles = selectedAisles;
        }

        public ALNSSolution(ALNSSolution other) {
            this.selectedAisles = new ArrayList<>(other.selectedAisles);
            this.selectedOrders = new ArrayList<>(other.selectedOrders);
            this.objectiveValue = other.objectiveValue;
            this.itensLeftInAisles = new HashMap<>(other.itensLeftInAisles);
            this.totalItemsPicked = other.totalItemsPicked;
        }

        public void calcItemsLeftInAisles() {
            itensLeftInAisles.clear();

            for (int aisle : selectedAisles) {
                for (Map.Entry<Integer, Integer> entry : aisles.get(aisle).entrySet()) {
                    itensLeftInAisles.merge(entry.getKey(), entry.getValue(), Integer::sum);
                }
            }

            for (int order : selectedOrders) {
                for (Map.Entry<Integer, Integer> entry : orders.get(order).entrySet()) {
                    itensLeftInAisles.merge(entry.getKey(), -entry.getValue(), Integer::sum);
                }
            }

            totalItemsPicked = selectedOrders.stream()
                    .mapToInt(o -> orders.get(o).values().stream().mapToInt(Integer::intValue).sum())
                    .sum();
        }

        public String toString() {
            return "Orders: " + selectedOrders.toString() + ", Aisles: " + selectedAisles.toString() + ", Obj: " + objectiveValue;
        }
    }

    public ChallengeSolution solve(StopWatch stopWatch) {
        ALNSSolution bestSolution = getInitialViableSolution();

        ALNSSolution currentSolution = new ALNSSolution(bestSolution);
        ALNSSolution tempSolution;
        int iteration = 0;
        int noImprovementIterations = 0;
        Neighborhood neighborhood;
        double randomFactor = 0.7;
        while (MAX_RUNTIME - stopWatch.getTime(TimeUnit.SECONDS) > 5) {
            iteration++;
            noImprovementIterations++;

            tempSolution = new ALNSSolution(currentSolution);

            //ORDER MOVES

            //Remove Order Moves
            neighborhood = selectNeighborhood(weightedRemoveOrdersNeighborhoods);
            neighborhood.move(currentSolution, currentPercentage, randomFactor);

            if (currentSolution.objectiveValue > bestSolution.objectiveValue) {
                System.out.println("New best solution at iteration " + iteration + ": " + currentSolution.objectiveValue);
                bestSolution = new ALNSSolution(currentSolution);
                noImprovementIterations = 0;

                weightedRemoveOrdersNeighborhoods.merge(neighborhood, 5.0, Double::sum);
            } else if (currentSolution.objectiveValue > tempSolution.objectiveValue) {
                tempSolution = new ALNSSolution(currentSolution);

                weightedRemoveOrdersNeighborhoods.merge(neighborhood, 2.0, Double::sum);
            }

            //Add Order Moves
            neighborhood = selectNeighborhood(weightedAddOrdersNeighborhoods);
            neighborhood.move(currentSolution, currentPercentage, randomFactor);

            if (currentSolution.objectiveValue > bestSolution.objectiveValue) {
                System.out.println("New best solution at iteration " + iteration + ": " + currentSolution.objectiveValue);
                bestSolution = new ALNSSolution(currentSolution);
                noImprovementIterations = 0;

                weightedAddOrdersNeighborhoods.merge(neighborhood, 5.0, Double::sum);
            } else if (currentSolution.objectiveValue > tempSolution.objectiveValue) {
                tempSolution = new ALNSSolution(currentSolution);

                weightedAddOrdersNeighborhoods.merge(neighborhood, 2.0, Double::sum);
            } else {
                double delta = currentSolution.objectiveValue - tempSolution.objectiveValue;
                double denominator = Math.max(1000, noImprovementIterations);
                double acceptanceProbability = Math.exp(delta / denominator);

                if (Math.random() < acceptanceProbability && currentSolution.objectiveValue > 0) {
                    tempSolution = new ALNSSolution(currentSolution);
                } else {
                    currentSolution = new ALNSSolution(tempSolution);
                }
            }

            //AISLE MOVES

            //Remove Aisle Moves
            neighborhood = selectNeighborhood(weightedRemoveAislesNeighborhoods);
            neighborhood.move(currentSolution, currentPercentage, randomFactor);

            if (currentSolution.objectiveValue > bestSolution.objectiveValue) {
                System.out.println("New best solution at iteration " + iteration + ": " + currentSolution.objectiveValue);
                bestSolution = new ALNSSolution(currentSolution);
                noImprovementIterations = 0;

                weightedRemoveAislesNeighborhoods.merge(neighborhood, 5.0, Double::sum);
            } else if (currentSolution.objectiveValue > tempSolution.objectiveValue) {
                tempSolution = new ALNSSolution(currentSolution);

                weightedRemoveAislesNeighborhoods.merge(neighborhood, 2.0, Double::sum);
            }

            //Add Aisle Moves
            neighborhood = selectNeighborhood(weightedAddAislesNeighborhoods);
            neighborhood.move(currentSolution, currentPercentage, randomFactor);

            if (currentSolution.objectiveValue > bestSolution.objectiveValue) {
                System.out.println("New best solution at iteration " + iteration + ": " + currentSolution.objectiveValue);
                bestSolution = new ALNSSolution(currentSolution);
                noImprovementIterations = 0;

                weightedAddAislesNeighborhoods.merge(neighborhood, 5.0, Double::sum);
            } else if (currentSolution.objectiveValue > tempSolution.objectiveValue) {
                tempSolution = new ALNSSolution(currentSolution);

                weightedAddAislesNeighborhoods.merge(neighborhood, 2.0, Double::sum);
            } else {
                double delta = currentSolution.objectiveValue - tempSolution.objectiveValue;
                double denominator = Math.max(1000, noImprovementIterations);
                double acceptanceProbability = Math.exp(delta / denominator);

                if (Math.random() < acceptanceProbability && currentSolution.objectiveValue > 0) {
                    tempSolution = new ALNSSolution(currentSolution);
                } else {
                    currentSolution = new ALNSSolution(tempSolution);
                }
            }

            currentPercentage = Math.min(maxPercentage,
                    minPercentage + (noImprovementIterations / (maxNoImprovementIterations / (maxPercentage - minPercentage))));

            evaporateAllWeights(); 
        }

        System.out.println("Is viable: " + isSolutionFeasible(bestSolution));

        return new ChallengeSolution(new HashSet<>(bestSolution.selectedOrders), new HashSet<>(bestSolution.selectedAisles));
    }


    public void evaporateAllWeights() {
        evaporateWeights(weightedAddAislesNeighborhoods);
        evaporateWeights(weightedRemoveAislesNeighborhoods);
        evaporateWeights(weightedAddOrdersNeighborhoods);
        evaporateWeights(weightedRemoveOrdersNeighborhoods);
    }

    public void evaporateWeights(HashMap<Neighborhood, Double> neighborhoods) {
        for (Map.Entry<Neighborhood, Double> entry : neighborhoods.entrySet()) {
            neighborhoods.put(entry.getKey(), Math.max(0.1, entry.getValue() * 0.9));
        }
    }

    public Neighborhood selectNeighborhood(HashMap<Neighborhood, Double> neighborhoods) {
        double totalWeight = neighborhoods.values().stream().mapToDouble(Double::doubleValue).sum();
        double r = Math.random() * totalWeight;
        double cumulative = 0.0;

        for (Map.Entry<Neighborhood, Double> entry : neighborhoods.entrySet()) {
            cumulative += entry.getValue();
            if (r <= cumulative) {
                return entry.getKey();
            }
        }

        return neighborhoods.keySet().iterator().next();
    }

    private ALNSSolution getInitialViableSolution() {
        List<Integer> selectedOrders = new ArrayList<>();
        List<Integer> selectedAisles = new ArrayList<>();
        int totalItems = 0;
        HashMap<Integer, Integer> aisleCapacities = new HashMap<>();

        for (int i = 0; i < aisles.size(); i++) {
            selectedAisles.add(i);
            for (Map.Entry<Integer, Integer> entry : aisles.get(i).entrySet()) {
                aisleCapacities.put(entry.getKey(), entry.getValue() + aisleCapacities.getOrDefault(entry.getKey(), 0));
            }
        }

        for (int i = 0; i < orders.size(); i++) {
            int orderTotalItems = orders.get(i).values().stream().mapToInt(Integer::intValue).sum();
            if (totalItems + orderTotalItems > waveSizeUB) {
                continue;
            }

            boolean canFulfill = true;
            for (Map.Entry<Integer, Integer> entry : orders.get(i).entrySet()) {
                int itemIndex = entry.getKey();
                int itemQuantity = entry.getValue();
                if (aisleCapacities.getOrDefault(itemIndex, 0) < itemQuantity) {
                    canFulfill = false;
                    break;
                }
            }

            if (canFulfill) {
                selectedOrders.add(i);
                totalItems += orderTotalItems;

                for (Map.Entry<Integer, Integer> entry : orders.get(i).entrySet()) {
                    int itemIndex = entry.getKey();
                    int itemQuantity = entry.getValue();
                    aisleCapacities.put(itemIndex, aisleCapacities.get(itemIndex) - itemQuantity);
                }

                if (totalItems >= waveSizeLB) {
                    break;
                }
            }
        }

        ALNSSolution initialSolution = new ALNSSolution(selectedOrders, selectedAisles);
        initialSolution.objectiveValue = computeObjectiveFunction(initialSolution);
        initialSolution.calcItemsLeftInAisles();

        return initialSolution;
    }

    class Neighborhood {

        public void move(ALNSSolution currentSolution, double percentage, double randomFactor) {

            if (!isSolutionFeasible(currentSolution)) {
                currentSolution.objectiveValue = 0.0;
                return;
            }
            double quality = computeObjectiveFunction(currentSolution);
            currentSolution.objectiveValue = quality;
        }
    }

    class RemoveRandomAisles extends Neighborhood {
        @Override
        public void move(ALNSSolution currentSolution, double percentage, double randomFactor) {
            if (currentSolution.selectedAisles.size() <= 1) {
                return;
            }

            Collections.shuffle(currentSolution.selectedAisles);

            int nToRemove = (int) Math.ceil(currentSolution.selectedAisles.size() * (percentage / 100.0));
            nToRemove = Math.max(1, nToRemove);

            for (int i = 0; i < nToRemove; i++) {
                int aisle = currentSolution.selectedAisles.remove(0);

                for (Map.Entry<Integer, Integer> entry : aisles.get(aisle).entrySet()) {
                    currentSolution.itensLeftInAisles.merge(entry.getKey(), -entry.getValue(), Integer::sum);
                }
            }

            super.move(currentSolution, percentage, randomFactor);
        }
    }

    class RemoveRandomOrders extends Neighborhood {
        @Override
        public void move(ALNSSolution currentSolution, double percentage, double randomFactor) {
            Collections.shuffle(currentSolution.selectedOrders);

            int nToRemove = (int) Math.ceil(currentSolution.selectedOrders.size() * (percentage / 100.0));
            nToRemove = Math.max(1, nToRemove);

            for (int i = 0; i < nToRemove; i++) {
                int order = currentSolution.selectedOrders.remove(0);

                for (Map.Entry<Integer, Integer> entry : orders.get(order).entrySet()) {
                    currentSolution.itensLeftInAisles.merge(entry.getKey(), entry.getValue(), Integer::sum);
                    currentSolution.totalItemsPicked -= entry.getValue();
                }
            }

            super.move(currentSolution, percentage, randomFactor);
        }
    }

    class AddBestOrders extends Neighborhood {
        @Override
        public void move(ALNSSolution currentSolution, double percentage, double randomFactor) {
            Set<Integer> selectedSet = new HashSet<>(currentSolution.selectedOrders);
            List<Integer> candidateOrders = bestOrdersByItemNumber.stream()
                .filter(o -> !selectedSet.contains(o))
                .collect(Collectors.toList());

            if (candidateOrders.isEmpty()) {
                return;
            }

            for (int order : candidateOrders) {
                Map<Integer, Integer> orderItems = orders.get(order);
                int orderTotalItems = orderItems.values().stream().mapToInt(Integer::intValue).sum();

                if (currentSolution.totalItemsPicked + orderTotalItems > waveSizeUB) {
                    continue;
                }

                boolean canFulfill = true;
                for (Map.Entry<Integer, Integer> entry : orderItems.entrySet()) {
                    int itemIndex = entry.getKey();
                    int itemQuantity = entry.getValue();
                    if (currentSolution.itensLeftInAisles.getOrDefault(itemIndex, 0) < itemQuantity) {
                        canFulfill = false;
                        break;
                    }
                }

                if (canFulfill && Math.random() < randomFactor) {
                    currentSolution.selectedOrders.add(order);
                    currentSolution.totalItemsPicked += orderTotalItems;
                    for (Map.Entry<Integer, Integer> entry : orderItems.entrySet()) {
                        int itemIndex = entry.getKey();
                        int itemQuantity = entry.getValue();
                        currentSolution.itensLeftInAisles.merge(itemIndex, -itemQuantity, Integer::sum);
                    }
                }
            }

            super.move(currentSolution, percentage, randomFactor);
        }
    }

    class AddBestAisles extends Neighborhood {
        @Override
        public void move(ALNSSolution currentSolution, double percentage, double randomFactor) {
            Set<Integer> selectedSet = new HashSet<>(currentSolution.selectedAisles);
            List<Integer> candidateAisles = bestAislesByItemNumber.stream()
                .filter(a -> !selectedSet.contains(a))
                .collect(Collectors.toList());

            if (candidateAisles.isEmpty()) {
                return;
            }

            boolean valid = false;

            for (int aisle : candidateAisles) {
                if (Math.random() < randomFactor) {
                    currentSolution.selectedAisles.add(aisle);
                    for (Map.Entry<Integer, Integer> entry : aisles.get(aisle).entrySet()) {
                        currentSolution.itensLeftInAisles.merge(entry.getKey(), entry.getValue(), Integer::sum);
                    }
                }

                if (valid && Math.random() < randomFactor) {
                    break;
                } else if (currentSolution.itensLeftInAisles.values().stream().allMatch(v -> v >= 0)) {
                    valid = true;
                }
            }

            super.move(currentSolution, percentage, randomFactor);
        }
    }
    /*
     * Get the remaining time in seconds
     */
    protected long getRemainingTime(StopWatch stopWatch) {
        return Math.max(
                TimeUnit.SECONDS.convert(MAX_RUNTIME - stopWatch.getTime(TimeUnit.MILLISECONDS), TimeUnit.MILLISECONDS),
                0);
    }

    protected boolean isSolutionFeasible(ALNSSolution challengeSolution) {
        List<Integer> selectedOrders = challengeSolution.selectedOrders;
        List<Integer> visitedAisles = challengeSolution.selectedAisles;
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
        if (totalUnits < waveSizeLB || totalUnits > waveSizeUB) {
            return false;
        }

        // Check if the units picked do not exceed the units available
        for (int i = 0; i < nItems; i++) {
            if (totalUnitsPicked[i] > totalUnitsAvailable[i]) {
                return false;
            }
        }

        return true;
    }

    protected double computeObjectiveFunction(ALNSSolution challengeSolution) {
        List<Integer> selectedOrders = challengeSolution.selectedOrders;
        List<Integer> visitedAisles = challengeSolution.selectedAisles;
        if (selectedOrders == null || visitedAisles == null || selectedOrders.isEmpty() || visitedAisles.isEmpty()) {
            return 0.0;
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
        return (double) totalUnitsPicked / numVisitedAisles;
    }
}
