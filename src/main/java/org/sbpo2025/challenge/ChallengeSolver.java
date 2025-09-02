package org.sbpo2025.challenge;

import org.apache.commons.lang3.time.StopWatch;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.TimeUnit;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class ChallengeSolver {
    private final long MAX_RUNTIME = 60; // 1 minute

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

    class Ant {
        List<Integer> selectedAisles;
        List<Integer> selectedOrders;
        List<Double> heuristicValues;
        List<List<Integer>> aislesForThisOrder;
        List<Double> probabilityForThisOrder;
        Double currentHeuristicValue;
        int currentIntervalIndex;
        int currentOrderIndex;

        public Ant(List<Integer> selectedAisles, List<Integer> selectedOrders) {
            this.selectedAisles = selectedAisles;
            this.selectedOrders = selectedOrders;
        }

        public boolean chooseNextOrder(double[][] T, List<Integer> itemsPercentages, int alpha, int beta) {
            int currentTotalItems = selectedOrders.stream()
                .mapToInt(o -> orders.get(o).values().stream().mapToInt(Integer::intValue).sum())
                .sum();

            int intervalIndex = 0;
            for (int i = 1; i < itemsPercentages.size(); i++) {
                if (itemsPercentages.get(i) <= currentTotalItems) {
                    intervalIndex = i-1;
                } else {
                    break;
                }
            }

            updateHeuristicValues(intervalIndex, T, alpha, beta);

            double totalProbability = probabilityForThisOrder.stream().mapToDouble(Double::doubleValue).sum();

            if (totalProbability == 0.0) {
                return false; 
            }

            double r = Math.random() * totalProbability;

            double cumulativeProbability = 0.0;
            for (int i = 0; i < probabilityForThisOrder.size(); i++) {
                cumulativeProbability += probabilityForThisOrder.get(i);
                if (cumulativeProbability >= r) {
                    selectedOrders.add(i);
                    selectedAisles = aislesForThisOrder.get(i);
                    currentHeuristicValue = heuristicValues.get(i);
                    currentIntervalIndex = intervalIndex;
                    currentOrderIndex = i;
                    break;
                }
            }



            return true;
        }

        public void updateHeuristicValues(int itemIntervalIndex, double[][] T, int alpha, int beta) {
            for (int i = 0; i < orders.size(); i++) {

                if (selectedOrders.contains(i)) {
                    heuristicValues.set(i, 0.0);
                    probabilityForThisOrder.set(i, 0.0);
                    continue;
                }

                int totalItems = selectedOrders.stream()
                    .mapToInt(o -> orders.get(o).values().stream().mapToInt(Integer::intValue).sum())
                    .sum();
                
                totalItems += orders.get(i).values().stream().mapToInt(Integer::intValue).sum();

                if (totalItems > waveSizeUB) {
                    heuristicValues.set(i, 0.0);
                    probabilityForThisOrder.set(i, 0.0);
                    continue;
                }

                List<Integer> candidateAisles = selectAislesForOrders(i);
                if (candidateAisles == null) {
                    heuristicValues.set(i, 0.0);
                    probabilityForThisOrder.set(i, 0.0);
                    continue;
                }

                int numAisles = candidateAisles.size();

                Double heuristicValue = (double) totalItems / numAisles;

                heuristicValues.set(i, heuristicValue);
                aislesForThisOrder.set(i, candidateAisles);

                double pheromoneComp = Math.pow(T[itemIntervalIndex][i], alpha);
                double heuristicComp = Math.pow(heuristicValues.get(i), beta);
                probabilityForThisOrder.set(i, pheromoneComp * heuristicComp);
            }
        }

        public List<Integer> selectAislesForOrders(int candidateOrder) {
            List<Integer> candidateOrders = new ArrayList<>(selectedOrders);
            candidateOrders.add(candidateOrder);

            List<Integer> candidateAisles = new ArrayList<>(selectedAisles);

            List<Integer> possibleAisles = IntStream.range(0, aisles.size())
                .filter(i -> !selectedAisles.contains(i))
                .boxed()
                .collect(Collectors.toList());

            for (int i = 0; i < nItems; i++) {
                final int index = i;
                possibleAisles.removeAll(candidateAisles);

                int itensNeeded = candidateOrders.stream()
                    .mapToInt(order -> orders.get(order).getOrDefault(index, 0))
                    .sum();

                if (itensNeeded == 0) {
                    continue; 
                }

                int itensAvailable = candidateAisles.stream()
                    .mapToInt(aisle -> aisles.get(aisle).getOrDefault(index, 0))
                    .sum();

                int diff = itensNeeded - itensAvailable;

                if (diff > 0) {
                   possibleAisles.sort((a1, a2) -> aisles.get(a2).getOrDefault(index, 0) - aisles.get(a1).getOrDefault(index, 0));

                   for (int aisle : possibleAisles) {
                       int available = aisles.get(aisle).getOrDefault(i, 0);
                       if (available > 0) {
                           candidateAisles.add(aisle);
                           diff -= available;
                           if (diff <= 0) {
                               break;
                           }
                       }
                   }
                }

                if (diff > 0) {
                    return null; 
                }
            }

            return candidateAisles;
        }

        public ChallengeSolution getCurrentSolution() {
            return new ChallengeSolution(new HashSet<>(selectedOrders), new HashSet<>(selectedAisles));
        }

        public void updatePheromones(double[][] T, Double q) {
            if (currentHeuristicValue == null) {
                return;
            }
            T[currentIntervalIndex][currentOrderIndex] += q * currentHeuristicValue;
        }
    }

    public ChallengeSolution solve(StopWatch stopWatch) {
        ChallengeSolution bestSolution = null;
        Double bestValue = Double.NEGATIVE_INFINITY;

        int totalItems = orders.stream()
                .mapToInt(order -> order.values().stream().mapToInt(Integer::intValue).sum())
                .sum();

        int antNumber = 20;
        int alpha = 1;
        int beta = 2;
        double evaporationRate = 0.1;
        Double q = totalItems / 20.0;

        List<Integer> itemPercentages = IntStream.range(0, 20)
            .map(i -> (int) Math.round(totalItems * i / 20.0))
            .boxed()
            .collect(Collectors.toList());

        int nIntervals = 20;

        double[][] T = new double[nIntervals][orders.size()];

        for (int i = 0; i < nIntervals; i++) {
            for (int j = 0; j < orders.size(); j++) {
                T[i][j] = 1.0;
            }
        }

        int iteration = 0;
        while (stopWatch.getTime(TimeUnit.SECONDS) < MAX_RUNTIME - 1) {
            iteration++;
            List<Ant> ants = initAnts(antNumber);

            for (Ant ant : ants) {
                while (ant.chooseNextOrder(T, itemPercentages, alpha, beta)) { }

                if (ant.currentHeuristicValue != null && ant.currentHeuristicValue > bestValue) {
                    bestValue = ant.currentHeuristicValue;
                    bestSolution = ant.getCurrentSolution();
                }
            }

            for (int i = 0; i < nIntervals; i++) {
                for (int j = 0; j < orders.size(); j++) {
                    T[i][j] = (1 - evaporationRate) * T[i][j];
                }
            }

            for (Ant ant : ants) {
                ant.updatePheromones(T, q);
            }

            System.out.println("Current Best Value for iteration " + iteration + ": " + bestValue);
        }

        return bestSolution;
    }

    private List<Ant> initAnts(int antNumber) {
        List<Ant> ants = new ArrayList<>();
        for (int k = 0; k < antNumber; k++) {

            List<Integer> initialSelectedOrders = new ArrayList<>();
            List<Integer> initialSelectedAisles = new ArrayList<>();

            Ant ant = new Ant(initialSelectedAisles, initialSelectedOrders);

            ant.heuristicValues = new ArrayList<>(Collections.nCopies(orders.size(), 0.0));
            ant.probabilityForThisOrder = new ArrayList<>(Collections.nCopies(orders.size(), 0.0));
            ant.aislesForThisOrder = new ArrayList<>();
            for (int i = 0; i < orders.size(); i++) {
                ant.aislesForThisOrder.add(new ArrayList<>());
            }

            ants.add(ant);
        }

        return ants;
    }

    /*
    * Get the remaining time in seconds
    */
    protected long getRemainingTime(StopWatch stopWatch) {
        return Math.max(
                TimeUnit.SECONDS.convert(MAX_RUNTIME - stopWatch.getTime(TimeUnit.MILLISECONDS), TimeUnit.MILLISECONDS),
                0);
    }

    protected boolean isSolutionFeasible(ChallengeSolution challengeSolution) {
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

    protected double computeObjectiveFunction(ChallengeSolution challengeSolution) {
        Set<Integer> selectedOrders = challengeSolution.orders();
        Set<Integer> visitedAisles = challengeSolution.aisles();
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
