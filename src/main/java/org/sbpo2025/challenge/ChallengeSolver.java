package org.sbpo2025.challenge;

import org.apache.commons.lang3.time.StopWatch;
import org.apache.commons.lang3.tuple.Pair;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class ChallengeSolver {
    private final long MAX_RUNTIME = 60; // 1 minute
    private final int nThreads = 4;

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

    class AntResult {
        ChallengeSolution solution;
        double value;

        public AntResult(ChallengeSolution solution, double value) {
            this.solution = solution;
            this.value = value;
        }
    }

    class Ant {
        Set<Integer> selectedAisles;
        Set<Integer> selectedOrders;
        List<Double> heuristicValues;
        List<Set<Integer>> aislesForThisOrder;
        List<Double> probabilityForThisOrder;
        Double currentHeuristicValue;
        Double epsilon;
        List<Pair<Integer, Integer>> chosenOrders = new ArrayList<>();

        public Ant(Set<Integer> selectedAisles, Set<Integer> selectedOrders, Double epsilon) {
            this.selectedAisles = selectedAisles;
            this.selectedOrders = selectedOrders;
            this.epsilon = epsilon;
        }

        public boolean chooseNextOrder(double[][] T, List<Double> itemsPercentages, double alpha, double beta) {
            Random rng = new Random();

            int currentTotalItems = selectedOrders.stream()
                .mapToInt(o -> orders.get(o).values().stream().mapToInt(Integer::intValue).sum())
                .sum();

            int intervalIndex = 0;
            for (int i = 1; i < itemsPercentages.size(); i++) {
                if (currentTotalItems < itemsPercentages.get(i)) {
                    intervalIndex = i-1;
                    break;
                }
            }

            updateHeuristicValues(intervalIndex, T, alpha, beta);

            double totalProbability = probabilityForThisOrder.stream().mapToDouble(Double::doubleValue).sum();

            if (totalProbability == 0.0) {
                return false; 
            }

            //Explorer Decision
            if (rng.nextDouble() < epsilon) {

                List<Integer> candidates = IntStream.range(0, orders.size())
                    .filter(i -> probabilityForThisOrder.get(i) > 0.0)
                    .boxed()
                    .collect(Collectors.toList());

                int chosenIndex = candidates.get(rng.nextInt(candidates.size()));
                selectedOrders.add(chosenIndex);
                selectedAisles = aislesForThisOrder.get(chosenIndex);
                currentHeuristicValue = heuristicValues.get(chosenIndex);
                chosenOrders.add(Pair.of(intervalIndex, chosenIndex));
            
            // Heuristic Ant
            } else {

                double r = rng.nextDouble() * totalProbability;

                double cumulativeProbability = 0.0;
                for (int i = 0; i < probabilityForThisOrder.size(); i++) {
                    cumulativeProbability += probabilityForThisOrder.get(i);
                    if (cumulativeProbability >= r) {
                        selectedOrders.add(i);
                        selectedAisles = aislesForThisOrder.get(i);
                        currentHeuristicValue = heuristicValues.get(i);
                        chosenOrders.add(Pair.of(intervalIndex, i));
                        break;
                    }
                }
            }



            return true;
        }

        public void updateHeuristicValues(int itemIntervalIndex, double[][] T, double alpha, double beta) {
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

                Set<Integer> candidateAisles = selectAislesForOrders(i);
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

        public Set<Integer> selectAislesForOrders(int candidateOrder) {
            List<Integer> candidateOrders = new ArrayList<>(selectedOrders);
            candidateOrders.add(candidateOrder);

            HashSet<Integer> candidateAisles = new HashSet<>(selectedAisles);

            List<Integer> possibleAisles = IntStream.range(0, aisles.size())
                .filter(i -> !candidateAisles.contains(i))
                .boxed()
                .collect(Collectors.toList());

            for (int i = 0; i < nItems; i++) {
                final int index = i;

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
                    int possibleAislesSize = possibleAisles.size();

                    for (int j = 0; j < possibleAislesSize; j++) {
                        int aisle = possibleAisles.get((int) (Math.random() * possibleAislesSize));
                        int available = aisles.get(aisle).getOrDefault(i, 0);
                        if (available > 0) {
                            candidateAisles.add(aisle);
                            possibleAisles.remove((Integer) aisle);
                            possibleAislesSize--;
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
            int options = T[0].length;
            for (Pair<Integer, Integer> p : chosenOrders) {
                int intervalIndex = p.getLeft();
                int orderIndex = p.getRight();
                T[intervalIndex][orderIndex] += Math.min(options - 1, q);
            }
        }
    }

    public ChallengeSolution solve(StopWatch stopWatch, int antNumber, double alpha, double beta, double evaporationRate, double epsilon) throws InterruptedException, ExecutionException {
        ChallengeSolution bestSolution = null;
        Double bestValue = Double.NEGATIVE_INFINITY;

        int totalItems = orders.stream()
                .mapToInt(order -> order.values().stream().mapToInt(Integer::intValue).sum())
                .sum();

        /* int antNumber = 20; */
        int antsPerThread = antNumber / nThreads;
        /* int alpha = 1;
        int beta = 2;
        double evaporationRate = 0.5; */
        Double q = 1.0;
        /* Double epsilon = 0.10; */

        List<Double> itemPercentages = IntStream.range(0, 20)
            .mapToDouble(i -> totalItems * i / 20.0)
            .boxed()
            .collect(Collectors.toList());

        int nIntervals = 20;

        double[][] T = new double[nIntervals][orders.size()];

        for (int i = 0; i < nIntervals; i++) {
            for (int j = 0; j < orders.size(); j++) {
                T[i][j] = 1.0;
            }
        }

        ExecutorService executor = Executors.newFixedThreadPool(nThreads);

        int iteration = 0;
        int nIterationsWithoutImprovement = 0;
        while (stopWatch.getTime(TimeUnit.SECONDS) < MAX_RUNTIME - 1) {
            List<Future<AntResult>> futures = new ArrayList<>();

            iteration++;
            nIterationsWithoutImprovement++;

            List<Ant> ants = initAnts(antNumber, epsilon);

            for (int t = 0; t < nThreads; t++) {
                final int start = t * antsPerThread;
                final int end = (t == nThreads - 1) ? ants.size() : start + antsPerThread;

                futures.add(executor.submit(() -> {
                    AntResult localBest = null;

                    for (int i = start; i < end; i++) {
                        Ant ant = ants.get(i);
                        while (ant.chooseNextOrder(T, itemPercentages, alpha, beta)) { }

                        if (localBest == null || ant.currentHeuristicValue > localBest.value) {
                            localBest = new AntResult(ant.getCurrentSolution(), ant.currentHeuristicValue);
                        }
                    }

                    return localBest;
                }));
            }

            for (Future<AntResult> f : futures) {
                AntResult r = f.get();
                if (r != null && (bestSolution == null || r.value > bestValue)) {
                    bestSolution = r.solution;
                    bestValue = r.value;
                    nIterationsWithoutImprovement = 0;
                }
            }

            for (int i = 0; i < nIntervals; i++) {
                for (int j = 0; j < orders.size(); j++) {
                    T[i][j] = Math.max(1.0, (1 - evaporationRate) * T[i][j]);
                }
            }

            for (Ant ant : ants) {
                ant.updatePheromones(T, q);
            }

            if (nIterationsWithoutImprovement >= 10) {
                rainEvent(T);
                nIterationsWithoutImprovement = 0;
            }

            /* System.out.println("Iteration " + iteration + ": Best Value = " + bestValue + ", Time Elapsed = " + stopWatch.getTime(TimeUnit.SECONDS) + "s"); */
        }

        /* System.out.println("Is Solution Feasible? " + isSolutionFeasible(bestSolution)); */
        System.out.println(bestValue);

        executor.shutdown();

        return bestSolution;
    }

    private void rainEvent(double[][] T) {
        for (int i = 0; i < T.length; i++) {
            for (int j = 0; j < T[i].length; j++) {
                T[i][j] = Math.max(1.0, T[i][j] * 0.5);
            }
        }
    }

    private List<Ant> initAnts(int antNumber, Double epsilon) {
        List<Ant> ants = new ArrayList<>();
        for (int k = 0; k < antNumber; k++) {

            HashSet<Integer> initialSelectedOrders = new HashSet<>();
            HashSet<Integer> initialSelectedAisles = new HashSet<>();

            Ant ant = new Ant(initialSelectedAisles, initialSelectedOrders, epsilon);

            ant.heuristicValues = new ArrayList<>(Collections.nCopies(orders.size(), 0.0));
            ant.probabilityForThisOrder = new ArrayList<>(Collections.nCopies(orders.size(), 0.0));
            ant.aislesForThisOrder = new ArrayList<>();
            for (int i = 0; i < orders.size(); i++) {
                ant.aislesForThisOrder.add(new HashSet<>());
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
